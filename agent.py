from langgraph.graph import StateGraph, END, START
from shared_store import url_time
import time
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import ToolNode
from tools import (
    get_rendered_html,
    download_file,
    post_request,
    run_code,
    add_dependencies,
    ocr_image_tool,
    transcribe_audio,
    encode_image_to_base64,
    read_file,
    take_screenshot,
)
from typing import TypedDict, Annotated, List
from langchain_core.messages import trim_messages, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv

load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

RECURSION_LIMIT = 5000
MAX_TOKENS = 60000


# -------------------------------------------------
# STATE
# -------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


TOOLS = [
    run_code,
    get_rendered_html,
    download_file,
    post_request,
    add_dependencies,
    ocr_image_tool,
    transcribe_audio,
    encode_image_to_base64,
    read_file,
    take_screenshot,
]


# -------------------------------------------------
# LLM INIT
# -------------------------------------------------
rate_limiter = InMemoryRateLimiter(
    requests_per_second=10 / 60, check_every_n_seconds=0.1, max_bucket_size=10
)

llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.5-flash",
    rate_limiter=rate_limiter,
    api_key=os.getenv("GOOGLE_API_KEY"),
).bind_tools(TOOLS)


# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""
You are an advanced autonomous agent designed to solve complex data science and automation tasks.

Your capabilities:
1.  **Analyze**: You can read files, extract text, and inspect code.
2.  **Think**: You plan your actions, decompose problems, and reason about failures.
3.  **Execute**: You use tools to run code, fetch URLs, and process data.

Your job is to:
1.  **Explore**: Load the quiz page from the given URL. Extract instructions and the submit endpoint.
2.  **Plan**: Break down the problem. what tools do you need? Do you need to write a script?
3.  **Solve**:
    -   For **Simple** tasks (text extraction, git): Use the direct tools.
    -   For **Visual** tasks:
        -   **Images**: Use `encode_image_to_base64` to "see" the image.
        -   **Canvas/Hidden Text**: If you see a `<canvas>` element or suspect text is rendered visually, use `take_screenshot` to save it as an image, then use `ocr_image_tool(filename)` to read it.
    -   For **Multimedia/Documents** (Video, PDF, ZIP): **WRITE PYTHON CODE**.
        -   **Video**: Use `cv2` (opencv-python) to extract frames, then use `encode_image_to_base64`.
        -   **PDF**: Use `pypdf` or `pdfplumber` to extract text/images.
        -   **Archives**: Use `zipfile` or `tarfile` to extract.
    -   For **Complex** tasks (F1 score, sorting, large data): **WRITE PYTHON CODE**. Do not try to do math in your head. Write a script to parse the file and calculate the result.
    -   If you see 429 or 5xx errors, **WAIT** and retry.
4.  **Submit**: POST the answer to the submit endpoint.

**CRITICAL RULES**:
-   **Chain of Thought**: Before using a tool, explain your reasoning. "I need to calculate the F1 score, so I will write a script..."
-   **Code Execution**: When writing python code, assume you are in a standard environment. You can install packages with `add_dependencies` if needed (e.g. `scikit-learn`, `pandas`, `numpy`).
-   **Files**: You can read local files using `read_file`.
-   **Time**: You have a limited time. If you are stuck, try a different approach immediately.
-   **Credentials**:
    email = {EMAIL}
    secret = {SECRET}

**Multimodal Instructions**:
-   If you need to analyze an image, use `encode_image_to_base64` to get the base64 string. 
-   If the tool output is an image URL or path, use your vision tools to process it.

**TERMINATION**:
-   When you have successfully submitted the answer and received a verification message, you **MUST** output the single word: "END" to finish the task. Do not say anything else. Just "END".
"""


# -------------------------------------------------
# NEW NODE: HANDLE MALFORMED JSON
# -------------------------------------------------
def handle_malformed_node(state: AgentState):
    """
    If the LLM generates invalid JSON, this node sends a correction message
    so the LLM can try again.
    """
    print("--- DETECTED MALFORMED JSON. ASKING AGENT TO RETRY ---")
    return {
        "messages": [
            {
                "role": "user",
                "content": "SYSTEM ERROR: Your last tool call was Malformed (Invalid JSON). Please rewrite the code and try again. Ensure you escape newlines and quotes correctly inside the JSON.",
            }
        ]
    }


# -------------------------------------------------
# AGENT NODE
# -------------------------------------------------
def agent_node(state: AgentState):
    # --- TIME HANDLING START ---
    cur_time = time.time()
    cur_url = os.getenv("url")

    # SAFE GET: Prevents crash if url is None or not in dict
    prev_time = url_time.get(cur_url)
    offset = os.getenv("offset", "0")

    if prev_time is not None:
        prev_time = float(prev_time)
        # Timeout check disabled as per user request (24 questions requires > 10m)
        # if diff >= 600 or (offset != "0" and (cur_time - float(offset)) > 300):
        #     print(
        #         f"Timeout exceeded ({diff}s) — instructing LLM to purposely submit wrong answer."
        #     )

        #     fail_instruction = f"""
        #     You have exceeded the time limit for this task.
        #     Immediately call the `post_request` tool and submit a WRONG answer for the CURRENT quiz.
        #     USE THIS URL: {cur_url}
        #     """
        fail_instruction = ""

        # Using HumanMessage (as you correctly implemented)
        fail_msg = HumanMessage(content=fail_instruction)

        # We invoke the LLM immediately with this new instruction
        result = llm.invoke(state["messages"] + [fail_msg])
        return {"messages": [result]}
    # --- TIME HANDLING END ---

    trimmed_messages = trim_messages(
        messages=state["messages"],
        max_tokens=MAX_TOKENS,
        strategy="last",
        include_system=True,
        start_on="human",
        token_counter=len,  # Use message count or simple length strategy
    )

    # Better check: Does it have a HumanMessage?
    has_human = any(msg.type == "human" for msg in trimmed_messages)

    if not has_human:
        print("WARNING: Context was trimmed too far. Injecting state reminder.")
        # We remind the agent of the current URL from the environment
        current_url = os.getenv("url", "Unknown URL")
        reminder = HumanMessage(
            content=f"Context cleared due to length. Continue processing URL: {current_url}"
        )

        # We append this to the trimmed list (temporarily for this invoke)
        trimmed_messages.append(reminder)
    # ----------------------------------------

    print(f"--- INVOKING AGENT (Context: {len(trimmed_messages)} items) ---")

    result = llm.invoke(trimmed_messages)

    # --- VERBOSE LOGGING FOR USER ---
    print(f"\n🤖 AGENT THOUGHTS:\n{result.content}\n")
    if result.tool_calls:
        print(f"🛠️  AGENT TOOLS: {[t['name'] for t in result.tool_calls]}\n")
    # --------------------------------

    return {"messages": [result]}


# -------------------------------------------------
# ROUTE LOGIC (UPDATED FOR MALFORMED CALLS)
# -------------------------------------------------
def route(state):
    last = state["messages"][-1]

    # 1. CHECK FOR MALFORMED FUNCTION CALLS
    if "finish_reason" in last.response_metadata:
        if last.response_metadata["finish_reason"] == "MALFORMED_FUNCTION_CALL":
            return "handle_malformed"

    # 2. CHECK FOR VALID TOOLS
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        print("Route → tools")
        return "tools"

    # 3. CHECK FOR END
    content = getattr(last, "content", None)
    if isinstance(content, str):
        if "END" in content or "Task completed" in content:
            return END

    if isinstance(content, list) and len(content) and isinstance(content[0], dict):
        text = content[0].get("text", "")
        if "END" in text or "Task completed" in text:
            return END

    print("Route → agent")
    return "agent"


# -------------------------------------------------
# GRAPH
# -------------------------------------------------
graph = StateGraph(AgentState)

# Add Nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_node("handle_malformed", handle_malformed_node)  # Add the repair node

# Add Edges
graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_edge("handle_malformed", "agent")  # Retry loop

# Conditional Edges
graph.add_conditional_edges(
    "agent",
    route,
    {
        "tools": "tools",
        "agent": "agent",
        "handle_malformed": "handle_malformed",  # Map the new route
        END: END,
    },
)

app = graph.compile()


# -------------------------------------------------
# RUNNER
# -------------------------------------------------
def run_agent(url: str):
    # system message is seeded ONCE here
    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url},
    ]

    app.invoke(
        {"messages": initial_messages}, config={"recursion_limit": RECURSION_LIMIT}
    )

    print("Tasks completed successfully!")

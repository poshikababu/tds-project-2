import logging
import json
import os
from urllib.parse import urljoin
import httpx
from playwright.async_api import async_playwright
from openai import OpenAI


logger = logging.getLogger(__name__)

# --- Tools ---


async def read_page(url: str, page=None):
    """
    Navigates to the URL and returns the text content.
    If it's a file (CSV, Audio, PDF), it downloads it and returns a summary or the content.
    """
    logger.info(f"Reading {url}")

    # Check Content-Type first
    try:
        async with httpx.AsyncClient() as client:
            head_resp = await client.head(url, follow_redirects=True, timeout=10)
            content_type = head_resp.headers.get("content-type", "").lower()

            # If it's not HTML, treat as file download
            if "text/html" not in content_type:
                logger.info(
                    f"URL {url} has content-type {content_type}. Downloading directly."
                )
                resp = await client.get(url, follow_redirects=True)
                return f"File content ({len(resp.content)} bytes). Type: {content_type}"
    except Exception as e:
        logger.warning(
            f"HEAD request failed for {url}: {e}. Proceeding with Playwright."
        )

    if not page:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                await page.goto(url)
                await page.wait_for_load_state("networkidle")
                content = await page.evaluate("document.body.innerText")
                logger.info(
                    f"Read page content ({len(content)} chars): {content[:200]}..."
                )
            except Exception as e:
                logger.warning(f"Navigation failed: {e}. Trying httpx fallback.")
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, follow_redirects=True)
                    content = f"Navigation failed, fell back to raw request. Content length: {len(resp.content)}"
            finally:
                await browser.close()
            return content

    logger.info(f"Navigating to {url}")
    try:
        await page.goto(url)
        await page.wait_for_load_state("networkidle")

        # Extract content with links
        content = await page.evaluate("""() => {
            let text = "";
            document.querySelectorAll("body *").forEach(el => {
                if (el.tagName === "A" && el.href) {
                    text += ` [${el.innerText}](${el.href}) `;
                } else if (el.childNodes.length === 1 && el.childNodes[0].nodeType === Node.TEXT_NODE) {
                    text += el.innerText + " ";
                }
            });
            // Fallback if the above is too messy, just get body text and append links
            if (text.length < 50) {
                 return document.body.innerText + "\\n\\nLinks:\\n" + Array.from(document.querySelectorAll("a")).map(a => `[${a.innerText}](${a.href})`).join("\\n");
            }
            return document.body.innerText + "\\n\\nLinks:\\n" + Array.from(document.querySelectorAll("a")).map(a => `[${a.innerText}](${a.href})`).join("\\n");
        }""")

        return content

    except Exception as e:
        if "Download is starting" in str(e):
            logger.info("Download detected during navigation. Fetching with httpx.")
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, follow_redirects=True)
                return f"File downloaded. Size: {len(resp.content)} bytes."
        raise e


async def run_python(code: str):
    """
    Executes the given Python code and returns the output (stdout) or error.
    """
    import io
    import sys

    # Capture stdout
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()

    local_scope = {}
    try:
        # We wrap in a function to allow return statements if needed, but exec works best with print
        exec(code, {}, local_scope)
        output = redirected_output.getvalue()
        return f"Output:\n{output}"
    except Exception as e:
        logger.error(f"Python Error: {e}")
        return f"Error: {e}"
    finally:
        sys.stdout = old_stdout


async def transcribe_audio(file_url: str):
    """
    Downloads an audio file and transcribes it using SpeechRecognition.
    """
    logger.info(f"Transcribing audio from {file_url}")
    import speech_recognition as sr
    from pydub import AudioSegment
    import tempfile

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(file_url)
            resp.raise_for_status()
            audio_data = resp.content

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            tmp_mp3.write(audio_data)
            tmp_mp3_path = tmp_mp3.name

        # Convert to WAV for SpeechRecognition
        wav_path = tmp_mp3_path + ".wav"
        AudioSegment.from_file(tmp_mp3_path).export(wav_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

        # Cleanup
        os.remove(tmp_mp3_path)
        os.remove(wav_path)

        return f"Transcription: {text}"
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return f"Error transcribing audio: {e}"


# --- Agent Loop ---


async def solve_quiz(url: str, email: str, secret: str):
    token = os.environ.get("OPENROUTER_API_KEY")
    if not token:
        raise ValueError("No API token found (OPENROUTER_API_KEY)")

    client = OpenAI(
        api_key=token,
        base_url="https://openrouter.ai/api/v1",
    )

    model = "x-ai/grok-4.1-fast:free"  # Or "google/gemini-2.0-flash-exp:free"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_page",
                "description": "Read the text content of a webpage.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL to read."}
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_python",
                "description": "Execute Python code to solve a problem. Use this for calculations, data processing (pandas), etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute.",
                        }
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "transcribe_audio",
                "description": "Transcribe an audio file from a URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_url": {
                            "type": "string",
                            "description": "The URL of the audio file.",
                        }
                    },
                    "required": ["file_url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_answer",
                "description": "Submit the answer to the quiz endpoint.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "submission_url": {
                            "type": "string",
                            "description": "The URL to submit to.",
                        },
                        "answer": {
                            "type": "string",
                            "description": "The answer to submit (can be number or string).",
                        },
                    },
                    "required": ["submission_url", "answer"],
                },
            },
        },
    ]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        messages = [
            {
                "role": "system",
                "content": f"""You are an autonomous agent solving a quiz. 
                Your goal is to navigate to the quiz URL, understand the question, perform necessary actions (read page, run code, transcribe audio), and submit the answer.
                
                You have access to the following tools:
                - read_page: Read the content of the current or other URLs.
                - run_python: Execute Python code. Use this for CSV analysis, math, etc.
                - transcribe_audio: Transcribe audio files.
                - submit_answer: Submit the final answer.
                
                The current email is: {email}
                The current secret is: {secret}
                
                When you submit an answer, if it is correct, you might get a new URL. You must continue to that URL and solve the next question.
                If the submission returns a new URL, the `submit_answer` tool will return it. You should then `read_page` that new URL.
                If `submit_answer` returns "Quiz completed", you are done.
                
                SPECIFIC HINTS:
                - For CSV tasks:
                  - Check if the first row is a header or data. If it looks like a number, use `header=None`.
                  - "Cutoff: X" usually means "Sum of values in the column where value > X". Try that first. If that fails, try "Count of values > X".
                - For Audio tasks:
                  - Transcribe the audio to get the text/number.
                - For Image tasks:
                  - Use `run_python` with `pillow` and `pytesseract` to extract text (OCR).
                  - Libraries available: `pandas`, `numpy`, `scikit-learn`, `scipy`, `pillow` (PIL), `pytesseract`.
                """,
            },
            {"role": "user", "content": f"Start the quiz at {url}"},
        ]

        current_url = url
        quiz_url = url  # Track the actual quiz URL for submission

        while True:
            # Call LLM
            try:
                response = client.chat.completions.create(
                    model=model, messages=messages, tools=tools, tool_choice="auto"
                )
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                break

            msg = response.choices[0].message
            messages.append(msg)

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    result = None

                    if func_name == "read_page":
                        # Handle relative URLs if needed, but usually LLM should handle or we pass absolute
                        target_url = args["url"]
                        if not target_url.startswith("http"):
                            target_url = urljoin(current_url, target_url)
                        result = await read_page(target_url, page)
                        current_url = (
                            target_url  # Update current URL context (browser location)
                        )

                    elif func_name == "run_python":
                        result = await run_python(args["code"])

                    elif func_name == "transcribe_audio":
                        file_url = args["file_url"]
                        if not file_url.startswith("http"):
                            file_url = urljoin(current_url, file_url)
                        result = await transcribe_audio(file_url)

                    elif func_name == "submit_answer":
                        sub_url = args["submission_url"]
                        if not sub_url.startswith("http"):
                            sub_url = urljoin(current_url, sub_url)

                        logger.info(
                            f"Submitting to {sub_url} with answer: {args['answer']}"
                        )

                        payload = {
                            "email": email,
                            "secret": secret,
                            "url": quiz_url,  # Use the QUIZ URL, not the current browser URL
                            "answer": args["answer"],
                        }

                        async with httpx.AsyncClient() as http_client:
                            try:
                                resp = await http_client.post(
                                    sub_url, json=payload, timeout=30
                                )
                                if resp.status_code == 200:
                                    resp_data = resp.json()
                                    result = json.dumps(resp_data)

                                    if resp_data.get("correct"):
                                        next_url = resp_data.get("url")
                                        if next_url:
                                            result = f"Correct! Next URL: {next_url}"
                                            # Update both URLs for the next level
                                            quiz_url = next_url
                                            current_url = next_url
                                        else:
                                            result = "Quiz completed! No more URLs."
                                            return resp_data
                                    else:
                                        result = f"Incorrect. Reason: {resp_data.get('reason')}"
                                else:
                                    result = f"Error {resp.status_code}: {resp.text}"
                            except Exception as e:
                                result = f"Submission error: {e}"

                    # Append tool result
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
                    )
            else:
                # No tool calls, maybe just chatter.
                logger.info(f"Agent message: {msg.content}")
                if "Quiz completed" in str(msg.content):
                    break

        await browser.close()
        return {"status": "finished"}

from langchain_core.tools import tool
import os


@tool
def read_file(file_path: str) -> str:
    """
    Read the contents of a file.

    Args:
        file_path (str): The path to the file to read (relative to LLMFiles or absolute).

    Returns:
        str: The content of the file or an error message.
    """
    try:
        # Securely handle paths - prefer LLMFiles but allow absolute if needed/safe
        # For simplicity in this agent, we'll try direct access but default to LLMFiles if relative
        if not os.path.isabs(file_path):
            file_path = os.path.join("LLMFiles", file_path)

        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

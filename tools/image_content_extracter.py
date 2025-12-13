import pytesseract
from PIL import Image
from io import BytesIO
import base64
import os


from langchain_core.tools import tool


def load_image(image_input):
    """Internal helper to load an image from bytes, file path, base64, or PIL.Image."""
    if isinstance(image_input, bytes):
        return Image.open(BytesIO(image_input)).convert("RGB")
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, str):
        if image_input.startswith("data:"):  # base64 data URL
            _, b64 = image_input.split(",", 1)
            return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        return Image.open(os.path.join("LLMFiles", image_input)).convert("RGB")
    raise ValueError("Unsupported image input type")


@tool
def ocr_image_tool(image: str, lang: str = "eng") -> dict:
    """
    Extract text from an image using pytesseract OCR.

    Parameters:
        image (str): The image source. Can be a file path (relative to LLMFiles) or a base64 data URL.
        lang (str): Language code (default "eng").

    Returns:
        dict:
        {
            "text": "<extracted text>",
            "engine": "pytesseract"
        }
    """
    try:
        img = load_image(image)
        text = pytesseract.image_to_string(img, lang=lang)

        return {"text": text.strip(), "engine": "pytesseract"}
    except Exception as e:
        return f"Error occurred: {e}"

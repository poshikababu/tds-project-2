from langchain_core.tools import tool
from playwright.sync_api import sync_playwright
import os
import time


@tool
def take_screenshot(url: str) -> str:
    """
    Take a screenshot of the given URL.
    Useful for <canvas> elements or when text is hidden in images.
    Returns the filename of the saved screenshot (relative to LLMFiles).
    """
    print(f"\n📸 Taking screenshot of: {url}")
    filename = "screenshot.png"
    filepath = os.path.join("LLMFiles", filename)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1280, "height": 1024})
            page = context.new_page()

            page.goto(url, wait_until="networkidle")
            # Slight delay to ensure canvas renders
            time.sleep(1.0)

            page.screenshot(path=filepath)
            browser.close()

        print(f"Screenshot saved to {filepath}")
        return filename
    except Exception as e:
        return f"Error taking screenshot: {str(e)}"

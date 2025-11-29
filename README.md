# Agentic Quiz Solver

An autonomous AI agent designed to solve multi-step quizzes requiring web navigation, data scraping, audio transcription, and data analysis.

## Features

- **Agentic Loop**: Uses an LLM (via OpenRouter) to determine the next best action based on the current state.
- **Tool Use**: Equipped with tools to:
    - `read_page`: Navigate and extract text/links from web pages (Playwright).
    - `run_python`: Execute Python code for data analysis (Pandas).
    - `transcribe_audio`: Download and transcribe audio files (SpeechRecognition, pydub).
    - `submit_answer`: Submit solutions to the quiz endpoint.
- **Resilience**: Handles dynamic content, file downloads, and error retries.

## Prerequisites

- Python 3.12+
- `uv` package manager (recommended) or `pip`
- `ffmpeg` (required for audio processing)
    - Linux: `sudo apt-get install ffmpeg`
    - macOS: `brew install ffmpeg`

## Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd tds-project-2
    ```

2.  **Install dependencies**:
    Using `uv`:
    ```bash
    uv sync
    ```
    Or using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Playwright browsers**:
    ```bash
    uv run playwright install chromium
    # or
    playwright install chromium
    ```

4.  **Configuration**:
    Create a `.env` file in the root directory:
    ```env
    OPENROUTER_API_KEY=your_openrouter_key
    AIPROXY_TOKEN=your_secret_token
    ```

## Usage

1.  **Start the API Server**:
    ```bash
    uv run uvicorn main:app --host 0.0.0.0 --port 8000
    ```

2.  **Trigger the Solver**:
    Send a POST request to `/run`:
    ```bash
    curl -X POST http://localhost:8000/run \
      -H "Content-Type: application/json" \
      -d '{
        "email": "student@example.com",
        "secret": "your_secret_token",
        "url": "https://tds-llm-analysis.s-anand.net/demo"
      }'
    ```

## Architecture

- `main.py`: FastAPI entry point.
- `solver.py`: Core logic containing the agent loop and tool definitions.
- `prompts.py`: System and user prompts for the LLM.

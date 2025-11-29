from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
import os
import logging
from solver import solve_quiz
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()


class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str


@app.post("/run")
async def run_quiz(request: QuizRequest):
    # Verify secret
    expected_secret = os.getenv("AIPROXY_TOKEN")
    if expected_secret and request.secret != expected_secret:
        # If AIPROXY_TOKEN is set, we must match it.
        # If not set, we might be in a dev mode where we accept anything, or we should reject.
        # Given the instructions imply a specific token, we should probably enforce it if known.
        # But for local testing without the token set, we might want to be lenient or require it to be set.
        # Let's enforce it if the env var is present.
        raise HTTPException(status_code=401, detail="Invalid secret")

    logger.info(f"Received request for {request.email} with URL {request.url}")

    # Basic validation
    if not request.email or not request.secret or not request.url:
        raise HTTPException(status_code=400, detail="Missing required fields")

    # Execute the solver
    try:
        result = await solve_quiz(request.url, request.email, request.secret)
        return result
    except Exception as e:
        logger.error(f"Error solving quiz: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

import requests
import os

# URL of your local API
API_URL = "http://localhost:3000/run"


# Payload for the demo quiz
payload = {
    "email": "student@example.com",
    "secret": "test_secret",
    "url": "https://tds-llm-analysis.s-anand.net/demo",
}

print(f"Sending request to {API_URL}...")
try:
    response = requests.post(
        API_URL, json=payload, timeout=300
    )  # Long timeout for the whole process
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
    if hasattr(e, "response") and e.response:
        print(f"Error Response: {e.response.text}")

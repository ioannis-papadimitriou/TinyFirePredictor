import requests
import json

def test_ollama():
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    
    # Simple test prompt
    payload = {
        "model": "deepseek-r1:32b",  # We'll also test if this model is available
        "prompt": "Say hello!",
        "max_tokens": 50
    }
    
    try:
        print("Testing Ollama connection...")
        response = requests.post(url, headers=headers, json=payload)
        print(f"Status code: {response.status_code}")
        print(f"Response text: {response.text[:200]}")  # Print first 200 chars
        return True
    except Exception as e:
        print(f"Error connecting to Ollama: {str(e)}")
        return False

if __name__ == "__main__":
    test_ollama()
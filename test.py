import requests
import json

def ask_local_ai(question):
    # Ollama's local API endpoint
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "qwen2.5:3b",
        "messages": [{"role": "user", "content": question}],
        "stream": False  # Wait for full answer before returning
    }
    
    try:
        # Send request to Ollama
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Stop if there's an error
        
        # Extract AI's reply
        reply = response.json()["message"]["content"]
        return reply
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("🤖 Asking Qwen2.5:3b a test question...")
    question = "What is the melting point of pure iron in Celsius? Answer in one sentence."
    answer = ask_local_ai(question)
    print(f"\n✅ AI Response:\n{answer}")
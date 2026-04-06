import os
import json
import requests
from openai import OpenAI
from typing import Dict, Any

# Initialize OpenAI client (if API key is set)
openai_client = None
if os.environ.get("OPENAI_API_KEY"):
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# DeepSeek API endpoint
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

# Gemma can be accessed via Ollama local or external API; here we use a mock or you can set up a URL
GEMMA_API_URL = os.environ.get("GEMMA_API_URL", "http://localhost:11434/api/generate")  # Ollama default
GEMMA_API_KEY = os.environ.get("GEMMA_API_KEY")  # optional

def call_openai(system_prompt: str, user_prompt: str, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
    """Call OpenAI GPT model."""
    if not openai_client:
        raise Exception("OpenAI API key not set")
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )
    return {
        "text": response.choices[0].message.content,
        "model": model,
        "usage": response.usage.dict() if response.usage else None
    }

def call_deepseek(system_prompt: str, user_prompt: str, model: str = "deepseek-chat") -> Dict[str, Any]:
    """Call DeepSeek API."""
    if not DEEPSEEK_API_KEY:
        raise Exception("DeepSeek API key not set")
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7
    }
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return {
        "text": data["choices"][0]["message"]["content"],
        "model": model,
        "usage": data.get("usage")
    }

def call_gemma(system_prompt: str, user_prompt: str, model: str = "gemma2:2b") -> Dict[str, Any]:
    """Call Gemma via Ollama API (or other endpoint)."""
    if not GEMMA_API_URL:
        raise Exception("Gemma API URL not set")
    payload = {
        "model": model,
        "prompt": f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:",
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    if GEMMA_API_KEY:
        headers["Authorization"] = f"Bearer {GEMMA_API_KEY}"
    response = requests.post(GEMMA_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    # Ollama returns {"response": "..."}
    return {
        "text": data.get("response", ""),
        "model": model
    }

def test_prompt(ai_provider: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Route to appropriate AI provider."""
    if ai_provider == "openai":
        return call_openai(system_prompt, user_prompt)
    elif ai_provider == "deepseek":
        return call_deepseek(system_prompt, user_prompt)
    elif ai_provider == "gemma":
        return call_gemma(system_prompt, user_prompt)
    else:
        raise ValueError(f"Unknown AI provider: {ai_provider}")
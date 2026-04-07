import os
import json
import requests
from openai import OpenAI
from typing import Dict, Any, Optional

# ---------- OpenAI ----------
openai_client = None
if os.environ.get("OPENAI_API_KEY"):
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def call_openai(system_prompt: str, user_prompt: str, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
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

# ---------- DeepSeek ----------
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

def call_deepseek(system_prompt: str, user_prompt: str, model: str = "deepseek-chat") -> Dict[str, Any]:
    if not DEEPSEEK_API_KEY:
        raise Exception("DeepSeek API key not set")
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
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
    return {"text": data["choices"][0]["message"]["content"], "model": model, "usage": data.get("usage")}

# ---------- Gemma (Ollama) ----------
GEMMA_API_URL = os.environ.get("GEMMA_API_URL", "http://localhost:11434/api/generate")

def call_gemma(system_prompt: str, user_prompt: str, model: str = "gemma2:2b") -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:",
        "stream": False
    }
    response = requests.post(GEMMA_API_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    return {"text": data.get("response", ""), "model": model}

# ---------- Mock responses when keys are missing ----------
def get_mock_response(ai_provider: str, essay_text: str) -> str:
    """Return a plausible evaluation when real API keys are absent."""
    return f"""
⚠️ **Mock Evaluation (no real API key for {ai_provider})**

Based on your essay, here is a simulated feedback:

- **Grammar score:** 76/100  
- **Coherence score:** 72/100  
- **Content score:** 74/100  

**Feedback:**  
Your essay has a clear structure, but some sentences are too long. Consider breaking them into shorter, clearer statements. Add more specific examples to support your arguments. The introduction is good, but the conclusion could be stronger.

*To get real AI evaluations, add your {ai_provider.upper()} API key to the backend environment variables.*
"""

# ---------- Main test function with safe fallback ----------
def test_prompt(ai_provider: str, system_prompt: str, user_prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Call the AI provider safely. If keys are missing or any error occurs, return a mock response.
    """
    try:
        if ai_provider == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                return {"text": get_mock_response("openai", user_prompt), "model": "mock", "error": "missing_key"}
            return call_openai(system_prompt, user_prompt, model=model or "gpt-3.5-turbo")
        elif ai_provider == "deepseek":
            if not DEEPSEEK_API_KEY:
                return {"text": get_mock_response("deepseek", user_prompt), "model": "mock", "error": "missing_key"}
            return call_deepseek(system_prompt, user_prompt, model=model or "deepseek-chat")
        elif ai_provider == "gemma":
            # Gemma might be local – we don't mock it, but catch errors
            return call_gemma(system_prompt, user_prompt, model=model or "gemma2:2b")
        else:
            raise ValueError(f"Unknown AI provider: {ai_provider}")
    except Exception as e:
        # Any error (network, missing key, etc.) -> return mock
        return {"text": get_mock_response(ai_provider, user_prompt), "model": "mock", "error": str(e)}
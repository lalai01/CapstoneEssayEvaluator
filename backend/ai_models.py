import os
import json
import requests
from openai import OpenAI

# ---------- OpenAI ----------
def call_openai(system_prompt, user_prompt, model="gpt-3.5-turbo"):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    return {
        "text": response.choices[0].message.content,
        "model": model,
        "provider": "openai"
    }

# ---------- DeepSeek ----------
def call_deepseek(system_prompt, user_prompt, model="deepseek-chat"):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3
    }
    response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return {
        "text": data["choices"][0]["message"]["content"],
        "model": model,
        "provider": "deepseek"
    }

# ---------- Ollama (Gemma) ----------
def call_ollama(system_prompt, user_prompt, model="gemma2:2b"):
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.3
        }
    }
    response = requests.post(f"{ollama_url}/api/chat", json=payload)
    response.raise_for_status()
    data = response.json()
    return {
        "text": data["message"]["content"],
        "model": model,
        "provider": "ollama"
    }

# ---------- Llama.cpp (OpenAI-compatible) ----------
def call_llamacpp(system_prompt, user_prompt, model=None):
    llamacpp_url = os.environ.get("LLAMACPP_URL", "http://llamacpp:8080/v1")
    default_model = os.environ.get("LLAMACPP_MODEL", "Phi-3.5-mini-instruct-Q4_K_M.gguf")
    model = model or default_model

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3
    }
    response = requests.post(f"{llamacpp_url}/chat/completions", json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return {
        "text": data["choices"][0]["message"]["content"],
        "model": model,
        "provider": "llamacpp"
    }

# ---------- Router ----------
def test_prompt(ai_provider, system_prompt, user_prompt, model=None):
    provider = ai_provider.lower()
    
    if provider == "openai":
        return call_openai(system_prompt, user_prompt, model or "gpt-3.5-turbo")
    elif provider == "deepseek":
        return call_deepseek(system_prompt, user_prompt, model or "deepseek-chat")
    elif provider in ("gemma", "ollama"):
        return call_ollama(system_prompt, user_prompt, model or "gemma2:2b")
    elif provider == "llamacpp":
        return call_llamacpp(system_prompt, user_prompt, model)
    else:
        raise ValueError(f"Unsupported AI provider: {ai_provider}")
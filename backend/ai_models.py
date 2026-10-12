import os
import json
import time
import requests

# Local models (Ollama, llama.cpp) run on the VPS CPU, so responses are
# streamed: the read timeout then applies between tokens rather than to the
# whole answer, and a slow-but-working model is not cut off. TOTAL_DEADLINE
# stays under the frontend's 5-minute request timeout.
CONNECT_TIMEOUT = 10
FIRST_TOKEN_TIMEOUT = 150   # model load + reading the prompt
TOTAL_DEADLINE = 280

# Enough for the playground's JSON (scores, short per-criterion notes, a few
# grammar fixes and recommendations) without letting a model ramble.
MAX_OUTPUT_TOKENS = 1024
# Prompt + answer must fit; larger contexts cost memory and speed on CPU.
OLLAMA_CONTEXT = 4096
# Keep models in memory between runs so the next run skips the load time.
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")


def _stats(started, output_tokens=None):
    seconds = time.time() - started
    stats = {"seconds": round(seconds, 1)}
    if output_tokens:
        stats["output_tokens"] = output_tokens
        stats["tokens_per_second"] = round(output_tokens / seconds, 1) if seconds else None
    return stats


def _error_reason(response):
    """The server's own explanation of an error response, kept short."""
    try:
        body = response.json()
        error = body.get("error", body)
        message = error.get("message") if isinstance(error, dict) else error
        return str(message)[:300]
    except Exception:
        return (response.text or f"HTTP {response.status_code}")[:300]


def _check_deadline(started, provider):
    if time.time() - started > TOTAL_DEADLINE:
        raise TimeoutError(
            f"{provider} took longer than {TOTAL_DEADLINE}s. The VPS CPU is likely busy; "
            "try again, or run fewer local models at once."
        )


# ---------- DeepSeek ----------
def call_deepseek(system_prompt, user_prompt, model="deepseek-chat", json_mode=False):
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
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    started = time.time()
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    usage = data.get("usage") or {}
    return {
        "text": data["choices"][0]["message"]["content"],
        "model": model,
        "provider": "deepseek",
        "stats": _stats(started, usage.get("completion_tokens")),
    }


# ---------- Ollama (Gemma / Llama) ----------
def call_ollama(system_prompt, user_prompt, model="llama3.2:3b", json_mode=False):
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": True,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0.3,
            "num_predict": MAX_OUTPUT_TOKENS,
            "num_ctx": OLLAMA_CONTEXT,
        },
    }
    if json_mode:
        # Constrains output to valid JSON: no prose to strip, no retries.
        payload["format"] = "json"

    started = time.time()
    parts = []
    output_tokens = None
    with requests.post(
        f"{ollama_url}/api/chat",
        json=payload,
        stream=True,
        timeout=(CONNECT_TIMEOUT, FIRST_TOKEN_TIMEOUT),
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            _check_deadline(started, f"Ollama ({model})")
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            parts.append((chunk.get("message") or {}).get("content", ""))
            if chunk.get("done"):
                output_tokens = chunk.get("eval_count")
                break

    return {
        "text": "".join(parts),
        "model": model,
        "provider": "ollama",
        "stats": _stats(started, output_tokens),
    }


# ---------- Llama.cpp (OpenAI-compatible) ----------
def call_llamacpp(system_prompt, user_prompt, model=None, json_mode=False):
    llamacpp_url = os.environ.get("LLAMACPP_URL", "http://llamacpp:8080/v1")
    default_model = os.environ.get(
        "LLAMACPP_MODEL", "Phi-3.5-mini-instruct-Q4_K_M.gguf"
    )
    model = model or default_model

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": True,
        # Reuse the processed system prompt across runs.
        "cache_prompt": True,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    # Not every llama.cpp build / model accepts every option, and a long
    # prompt plus max_tokens can exceed the server's context size (-c). On a
    # 400, retry with progressively plainer requests instead of failing.
    attempts = [
        payload,
        {k: v for k, v in payload.items() if k not in ("response_format", "cache_prompt")},
        {**{k: v for k, v in payload.items() if k not in ("response_format", "cache_prompt")},
         "max_tokens": MAX_OUTPUT_TOKENS // 2},
    ]

    started = time.time()
    parts = []
    output_tokens = 0
    response = None
    errors = []
    for attempt in attempts:
        response = requests.post(
            f"{llamacpp_url}/chat/completions",
            json=attempt,
            stream=True,
            timeout=(CONNECT_TIMEOUT, FIRST_TOKEN_TIMEOUT),
        )
        if response.status_code != 400:
            break
        reason = _error_reason(response)
        errors.append(reason)
        print(f"[llama.cpp] 400 Bad Request ({reason}); retrying with a simpler request")
        response.close()
    else:
        raise RuntimeError(
            f"llama.cpp rejected the request: {errors[-1]}. If it mentions the context size, "
            "start llama.cpp with a larger -c (e.g. -c 4096)."
        )

    with response:
        response.raise_for_status()
        for raw in response.iter_lines():
            _check_deadline(started, "llama.cpp")
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content")
            if delta:
                parts.append(delta)
                output_tokens += 1  # llama.cpp streams roughly one token per chunk

    return {
        "text": "".join(parts),
        "model": model,
        "provider": "llamacpp",
        "stats": _stats(started, output_tokens or None),
    }


# ---------- Router ----------
def test_prompt(ai_provider, system_prompt, user_prompt, model=None, json_mode=False):
    provider = ai_provider.lower()

    if provider == "deepseek":
        return call_deepseek(system_prompt, user_prompt, model or "deepseek-chat", json_mode)
    elif provider == "gemma":
        return call_ollama(system_prompt, user_prompt, model or "gemma2:2b", json_mode)
    elif provider == "ollama":
        return call_ollama(system_prompt, user_prompt, model or "llama3.2:3b", json_mode)
    elif provider == "llamacpp":
        return call_llamacpp(system_prompt, user_prompt, model, json_mode)
    else:
        raise ValueError(f"Unsupported AI provider: {ai_provider}")

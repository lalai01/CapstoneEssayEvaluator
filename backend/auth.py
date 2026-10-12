import os
import time
import hashlib
import threading
import requests
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL environment variable is required")

security = HTTPBearer()
optional_security = HTTPBearer(auto_error=False)

# Verified tokens are cached briefly so each request doesn't call Supabase.
_CACHE_SECONDS = 120
_cache = {}
_cache_lock = threading.Lock()


def _verify_with_supabase(token: str) -> dict:
    """
    Ask Supabase Auth who this token belongs to. This verifies the signature
    and expiry for every JWT signing scheme (legacy HS256 secret or the newer
    asymmetric keys), and returns server-controlled fields such as
    app_metadata and email_confirmed_at.
    """
    response = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={"apikey": SUPABASE_KEY or "", "Authorization": f"Bearer {token}"},
        timeout=10,
    )
    if response.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    data = response.json()
    return {
        "id": data.get("id"),
        "email": (data.get("email") or "").lower() or None,
        # Only app_metadata is trusted for roles: users can edit their own
        # user_metadata from the browser, so a role there means nothing.
        "role": (data.get("app_metadata") or {}).get("role"),
        "email_confirmed": bool(data.get("email_confirmed_at") or data.get("confirmed_at")),
        "full_name": (data.get("user_metadata") or {}).get("full_name"),
    }


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    key = hashlib.sha256(token.encode()).hexdigest()
    now = time.time()

    with _cache_lock:
        cached = _cache.get(key)
        if cached and cached[0] > now:
            return cached[1]

    try:
        user = _verify_with_supabase(token)
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Could not verify token")

    with _cache_lock:
        if len(_cache) > 5000:
            _cache.clear()
        _cache[key] = (now + _CACHE_SECONDS, user)
    return user


async def get_optional_user(credentials: HTTPAuthorizationCredentials = Depends(optional_security)):
    """Like get_current_user, but returns None for anonymous requests."""
    if not credentials:
        return None
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None

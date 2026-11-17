import os
import json
import time
import base64
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


def _token_claims(token: str) -> dict:
    """Claims of a JWT that Supabase has just accepted (so they can be
    trusted). Used for `aal`: aal2 means this session passed two-factor
    authentication."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}


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
    if response.status_code in (401, 403):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if response.status_code != 200:
        # Supabase itself is having trouble; the token may be fine.
        raise HTTPException(status_code=503, detail="Sign-in check temporarily unavailable. Please try again.")
    data = response.json()
    factors = data.get("factors") or []
    return {
        "id": data.get("id"),
        "email": (data.get("email") or "").lower() or None,
        # Only app_metadata is trusted for roles: users can edit their own
        # user_metadata from the browser, so a role there means nothing.
        "role": (data.get("app_metadata") or {}).get("role"),
        "email_confirmed": bool(data.get("email_confirmed_at") or data.get("confirmed_at")),
        "full_name": (data.get("user_metadata") or {}).get("full_name"),
        # Two-factor authentication: "aal2" once this session entered an
        # authenticator code; mfa_enrolled when the account has an
        # authenticator set up (then every request needs aal2).
        "aal": _token_claims(token).get("aal") or "aal1",
        "mfa_enrolled": any(f.get("status") == "verified" for f in factors if isinstance(f, dict)),
    }


MFA_REQUIRED = {"code": "mfa_required",
                "message": "Enter the code from your authenticator app to continue (two-factor authentication)."}


async def get_current_user_any_aal(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """The verified user, even before their two-factor code is entered. Only
    for GET /me, which tells the website that a code is still needed."""
    return _user_from_token(credentials.credentials)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """The verified user. An account with an authenticator set up must have
    entered its code in this session (aal2); a stolen password alone is not
    enough to use the API."""
    user = _user_from_token(credentials.credentials)
    if user.get("mfa_enrolled") and user.get("aal") != "aal2":
        raise HTTPException(status_code=401, detail=MFA_REQUIRED)
    return user


def _user_from_token(token: str) -> dict:
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
        # Network error or timeout reaching Supabase: not the user's fault.
        print(f"❌ Token verification failed: {e}")
        raise HTTPException(status_code=503, detail="Sign-in check temporarily unavailable. Please try again.")

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

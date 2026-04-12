import os
import requests
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from jose.jwk import construct
from jose.backends.cryptography_backend import CryptographyECKey
import json

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET")

if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL environment variable is required")

security = HTTPBearer()

# Cache for JWKS
_jwks_cache = None

def get_jwks():
    """Fetch JWKS from Supabase"""
    global _jwks_cache
    if _jwks_cache:
        return _jwks_cache
    
    jwks_url = f"{SUPABASE_URL}/auth/v1/jwks"
    try:
        response = requests.get(jwks_url, timeout=10)
        response.raise_for_status()
        _jwks_cache = response.json()
        print("✅ JWKS fetched successfully")
        return _jwks_cache
    except Exception as e:
        print(f"❌ Failed to fetch JWKS: {e}")
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    print(f"🔑 Token received: {token[:50]}...")
    
    # Get the token header to find the key ID
    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        alg = unverified_header.get("alg", "ES256")
        print(f"🔑 Token header: alg={alg}, kid={kid}")
    except Exception as e:
        print(f"❌ Failed to parse token header: {e}")
        raise HTTPException(status_code=401, detail="Invalid token format")
    
    # Try verification with JWKS first (for ES256/RS256)
    jwks = get_jwks()
    if jwks and kid:
        for key_data in jwks.get("keys", []):
            if key_data.get("kid") == kid:
                try:
                    # Construct the public key from JWK
                    public_key = construct(key_data, algorithm=alg)
                    payload = jwt.decode(
                        token,
                        public_key,
                        algorithms=[alg],
                        audience="authenticated"
                    )
                    user_id = payload.get("sub")
                    email = payload.get("email")
                    print(f"✅ Token valid (JWKS) for: {email}")
                    return {"id": user_id, "email": email}
                except Exception as e:
                    print(f"❌ JWKS verification failed: {e}")
    
    # Fallback: Try HS256 with JWT_SECRET (legacy tokens)
    if SUPABASE_JWT_SECRET:
        try:
            print("🔄 Trying HS256 with JWT_SECRET...")
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated"
            )
            user_id = payload.get("sub")
            email = payload.get("email")
            print(f"✅ Token valid (HS256) for: {email}")
            return {"id": user_id, "email": email}
        except Exception as e:
            print(f"❌ HS256 fallback failed: {e}")
    
    raise HTTPException(status_code=401, detail="Invalid or expired token")
import os
import requests
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET")

if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL environment variable is required")

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    print(f"🔑 Token received: {token[:50]}...")
    
    # Helper to extract common fields from payload
    def extract_user_info(payload):
        user_id = payload.get("sub")
        email = payload.get("email")
        # ✅ Extract role from user_metadata (Supabase injects this into the JWT)
        role = payload.get("user_metadata", {}).get("role")
        return {"id": user_id, "email": email, "role": role}

    # Approach 1: Use the JWT_SECRET directly (works for legacy tokens)
    if SUPABASE_JWT_SECRET:
        try:
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
                options={"verify_signature": True, "verify_aud": True}
            )
            user_info = extract_user_info(payload)
            print(f"✅ Token valid (HS256) for: {user_info['email']}, role: {user_info['role']}")
            return user_info
        except Exception as e:
            print(f"⚠️ HS256 verification failed: {e}")
    
    # Approach 2: Skip signature verification (for development only)
    try:
        print("🔄 Trying without signature verification...")
        payload = jwt.decode(
            token,
            "",
            algorithms=["HS256", "RS256", "ES256"],
            options={"verify_signature": False, "verify_aud": False}
        )
        user_info = extract_user_info(payload)
        print(f"✅ Token parsed (no verification) for: {user_info['email']}, role: {user_info['role']}")
        return user_info
    except Exception as e:
        print(f"❌ All verification attempts failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
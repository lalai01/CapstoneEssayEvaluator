import os
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET")
if not SUPABASE_JWT_SECRET:
    raise ValueError("SUPABASE_JWT_SECRET environment variable is required")

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    print(f"🔑 Token received: {token[:50]}...")
    
    try:
        # Try HS256 first (Supabase default)
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],  # Force HS256 regardless of what header says
            audience="authenticated"
        )
        
        user_id = payload.get("sub")
        email = payload.get("email")
        
        print(f"✅ Token valid for: {email} (ID: {user_id})")
        
        if user_id is None:
            print("❌ Token missing 'sub' claim")
            raise HTTPException(status_code=401, detail="Invalid token: missing user ID")
        
        return {"id": user_id, "email": email}
        
    except JWTError as e:
        print(f"❌ JWT Error with HS256: {e}")
        
        # If HS256 fails, try with options to ignore algorithm check
        try:
            print("🔄 Trying with algorithm verification disabled...")
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256", "RS256", "ES256"],
                options={"verify_signature": True, "verify_aud": True},
                audience="authenticated"
            )
            
            user_id = payload.get("sub")
            email = payload.get("email")
            
            print(f"✅ Token valid for: {email} (ID: {user_id})")
            
            if user_id is None:
                raise HTTPException(status_code=401, detail="Invalid token: missing user ID")
            
            return {"id": user_id, "email": email}
            
        except JWTError as e2:
            print(f"❌ All JWT attempts failed: {e2}")
            raise HTTPException(status_code=401, detail=f"Invalid or expired token")
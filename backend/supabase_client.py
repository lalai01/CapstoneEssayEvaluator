import os
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")          # anon key
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")  # service_role key

if not SUPABASE_URL or not SUPABASE_KEY:
    print("⚠️ Supabase credentials not set. Knowledge base features will not work.")
    supabase = None
    supabase_admin = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    if SUPABASE_SERVICE_KEY:
        supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    else:
        supabase_admin = None
        print("⚠️ SUPABASE_SERVICE_KEY not set. Admin operations will be limited.")
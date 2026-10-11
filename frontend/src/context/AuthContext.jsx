import { createContext, useContext, useEffect, useState } from 'react'
import { supabase } from '../lib/supabase'

const AuthContext = createContext({})

// Mirrors backend is_admin(): role "admin" in Supabase user metadata, or an
// email listed in VITE_ADMIN_EMAILS (comma-separated). Note: Supabase's
// top-level `user.role` is always "authenticated", so it is not checked.
const ADMIN_EMAILS = (
  import.meta.env.VITE_ADMIN_EMAILS || 'admin_essay_capstone@gmail.com,galangec24@gmail.com'
)
  .split(',')
  .map((e) => e.trim().toLowerCase())
  .filter(Boolean)

function isAdminUser(user) {
  if (!user) return false
  return (
    user.user_metadata?.role === 'admin' ||
    user.app_metadata?.role === 'admin' ||
    ADMIN_EMAILS.includes((user.email || '').toLowerCase())
  )
}

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null)
      setLoading(false)
    })

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null)
    })
    return () => listener.subscription.unsubscribe()
  }, [])
  
  const signInWithEmail = async (email, password) => {
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  });
  if (error) throw error;
  return data;
};

  const signInWithGoogle = () => supabase.auth.signInWithOAuth({ provider: 'google' })
  const signInWithYahoo = () => supabase.auth.signInWithOAuth({ provider: 'custom:yahoo' })
  const signOut = () => supabase.auth.signOut()

  return (
    <AuthContext.Provider value={{ user, isAdmin: isAdminUser(user), loading, signInWithGoogle, signInWithYahoo, signInWithEmail, signOut }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
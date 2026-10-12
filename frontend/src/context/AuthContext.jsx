import { createContext, useContext, useEffect, useState } from 'react'
import { supabase } from '../lib/supabase'
import { getMe } from '../api'

const AuthContext = createContext({})

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  // Decided by the backend (GET /me), which checks the server-side role,
  // ADMIN_EMAILS and the admin_users table. The browser never decides this.
  // Stored with the user id it belongs to, so a previous user's answer never
  // leaks to the next one while /me is loading.
  const [adminCheck, setAdminCheck] = useState({ userId: null, isAdmin: false, isTeacher: false, role: null })

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

  const userId = user?.id
  useEffect(() => {
    if (!userId) return
    let cancelled = false
    getMe()
      .then((me) => {
        if (!cancelled) {
          setAdminCheck({ userId, isAdmin: !!me?.is_admin, isTeacher: !!me?.is_teacher, role: me?.role || 'student' })
        }
      })
      .catch(() => { if (!cancelled) setAdminCheck({ userId, isAdmin: false, isTeacher: false, role: 'student' }) })
    return () => { cancelled = true }
  }, [userId])

  const checked = !!userId && adminCheck.userId === userId
  const isAdmin = checked && adminCheck.isAdmin
  // Teachers (approvers) can override and approve evaluations; admins are teachers too.
  const isTeacher = checked && adminCheck.isTeacher
  const role = checked ? adminCheck.role : null

  const signInWithEmail = async (email, password) => {
    const { data, error } = await supabase.auth.signInWithPassword({ email, password })
    if (error) throw error
    return data
  }

  // Passwordless: Supabase emails a one-time sign-in link to this address.
  const signInWithEmailLink = async (email) => {
    const { error } = await supabase.auth.signInWithOtp({
      email,
      options: { emailRedirectTo: window.location.origin },
    })
    if (error) throw error
  }

  // prompt=select_account makes Google show its account chooser every time,
  // instead of silently reusing the last Google account used in this browser.
  const signInWithGoogle = () => supabase.auth.signInWithOAuth({
    provider: 'google',
    options: {
      redirectTo: window.location.origin,
      queryParams: { prompt: 'select_account' },
    },
  })
  const signInWithYahoo = () => supabase.auth.signInWithOAuth({ provider: 'custom:yahoo' })
  const signOut = () => supabase.auth.signOut()

  return (
    <AuthContext.Provider value={{
      user, isAdmin, isTeacher, role, loading,
      signInWithGoogle, signInWithYahoo, signInWithEmail, signInWithEmailLink, signOut,
    }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)

import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { supabase } from '../lib/supabase'
import { getMe } from '../api'
import { clearDrafts } from '../lib/draft'

const AuthContext = createContext({})

// After signing in, come back to the page the user was on (e.g. /my-rubrics).
// Supabase only allows URLs in its Redirect URLs list and otherwise uses the
// Site URL, so the worst case is landing on Home as before.
const returnUrl = () => window.location.origin + window.location.pathname

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  // Decided by the backend (GET /me), which checks the server-side role,
  // ADMIN_EMAILS and the admin_users table. The browser never decides this.
  // Stored with the user id it belongs to, so a previous user's answer never
  // leaks to the next one while /me is loading.
  const [adminCheck, setAdminCheck] = useState({ userId: null, isAdmin: false, isTeacher: false, role: null, mfaForRole: false })
  // Two-factor authentication: currentLevel 'aal2' once this session entered
  // an authenticator code; nextLevel 'aal2' when the account has one set up.
  const [aal, setAal] = useState({ userId: null, currentLevel: null, nextLevel: null })

  const refreshMfa = useCallback(async () => {
    const { data: { session } } = await supabase.auth.getSession()
    if (!session) {
      setAal({ userId: null, currentLevel: null, nextLevel: null })
      return
    }
    const { data, error } = await supabase.auth.mfa.getAuthenticatorAssuranceLevel()
    if (!error && data) setAal({ userId: session.user.id, currentLevel: data.currentLevel, nextLevel: data.nextLevel })
  }, [])

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null)
      setLoading(false)
      refreshMfa()
    })

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null)
      // Outside the callback: Supabase calls inside it can deadlock.
      setTimeout(refreshMfa, 0)
    })
    return () => listener.subscription.unsubscribe()
  }, [refreshMfa])

  const userId = user?.id
  // The role is asked again once the session enters its code (aal2), since
  // teacher and admin powers need it.
  const sessionLevel = aal.userId === userId ? aal.currentLevel : null
  useEffect(() => {
    if (!userId) return
    let cancelled = false
    let timer = null
    let attempt = 0
    // A failed check (backend restarting, network blip) must never be read as
    // "this user is a student". Retry with backoff; if it still fails, keep
    // the last known role for this user, or leave the role unknown.
    const check = () => {
      getMe()
        .then((me) => {
          if (!cancelled) {
            setAdminCheck({
              userId, isAdmin: !!me?.is_admin, isTeacher: !!me?.is_teacher, role: me?.role || 'student',
              mfaForRole: !!me?.mfa_required_for_role,
            })
          }
        })
        .catch((err) => {
          if (cancelled) return
          attempt += 1
          if (attempt < 4) {
            timer = setTimeout(check, 1500 * attempt)
            return
          }
          console.warn('[auth] role check failed; keeping the last known role', err)
          setAdminCheck((prev) => (prev.userId === userId
            ? prev
            : { userId, isAdmin: false, isTeacher: false, role: null, mfaForRole: false }))
        })
    }
    check()
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [userId, sessionLevel])

  const checked = !!userId && adminCheck.userId === userId
  const isAdmin = checked && adminCheck.isAdmin
  // Teachers (approvers) can override and approve evaluations; admins are teachers too.
  const isTeacher = checked && adminCheck.isTeacher
  const role = checked ? adminCheck.role : null

  // What two-factor authentication still needs in this session:
  //   needsCode:  an authenticator is set up, but its code wasn't entered yet
  //   mustEnroll: teachers and admins must set one up before continuing
  const aalKnown = !!userId && aal.userId === userId
  const mfaEnrolled = aalKnown && aal.nextLevel === 'aal2'
  const mfa = {
    known: aalKnown,
    enrolled: mfaEnrolled,
    verified: aalKnown && aal.currentLevel === 'aal2',
    needsCode: mfaEnrolled && aal.currentLevel !== 'aal2',
    mustEnroll: aalKnown && checked && adminCheck.mfaForRole && !mfaEnrolled,
    requiredForRole: checked && adminCheck.mfaForRole,
    refresh: refreshMfa,
  }

  const signInWithEmail = async (email, password) => {
    const { data, error } = await supabase.auth.signInWithPassword({ email, password })
    if (error) throw error
    return data
  }

  // Passwordless: Supabase emails a one-time sign-in link to this address.
  const signInWithEmailLink = async (email) => {
    const { error } = await supabase.auth.signInWithOtp({
      email,
      options: { emailRedirectTo: returnUrl() },
    })
    if (error) throw error
  }

  // prompt=select_account makes Google show its account chooser every time,
  // instead of silently reusing the last Google account used in this browser.
  const signInWithGoogle = () => supabase.auth.signInWithOAuth({
    provider: 'google',
    options: {
      redirectTo: returnUrl(),
      queryParams: { prompt: 'select_account' },
    },
  })
  const signInWithYahoo = () => supabase.auth.signInWithOAuth({ provider: 'custom:yahoo' })
  // Drafts belong to whoever wrote them: clear them and reload, so the next
  // person on a shared computer starts with an empty page.
  const signOut = async () => {
    await supabase.auth.signOut()
    clearDrafts()
    window.location.assign('/')
  }
  // Signed in, but the role (admin / teacher) isn't known yet.
  const roleLoading = !!userId && !checked

  return (
    <AuthContext.Provider value={{
      user, isAdmin, isTeacher, role, loading, roleLoading, mfa,
      signInWithGoogle, signInWithYahoo, signInWithEmail, signInWithEmailLink, signOut,
    }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)

import { useEffect } from 'react';

// Automatic sign-out, so an unattended or forgotten session can't be used:
//   - after IDLE_HOURS without any activity in the app (default 24 hours),
//     checked every minute and when the site is opened again;
//   - SESSION_MAX_DAYS after the last real sign-in (default 7 days), however
//     active the user was. The backend enforces the same maximum.
// Both can be changed with VITE_IDLE_LOGOUT_HOURS and VITE_SESSION_MAX_DAYS.
export const IDLE_HOURS = Number(import.meta.env.VITE_IDLE_LOGOUT_HOURS) || 24;
export const SESSION_MAX_DAYS = Number(import.meta.env.VITE_SESSION_MAX_DAYS) || 7;

const LAST_ACTIVE = 'essay-evaluator:last-active';
const REASON = 'essay-evaluator:signed-out-reason';
const EVENTS = ['pointerdown', 'keydown', 'scroll', 'touchstart', 'mousemove'];

const read = (store, key) => { try { return store.getItem(key); } catch { return null; } };
const write = (store, key, value) => {
  try {
    if (value === null) store.removeItem(key);
    else store.setItem(key, value);
  } catch {
    // storage blocked: the timers below still work for this tab
  }
};

export function markActive() {
  write(localStorage, LAST_ACTIVE, String(Date.now()));
}

// Message to show after the reload that follows an automatic sign-out.
export function takeSignOutReason() {
  const reason = read(sessionStorage, REASON);
  write(sessionStorage, REASON, null);
  return reason;
}

export function rememberSignOutReason(reason) {
  write(sessionStorage, REASON, reason);
}

// user: the signed-in Supabase user (or null); signOut(reason) signs out.
export default function useSessionGuard(user, signOut) {
  useEffect(() => {
    if (!user) return undefined;
    const idleMs = IDLE_HOURS * 3600 * 1000;
    const maxMs = SESSION_MAX_DAYS * 86400 * 1000;

    const check = () => {
      const now = Date.now();
      const signedIn = Date.parse(user.last_sign_in_at || '') || 0;
      // A fresh sign-in counts as activity (the stored time may be from an
      // earlier session on this computer).
      const last = Math.max(Number(read(localStorage, LAST_ACTIVE)) || 0, signedIn);
      if (last && now - last > idleMs) {
        signOut(`You were signed out after ${IDLE_HOURS} hours without activity. Please sign in again.`);
        return true;
      }
      if (signedIn && now - signedIn > maxMs) {
        signOut(`For your security, sign-ins last ${SESSION_MAX_DAYS} days. Please sign in again.`);
        return true;
      }
      return false;
    };

    // Opened again after a long time away: sign out before anything else.
    if (check()) return undefined;
    markActive();

    let lastWrite = Date.now();
    const onActivity = () => {
      if (Date.now() - lastWrite > 30000) {   // at most every 30 s
        lastWrite = Date.now();
        markActive();
      }
    };
    EVENTS.forEach((e) => window.addEventListener(e, onActivity, { passive: true }));
    const onVisible = () => { if (document.visibilityState === 'visible') check(); };
    document.addEventListener('visibilitychange', onVisible);
    const timer = setInterval(check, 60000);
    return () => {
      EVENTS.forEach((e) => window.removeEventListener(e, onActivity));
      document.removeEventListener('visibilitychange', onVisible);
      clearInterval(timer);
    };
  }, [user, signOut]);
}

import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import toast from 'react-hot-toast';

export default function LoginModal({ isOpen, onClose }) {
  const { signInWithGoogle, signInWithYahoo, signInWithEmail, signInWithEmailLink } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showAdminForm, setShowAdminForm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [linkSentTo, setLinkSentTo] = useState('');

  const handleAdminLogin = async (e) => {
    e.preventDefault();
    if (!email || !password) {
      toast.error('Enter your email and password, or use "Email me a sign-in link".');
      return;
    }
    setLoading(true);
    try {
      await signInWithEmail(email, password);
      // Admin tabs appear once the server confirms this account is an admin.
      toast.success('Signed in');
      onClose();
    } catch (err) {
      toast.error(err.message || 'Sign-in failed');
    } finally {
      setLoading(false);
    }
  };

  const handleEmailLink = async () => {
    const address = email.trim();
    if (!address) {
      toast.error('Enter your email first.');
      return;
    }
    setLoading(true);
    try {
      await signInWithEmailLink(address);
      setLinkSentTo(address);
    } catch (err) {
      toast.error(err.message || 'Could not send the sign-in link');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm animate-fade-in">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 relative">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 text-xl"
        >
          ✕
        </button>

        <h2 className="text-2xl font-bold text-gray-800 mb-4 text-center">Sign In</h2>

        {!showAdminForm ? (
          <>
            <p className="text-sm text-gray-600 mb-4 text-center">
              Choose a sign-in method
            </p>
            <div className="space-y-3">
              <button
                onClick={signInWithGoogle}
                className="w-full flex items-center justify-center gap-3 bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 font-medium py-3 px-4 rounded-xl transition shadow-sm"
              >
                <span className="text-xl">G</span> Continue with Google
              </button>
              <button
                onClick={signInWithYahoo}
                className="w-full flex items-center justify-center gap-3 bg-purple-600 hover:bg-purple-700 text-white font-medium py-3 px-4 rounded-xl transition shadow-sm"
              >
                <span className="text-xl">Y!</span> Continue with Yahoo
              </button>
              <div className="flex items-center gap-3 text-xs text-gray-500" aria-hidden="true">
                <span className="flex-1 h-px bg-gray-300" />
                or
                <span className="flex-1 h-px bg-gray-300" />
              </div>
              <button
                onClick={() => setShowAdminForm(true)}
                className="w-full flex items-center justify-center gap-3 bg-gray-100 hover:bg-gray-200 border border-gray-200 text-gray-700 font-medium py-3 px-4 rounded-xl transition shadow-sm"
              >
                🔐 Admin Login
              </button>
            </div>
          </>
        ) : (
          <>
            <p className="text-sm text-gray-600 mb-4 text-center">
              Admin sign‑in. Admins can also use Google or Yahoo if their email is on the admin list.
            </p>
            {linkSentTo ? (
              <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-4 text-sm text-emerald-900 mb-4" role="status">
                📧 We sent a sign-in link to <strong>{linkSentTo}</strong>. Open it on this device to finish signing in.
              </div>
            ) : null}
            <form onSubmit={handleAdminLogin} className="space-y-4">
              <div>
                <label htmlFor="admin-email" className="block text-sm font-medium text-gray-700 mb-1">
                  Email
                </label>
                <input
                  id="admin-email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="admin@example.com"
                  className="w-full border border-gray-300 rounded-lg p-2 focus:ring-2 focus:ring-indigo-500"
                  required
                />
              </div>
              <div>
                <label htmlFor="admin-password" className="block text-sm font-medium text-gray-700 mb-1">
                  Password <span className="font-normal text-gray-400">(not needed for an email link)</span>
                </label>
                <input
                  id="admin-password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full border border-gray-300 rounded-lg p-2 focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-3 px-4 rounded-xl transition disabled:opacity-50"
              >
                {loading ? 'Signing in...' : 'Sign In with Password'}
              </button>
              <button
                type="button"
                onClick={handleEmailLink}
                disabled={loading}
                className="w-full bg-white border border-indigo-300 hover:bg-indigo-50 text-indigo-700 font-medium py-3 px-4 rounded-xl transition disabled:opacity-50"
              >
                📧 Email Me a Sign-in Link
              </button>
              <button
                type="button"
                onClick={() => setShowAdminForm(false)}
                className="w-full text-sm text-indigo-600 hover:underline py-1"
              >
                ← Back to options
              </button>
            </form>
          </>
        )}
      </div>
    </div>
  );
}
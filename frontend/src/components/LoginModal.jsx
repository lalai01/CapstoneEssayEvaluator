import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import toast from 'react-hot-toast';

export default function LoginModal({ isOpen, onClose }) {
  const { signInWithGoogle, signInWithYahoo, signInWithEmail } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showAdminForm, setShowAdminForm] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleAdminLogin = async (e) => {
    e.preventDefault();
    if (!email || !password) {
      toast.error('Please enter email and password');
      return;
    }
    setLoading(true);
    try {
      await signInWithEmail(email, password);
      toast.success('Logged in as Admin');
      onClose();
    } catch (err) {
      toast.error(err.message || 'Admin login failed');
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
              <div className="relative my-2">
                <hr className="border-gray-300" />
                <span className="absolute left-1/2 -translate-x-1/2 bg-white px-2 text-xs text-gray-500">or</span>
              </div>
              <button
                onClick={() => setShowAdminForm(true)}
                className="w-full flex items-center justify-center gap-2 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium py-3 px-4 rounded-xl transition"
              >
                🔐 Admin Login
              </button>
            </div>
          </>
        ) : (
          <>
            <p className="text-sm text-gray-600 mb-4 text-center">
              Admin sign‑in (email/password)
            </p>
            <form onSubmit={handleAdminLogin} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Email
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="admin@example.com"
                  className="w-full border border-gray-300 rounded-lg p-2 focus:ring-2 focus:ring-indigo-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Password
                </label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full border border-gray-300 rounded-lg p-2 focus:ring-2 focus:ring-indigo-500"
                  required
                />
              </div>
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-3 px-4 rounded-xl transition disabled:opacity-50"
              >
                {loading ? 'Signing in...' : 'Sign In as Admin'}
              </button>
              <button
                type="button"
                onClick={() => setShowAdminForm(false)}
                className="w-full text-sm text-indigo-600 hover:underline mt-2"
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
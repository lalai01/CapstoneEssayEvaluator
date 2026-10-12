import React, { useEffect, useState } from 'react';
import toast from 'react-hot-toast';
import { listAdmins, addAdmin, removeAdmin } from '../api';
import { useAuth } from '../context/AuthContext';
import { errorMessage } from '../lib/scoring';

// Grant or revoke admin access by email. The person can then sign in with
// any method (Google, Yahoo, email + password, or an emailed link) using that
// address, even if they have never signed in before.
export default function AdminUsers() {
  const { user, isAdmin } = useAuth();
  const [admins, setAdmins] = useState([]);
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      setAdmins(await listAdmins());
    } catch (err) {
      toast.error('Failed to load admins: ' + errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isAdmin) load();
  }, [isAdmin]);

  const add = async (e) => {
    e.preventDefault();
    const value = email.trim().toLowerCase();
    if (!value) return;
    setSaving(true);
    try {
      await addAdmin(value);
      toast.success(`${value} is now an admin`);
      setEmail('');
      load();
    } catch (err) {
      toast.error('Could not add admin: ' + errorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  const remove = async (target) => {
    if (!confirm(`Remove admin access for ${target}?`)) return;
    try {
      await removeAdmin(target);
      toast.success(`${target} is no longer an admin`);
      load();
    } catch (err) {
      toast.error('Could not remove admin: ' + errorMessage(err));
    }
  };

  if (!isAdmin) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 text-center">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Access Denied</h2>
        <p className="text-gray-600">Administrator access required.</p>
      </div>
    );
  }

  const myEmail = (user?.email || '').toLowerCase();

  return (
    <div className="glass-card rounded-2xl shadow-xl p-6 max-w-3xl mx-auto">
      <h2 className="text-2xl font-bold text-indigo-700 mb-1">👥 Admins</h2>
      <p className="text-sm text-gray-600 mb-6">
        Admins manage the school-wide rubric, surveys, OCR labels and other admins. Add someone by email.
        They can then sign in with Google, Yahoo, email and password, or an emailed sign-in link using that address.
      </p>

      <form onSubmit={add} className="flex flex-col sm:flex-row gap-2 mb-6">
        <label htmlFor="new-admin-email" className="sr-only">Admin email</label>
        <input
          id="new-admin-email"
          type="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="teacher@school.edu"
          className="flex-1 border border-gray-300 rounded-lg p-2.5 focus:ring-2 focus:ring-indigo-500"
        />
        <button
          type="submit"
          disabled={saving}
          className="bg-indigo-600 text-white px-4 py-2.5 rounded-lg hover:bg-indigo-700 transition disabled:opacity-50"
        >
          {saving ? 'Adding…' : '+ Add admin'}
        </button>
      </form>

      {loading ? (
        <p className="text-gray-500">Loading admins…</p>
      ) : (
        <ul className="divide-y border rounded-xl bg-white">
          {admins.map((a) => (
            <li key={a.email} className="flex flex-wrap items-center justify-between gap-2 p-3">
              <div className="min-w-0">
                <div className="font-medium text-gray-800 truncate">
                  {a.email}
                  {a.email === myEmail && <span className="ml-2 text-xs text-indigo-600">(you)</span>}
                </div>
                <div className="text-xs text-gray-500">
                  {a.builtin ? 'Built-in (server setting)' : `Added by ${a.added_by || 'unknown'}${a.created_at ? ` · ${new Date(a.created_at).toLocaleDateString()}` : ''}`}
                </div>
              </div>
              {a.builtin || a.email === myEmail ? (
                <span className="text-xs text-gray-400">{a.builtin ? 'Can’t remove here' : ''}</span>
              ) : (
                <button onClick={() => remove(a.email)} className="text-red-600 text-sm hover:underline">
                  Remove
                </button>
              )}
            </li>
          ))}
        </ul>
      )}

      <p className="text-xs text-gray-500 mt-4">
        Access only counts once the person's email is confirmed, so nobody can claim an admin address they don't own.
        Changes apply within a minute; the person may need to refresh the page.
      </p>
    </div>
  );
}

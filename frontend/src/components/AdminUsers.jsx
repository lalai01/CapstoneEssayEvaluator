import React, { useEffect, useMemo, useState } from 'react';
import toast from 'react-hot-toast';
import {
  listAdmins, addAdmin, removeAdmin,
  listTeachers, addTeacher, removeTeacher, setTeacherStatus,
  listRoleLog,
} from '../api';
import { useAuth } from '../context/AuthContext';
import { errorMessage } from '../lib/scoring';

const STATUS_STYLE = {
  active: { label: 'Active', className: 'bg-emerald-100 text-emerald-800' },
  inactive: { label: 'Inactive', className: 'bg-amber-100 text-amber-800' },
  removed: { label: 'Removed', className: 'bg-gray-200 text-gray-700' },
};

const ACTION_TEXT = {
  added: { verb: 'added', icon: '➕' },
  removed: { verb: 'removed', icon: '🗑️' },
  restored: { verb: 'restored', icon: '♻️' },
  activated: { verb: 'reactivated', icon: '✅' },
  deactivated: { verb: 'deactivated', icon: '⏸️' },
};

function when(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleString();
}

function SummaryCard({ label, value, tone }) {
  return (
    <div className={`rounded-xl border p-4 ${tone}`}>
      <div className="text-xs uppercase tracking-wide opacity-80">{label}</div>
      <div className="text-3xl font-bold">{value ?? '—'}</div>
    </div>
  );
}

function EmailForm({ inputId, label, buttonText, onSubmit }) {
  const [email, setEmail] = useState('');
  const [saving, setSaving] = useState(false);
  const submit = async (e) => {
    e.preventDefault();
    const value = email.trim().toLowerCase();
    if (!value) return;
    setSaving(true);
    try {
      await onSubmit(value);
      setEmail('');
    } finally {
      setSaving(false);
    }
  };
  return (
    <form onSubmit={submit} className="flex flex-col sm:flex-row gap-2 mb-3">
      <label htmlFor={inputId} className="sr-only">{label}</label>
      <input
        id={inputId}
        type="email"
        required
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="name@school.edu"
        className="flex-1 border border-gray-300 rounded-lg p-2.5 focus:ring-2 focus:ring-indigo-500"
      />
      <button
        type="submit"
        disabled={saving}
        className="bg-indigo-600 text-white px-4 py-2.5 rounded-lg hover:bg-indigo-700 transition disabled:opacity-50"
      >
        {saving ? 'Saving…' : buttonText}
      </button>
    </form>
  );
}

export default function AdminUsers() {
  const { user, isAdmin } = useAuth();
  const [teachers, setTeachers] = useState(null);
  const [admins, setAdmins] = useState(null);
  const [log, setLog] = useState(null);
  const [filter, setFilter] = useState('current');
  const [search, setSearch] = useState('');
  const [busy, setBusy] = useState('');

  const refresh = async () => {
    const [t, a, l] = await Promise.allSettled([listTeachers(true), listAdmins(), listRoleLog(100)]);
    if (t.status === 'fulfilled') setTeachers(t.value); else toast.error('Teachers: ' + errorMessage(t.reason));
    if (a.status === 'fulfilled') setAdmins(a.value); else toast.error('Admins: ' + errorMessage(a.reason));
    if (l.status === 'fulfilled') setLog(l.value); else setLog([]);
  };

  useEffect(() => {
    if (isAdmin) refresh();
  }, [isAdmin]);

  const counts = useMemo(() => {
    const c = { active: 0, inactive: 0, removed: 0 };
    for (const t of teachers || []) c[t.status] = (c[t.status] || 0) + 1;
    return c;
  }, [teachers]);

  const visibleTeachers = useMemo(() => {
    const q = search.trim().toLowerCase();
    return (teachers || []).filter((t) =>
      (filter === 'all' || (filter === 'current' ? t.status !== 'removed' : t.status === filter)) &&
      (!q || t.email.includes(q))
    );
  }, [teachers, filter, search]);

  const run = async (key, action, success) => {
    setBusy(key);
    try {
      await action();
      toast.success(success);
      await refresh();
    } catch (err) {
      toast.error(errorMessage(err));
    } finally {
      setBusy('');
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
    <div className="glass-card rounded-2xl shadow-xl p-6 max-w-5xl mx-auto space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-indigo-700 mb-1">🛡️ Admin Dashboard</h2>
        <p className="text-sm text-gray-600">
          Everyone who signs in is a <strong>Student</strong> unless listed here. <strong>Teachers</strong> are approvers:
          they can override AI scores, approve evaluations into the Knowledge Base, and create rubrics.
          Add people by email; they can then sign in with Google, Yahoo, password or an email link.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <SummaryCard label="Active teachers" value={teachers ? counts.active : null} tone="bg-emerald-50 border-emerald-200 text-emerald-900" />
        <SummaryCard label="Inactive teachers" value={teachers ? counts.inactive : null} tone="bg-amber-50 border-amber-200 text-amber-900" />
        <SummaryCard label="Removed teachers" value={teachers ? counts.removed : null} tone="bg-gray-50 border-gray-200 text-gray-800" />
        <SummaryCard label="Admins" value={admins ? admins.length : null} tone="bg-indigo-50 border-indigo-200 text-indigo-900" />
      </div>

      {/* ---------------- Teachers ---------------- */}
      <section>
        <h3 className="text-lg font-semibold text-gray-800 mb-1">👩‍🏫 Teachers (approvers)</h3>
        <p className="text-sm text-gray-600 mb-3">
          <strong>Inactive</strong> keeps the teacher on the list without teacher rights until you reactivate them.
          <strong> Remove</strong> is a soft delete: the record and who removed it are kept, and you can restore it.
        </p>
        <EmailForm
          inputId="new-teacher-email"
          label="Teacher email"
          buttonText="+ Add teacher"
          onSubmit={(email) => run(`add:${email}`, () => addTeacher(email), `${email} can now approve and override`)}
        />

        <div className="flex flex-wrap items-center gap-2 mb-3">
          {[
            { id: 'current', label: `Current (${counts.active + counts.inactive})` },
            { id: 'active', label: `Active (${counts.active})` },
            { id: 'inactive', label: `Inactive (${counts.inactive})` },
            { id: 'removed', label: `Removed (${counts.removed})` },
            { id: 'all', label: 'All' },
          ].map((f) => (
            <button
              key={f.id}
              onClick={() => setFilter(f.id)}
              className={`text-xs px-2.5 py-1 rounded-full border transition ${
                filter === f.id ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50'
              }`}
            >
              {f.label}
            </button>
          ))}
          <input
            type="search"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search email…"
            aria-label="Search teachers"
            className="ml-auto border border-gray-200 rounded-lg px-3 py-1.5 text-sm"
          />
        </div>

        {teachers === null ? (
          <p className="text-gray-500 text-sm">Loading…</p>
        ) : visibleTeachers.length === 0 ? (
          <p className="text-gray-500 text-sm">No teachers in this view.</p>
        ) : (
          <div className="overflow-x-auto border rounded-xl bg-white">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-gray-600">
                <tr>
                  <th className="text-left font-medium p-3">Teacher</th>
                  <th className="text-left font-medium p-3">Status</th>
                  <th className="text-left font-medium p-3">Added</th>
                  <th className="text-left font-medium p-3">Last change</th>
                  <th className="text-right font-medium p-3">Actions</th>
                </tr>
              </thead>
              <tbody>
                {visibleTeachers.map((t) => {
                  const style = STATUS_STYLE[t.status] || STATUS_STYLE.active;
                  const lastBy = t.status === 'removed' ? t.deleted_by : t.updated_by;
                  const lastAt = t.status === 'removed' ? t.deleted_at : t.updated_at;
                  const key = (k) => `${k}:${t.email}`;
                  return (
                    <tr key={t.email} className="border-t align-top">
                      <td className="p-3 font-medium text-gray-800 break-all">{t.email}</td>
                      <td className="p-3">
                        <span className={`text-xs px-2 py-0.5 rounded-full ${style.className}`}>{style.label}</span>
                      </td>
                      <td className="p-3 text-xs text-gray-600">
                        {t.added_by || 'unknown'}
                        <div className="text-gray-400">{when(t.created_at)}</div>
                      </td>
                      <td className="p-3 text-xs text-gray-600">
                        {lastBy ? (
                          <>
                            {t.status === 'removed' ? 'Removed by ' : 'Changed by '}{lastBy}
                            <div className="text-gray-400">{when(lastAt)}</div>
                          </>
                        ) : '—'}
                      </td>
                      <td className="p-3 text-right whitespace-nowrap space-x-3">
                        {t.status === 'active' && (
                          <button
                            disabled={!!busy}
                            onClick={() => run(key('off'), () => setTeacherStatus(t.email, 'inactive'), `${t.email} is now inactive`)}
                            className="text-amber-700 hover:underline disabled:opacity-40"
                          >
                            Deactivate
                          </button>
                        )}
                        {t.status === 'inactive' && (
                          <button
                            disabled={!!busy}
                            onClick={() => run(key('on'), () => setTeacherStatus(t.email, 'active'), `${t.email} is active again`)}
                            className="text-emerald-700 hover:underline disabled:opacity-40"
                          >
                            Activate
                          </button>
                        )}
                        {t.status !== 'removed' ? (
                          <button
                            disabled={!!busy}
                            onClick={() => {
                              if (confirm(`Remove ${t.email} as a teacher? The record is kept and can be restored.`)) {
                                run(key('rm'), () => removeTeacher(t.email), `${t.email} was removed`);
                              }
                            }}
                            className="text-red-600 hover:underline disabled:opacity-40"
                          >
                            Remove
                          </button>
                        ) : (
                          <button
                            disabled={!!busy}
                            onClick={() => run(key('restore'), () => addTeacher(t.email), `${t.email} was restored`)}
                            className="text-indigo-700 hover:underline disabled:opacity-40"
                          >
                            Restore
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* ---------------- Admins ---------------- */}
      <section>
        <h3 className="text-lg font-semibold text-gray-800 mb-1">🛡️ Admins</h3>
        <p className="text-sm text-gray-600 mb-3">
          Everything a teacher can do, plus this dashboard, surveys, OCR labels and the school-wide rubric.
        </p>
        <EmailForm
          inputId="new-admin-email"
          label="Admin email"
          buttonText="+ Add admin"
          onSubmit={(email) => run(`admin:${email}`, () => addAdmin(email), `${email} is now an admin`)}
        />
        {admins === null ? (
          <p className="text-gray-500 text-sm">Loading…</p>
        ) : (
          <ul className="divide-y border rounded-xl bg-white">
            {admins.map((a) => (
              <li key={a.email} className="flex flex-wrap items-center justify-between gap-2 p-3">
                <div className="min-w-0">
                  <div className="font-medium text-gray-800 break-all">
                    {a.email}
                    {a.email === myEmail && <span className="ml-2 text-xs text-indigo-600">(you)</span>}
                  </div>
                  <div className="text-xs text-gray-500">
                    {a.builtin ? 'Built-in (server setting)' : `Added by ${a.added_by || 'unknown'}${a.created_at ? ` · ${when(a.created_at)}` : ''}`}
                  </div>
                </div>
                {a.builtin || a.email === myEmail ? (
                  <span className="text-xs text-gray-400">{a.builtin ? 'Can’t remove here' : ''}</span>
                ) : (
                  <button
                    disabled={!!busy}
                    onClick={() => {
                      if (confirm(`Remove admin access for ${a.email}?`)) {
                        run(`admin-rm:${a.email}`, () => removeAdmin(a.email), `${a.email} is no longer an admin`);
                      }
                    }}
                    className="text-red-600 text-sm hover:underline disabled:opacity-40"
                  >
                    Remove
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* ---------------- Activity log ---------------- */}
      <section>
        <h3 className="text-lg font-semibold text-gray-800 mb-1">📜 Activity log</h3>
        <p className="text-sm text-gray-600 mb-3">Who added, removed, activated or deactivated teachers and admins (latest 100).</p>
        {log === null ? (
          <p className="text-gray-500 text-sm">Loading…</p>
        ) : log.length === 0 ? (
          <p className="text-gray-500 text-sm">No changes recorded yet.</p>
        ) : (
          <ol className="border rounded-xl bg-white divide-y max-h-96 overflow-y-auto">
            {log.map((e, i) => {
              const a = ACTION_TEXT[e.action] || { verb: e.action, icon: '•' };
              return (
                <li key={e.id ?? i} className="flex flex-wrap items-baseline gap-x-2 p-3 text-sm">
                  <span aria-hidden="true">{a.icon}</span>
                  <span className="font-medium text-gray-800">{e.actor || 'Someone'}</span>
                  <span className="text-gray-600">{a.verb} {e.role}</span>
                  <span className="font-medium text-gray-800 break-all">{e.email}</span>
                  <span className="ml-auto text-xs text-gray-400">{when(e.created_at)}</span>
                </li>
              );
            })}
          </ol>
        )}
      </section>
    </div>
  );
}

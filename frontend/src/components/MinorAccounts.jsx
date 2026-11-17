import React, { useEffect, useState } from 'react';
import toast from 'react-hot-toast';
import { listMinorAccounts, updateMinorAccount } from '../api';
import { errorMessage, matchesSearch } from '../lib/scoring';

// Teachers and admins: student accounts under 18 (RA 10173). A minor can't
// evaluate or upload until someone here confirms that the school has the
// parent's or guardian's consent (for example, the form signed at
// enrollment). Every change is logged in the admin's activity log.

const when = (iso) => {
  const d = iso ? new Date(iso) : null;
  return d && !Number.isNaN(d.getTime()) ? d.toLocaleString() : '';
};

export default function MinorAccounts() {
  const [rows, setRows] = useState(null);
  const [reload, setReload] = useState(0);
  const [busy, setBusy] = useState('');
  const [search, setSearch] = useState('');
  const [notes, setNotes] = useState({});   // email → note typed for the consent

  useEffect(() => {
    let alive = true;
    listMinorAccounts()
      .then((d) => { if (alive) setRows(d); })
      .catch((err) => { if (alive) setRows({ error: errorMessage(err) }); });
    return () => { alive = false; };
  }, [reload]);

  const run = async (email, action, message, note) => {
    setBusy(`${action}:${email}`);
    try {
      await updateMinorAccount(email, action, note);
      toast.success(message);
      setReload((n) => n + 1);
    } catch (err) {
      toast.error(errorMessage(err));
    } finally {
      setBusy('');
    }
  };

  const list = Array.isArray(rows) ? rows.filter((r) => matchesSearch(search, r.email, r.name)) : [];
  const pending = list.filter((r) => !r.guardian_consent);
  const confirmed = list.filter((r) => r.guardian_consent);

  const renderRow = (r) => {
    const c = r.guardian_consent;
    return (
      <li key={r.email} className="flex flex-wrap items-start justify-between gap-3 p-3">
        <div className="min-w-0">
          <div className="font-medium text-gray-800 break-all">{r.name ? `${r.name} · ` : ''}{r.email}</div>
          <div className="text-xs text-gray-500">
            Said they are under 18{r.declared_at ? ` on ${when(r.declared_at)}` : ''}
            {c && <> · consent confirmed by {c.by || 'a teacher'}{c.at ? ` on ${when(c.at)}` : ''}{c.note ? ` (${c.note})` : ''}</>}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {!c && (
            <>
              <input
                value={notes[r.email] || ''}
                onChange={(e) => setNotes({ ...notes, [r.email]: e.target.value })}
                placeholder="Note, e.g. form on file, 7-Rizal"
                aria-label={`Note for ${r.email}`}
                className="w-56 rounded-lg border border-gray-300 p-1.5 text-sm"
              />
              <button
                disabled={!!busy}
                onClick={() => {
                  if (confirm(`Confirm that the school has the parent's or guardian's consent for ${r.email}?`)) {
                    run(r.email, 'confirm_consent', `${r.email} can now evaluate essays`, notes[r.email] || null);
                  }
                }}
                className="rounded-lg bg-emerald-600 px-3 py-1.5 text-sm font-semibold text-white hover:bg-emerald-700 disabled:opacity-50"
              >
                ✓ Consent on file
              </button>
            </>
          )}
          {c && (
            <button
              disabled={!!busy}
              onClick={() => { if (confirm(`Withdraw the consent of ${r.email}? They won't be able to evaluate essays.`)) run(r.email, 'withdraw_consent', `Consent withdrawn for ${r.email}`); }}
              className="rounded-lg border border-red-300 px-3 py-1.5 text-sm text-red-700 hover:bg-red-50 disabled:opacity-50"
            >
              Withdraw consent
            </button>
          )}
          <button
            disabled={!!busy}
            onClick={() => { if (confirm(`Mark ${r.email} as 18 or older? Do this only if you have checked their age.`)) run(r.email, 'set_adult', `${r.email} is now marked 18 or older`); }}
            className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50 disabled:opacity-50"
          >
            Is 18 or older
          </button>
        </div>
      </li>
    );
  };

  return (
    <div className="glass-card rounded-2xl shadow-xl p-6 bg-white/95 space-y-5 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-gray-800">🧒 Student Consent (under 18)</h1>
        <p className="mt-1 text-sm text-gray-600">
          Under the Data Privacy Act (RA 10173), students under 18 need a parent’s or guardian’s consent before their
          essays are processed. These students said they are under 18. Confirm consent only when the school has it on
          file (for example, the form signed at enrollment). Until then they can’t evaluate, upload or answer surveys.
          Their essays always get the minor’s protections: personal details removed, photos not kept, deletion after
          about one school year.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search by name or email"
          className="flex-1 min-w-[12rem] rounded-lg border border-gray-300 p-2.5"
        />
        <button onClick={() => setReload((n) => n + 1)} className="rounded-lg border px-3 py-2 text-sm text-gray-700 hover:bg-gray-50">
          ↻ Refresh
        </button>
      </div>

      {rows === null && <p className="text-gray-500">Loading…</p>}
      {rows?.error && <p className="text-red-600">{rows.error}</p>}

      {Array.isArray(rows) && (
        <>
          <section>
            <h2 className="text-lg font-semibold text-amber-800">⏳ Waiting for consent ({pending.length})</h2>
            {pending.length === 0
              ? <p className="mt-1 text-sm text-gray-500">No student is waiting.</p>
              : <ul className="mt-2 divide-y rounded-xl border bg-white">{pending.map(renderRow)}</ul>}
          </section>
          <section>
            <h2 className="text-lg font-semibold text-emerald-800">✓ Consent on file ({confirmed.length})</h2>
            {confirmed.length === 0
              ? <p className="mt-1 text-sm text-gray-500">None yet.</p>
              : <ul className="mt-2 divide-y rounded-xl border bg-white">{confirmed.map(renderRow)}</ul>}
          </section>
        </>
      )}
    </div>
  );
}

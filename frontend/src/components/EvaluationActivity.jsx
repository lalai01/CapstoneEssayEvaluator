import React, { useEffect, useMemo, useState } from 'react';
import toast from 'react-hot-toast';
import { getEvaluationActivity, getEvaluationEssay } from '../api';
import { useAuth } from '../context/AuthContext';
import { errorMessage, getCriteria, matchesSearch } from '../lib/scoring';
import { downloadText, toCsv } from '../lib/surveyKit';

// Status of an evaluated essay: icon + label + colour (never colour alone).
const STATUS = {
  draft: { icon: '📝', label: 'Draft (not saved)', cls: 'bg-gray-100 text-gray-700 border-gray-200' },
  saved: { icon: '💾', label: 'Saved', cls: 'bg-sky-50 text-sky-800 border-sky-200' },
  approved: { icon: '✅', label: 'Approved (teacher)', cls: 'bg-emerald-50 text-emerald-800 border-emerald-200' },
  overridden: { icon: '✏️', label: 'Overridden', cls: 'bg-amber-50 text-amber-800 border-amber-200' },
  rejected: { icon: '⛔', label: 'Rejected & overridden (teacher)', cls: 'bg-red-50 text-red-700 border-red-200' },
};
const ROLE = { teacher: 'Teacher', admin: 'Admin', student: 'Student', unknown: '—' };
const fmt = (at) => (at ? new Date(at).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '—');

export function StatusBadge({ status }) {
  const s = STATUS[status] || STATUS.draft;
  return <span className={`inline-flex items-center gap-1 whitespace-nowrap rounded-full border px-2 py-0.5 text-xs font-medium ${s.cls}`}>{s.icon} {s.label}</span>;
}

function Who({ row }) {
  return (
    <div className="min-w-0">
      <div className="truncate text-gray-800">{row.name || row.email || 'Unknown user'}</div>
      <div className="text-xs text-gray-500">{ROLE[row.role] || row.role}{row.name && row.email ? ` · ${row.email}` : ''}</div>
    </div>
  );
}

// Scores of one AI run, criterion by criterion.
export function ScoreList({ scores }) {
  if (!scores) return null;
  if (typeof scores.holistic_score === 'number') {
    return <p className="text-sm">Holistic: <strong>{scores.holistic_points ?? scores.holistic_score}/{scores.holistic_max ?? 5}</strong> (level {scores.holistic_score} of 5)</p>;
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {getCriteria(scores).map((c) => (
        <span key={c.key} className="rounded-md bg-gray-100 px-2 py-0.5 text-xs text-gray-700">
          {c.label}: <strong>{scores[c.key] ?? '—'}</strong>/{c.max_score}
        </span>
      ))}
    </div>
  );
}

function EssayDetail({ essay, onClose }) {
  const [data, setData] = useState(null);
  useEffect(() => {
    let alive = true;
    getEvaluationEssay(essay.user_id, essay.essay_hash)
      .then((d) => { if (alive) setData(d); })
      .catch((err) => { if (alive) setData({ error: errorMessage(err) }); });
    return () => { alive = false; };
  }, [essay]);

  return (
    <div className="fixed inset-0 z-[60] flex justify-end bg-black/40" onMouseDown={onClose}>
      <aside className="h-full w-full max-w-2xl overflow-y-auto bg-white p-6 shadow-2xl" onMouseDown={(e) => e.stopPropagation()}
        role="dialog" aria-modal="true" aria-labelledby="essay-detail-title">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 id="essay-detail-title" className="text-xl font-bold text-gray-900">{essay.title}</h3>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-sm text-gray-600">
              <StatusBadge status={essay.status} /> <span>{essay.name || essay.email} · {ROLE[essay.role] || essay.role}</span>
            </div>
          </div>
          <button onClick={onClose} className="rounded-lg px-2 py-1 text-gray-500 hover:bg-gray-100" aria-label="Close">✕</button>
        </div>

        {!data ? <p className="mt-6 text-gray-500">Loading…</p> : data.error ? <p className="mt-6 text-red-600">{data.error}</p> : (
          <div className="mt-5 space-y-6">
            <section>
              <h4 className="font-semibold text-gray-800">Essay</h4>
              <p className="mt-2 max-h-64 overflow-y-auto whitespace-pre-wrap rounded-lg border bg-gray-50 p-3 text-sm text-gray-800">{data.essay || '—'}</p>
            </section>

            <section>
              <h4 className="font-semibold text-gray-800">AI evaluations ({data.runs.length})</h4>
              {data.runs.length === 0 && <p className="mt-1 text-sm text-gray-500">No runs recorded (made before activity logging started).</p>}
              <div className="mt-2 space-y-2">
                {data.runs.map((r) => (
                  <div key={r.id} className="rounded-lg border p-3">
                    <div className="flex flex-wrap justify-between gap-2 text-sm">
                      <strong className="text-gray-800">{r.ai_label || r.ai_choice}</strong>
                      <span className="text-gray-500">{fmt(r.created_at)}</span>
                    </div>
                    <div className="mt-1 text-xs text-gray-500">
                      {r.eval_type} · overall <strong className="text-gray-800">{r.overall || '—'}</strong>
                      {r.rubric_title ? ` · rubric: ${r.rubric_title}` : ''}{r.used_rag ? ' · with RAG' : ''}
                    </div>
                    <div className="mt-2"><ScoreList scores={r.scores} /></div>
                  </div>
                ))}
              </div>
            </section>

            {data.kb.length > 0 && (
              <section>
                <h4 className="font-semibold text-gray-800">Saved to the Knowledge Base</h4>
                {data.kb.map((k) => (
                  <div key={k.id} className="mt-2 rounded-lg border border-sky-200 bg-sky-50/50 p-3 text-sm">
                    {fmt(k.created_at)} · {k.approved_by ? `✅ approved by ${k.approved_by}` : 'not approved'}
                    {k.satisfaction ? ` · satisfaction ${k.satisfaction}/10` : ''}
                  </div>
                ))}
              </section>
            )}

            {data.overrides.length > 0 && (
              <section>
                <h4 className="font-semibold text-gray-800">Overrides</h4>
                {data.overrides.map((o) => (
                  <div key={o.id} className={`mt-2 rounded-lg border p-3 text-sm ${o.approved_by ? 'border-red-200 bg-red-50/50' : 'border-amber-200 bg-amber-50/50'}`}>
                    <div>{fmt(o.created_at)} · {o.approved_by ? `⛔ rejected & overridden by teacher ${o.approved_by}` : '✏️ overridden'}</div>
                    <div className="mt-1"><span className="text-gray-500">Remarks:</span> {o.teacher_feedback || '—'}</div>
                    {o.suggested_changes && <div><span className="text-gray-500">Suggestions:</span> {o.suggested_changes}</div>}
                  </div>
                ))}
              </section>
            )}
          </div>
        )}
      </aside>
    </div>
  );
}

export default function EvaluationActivity() {
  const { isAdmin } = useAuth();
  const [data, setData] = useState(null);
  const [days, setDays] = useState(180);
  const [tab, setTab] = useState('essays');
  const [status, setStatus] = useState('all');
  const [role, setRole] = useState('all');
  const [person, setPerson] = useState('all');
  const [search, setSearch] = useState('');
  const [open, setOpen] = useState(null);

  // Loads when the period changes or Refresh is pressed; the previous result
  // stays visible until the new one arrives.
  const [reload, setReload] = useState(0);
  const want = `${days}:${reload}`;
  useEffect(() => {
    if (!isAdmin) return undefined;
    let alive = true;
    getEvaluationActivity(days)
      .then((d) => { if (alive) setData({ ...d, key: want }); })
      .catch((err) => {
        if (!alive) return;
        toast.error('Could not load the activity: ' + errorMessage(err));
        setData({ essays: [], overrides: [], saved: [], counts: {}, error: true, key: want });
      });
    return () => { alive = false; };
  }, [isAdmin, days, want]);
  const loading = !data || data.key !== want;

  const people = useMemo(() => {
    const all = [...(data?.essays || []), ...(data?.overrides || []), ...(data?.saved || [])];
    return [...new Map(all.filter((r) => r.user_id).map((r) => [r.user_id, r.name || r.email || r.user_id])).entries()]
      .sort((a, b) => String(a[1]).localeCompare(String(b[1])));
  }, [data]);

  const keep = (r, extra = []) => (role === 'all' || r.role === role)
    && (person === 'all' || r.user_id === person)
    && matchesSearch(search, r.title, r.email, r.name, r.ai, ...extra);
  const essays = (data?.essays || []).filter((e) => (status === 'all' || e.status === status) && keep(e, [e.essay, ...(e.ais || [])]));
  const overrides = (data?.overrides || []).filter((o) => keep(o, [o.remarks, o.suggestions]));
  const saved = (data?.saved || []).filter((s) => keep(s));

  const exportCsv = () => {
    const rows = tab === 'essays'
      ? [['Last activity', 'Person', 'Role', 'Title', 'Status', 'AI runs', 'Words'],
        ...essays.map((e) => [e.last_at, e.email || e.name, e.role, e.title, STATUS[e.status].label,
          e.runs.map((r) => `${r.ai}: ${r.overall || '—'}`).join('; '), e.essay_length])]
      : tab === 'overrides'
        ? [['Date', 'Person', 'Role', 'Title', 'Type', 'AI', 'AI overall', 'Corrected overall', 'Changes', 'Remarks', 'Suggestions'],
          ...overrides.map((o) => [o.at, o.email || o.name, o.role, o.title, o.kind === 'rejected' ? 'Rejected & overridden (teacher)' : 'Overridden',
            o.ai, o.ai_overall, o.new_overall, o.changes.map((c) => `${c.label} ${c.ai}→${c.new}`).join('; '), o.remarks, o.suggestions])]
        : [['Date', 'Person', 'Role', 'Title', 'AI', 'Overall', 'Approved', 'Approved by', 'Satisfaction'],
          ...saved.map((s) => [s.at, s.email || s.name, s.role, s.title, s.ai, s.overall, s.approved ? 'Yes' : 'No', s.approved_by, s.satisfaction])];
    downloadText(`evaluation-activity-${tab}.csv`, toCsv(rows));
  };

  if (!isAdmin) {
    return <div className="glass-card rounded-2xl shadow-xl p-12 text-center"><h2 className="text-2xl font-bold text-gray-800">Admins only</h2></div>;
  }

  const counts = data?.counts || {};
  const tabs = [
    ['essays', `Essays evaluated (${data?.essays?.length ?? 0})`],
    ['overrides', `Overrides & rejections (${data?.overrides?.length ?? 0})`],
    ['saved', `Saved to Knowledge Base (${data?.saved?.length ?? 0})`],
  ];

  return (
    <div className="glass-card rounded-2xl shadow-xl p-6 bg-white/95 space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">📊 Evaluation Activity</h2>
          <p className="text-sm text-gray-600">Every essay evaluated by signed-in users, and what happened to it: left as a draft, saved, approved, overridden or rejected.</p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <select value={days} onChange={(e) => setDays(Number(e.target.value))} className="rounded-lg border p-2" aria-label="Period">
            <option value={7}>Last 7 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
            <option value={180}>Last 6 months</option>
            <option value={365}>Last 12 months</option>
            <option value={3650}>All time</option>
          </select>
          <button onClick={() => setReload((n) => n + 1)} disabled={loading} className="rounded-lg bg-gray-100 px-3 py-2 hover:bg-gray-200 disabled:opacity-50">{loading ? '⏳ Loading…' : '🔄 Refresh'}</button>
          <button onClick={exportCsv} disabled={!data} className="rounded-lg bg-emerald-600 px-3 py-2 text-white hover:bg-emerald-700 disabled:opacity-50">⬇ CSV</button>
        </div>
      </div>

      {data && data.table_ready === false && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
          ⚠️ Evaluation runs aren’t being recorded yet, so drafts (essays evaluated but not saved) can’t be shown.
          {' '}{data.setup_hint} Overrides and Knowledge Base saves below are complete.
        </div>
      )}

      {/* Status tiles: also a filter */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
        <button onClick={() => { setStatus('all'); setTab('essays'); }}
          className={`rounded-xl border p-3 text-left ${status === 'all' && tab === 'essays' ? 'ring-2 ring-indigo-400' : ''}`}>
          <div className="text-xs text-gray-500">All essays</div>
          <div className="text-2xl font-bold text-gray-900 tabular-nums">{data?.essays?.length ?? '…'}</div>
        </button>
        {Object.entries(STATUS).map(([key, s]) => (
          <button key={key} onClick={() => { setStatus(key); setTab('essays'); }}
            className={`rounded-xl border p-3 text-left ${s.cls} ${status === key && tab === 'essays' ? 'ring-2 ring-indigo-400' : ''}`}>
            <div className="text-xs">{s.icon} {s.label}</div>
            <div className="text-2xl font-bold tabular-nums">{data ? counts[key] ?? 0 : '…'}</div>
          </button>
        ))}
      </div>

      {/* Tabs and filters */}
      <div className="flex flex-wrap gap-1 border-b">
        {tabs.map(([key, label]) => (
          <button key={key} onClick={() => setTab(key)}
            className={`-mb-px border-b-2 px-3 py-2 text-sm ${tab === key ? 'border-indigo-600 font-semibold text-indigo-700' : 'border-transparent text-gray-600 hover:text-gray-900'}`}>
            {label}
          </button>
        ))}
      </div>
      <div className="flex flex-wrap gap-2 text-sm">
        <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search title, person, essay, remarks…"
          className="min-w-[14rem] flex-1 rounded-lg border p-2" />
        <select value={role} onChange={(e) => setRole(e.target.value)} className="rounded-lg border p-2" aria-label="Role">
          <option value="all">All roles</option>
          <option value="teacher">Teachers</option>
          <option value="student">Students</option>
          <option value="admin">Admins</option>
        </select>
        <select value={person} onChange={(e) => setPerson(e.target.value)} className="max-w-[16rem] rounded-lg border p-2" aria-label="Person">
          <option value="all">Everyone</option>
          {people.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
        </select>
        {tab === 'essays' && status !== 'all' && (
          <button onClick={() => setStatus('all')} className="rounded-lg border px-3 py-2 text-gray-600 hover:bg-gray-50">✕ {STATUS[status].label}</button>
        )}
      </div>

      {!data ? <p className="text-gray-500">Loading…</p> : (
        <div className={`overflow-x-auto transition-opacity ${loading ? 'opacity-50' : ''}`}>
          {tab === 'essays' && (
            <table className="min-w-full text-sm">
              <thead><tr className="border-b text-left text-xs text-gray-500">
                <th className="py-2 pr-3">Essay</th><th className="py-2 pr-3">Person</th><th className="py-2 pr-3">AI runs (overall)</th>
                <th className="py-2 pr-3">Status</th><th className="py-2">Last activity</th>
              </tr></thead>
              <tbody>
                {essays.length === 0 && <tr><td colSpan={5} className="py-6 text-center text-gray-500">No essays match.</td></tr>}
                {essays.map((e) => (
                  <tr key={e.key} onClick={() => setOpen(e)} className="cursor-pointer border-b align-top hover:bg-indigo-50/40">
                    <td className="max-w-xs py-2 pr-3">
                      <div className="font-medium text-gray-900">{e.title}</div>
                      <div className="line-clamp-2 text-xs text-gray-500">{e.essay}</div>
                    </td>
                    <td className="py-2 pr-3"><Who row={e} /></td>
                    <td className="py-2 pr-3">
                      {e.runs.length === 0 ? <span className="text-xs text-gray-400">not recorded</span> : (
                        <div className="flex flex-wrap gap-1">
                          {e.runs.map((r, i) => <span key={i} className="rounded bg-gray-100 px-1.5 py-0.5 text-xs">{r.ai}: <strong>{r.overall || '—'}</strong></span>)}
                        </div>
                      )}
                    </td>
                    <td className="py-2 pr-3"><StatusBadge status={e.status} /></td>
                    <td className="whitespace-nowrap py-2 text-xs text-gray-600">{fmt(e.last_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {tab === 'overrides' && (
            <table className="min-w-full text-sm">
              <thead><tr className="border-b text-left text-xs text-gray-500">
                <th className="py-2 pr-3">Essay</th><th className="py-2 pr-3">Person</th><th className="py-2 pr-3">Type</th>
                <th className="py-2 pr-3">AI → corrected</th><th className="py-2 pr-3">Remarks</th><th className="py-2">Date</th>
              </tr></thead>
              <tbody>
                {overrides.length === 0 && <tr><td colSpan={6} className="py-6 text-center text-gray-500">No overrides match.</td></tr>}
                {overrides.map((o) => (
                  <tr key={o.id} className="border-b align-top">
                    <td className="max-w-[14rem] py-2 pr-3"><div className="font-medium text-gray-900">{o.title}</div><div className="text-xs text-gray-500">{o.ai || 'AI not recorded'}</div></td>
                    <td className="py-2 pr-3"><Who row={o} /></td>
                    <td className="py-2 pr-3"><StatusBadge status={o.kind} /></td>
                    <td className="py-2 pr-3">
                      <div className="whitespace-nowrap"><strong>{o.ai_overall || '—'}</strong> → <strong>{o.new_overall || '—'}</strong></div>
                      <div className="text-xs text-gray-600">{o.changes.map((c) => `${c.label} ${c.ai}→${c.new}`).join(', ') || 'no score changes'}</div>
                    </td>
                    <td className="max-w-xs py-2 pr-3 text-xs text-gray-700">
                      <div className="line-clamp-3">{o.remarks || '—'}</div>
                      {o.suggestions && <div className="mt-1 line-clamp-2 text-gray-500">Suggestions: {o.suggestions}</div>}
                    </td>
                    <td className="whitespace-nowrap py-2 text-xs text-gray-600">{fmt(o.at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {tab === 'saved' && (
            <table className="min-w-full text-sm">
              <thead><tr className="border-b text-left text-xs text-gray-500">
                <th className="py-2 pr-3">Essay</th><th className="py-2 pr-3">Person</th><th className="py-2 pr-3">AI</th>
                <th className="py-2 pr-3">Overall</th><th className="py-2 pr-3">Approval</th><th className="py-2 pr-3">Satisfaction</th><th className="py-2">Date</th>
              </tr></thead>
              <tbody>
                {saved.length === 0 && <tr><td colSpan={7} className="py-6 text-center text-gray-500">No saved evaluations match.</td></tr>}
                {saved.map((s) => (
                  <tr key={s.id} className="border-b align-top">
                    <td className="py-2 pr-3 font-medium text-gray-900">{s.title}</td>
                    <td className="py-2 pr-3"><Who row={s} /></td>
                    <td className="py-2 pr-3 text-xs">{s.ai || '—'}</td>
                    <td className="py-2 pr-3 font-semibold">{s.overall || '—'}</td>
                    <td className="py-2 pr-3"><StatusBadge status={s.approved ? 'approved' : 'saved'} /></td>
                    <td className="py-2 pr-3">{s.satisfaction ? `${s.satisfaction}/10` : '—'}</td>
                    <td className="whitespace-nowrap py-2 text-xs text-gray-600">{fmt(s.at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {open && <EssayDetail essay={open} onClose={() => setOpen(null)} />}
    </div>
  );
}

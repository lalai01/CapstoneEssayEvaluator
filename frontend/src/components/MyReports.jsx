import React, { useEffect, useState } from 'react';
import toast from 'react-hot-toast';
import { getMyReportEssay, getMyReports } from '../api';
import { useAuth } from '../context/AuthContext';
import { errorMessage, matchesSearch } from '../lib/scoring';
import ReportDialog from './ReportDialog';
import { ScoreList, StatusBadge } from './EvaluationActivity';

// Every essay the signed-in user evaluated (teachers and students alike), with
// what happened to it, and the report of each AI result to view or download again.
const fmt = (at) => (at ? new Date(at).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '—');
const FILTERS = [
  ['all', 'All'],
  ['draft', '📝 Not saved'],
  ['saved', '💾 Saved'],
  ['approved', '✅ Approved'],
  ['overridden', '✏️ Overridden'],
  ['rejected', '⛔ Rejected'],
];

// The user's own corrected scores for this essay, if they overrode the AI.
function overrideScores(overrides) {
  const last = overrides[overrides.length - 1];
  if (!last) return null;
  const scores = last.teacher_scores || last.original_scores?.teacher_scores;
  return scores && Object.keys(scores).length ? scores : null;
}

function ReportDetail({ essay, onClose }) {
  const { user } = useAuth();
  const [data, setData] = useState(null);
  const [report, setReport] = useState(null);   // the run whose PDF is being made

  useEffect(() => {
    let alive = true;
    getMyReportEssay(essay.essay_hash)
      .then((d) => { if (alive) setData(d); })
      .catch((err) => { if (alive) setData({ error: errorMessage(err) }); });
    return () => { alive = false; };
  }, [essay]);

  const override = data && !data.error ? data.overrides[data.overrides.length - 1] : null;

  return (
    <div className="fixed inset-0 z-[60] flex justify-end bg-black/40" onMouseDown={onClose}>
      <aside className="h-full w-full max-w-2xl overflow-y-auto bg-white p-6 shadow-2xl" onMouseDown={(e) => e.stopPropagation()}
        role="dialog" aria-modal="true" aria-labelledby="report-detail-title">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 id="report-detail-title" className="text-xl font-bold text-gray-900">{essay.title}</h3>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-sm text-gray-600">
              <StatusBadge status={essay.status} /> <span>{essay.essay_length} words · last activity {fmt(essay.last_at)}</span>
            </div>
          </div>
          <button onClick={onClose} className="rounded-lg px-2 py-1 text-gray-500 hover:bg-gray-100" aria-label="Close">✕</button>
        </div>

        {!data ? <p className="mt-6 text-gray-500">Loading…</p> : data.error ? <p className="mt-6 text-red-600">{data.error}</p> : (
          <div className="mt-5 space-y-6">
            <section>
              <h4 className="font-semibold text-gray-800">Reports ({data.runs.length})</h4>
              <p className="text-xs text-gray-500">One per AI result. Download any of them as a PDF again.</p>
              {data.runs.length === 0 && (
                <p className="mt-2 text-sm text-gray-500">This essay was evaluated before reports were recorded, so only its saved results are listed below.</p>
              )}
              <div className="mt-2 space-y-3">
                {data.runs.map((r) => (
                  <div key={r.id} className="rounded-xl border p-3">
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div>
                        <strong className="text-gray-800">{r.ai_label || r.ai_choice}</strong>
                        <div className="text-xs text-gray-500">
                          {fmt(r.created_at)} · {r.eval_type} · overall <strong className="text-gray-800">{r.overall || '—'}</strong>
                          {r.rubric_title ? ` · rubric: ${r.rubric_title}` : ''}{r.used_rag ? ' · with RAG' : ''}
                        </div>
                      </div>
                      <button onClick={() => setReport(r)} disabled={!r.scores}
                        className="rounded-lg bg-indigo-600 px-3 py-1.5 text-sm text-white hover:bg-indigo-700 disabled:opacity-50">
                        📄 Download report
                      </button>
                    </div>
                    <div className="mt-2"><ScoreList scores={r.scores} /></div>
                    {r.scores?.feedback_text && (
                      <details className="mt-2 rounded-lg bg-gray-50">
                        <summary className="cursor-pointer px-3 py-1.5 text-sm text-gray-700">Feedback</summary>
                        <p className="whitespace-pre-wrap px-3 pb-3 text-sm text-gray-700">{r.scores.feedback_text}</p>
                      </details>
                    )}
                  </div>
                ))}
              </div>
            </section>

            {override && (
              <section>
                <h4 className="font-semibold text-gray-800">{override.approved_by ? 'Teacher override' : 'Your override'}</h4>
                <div className={`mt-2 rounded-lg border p-3 text-sm ${override.approved_by ? 'border-red-200 bg-red-50/50' : 'border-amber-200 bg-amber-50/50'}`}>
                  <div className="text-gray-600">{fmt(override.created_at)}</div>
                  <div className="mt-1"><span className="text-gray-500">Remarks:</span> {override.teacher_feedback || '—'}</div>
                  {override.suggested_changes && <div><span className="text-gray-500">Suggestions:</span> {override.suggested_changes}</div>}
                  <p className="mt-1 text-xs text-gray-500">Downloaded reports include these corrected scores and remarks.</p>
                </div>
              </section>
            )}

            {data.kb.length > 0 && (
              <section>
                <h4 className="font-semibold text-gray-800">Saved to the Knowledge Base</h4>
                {data.kb.map((k) => (
                  <div key={k.id} className="mt-2 rounded-lg border border-sky-200 bg-sky-50/50 p-3 text-sm">
                    {fmt(k.created_at)} · {k.approved_by ? `✅ approved by ${k.approved_by}` : 'saved, not approved'}
                    {k.satisfaction ? ` · satisfaction ${k.satisfaction}/10` : ''}
                  </div>
                ))}
              </section>
            )}

            <section>
              <h4 className="font-semibold text-gray-800">Essay</h4>
              <p className="mt-2 max-h-72 overflow-y-auto whitespace-pre-wrap rounded-lg border bg-gray-50 p-3 text-sm text-gray-800">{data.essay || '—'}</p>
            </section>
          </div>
        )}
      </aside>

      {report && (
        <div onMouseDown={(e) => e.stopPropagation()}>
          <ReportDialog
            onClose={() => setReport(null)}
            defaults={{
              teacherName: user?.user_metadata?.full_name || user?.email || '',
              teacherComments: override
                ? [override.teacher_feedback, override.suggested_changes && `Suggestions: ${override.suggested_changes}`].filter(Boolean).join('\n\n')
                : '',
            }}
            reportData={{
              title: essay.title,
              essayText: data.essay,
              feedback: report.scores?.feedback_text || '',
              scores: report.scores,
              evalType: report.eval_type,
              teacherScores: overrideScores(data.overrides),
            }}
          />
        </div>
      )}
    </div>
  );
}

export default function MyReports() {
  const [data, setData] = useState(null);
  const [reload, setReload] = useState(0);
  const [status, setStatus] = useState('all');
  const [search, setSearch] = useState('');
  const [open, setOpen] = useState(null);

  useEffect(() => {
    let alive = true;
    getMyReports()
      .then((d) => { if (alive) setData({ ...d, key: reload }); })
      .catch((err) => {
        if (!alive) return;
        toast.error('Could not load your reports: ' + errorMessage(err));
        setData({ essays: [], counts: {}, key: reload, error: true });
      });
    return () => { alive = false; };
  }, [reload]);
  const loading = !data || data.key !== reload;

  const essays = (data?.essays || []).filter((e) => (status === 'all' || e.status === status)
    && matchesSearch(search, e.title, e.essay, ...(e.ais || [])));

  return (
    <div className="glass-card rounded-2xl shadow-xl p-6 bg-white/95 space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">📑 My Reports</h2>
          <p className="text-sm text-gray-600">Every essay you evaluated, with the result of each AI. Open one to read it again or download its report.</p>
        </div>
        <button onClick={() => setReload((n) => n + 1)} disabled={loading}
          className="rounded-lg bg-gray-100 px-3 py-2 text-sm hover:bg-gray-200 disabled:opacity-50">{loading ? '⏳ Loading…' : '🔄 Refresh'}</button>
      </div>

      {data && data.table_ready === false && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
          ⚠️ Evaluations aren’t being recorded on the server yet, so only essays you saved or overrode are listed.
        </div>
      )}

      <div className="flex flex-wrap gap-2 text-sm">
        <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search title, essay or AI…"
          className="min-w-[14rem] flex-1 rounded-lg border p-2" aria-label="Search reports" />
        <div className="flex flex-wrap gap-1" role="group" aria-label="Filter by status">
          {FILTERS.map(([key, label]) => (
            <button key={key} onClick={() => setStatus(key)} aria-pressed={status === key}
              className={`rounded-lg border px-3 py-2 ${status === key ? 'border-indigo-500 bg-indigo-50 font-semibold text-indigo-700' : 'text-gray-600 hover:bg-gray-50'}`}>
              {label}{key !== 'all' && data ? ` (${data.counts?.[key] ?? 0})` : ''}
            </button>
          ))}
        </div>
      </div>

      {!data ? <p className="text-gray-500">Loading…</p> : (
        <div className={`space-y-2 transition-opacity ${loading ? 'opacity-50' : ''}`}>
          {essays.length === 0 && (
            <p className="py-8 text-center text-gray-500">
              {data.essays.length === 0 ? 'No reports yet. Evaluate an essay while signed in and it will appear here.' : 'No reports match.'}
            </p>
          )}
          {essays.map((e) => (
            <button key={e.key} onClick={() => setOpen(e)}
              className="block w-full rounded-xl border p-4 text-left transition hover:border-indigo-300 hover:bg-indigo-50/40">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="font-semibold text-gray-900">{e.title}</div>
                  <div className="text-xs text-gray-500">{fmt(e.last_at)} · {e.essay_length} words</div>
                </div>
                <StatusBadge status={e.status} />
              </div>
              <p className="mt-1 line-clamp-2 text-sm text-gray-600">{e.essay}</p>
              {e.runs.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {e.runs.map((r, i) => <span key={i} className="rounded bg-gray-100 px-1.5 py-0.5 text-xs">{r.ai}: <strong>{r.overall || '—'}</strong></span>)}
                </div>
              )}
            </button>
          ))}
        </div>
      )}

      {open && <ReportDetail essay={open} onClose={() => setOpen(null)} />}
    </div>
  );
}

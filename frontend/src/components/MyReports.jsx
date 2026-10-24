import React, { useEffect, useState } from 'react';
import toast from 'react-hot-toast';
import { getMyReportEssay, getMyReports } from '../api';
import { useAuth } from '../context/AuthContext';
import { errorMessage, matchesSearch } from '../lib/scoring';
import ReportDialog from './ReportDialog';
import { ScoreList, StatusBadge } from './EvaluationActivity';

// Every essay the signed-in user evaluated (teachers and students alike), with
// what happened to it, and the report of each AI result to view or download again.
// Clicking an essay expands it in place, full width, so nothing is covered.
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

// A section inside an expanded essay that opens and closes on its own.
function Section({ title, defaultOpen = true, tone = 'border-gray-200', children }) {
  return (
    <details open={defaultOpen} className={`group rounded-xl border bg-white ${tone}`}>
      <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-4 py-3 font-semibold text-gray-800">
        <span>{title}</span>
        <span className="text-xs text-gray-500 transition-transform group-open:rotate-180" aria-hidden="true">▼</span>
      </summary>
      <div className="border-t px-4 py-3">{children}</div>
    </details>
  );
}

function ReportDetail({ essay }) {
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

  if (!data) return <p className="px-4 py-6 text-gray-500">Loading…</p>;
  if (data.error) return <p className="px-4 py-6 text-red-600">{data.error}</p>;
  const override = data.overrides[data.overrides.length - 1] || null;

  return (
    <div className="space-y-3 border-t bg-gray-50/70 p-4">
      <Section title={`📄 Reports (${data.runs.length}): one per AI result`}>
        {data.runs.length === 0 && (
          <p className="text-sm text-gray-500">This essay was evaluated before reports were recorded, so only its saved results are listed below.</p>
        )}
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2 2xl:grid-cols-3">
          {data.runs.map((r) => (
            <div key={r.id} className="flex flex-col rounded-xl border bg-white p-3">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div className="min-w-0">
                  <strong className="text-gray-800">{r.ai_label || r.ai_choice}</strong>
                  <div className="text-xs text-gray-500">
                    {fmt(r.created_at)} · {r.eval_type} · overall <strong className="text-gray-800">{r.overall || '—'}</strong>
                    {r.rubric_title ? ` · rubric: ${r.rubric_title}` : ''}{r.used_rag ? ' · with RAG' : ''}
                  </div>
                </div>
                <button onClick={() => setReport(r)} disabled={!r.scores}
                  className="shrink-0 rounded-lg bg-indigo-600 px-3 py-1.5 text-sm text-white hover:bg-indigo-700 disabled:opacity-50">
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
      </Section>

      {data.overrides.length > 0 && (
        <Section title={`✏️ ${override.approved_by ? 'Teacher override' : 'Your override'}${data.overrides.length > 1 ? `s (${data.overrides.length})` : ''}`}
          tone={override.approved_by ? 'border-red-200' : 'border-amber-200'}>
          <div className="space-y-3">
            {data.overrides.map((o) => (
              <div key={o.id} className="text-sm">
                <div className="text-xs text-gray-500">{fmt(o.created_at)}{o.approved_by ? ` · by teacher ${o.approved_by}` : ''}</div>
                <div className="mt-1"><span className="font-medium text-gray-600">Remarks:</span> <span className="whitespace-pre-wrap">{o.teacher_feedback || '—'}</span></div>
                {o.suggested_changes && (
                  <div className="mt-1"><span className="font-medium text-gray-600">Suggestions:</span> <span className="whitespace-pre-wrap">{o.suggested_changes}</span></div>
                )}
              </div>
            ))}
            <p className="text-xs text-gray-500">Downloaded reports include the latest corrected scores and remarks.</p>
          </div>
        </Section>
      )}

      {data.kb.length > 0 && (
        <Section title="💾 Saved to the Knowledge Base" tone="border-sky-200">
          {data.kb.map((k) => (
            <div key={k.id} className="text-sm">
              {fmt(k.created_at)} · {k.approved_by ? `✅ approved by ${k.approved_by}` : 'saved, not approved'}
              {k.satisfaction ? ` · satisfaction ${k.satisfaction}/10` : ''}
            </div>
          ))}
        </Section>
      )}

      <Section title={`📝 Essay (${essay.essay_length} words)`} defaultOpen={false}>
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-gray-800">{data.essay || '—'}</p>
      </Section>

      {report && (
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
      )}
    </div>
  );
}

export default function MyReports() {
  const [data, setData] = useState(null);
  const [reload, setReload] = useState(0);
  const [status, setStatus] = useState('all');
  const [search, setSearch] = useState('');
  const [expanded, setExpanded] = useState(() => new Set());

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
  const toggle = (key) => setExpanded((prev) => {
    const next = new Set(prev);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    return next;
  });

  return (
    <div className="glass-card rounded-2xl shadow-xl p-6 bg-white/95 space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">📑 My Reports</h2>
          <p className="text-sm text-gray-600">Every essay you evaluated, with the result of each AI. Click one to expand it, read it again or download its report.</p>
        </div>
        <div className="flex flex-wrap gap-2 text-sm">
          {expanded.size > 0 && (
            <button onClick={() => setExpanded(new Set())} className="rounded-lg border px-3 py-2 text-gray-700 hover:bg-gray-50">Collapse all</button>
          )}
          <button onClick={() => setReload((n) => n + 1)} disabled={loading}
            className="rounded-lg bg-gray-100 px-3 py-2 hover:bg-gray-200 disabled:opacity-50">{loading ? '⏳ Loading…' : '🔄 Refresh'}</button>
        </div>
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
        <div className={`space-y-3 transition-opacity ${loading ? 'opacity-50' : ''}`}>
          {essays.length === 0 && (
            <p className="py-8 text-center text-gray-500">
              {data.essays.length === 0 ? 'No reports yet. Evaluate an essay while signed in and it will appear here.' : 'No reports match.'}
            </p>
          )}
          {essays.map((e) => {
            const open = expanded.has(e.key);
            return (
              <div key={e.key} className={`overflow-hidden rounded-xl border transition ${open ? 'border-indigo-300 shadow-md' : 'hover:border-indigo-300'}`}>
                <button onClick={() => toggle(e.key)} aria-expanded={open}
                  className={`block w-full p-4 text-left transition ${open ? 'bg-indigo-50/60' : 'hover:bg-indigo-50/40'}`}>
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="font-semibold text-gray-900">{e.title}</div>
                      <div className="text-xs text-gray-500">{fmt(e.last_at)} · {e.essay_length} words</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <StatusBadge status={e.status} />
                      <span className={`text-gray-500 transition-transform ${open ? 'rotate-180' : ''}`} aria-hidden="true">▼</span>
                    </div>
                  </div>
                  {!open && <p className="mt-1 line-clamp-2 text-sm text-gray-600">{e.essay}</p>}
                  {e.runs.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {e.runs.map((r, i) => <span key={i} className="rounded bg-white/80 border px-1.5 py-0.5 text-xs">{r.ai}: <strong>{r.overall || '—'}</strong></span>)}
                    </div>
                  )}
                </button>
                {open && <ReportDetail essay={e} />}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

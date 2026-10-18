import React, { useMemo } from 'react';
import { computeStats, othersConsensus } from '../lib/aiComparison';
import { agreementLabel, evaluationRaters, notableDifferences, overallResult } from '../lib/evaluationStats';
import { isNum } from '../lib/scoring';

const fmt = (n, digits = 1) => (isNum(n) ? n.toFixed(digits) : '—');

export function AiChip({ ai }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-sm font-medium ${ai.chip}`}>
      {ai.icon} {ai.name}
    </span>
  );
}

// How much the three AIs agree on this essay. Each AI is compared with the
// average of the *other* AIs, never with itself.
export default function AiStatistics({ aiRuns, title = '📈 Agreement statistics' }) {
  const { raters, ruleBased, criteria } = useMemo(() => evaluationRaters(aiRuns), [aiRuns]);
  const stats = useMemo(
    () => (raters.length >= 2 ? computeStats(criteria, raters, othersConsensus(criteria, raters)) : null),
    [criteria, raters],
  );
  const running = Object.values(aiRuns || {}).some((r) => r?.status === 'running');
  if (!aiRuns) return null;

  const ranked = stats ? [...stats.perRater].sort((a, b) => (b.similarity ?? -1) - (a.similarity ?? -1)) : [];
  const best = ranked[0];
  const differences = stats ? notableDifferences(criteria, raters) : [];

  return (
    <section className="glass-card rounded-2xl shadow-xl p-5 bg-white/95" aria-labelledby="ai-stats-title">
      <h2 id="ai-stats-title" className="text-lg font-bold text-gray-800">{title}</h2>
      <p className="text-xs text-gray-500 mt-0.5">
        Each AI is compared with the average of the other AIs. Scores are put on a 0–100 scale per criterion, so
        criteria with different ranges are comparable.
      </p>

      <details className="mt-3 rounded-xl border border-sky-200 bg-sky-50/70 px-4 py-2 text-sm text-sky-900">
        <summary className="cursor-pointer font-semibold">📖 What do these numbers mean? (in plain words)</summary>
        <ul className="mt-2 mb-1 space-y-1 list-disc pl-5">
          <li><strong>Similarity</strong>: how close an AI’s scores are to what the other AIs gave. 100% means exactly the same scores.</li>
          <li><strong>Closest to consensus</strong>: the most “middle-of-the-road” grader, a safe choice when you’re unsure.</li>
          <li><strong>Balanced / Lenient / Strict</strong>: whether an AI usually scores the same as, higher than, or lower than the others.</li>
          <li><strong>Spread</strong>: how far apart the AIs are on one criterion. “Disagree” means they’re more than a rubric level apart; read that feedback carefully.</li>
          <li>This compares one essay only. Agreement doesn’t prove an AI is right; your own judgement still matters.</li>
        </ul>
      </details>

      {!stats ? (
        <p className="mt-4 text-sm text-gray-600">
          {running
            ? 'Statistics appear when at least two AIs have finished…'
            : 'At least two AIs need to answer to compare them.'}
          {ruleBased.length > 0 && ` ${ruleBased.join(' and ')} used rule-based scoring, so ${ruleBased.length > 1 ? 'they are' : 'it is'} left out.`}
        </p>
      ) : (
        <>
          <div className="mt-4 rounded-xl border border-indigo-200 bg-indigo-50/60 p-4">
            <div className="text-xs font-semibold uppercase tracking-wide text-indigo-700">Closest to consensus</div>
            <div className="mt-1 flex flex-wrap items-center gap-3">
              <AiChip ai={best} />
              <span className="text-3xl font-bold text-gray-900">{fmt(best.similarity)}%</span>
              <span className="text-sm text-gray-600">similar to the other AIs</span>
              <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${agreementLabel(best.similarity).className}`}>
                {agreementLabel(best.similarity).label}
              </span>
            </div>
            {stats.tie && ranked[1] && (
              <p className="mt-1 text-xs text-amber-800">Near-tie: {ranked[1].name} is almost as close.</p>
            )}

            <ol className="mt-4 space-y-2">
              {ranked.map((r, i) => (
                <li key={r.id} className="grid grid-cols-[1.5rem_auto_1fr_3rem_auto] items-center gap-2 text-sm">
                  <span className="text-gray-500">{i + 1}.</span>
                  <AiChip ai={r} />
                  <div className="h-2.5 rounded-full bg-white border border-indigo-100 overflow-hidden" aria-hidden="true">
                    <div className="h-full rounded-full bg-indigo-500" style={{ width: `${Math.max(0, r.similarity ?? 0)}%` }} />
                  </div>
                  <span className="text-right font-semibold text-gray-800">{fmt(r.similarity, 0)}%</span>
                  <span className={`rounded-full px-2 py-0.5 text-xs ${r.tendency.className}`}>{r.tendency.label}</span>
                </li>
              ))}
            </ol>

            {differences.length > 0 && (
              <ul className="mt-3 space-y-1 text-sm text-gray-700 list-disc pl-5">
                {differences.map(({ rater, criterion, diff }) => (
                  <li key={`${rater.id}-${criterion.key}`}>
                    <strong>{rater.name}</strong> rates <strong>{criterion.label}</strong> {fmt(Math.abs(diff))}{' '}
                    {Math.abs(diff) === 1 ? 'point' : 'points'} {diff > 0 ? 'higher' : 'lower'} than the other AIs
                    <span className="text-gray-500"> (on 1–{criterion.max_score})</span>.
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-gray-500 border-b">
                  <th className="py-2 pr-3 font-medium">Criterion</th>
                  {raters.map((r) => <th key={r.id} className="py-2 px-2 font-medium text-center">{r.icon} {r.name}</th>)}
                  <th className="py-2 pl-2 font-medium text-center">Spread</th>
                </tr>
              </thead>
              <tbody>
                {stats.perCriterion.map((c) => (
                  <tr key={c.key} className="border-b last:border-0">
                    <td className="py-2 pr-3 text-gray-800">{c.label} <span className="text-xs text-gray-400">/ {c.max_score}</span></td>
                    {raters.map((r) => (
                      <td key={r.id} className="py-2 px-2 text-center font-semibold text-gray-800">
                        {isNum(r.scores[c.key]) ? r.scores[c.key] : '—'}
                      </td>
                    ))}
                    <td className="py-2 pl-2 text-center">
                      <span className={`rounded-full px-2 py-0.5 text-xs ${
                        c.allAgree ? 'bg-emerald-100 text-emerald-800'
                          : c.highDisagreement ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-700'}`}
                      >
                        {c.allAgree ? 'All agree' : c.highDisagreement ? 'Disagree' : 'Close'}
                      </span>
                    </td>
                  </tr>
                ))}
                <tr className="text-xs text-gray-600">
                  <td className="pt-2 pr-3">Overall</td>
                  {raters.map((r) => <td key={r.id} className="pt-2 px-2 text-center">{overallResult(r.scores)?.text || '—'}</td>)}
                  <td />
                </tr>
              </tbody>
            </table>
          </div>
          {ruleBased.length > 0 && (
            <p className="mt-2 text-xs text-amber-700">
              ⚠️ {ruleBased.join(' and ')} used rule-based scoring instead of AI, so {ruleBased.length > 1 ? 'they are' : 'it is'} left out of these statistics.
            </p>
          )}
        </>
      )}
    </section>
  );
}

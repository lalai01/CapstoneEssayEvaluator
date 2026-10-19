import React from 'react';
import { EVALUATION_AIS } from '../lib/evaluationAis';
import { overallResult as overall } from '../lib/evaluationStats';

const seconds = (ms) => (ms ? `${Math.round(ms / 1000)} s` : '');

// runs: { [aiId]: { status: 'running' | 'done' | 'failed', scores, feedback, ms, error } }
export default function AiChoice({ runs, chosen, onChoose }) {
  if (!runs) return null;
  const finished = EVALUATION_AIS.filter((a) => runs[a.id]?.status !== 'running').length;

  return (
    <section className="glass-card rounded-2xl shadow-xl p-5 bg-white/95" aria-labelledby="ai-choice-title">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="ai-choice-title" className="block text-lg font-bold text-gray-800">🤖 Three AIs evaluated this essay</h2>
        <span className="text-xs text-gray-500" aria-live="polite">
          {finished < EVALUATION_AIS.length ? `${finished} of 3 finished…` : 'All finished'}
        </span>
      </div>
      <p className="text-sm text-gray-600 mt-1">
        Pick the result you prefer. The one you choose is shown below and is what you save or override.
      </p>

      <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3" role="radiogroup" aria-labelledby="ai-choice-title">
        {EVALUATION_AIS.map((ai) => {
          const run = runs[ai.id] || { status: 'running' };
          const usable = run.status === 'done' && run.scores;
          const ruleBased = run.scores?.evaluated_by?.provider === 'heuristic';
          const result = overall(run.scores);
          const selected = chosen === ai.id;
          return (
            <button
              key={ai.id}
              type="button"
              role="radio"
              aria-checked={selected}
              disabled={!usable}
              onClick={() => onChoose(ai.id)}
              className={`text-left rounded-xl border p-3 transition disabled:cursor-not-allowed ${
                selected
                  ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-300'
                  : usable ? 'border-gray-200 bg-white hover:border-blue-300 hover:bg-blue-50/40' : 'border-gray-200 bg-gray-50'
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-semibold text-gray-800">{ai.icon} {ai.name}</span>
                {selected && <span className="text-xs font-semibold text-blue-700">✓ Chosen</span>}
              </div>
              <div className="text-xs text-gray-500">{ai.where}</div>

              {run.status === 'running' && (
                <div className="mt-3 flex items-center gap-2 text-sm text-gray-500">
                  <span className="h-3 w-3 rounded-full border-2 border-blue-400 border-t-transparent animate-spin" aria-hidden="true" />
                  Evaluating…
                </div>
              )}
              {run.status === 'failed' && (
                <p className="mt-3 text-sm text-red-600">⚠️ {run.error || `${ai.name} didn’t respond.`}</p>
              )}
              {usable && (
                <div className="mt-3">
                  {result ? (
                    <div className="flex items-baseline gap-2">
                      <span className={`text-2xl font-bold ${result.grade.color}`}>{result.text}</span>
                      <span className={`text-sm font-medium ${result.grade.color}`}>{result.grade.label}</span>
                    </div>
                  ) : (
                    <span className="text-sm text-gray-500">No overall score</span>
                  )}
                  {ruleBased ? (
                    <p className="text-xs text-amber-700 mt-1">
                      ⚠️ {run.scores?.evaluated_by?.reason || `${ai.name} didn’t respond, so rule-based scoring was used.`}
                    </p>
                  ) : (
                    <>
                      {/* Another AI stood in (e.g. Gemini when Groq is over its limit). */}
                      {run.scores?.evaluated_by?.reason?.includes('was used instead') && (
                        <p className="text-xs text-amber-700 mt-1">
                          ↪ Answered by {run.scores.evaluated_by.label}: {run.scores.evaluated_by.reason.replace(/^You chose [^.]*\.\s*/, '')}
                        </p>
                      )}
                      <div className="text-xs text-gray-500 mt-1">Finished in {seconds(run.ms)}</div>
                    </>
                  )}
                </div>
              )}
            </button>
          );
        })}
      </div>
    </section>
  );
}

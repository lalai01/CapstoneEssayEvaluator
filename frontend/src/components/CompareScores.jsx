import React, { useMemo, useState } from 'react';
import toast from 'react-hot-toast';
import AiStatistics from './AiStatistics';
import useDraft from '../lib/draft';
import { EVALUATION_AIS } from '../lib/evaluationAis';
import { evaluationRaters, overallResult, sharedScale } from '../lib/evaluationStats';
import { essayKey, fractionOf, isNum } from '../lib/scoring';

const fmt = (n) => (isNum(n) ? n.toFixed(1) : '—');

// "Which AI scores closest to you?" The three AIs already evaluated the essay
// on the Evaluate page; here the user enters their own scores and sees which
// AI is nearest, then can use that AI's result.
export default function CompareScores({ aiRuns, chosenAi, onChooseAi, essayTitle, evaluatedEssay, onGoEvaluate }) {
  const { raters, criteria } = useMemo(() => evaluationRaters(aiRuns), [aiRuns]);
  const forEssay = essayKey(evaluatedEssay);
  // The user's scores are kept as a draft, for this essay only.
  const [draft, setDraft] = useDraft('myScores', null);
  const values = draft?.essay === forEssay ? draft.values : {};
  const setValue = (key, value) => setDraft({ essay: forEssay, values: { ...values, [key]: value } });
  const [picked, setPicked] = useState('');

  const doneRuns = EVALUATION_AIS.filter((ai) => aiRuns?.[ai.id]?.status === 'done' && aiRuns[ai.id].scores);
  const scale = sharedScale(criteria);

  // Valid entries only: whole numbers within the criterion's range.
  const mine = {};
  for (const c of criteria) {
    const n = Number(values[c.key]);
    if (values[c.key] !== '' && values[c.key] !== undefined && Number.isInteger(n) && n >= 1 && n <= c.max_score) mine[c.key] = n;
  }
  const entered = criteria.filter((c) => isNum(mine[c.key]));
  const total = entered.reduce((s, c) => s + mine[c.key], 0);
  const maxTotal = criteria.reduce((s, c) => s + c.max_score, 0);

  // Average difference per criterion: in points when every criterion has the
  // same range, otherwise as a percentage of each criterion's range.
  const differences = raters.map((r) => {
    const diffs = entered
      .filter((c) => isNum(r.scores[c.key]))
      .map((c) => (scale
        ? Math.abs(r.scores[c.key] - mine[c.key])
        : Math.abs(fractionOf(r.scores[c.key], c.max_score) - fractionOf(mine[c.key], c.max_score)) * 100));
    return { ...r, mae: diffs.length ? diffs.reduce((a, b) => a + b, 0) / diffs.length : null };
  }).filter((r) => r.mae !== null);
  const ranked = [...differences].sort((a, b) => a.mae - b.mae);
  const closest = ranked[0];
  const unit = scale ? 'points' : '% of the scale';
  const worst = scale ? scale - 1 : 100;

  const selectedId = picked || closest?.id || chosenAi || doneRuns[0]?.id || '';
  const selectedAi = EVALUATION_AIS.find((ai) => ai.id === selectedId);
  const selectedDiff = differences.find((r) => r.id === selectedId);

  if (!doneRuns.length) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-8 bg-white/95 max-w-2xl mx-auto text-center animate-fade-in">
        <div className="text-4xl" aria-hidden="true">⚖️</div>
        <h1 className="text-2xl font-bold text-gray-800 mt-2">Compare with My Scores</h1>
        <p className="text-gray-600 mt-2">
          Evaluate an essay first. Its three AI results (DeepSeek, Llama 3.2 and Gemma 2) appear here, and you can
          enter your own scores to see which AI grades closest to you.
        </p>
        <button
          onClick={onGoEvaluate}
          className="mt-5 bg-indigo-600 text-white px-5 py-2 rounded-lg hover:bg-indigo-700 transition"
        >
          ✨ Go to Evaluate
        </button>
      </div>
    );
  }

  const useSelected = () => {
    if (!selectedAi) return;
    onChooseAi(selectedId);
    toast.success(`${selectedAi.name}’s result is now the chosen one on the Evaluate page.`);
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <section className="glass-card rounded-2xl shadow-xl p-6 bg-white/95">
        <h1 className="text-2xl font-bold text-gray-800">⚖️ Which AI scores closest to you?</h1>
        <p className="text-gray-600 mt-1">Score the same essay with the same rubric, then compare your scores with the three AIs.</p>
        <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-sm text-gray-700">
          <span><strong>Essay:</strong> {essayTitle || 'Untitled essay'}</span>
          <span>
            <strong>Rubric:</strong>{' '}
            {aiRuns[doneRuns[0].id].scores?.rubric?.title ? `${aiRuns[doneRuns[0].id].scores.rubric.title} · ` : ''}
            {criteria.length} {criteria.length === 1 ? 'criterion' : 'criteria'}
            {scale ? ` × ${scale} points` : ''}
          </span>
        </div>
      </section>

      <AiStatistics aiRuns={aiRuns} title="📈 How the three AIs compare with each other" />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section className="glass-card rounded-2xl shadow-xl p-6 bg-white/95" aria-labelledby="your-scores-title">
          <h2 id="your-scores-title" className="text-lg font-bold text-gray-800">Your scores</h2>
          <p className="text-xs text-gray-500 mt-0.5">Whole numbers from 1 to each criterion’s maximum. Saved as a draft in this browser.</p>
          <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-4">
            {criteria.map((c) => {
              const raw = values[c.key] ?? '';
              const invalid = raw !== '' && !isNum(mine[c.key]);
              return (
                <label key={c.key} className="block">
                  <span className="text-sm text-gray-700">{c.label} / {c.max_score}</span>
                  <input
                    type="number"
                    inputMode="numeric"
                    min={1}
                    max={c.max_score}
                    step={1}
                    value={raw}
                    onChange={(e) => setValue(c.key, e.target.value)}
                    aria-invalid={invalid}
                    className={`mt-1 w-full rounded-lg border p-2.5 ${invalid ? 'border-red-500 bg-red-50' : 'border-gray-300 bg-gray-50'}`}
                  />
                  {invalid && <span className="text-xs text-red-600">Enter 1–{c.max_score}</span>}
                </label>
              );
            })}
          </div>
          <p className="mt-4 text-sm font-semibold text-gray-800">
            Your total: {total} / {maxTotal}
            {entered.length < criteria.length && (
              <span className="font-normal text-gray-500"> ({entered.length} of {criteria.length} scored)</span>
            )}
          </p>
        </section>

        <section className="glass-card rounded-2xl shadow-xl p-6 bg-white/95" aria-labelledby="choose-title">
          <h2 id="choose-title" className="text-lg font-bold text-gray-800">Choose an evaluator</h2>
          <p className="mt-3 border-l-4 border-indigo-400 bg-indigo-50/60 px-3 py-2 text-sm text-gray-800">
            {closest
              ? <>Closest to your scores: <strong>{closest.name}</strong>. Average difference: {fmt(closest.mae)} {unit} per criterion.</>
              : 'Enter your scores to see which AI is closest to you.'}
          </p>

          <label htmlFor="pick-ai" className="block mt-4 text-sm text-gray-700">AI result to use</label>
          <select
            id="pick-ai"
            value={selectedId}
            onChange={(e) => setPicked(e.target.value)}
            className="mt-1 w-full rounded-lg border border-gray-300 bg-gray-50 p-2.5"
          >
            {doneRuns.map((ai) => (
              <option key={ai.id} value={ai.id}>
                {ai.icon} {ai.name}{ai.id === closest?.id ? ' (closest to you)' : ''}{ai.id === chosenAi ? ' (in use)' : ''}
              </option>
            ))}
          </select>
          {selectedAi && (
            <p className="mt-2 text-sm text-gray-600">
              Overall: <strong>{overallResult(aiRuns[selectedId].scores)?.text || '—'}</strong>
              {selectedDiff && <> · {fmt(selectedDiff.mae)} {unit} from yours on average</>}
            </p>
          )}

          <button
            onClick={useSelected}
            disabled={!selectedAi || selectedId === chosenAi}
            className="mt-4 bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {selectedId === chosenAi ? '✓ In use' : 'Use this AI’s result'}
          </button>
          <button onClick={onGoEvaluate} className="mt-4 ml-3 text-sm text-indigo-700 underline">
            View results on Evaluate
          </button>

          {differences.length > 0 && (
            <div className="mt-6">
              <h3 className="text-sm font-semibold text-gray-800">Average difference from your scores</h3>
              <p className="text-xs text-gray-500">Lower is closer · {unit} per criterion</p>
              <ul className="mt-2 space-y-2">
                {differences.map((r) => (
                  <li key={r.id}>
                    <div className="flex justify-between text-sm">
                      <span>{r.icon} {r.name}</span>
                      <span className="font-semibold">{fmt(r.mae)} {scale ? 'pts' : '%'}</span>
                    </div>
                    <div className="h-2.5 rounded-full bg-gray-100 overflow-hidden" aria-hidden="true">
                      <div className={`h-full ${r.bar}`} style={{ width: `${Math.min(100, (r.mae / worst) * 100)}%`, minWidth: '4px' }} />
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      </div>

      {entered.length > 0 && (
        <section className="glass-card rounded-2xl shadow-xl p-6 bg-white/95" aria-labelledby="chart-title">
          <h2 id="chart-title" className="text-lg font-bold text-gray-800">Criterion scores compared with yours</h2>
          <div className="mt-2 flex flex-wrap gap-4 text-xs text-gray-600">
            <span className="flex items-center gap-1"><span className="h-3 w-3 rounded-sm bg-slate-400" /> You</span>
            {raters.map((r) => (
              <span key={r.id} className="flex items-center gap-1"><span className={`h-3 w-3 rounded-sm ${r.bar}`} /> {r.name}</span>
            ))}
          </div>
          <div className="mt-4 space-y-5">
            {entered.map((c) => (
              <div key={c.key} className="grid grid-cols-1 sm:grid-cols-[10rem_1fr] gap-2">
                <div className="text-sm font-medium text-gray-800">{c.label}</div>
                <div className="space-y-1.5">
                  {[{ id: 'you', name: 'You', bar: 'bg-slate-400', score: mine[c.key] },
                    ...raters.map((r) => ({ ...r, score: r.scores[c.key] }))].map((row) => (
                    <div key={row.id} className="grid grid-cols-[1fr_2.5rem] items-center gap-2" title={`${row.name}: ${row.score ?? '—'} / ${c.max_score}`}>
                      <div className="h-3 rounded bg-gray-100 overflow-hidden">
                        <div className={`h-full ${row.bar}`} style={{ width: `${isNum(row.score) ? (row.score / c.max_score) * 100 : 0}%` }} />
                      </div>
                      <span className="text-right text-sm font-semibold text-gray-800">{isNum(row.score) ? row.score : '—'}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <p className="text-xs text-gray-500">
        Agreement is based on this essay only; compare several essays for a broader picture. Read each AI’s feedback
        as well as its scores before choosing.
        {raters.length < doneRuns.length && ' AIs that used rule-based scoring are left out of the comparison.'}
      </p>
    </div>
  );
}

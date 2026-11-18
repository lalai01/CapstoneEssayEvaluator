import React, { useEffect, useState } from 'react';
import toast from 'react-hot-toast';
import { findHumanScore, listLearningFeedback, saveHumanScore } from '../api';
import { useAuth } from '../context/AuthContext';
import useDraft from '../lib/draft';
import { EVALUATION_AIS } from '../lib/evaluationAis';
import { evaluationRaters } from '../lib/evaluationStats';
import { errorMessage, essayKey, isNum } from '../lib/scoring';

// Human score beside each AI's score, criterion by criterion, with how far
// each AI is from the human. The human scores are the same draft as on the
// Compare page ("myScores"), so they are entered once for both. A human score
// already saved for this essay (an override's teacher scores, in the Learning
// Hub) is retrieved and filled in, and so is one saved from the Override
// dialog on this page. "Save human score" keeps it on the server beside the
// AIs' results, where admins see it in Evaluation Activity.

export const OVERRIDE_SAVED_EVENT = 'essay-evaluator:override-saved';

// Latest saved override of this essay that has teacher scores, or null.
function savedHumanScore(rows, forEssay) {
  for (const row of rows || []) {
    const scores = row.teacher_scores || row.original_scores?.teacher_scores;
    if (essayKey(row.original_essay) === forEssay && scores && Object.keys(scores).length) {
      return { scores, at: row.created_at, teacher: !!(row.approved_by || row.original_scores?.approved_by) };
    }
  }
  return null;
}

const asDraftValues = (scores) => Object.fromEntries(Object.entries(scores).map(([k, v]) => [k, String(v)]));
//   values / onChange: pass them to use scores kept by the parent (Compare
//   page); without them the table keeps its own draft and is editable.
export default function HumanVsAi({ runs, evaluatedEssay, essayTitle, values: givenValues, onChange, editable = true }) {
  const forEssay = essayKey(evaluatedEssay);
  const [draft, setDraft] = useDraft('myScores', null);
  const own = draft?.essay === forEssay ? draft.values : {};
  const values = givenValues || own;
  const setValue = onChange || ((key, value) => setDraft({ essay: forEssay, values: { ...own, [key]: value } }));
  const { user } = useAuth();
  const [saved, setSaved] = useState(null);   // { scores, at, teacher, source } retrieved for this essay
  const [saving, setSaving] = useState(false);
  const retrieving = !givenValues && !!user && !!forEssay;

  // Retrieve the human score saved for this essay; fill it in if none is typed yet.
  useEffect(() => {
    if (!retrieving) return undefined;
    let cancelled = false;
    // A saved human score first; otherwise the teacher scores of an override.
    findHumanScore(evaluatedEssay)
      .then((h) => (h?.scores && Object.keys(h.scores).length
        ? { scores: h.scores, at: h.at, source: 'saved', label: h.label }
        : listLearningFeedback().then((rows) => savedHumanScore(rows, forEssay))))
      .catch(() => listLearningFeedback().then((rows) => savedHumanScore(rows, forEssay)))
      .then((found) => {
        if (cancelled) return;
        setSaved(found);
        if (found) {
          setDraft((d) => (d?.essay === forEssay && Object.values(d.values || {}).some((v) => v !== '')
            ? d
            : { essay: forEssay, values: asDraftValues(found.scores) }));
        }
      })
      .catch(() => {});   // signed out or offline: the human score is typed instead
    return () => { cancelled = true; };
  }, [retrieving, forEssay, evaluatedEssay, setDraft]);

  // An override saved on this page becomes the human score right away.
  useEffect(() => {
    if (givenValues) return undefined;
    const onSaved = (e) => {
      const { essay, teacherScores, teacher } = e.detail || {};
      if (essayKey(essay) !== forEssay || !teacherScores) return;
      const found = { scores: teacherScores, at: new Date().toISOString(), teacher };
      setSaved(found);
      setDraft({ essay: forEssay, values: asDraftValues(teacherScores) });
    };
    window.addEventListener(OVERRIDE_SAVED_EVENT, onSaved);
    return () => window.removeEventListener(OVERRIDE_SAVED_EVENT, onSaved);
  }, [givenValues, forEssay, setDraft]);

  const matchesSaved = saved && Object.entries(saved.scores).every(([k, v]) => String(values[k] ?? '') === String(v));

  const { criteria } = evaluationRaters(runs);
  const ais = EVALUATION_AIS.map((ai) => ({ ...ai, run: runs?.[ai.id] }));
  if (!runs || !criteria.length) return null;

  const human = {};
  for (const c of criteria) {
    const n = Number(values[c.key]);
    if (values[c.key] !== '' && values[c.key] !== undefined && Number.isInteger(n) && n >= 1 && n <= c.max_score) human[c.key] = n;
  }
  const scoreOf = (ai, key) => (ai.run?.status === 'done' ? ai.run.scores?.[key] : undefined);
  const sum = (get) => {
    const vals = criteria.map((c) => get(c.key));
    return vals.every(isNum) ? vals.reduce((a, b) => a + b, 0) : null;
  };
  const maxTotal = criteria.reduce((s, c) => s + c.max_score, 0);
  const humanTotal = sum((k) => human[k]);
  const anyHuman = Object.keys(human).length > 0;
  const complete = criteria.every((c) => isNum(human[c.key]));

  const save = async () => {
    const first = EVALUATION_AIS.map((ai) => runs?.[ai.id]).find((r) => r?.status === 'done' && r.scores)?.scores || {};
    const holistic = criteria.length === 1 && criteria[0].key === 'holistic_points';
    setSaving(true);
    try {
      await saveHumanScore({
        text: evaluatedEssay,
        title: essayTitle || null,
        eval_type: holistic ? 'holistic' : 'analytic',
        scores: human,
        criteria_meta: criteria.map(({ key, label, max_score, weight }) => ({ key, label, max_score, weight })),
        holistic_max: holistic ? criteria[0].max_score : null,
        rubric: first.rubric || null,
        privacy: first.privacy || null,
      });
      setSaved({ scores: human, at: new Date().toISOString(), source: 'saved' });
      toast.success('Human score saved. Admins see it beside the AI scores in Evaluation Activity.');
    } catch (err) {
      toast.error('Could not save the human score: ' + errorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  // Average distance from the human, in points, over the criteria both scored.
  const avgDiff = (ai) => {
    const diffs = criteria
      .filter((c) => isNum(human[c.key]) && isNum(scoreOf(ai, c.key)))
      .map((c) => Math.abs(scoreOf(ai, c.key) - human[c.key]));
    return diffs.length ? diffs.reduce((a, b) => a + b, 0) / diffs.length : null;
  };
  const exact = (ai) => criteria.filter((c) => isNum(human[c.key]) && scoreOf(ai, c.key) === human[c.key]).length;
  const scoredByHuman = criteria.filter((c) => isNum(human[c.key])).length;
  const closest = ais
    .map((ai) => ({ ai, d: avgDiff(ai) }))
    .filter((x) => x.d !== null)
    .sort((a, b) => a.d - b.d)[0];

  const diffBadge = (aiScore, humanScore) => {
    if (!isNum(aiScore) || !isNum(humanScore)) return null;
    const d = aiScore - humanScore;
    if (d === 0) return <span className="ml-1 rounded bg-emerald-100 px-1 text-[11px] font-semibold text-emerald-800">=</span>;
    const far = Math.abs(d) >= 2;
    return (
      <span className={`ml-1 rounded px-1 text-[11px] font-semibold ${far ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-800'}`}>
        {d > 0 ? `+${d}` : d}
      </span>
    );
  };

  return (
    <section className="glass-card rounded-2xl shadow-xl p-5 bg-white/95" aria-labelledby="human-vs-ai-title">
      <h2 id="human-vs-ai-title" className="text-lg font-bold text-gray-800">👩‍🏫 Human score vs AI scores</h2>
      <p className="text-sm text-gray-600 mt-1">
        {editable
          ? 'Enter the teacher’s (human) score for each criterion. Each AI’s score is shown beside it, with the difference: = same, +1 the AI gave one point more, −1 one point less.'
          : 'Your scores beside each AI’s score, with the difference: = same, +1 the AI gave one point more, −1 one point less.'}
      </p>

      {saved && (
        <div className="mt-3 flex flex-wrap items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-700">
          <span>
            📥 Human score {saved.source === 'saved' ? 'saved' : `retrieved from ${saved.teacher ? 'the teacher’s' : 'your'} saved override`}
            {saved.at ? ` (${new Date(saved.at).toLocaleString()})` : ''}.
          </span>
          {!matchesSaved && (
            <button
              type="button"
              onClick={() => setDraft({ essay: forEssay, values: asDraftValues(saved.scores) })}
              className="font-semibold text-indigo-700 underline"
            >
              Use the saved score
            </button>
          )}
        </div>
      )}

      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[34rem] text-sm">
          <thead>
            <tr className="border-b border-gray-200 text-left text-gray-600">
              <th className="py-2 pr-2 font-semibold">Criterion</th>
              <th className="py-2 px-2 font-semibold text-slate-800 bg-slate-50">Human</th>
              {ais.map((ai) => (
                <th key={ai.id} className="py-2 px-2 font-semibold whitespace-nowrap">{ai.icon} {ai.name}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {criteria.map((c) => {
              const raw = values[c.key] ?? '';
              const invalid = raw !== '' && !isNum(human[c.key]);
              return (
                <tr key={c.key} className="border-b border-gray-100">
                  <td className="py-2 pr-2 text-gray-800">
                    {c.label} <span className="text-xs text-gray-500">/ {c.max_score}</span>
                  </td>
                  <td className="py-1.5 px-2 bg-slate-50">
                    {editable ? (
                      <input
                        type="number"
                        inputMode="numeric"
                        min={1}
                        max={c.max_score}
                        step={1}
                        value={raw}
                        onChange={(e) => setValue(c.key, e.target.value)}
                        aria-label={`Human score for ${c.label}, 1 to ${c.max_score}`}
                        aria-invalid={invalid}
                        className={`w-16 rounded-md border p-1.5 text-center ${invalid ? 'border-red-500 bg-red-50' : 'border-gray-300 bg-white'}`}
                      />
                    ) : (
                      <span className="font-semibold text-slate-800">{isNum(human[c.key]) ? human[c.key] : '—'}</span>
                    )}
                  </td>
                  {ais.map((ai) => {
                    const v = scoreOf(ai, c.key);
                    return (
                      <td key={ai.id} className="py-2 px-2 whitespace-nowrap">
                        {ai.run?.status === 'running'
                          ? <span className="text-gray-400">…</span>
                          : <><span className="font-semibold text-gray-800">{isNum(v) ? v : '—'}</span>{diffBadge(v, human[c.key])}</>}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
            <tr className="border-b border-gray-200 font-semibold">
              <td className="py-2 pr-2 text-gray-800">Total / {maxTotal}</td>
              <td className="py-2 px-2 bg-slate-50 text-slate-800">{isNum(humanTotal) ? humanTotal : '—'}</td>
              {ais.map((ai) => {
                const t = sum((k) => scoreOf(ai, k));
                return (
                  <td key={ai.id} className="py-2 px-2 whitespace-nowrap text-gray-800">
                    {isNum(t) ? t : '—'}{diffBadge(t, humanTotal)}
                  </td>
                );
              })}
            </tr>
            {anyHuman && (
              <tr className="text-gray-700">
                <td className="py-2 pr-2 text-xs">Agreement with human</td>
                <td className="py-2 px-2 bg-slate-50" />
                {ais.map((ai) => {
                  const d = avgDiff(ai);
                  return (
                    <td key={ai.id} className="py-2 px-2 text-xs whitespace-nowrap">
                      {d === null ? '—' : <>{exact(ai)}/{scoredByHuman} exact · ±{d.toFixed(1)} avg</>}
                    </td>
                  );
                })}
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <p className="mt-3 text-sm text-gray-700">
        {closest
          ? <>Closest to the human score: <strong>{closest.ai.icon} {closest.ai.name}</strong> (off by {closest.d.toFixed(1)} point{closest.d === 1 ? '' : 's'} per criterion on average).</>
          : editable ? 'Enter the human scores to see which AI agrees most with the teacher.' : ''}
      </p>
      {editable && user && (
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={save}
            disabled={!complete || saving || (saved?.source === 'saved' && matchesSaved)}
            className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-50"
          >
            {saving ? 'Saving…' : saved?.source === 'saved' && matchesSaved ? '✓ Human score saved' : '💾 Save human score'}
          </button>
          <span className="text-xs text-gray-500">
            {complete
              ? 'Saved beside the AI scores; admins see the comparison in Evaluation Activity. It isn’t sent to the AIs.'
              : 'Score every criterion to save it.'}
          </span>
        </div>
      )}
      {editable && !user && (
        <p className="mt-1 text-xs text-gray-500">Sign in to save the human score. Until then it is a draft in this browser.</p>
      )}
    </section>
  );
}

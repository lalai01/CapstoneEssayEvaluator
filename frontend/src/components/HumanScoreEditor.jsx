import React, { useState } from 'react';
import toast from 'react-hot-toast';
import { saveHumanScore } from '../api';
import { errorMessage, getCriteria, isNum } from '../lib/scoring';

// Enter or overwrite the human (teacher's) score of an essay that was
// already evaluated: in My Reports for your own essays, and in Evaluation
// Activity for admins (ownerUserId: the person who evaluated it). The new
// score replaces the earlier human score of the essay.
//   essayText, title: the essay; aiScores: one AI result of it (gives the
//   rubric's criteria); existing: the saved human scores ({ key: value }).
export default function HumanScoreEditor({ essayText, title, aiScores, existing, ownerUserId, onSaved }) {
  const holistic = isNum(aiScores?.holistic_score);
  const criteria = holistic
    ? [{ key: 'holistic_points', label: 'Overall (holistic)', max_score: aiScores.holistic_max || 5, weight: 1 }]
    : getCriteria(aiScores || {});
  const [values, setValues] = useState(() => Object.fromEntries(criteria.map((c) => [c.key, existing?.[c.key] ?? ''])));
  const [remarks, setRemarks] = useState('');
  const [saving, setSaving] = useState(false);

  if (!criteria.length) {
    return <p className="text-sm text-gray-500">No rubric was recorded for this essay, so it can’t be scored here.</p>;
  }

  const parsed = Object.fromEntries(criteria.map((c) => [c.key, Number(values[c.key])]));
  const invalid = (c) => values[c.key] !== '' && !(Number.isInteger(parsed[c.key]) && parsed[c.key] >= 1 && parsed[c.key] <= c.max_score);
  const complete = criteria.every((c) => values[c.key] !== '' && !invalid(c));
  const changed = criteria.some((c) => String(existing?.[c.key] ?? '') !== String(values[c.key]));

  const save = async () => {
    setSaving(true);
    try {
      await saveHumanScore({
        text: essayText,
        title: title || null,
        eval_type: holistic ? 'holistic' : 'analytic',
        scores: parsed,
        criteria_meta: criteria.map(({ key, label, max_score, weight }) => ({ key, label, max_score, weight })),
        holistic_max: holistic ? criteria[0].max_score : null,
        rubric: aiScores?.rubric || null,
        privacy: aiScores?.privacy || null,
        remarks: remarks.trim() || null,
        owner_user_id: ownerUserId || null,
      });
      toast.success(existing ? 'Human score updated (the previous one was replaced).' : 'Human score saved.');
      onSaved?.();
    } catch (err) {
      toast.error('Could not save the human score: ' + errorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {criteria.map((c) => (
          <label key={c.key} className="block text-sm">
            <span className="text-gray-700">{c.label} <span className="text-xs text-gray-500">/ {c.max_score}</span></span>
            <input
              type="number"
              inputMode="numeric"
              min={1}
              max={c.max_score}
              step={1}
              value={values[c.key]}
              onChange={(e) => setValues({ ...values, [c.key]: e.target.value })}
              aria-invalid={invalid(c)}
              className={`mt-1 w-full rounded-lg border p-2 ${invalid(c) ? 'border-red-500 bg-red-50' : 'border-gray-300 bg-white'}`}
            />
            {invalid(c) && <span className="text-xs text-red-600">Enter 1–{c.max_score}</span>}
          </label>
        ))}
      </div>
      <label className="block text-sm">
        <span className="text-gray-700">Remarks <span className="text-xs text-gray-400">(optional)</span></span>
        <input
          value={remarks}
          onChange={(e) => setRemarks(e.target.value)}
          maxLength={500}
          placeholder="e.g. Rechecked after the student's consultation"
          className="mt-1 w-full rounded-lg border border-gray-300 bg-white p-2"
        />
      </label>
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={save}
          disabled={!complete || saving || (existing && !changed && !remarks.trim())}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-50"
        >
          {saving ? 'Saving…' : existing ? '💾 Overwrite human score' : '💾 Save human score'}
        </button>
        <span className="text-xs text-gray-500">
          {existing ? 'Replaces the saved human score of this essay.' : 'Score every criterion to save.'}
        </span>
      </div>
    </div>
  );
}

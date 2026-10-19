// Statistics over the three AI results of one evaluation (Evaluate page and
// Compare with My Scores). The maths lives in aiComparison.js.
import { EVALUATION_AIS } from './evaluationAis';
import { averageFraction, getCriteria, gradeFromFraction, isNum, weightedFraction } from './scoring';

export const HOLISTIC_CRITERION = { key: 'holistic_score', label: 'Overall (holistic)', max_score: 5, weight: 1 };

export const isRuleBased = (scores) => scores?.evaluated_by?.provider === 'heuristic';

// AIs that answered, as raters for computeStats. A result scored by the
// rule-based fallback isn't that AI's judgement, so it's left out.
export function evaluationRaters(aiRuns) {
  const done = EVALUATION_AIS
    .map((ai) => ({ ...ai, run: aiRuns?.[ai.id] }))
    .filter((ai) => ai.run?.status === 'done' && ai.run.scores);
  const raters = done
    .filter((ai) => !isRuleBased(ai.run.scores))
    .map((ai) => ({ ...ai, scores: ai.run.scores }));
  const ruleBased = done.filter((ai) => isRuleBased(ai.run.scores)).map((ai) => ai.name);
  const first = raters[0]?.scores || done[0]?.run.scores || null;
  const holistic = isNum(first?.holistic_score);
  const criteria = !first ? [] : holistic ? [HOLISTIC_CRITERION] : getCriteria(first);
  return { raters, ruleBased, criteria, holistic };
}

// Overall result of one AI, as shown on its card: "3/5" or "78%".
export function overallResult(scores) {
  if (!scores) return null;
  if (isNum(scores.holistic_score)) {
    return { text: `${scores.holistic_score}/5`, grade: gradeFromFraction((scores.holistic_score - 1) / 4) };
  }
  const criteria = getCriteria(scores);
  const f = weightedFraction(scores, criteria) ?? averageFraction(scores, criteria);
  if (f === null) return null;
  return { text: `${Math.round(f * 100)}%`, grade: gradeFromFraction(f) };
}

// Points scale shared by all criteria ("1–4"), or null if they differ.
export function sharedScale(criteria) {
  const maxes = new Set(criteria.map((c) => c.max_score));
  return maxes.size === 1 ? criteria[0].max_score : null;
}

export function agreementLabel(similarity) {
  if (!isNum(similarity)) return { label: '—', className: 'bg-gray-100 text-gray-600' };
  if (similarity >= 90) return { label: 'Strong agreement', className: 'bg-emerald-100 text-emerald-800' };
  if (similarity >= 75) return { label: 'Fair agreement', className: 'bg-amber-100 text-amber-800' };
  return { label: 'Weak agreement', className: 'bg-red-100 text-red-700' };
}

// Criteria where one AI is at least a full rubric level away from the
// average of the others, e.g. "Qwen 2.5 rates Grammar 1.0 point higher".
export function notableDifferences(criteria, raters, limit = 4) {
  if (raters.length < 2) return [];
  const out = [];
  for (const r of raters) {
    for (const c of criteria) {
      const mine = r.scores[c.key];
      const others = raters.filter((o) => o.id !== r.id).map((o) => o.scores[c.key]).filter(isNum);
      if (!isNum(mine) || !others.length) continue;
      const diff = mine - others.reduce((a, b) => a + b, 0) / others.length;
      const oneLevel = Math.max(1, (c.max_score - 1) / 4);
      if (Math.abs(diff) >= oneLevel) out.push({ rater: r, criterion: c, diff });
    }
  }
  return out.sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff)).slice(0, limit);
}

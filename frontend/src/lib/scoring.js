// Rubric-aware scoring helpers shared by Results, LearningFeedback and the
// Prompt Playground. Criteria come from the backend (`scores.criteria_meta`
// or GET /rubric/active); DEFAULT_CRITERIA mirrors the built-in rubric.

export const DEFAULT_CRITERIA = [
  { key: 'main_statement', label: 'Main Statement', max_score: 4, weight: 1.0 },
  { key: 'organization',   label: 'Organization',   max_score: 4, weight: 1.0 },
  { key: 'evidence',       label: 'Evidence',       max_score: 4, weight: 1.5 },
  { key: 'analysis',       label: 'Analysis',       max_score: 4, weight: 1.5 },
  { key: 'grammar',        label: 'Grammar',        max_score: 4, weight: 1.0 },
];

// Keys in a scores object that are never criterion scores.
export const NON_CRITERION_KEYS = new Set([
  'criterion_feedback', 'grammar_issues', 'recommendations', 'criteria_meta',
  'feedback', 'holistic_score', 'level_description', 'teacher_scores',
]);

export function labelFromKey(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function getCriteria(scores) {
  const meta = scores?.criteria_meta;
  if (Array.isArray(meta) && meta.length > 0) return meta;
  if (!scores) return DEFAULT_CRITERIA;
  // Older saved results without criteria_meta: infer from numeric keys.
  const known = new Map(DEFAULT_CRITERIA.map((c) => [c.key, c]));
  const inferred = Object.entries(scores)
    .filter(([k, v]) => !NON_CRITERION_KEYS.has(k) && typeof v === 'number')
    .map(([k]) => known.get(k) || { key: k, label: labelFromKey(k), max_score: 4, weight: 1.0 });
  return inferred.length > 0 ? inferred : DEFAULT_CRITERIA;
}

export function isNum(v) {
  return typeof v === 'number' && !Number.isNaN(v);
}

// 1..max → 0..1
export function fractionOf(score, max) {
  if (!isNum(score) || !max || max <= 1) return null;
  return Math.max(0, Math.min(1, (score - 1) / (max - 1)));
}

export function totalScore(scores, criteria) {
  return criteria.reduce((sum, c) => sum + (isNum(scores?.[c.key]) ? scores[c.key] : 0), 0);
}

export function maxTotal(criteria) {
  return criteria.reduce((sum, c) => sum + c.max_score, 0);
}

export function weightedFraction(scores, criteria) {
  let sum = 0;
  let weight = 0;
  for (const c of criteria) {
    const f = fractionOf(scores?.[c.key], c.max_score);
    if (f !== null) {
      sum += f * (c.weight ?? 1);
      weight += c.weight ?? 1;
    }
  }
  return weight > 0 ? sum / weight : null;
}

export function averageFraction(scores, criteria) {
  const fs = criteria.map((c) => fractionOf(scores?.[c.key], c.max_score)).filter((f) => f !== null);
  return fs.length ? fs.reduce((a, b) => a + b, 0) / fs.length : null;
}

export function gradeFromFraction(f) {
  if (f === null || f === undefined) return { label: '—', color: 'text-gray-400', bg: 'bg-gray-50 border-gray-200' };
  if (f >= 0.833) return { label: 'Excellent', color: 'text-emerald-600', bg: 'bg-emerald-50 border-emerald-200' };
  if (f >= 0.5) return { label: 'Good', color: 'text-blue-600', bg: 'bg-blue-50 border-blue-200' };
  if (f >= 0.167) return { label: 'Developing', color: 'text-amber-600', bg: 'bg-amber-50 border-amber-200' };
  return { label: 'Beginning', color: 'text-red-600', bg: 'bg-red-50 border-red-200' };
}

export function colorForFraction(f) {
  return gradeFromFraction(f).color;
}

export function errorMessage(err) {
  const detail = err?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    // FastAPI 422 validation errors
    return detail.map((d) => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ');
  }
  return err?.message || 'Unknown error';
}

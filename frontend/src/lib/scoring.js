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
  'feedback', 'holistic_score', 'level_description', 'teacher_scores', 'rubric',
  'evaluated_by', 'title', 'ai_model', 'scores',
]);

// ---------- Saved entries (Knowledge Base rows and teacher overrides) ----------
// Newer rows have dedicated columns; on a database without the README
// migration the same values are nested in a jsonb column instead.

export function entryScores(entry) {
  // Knowledge Base row → full scores object (any rubric) when available.
  return entry?.scores || entry?.criterion_feedback?.scores || null;
}

export function entryTitle(entry) {
  const title = entry?.title || entry?.original_scores?.title;
  if (title) return title;
  const text = (entry?.essay || entry?.original_essay || '').trim();
  return text ? `${text.split(/\s+/).slice(0, 8).join(' ')}…` : `Essay #${entry?.id ?? '?'}`;
}

export const UNKNOWN_AI = 'Unknown AI (older entry)';

export function entryAiLabel(entry) {
  return (
    entry?.ai_model ||
    entryScores(entry)?.evaluated_by?.label ||
    entry?.criterion_feedback?.ai_model ||
    entry?.original_scores?.ai_model ||
    entry?.original_scores?.evaluated_by?.label ||
    UNKNOWN_AI
  );
}

export function entryRubricTitle(entry) {
  return (entryScores(entry) || entry?.original_scores)?.rubric?.title || null;
}

// Normalized text used to recognize the same essay across the KB and overrides.
export function essayKey(text) {
  return (text || '').toLowerCase().replace(/\s+/g, ' ').trim().slice(0, 500);
}

export function matchesSearch(query, ...fields) {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return fields.some((f) => (f || '').toLowerCase().includes(q));
}

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

// ---------- Satisfaction (1–10) ----------
// Same bands as the rating picker: 1–3 red, 4–6 amber, 7–10 green.
export function satisfactionStyle(value) {
  if (!isNum(value)) {
    return { label: 'Not rated', chip: 'bg-gray-100 text-gray-600 border-gray-200', text: 'text-gray-500', solid: 'bg-gray-400' };
  }
  if (value >= 9) return { label: 'Very satisfied', chip: 'bg-emerald-100 text-emerald-800 border-emerald-200', text: 'text-emerald-700', solid: 'bg-emerald-500' };
  if (value >= 7) return { label: 'Satisfied', chip: 'bg-emerald-50 text-emerald-700 border-emerald-200', text: 'text-emerald-600', solid: 'bg-emerald-500' };
  if (value >= 4) return { label: value >= 5 ? 'Neutral' : 'Unsatisfied', chip: 'bg-amber-50 text-amber-800 border-amber-200', text: 'text-amber-700', solid: 'bg-amber-500' };
  return { label: value >= 3 ? 'Unsatisfied' : 'Very unsatisfied', chip: 'bg-red-50 text-red-700 border-red-200', text: 'text-red-600', solid: 'bg-red-500' };
}

// Who approved a saved entry (teachers only), from its column or the
// jsonb fallback used before the README migration.
export function entryApprovedBy(entry) {
  return entry?.approved_by || entry?.criterion_feedback?.approved_by || entry?.original_scores?.approved_by || null;
}

export function errorMessage(err) {
  const detail = err?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  // Structured errors, e.g. { code: 'guest_limit', message }
  if (detail && typeof detail === 'object' && typeof detail.message === 'string') return detail.message;
  if (Array.isArray(detail)) {
    // FastAPI 422 validation errors
    return detail.map((d) => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ');
  }
  return err?.message || 'Unknown error';
}

// Helpers shared by both pages.
export function countByAi(entries, labelOf) {
  const counts = new Map();
  for (const e of entries) {
    const label = labelOf(e);
    counts.set(label, (counts.get(label) || 0) + 1);
  }
  return [...counts.entries()]
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count);
}

export function sortEntries(entries, sort, titleOf) {
  const list = [...entries];
  if (sort === 'title') return list.sort((a, b) => titleOf(a).localeCompare(titleOf(b)));
  const dir = sort === 'oldest' ? 1 : -1;
  return list.sort((a, b) => dir * (new Date(a.created_at) - new Date(b.created_at)));
}

// The AI key is always derived from the criterion name, so it matches what
// was set: "Main Statement" -> main_statement, "Gramática" -> gramatica.
export function slugify(text) {
  return String(text || '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

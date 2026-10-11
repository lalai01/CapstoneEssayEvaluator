// Statistics for comparing several AI raters on the same essay.
//
// Criteria can have different ranges (an admin may score Grammar 1–10 and
// Creativity 1–5), so every distance is measured on a normalized 0–100
// scale: 0 = the criterion's minimum (1), 100 = its maximum.

import { fractionOf, isNum, weightedFraction, averageFraction, NON_CRITERION_KEYS, labelFromKey } from './scoring';

// Mean signed difference (in normalized points) beyond which a model is
// called lenient or strict relative to the reference.
export const BIAS_THRESHOLD = 8;

function mean(xs) {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
}

function stdDev(xs) {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
}

export function pearson(xs, ys) {
  if (xs.length < 3) return null;
  const mx = mean(xs);
  const my = mean(ys);
  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < xs.length; i++) {
    num += (xs[i] - mx) * (ys[i] - my);
    dx += (xs[i] - mx) ** 2;
    dy += (ys[i] - my) ** 2;
  }
  if (dx === 0 || dy === 0) return null; // a rater gave identical scores everywhere
  return num / Math.sqrt(dx * dy);
}

export function tendency(bias) {
  if (bias === null) return { label: '—', className: 'bg-gray-100 text-gray-600' };
  if (bias > BIAS_THRESHOLD) return { label: 'Lenient', className: 'bg-emerald-100 text-emerald-800' };
  if (bias < -BIAS_THRESHOLD) return { label: 'Strict', className: 'bg-red-100 text-red-800' };
  return { label: 'Balanced', className: 'bg-blue-100 text-blue-800' };
}

// Criteria to compare: the active rubric's criteria that at least one model
// returned, then any other numeric keys the models returned.
export function buildCriteria(parsedList, rubricCriteria) {
  const present = new Set();
  const observedMax = {};
  for (const p of parsedList) {
    if (!p) continue;
    for (const [k, v] of Object.entries(p)) {
      if (NON_CRITERION_KEYS.has(k) || !isNum(v)) continue;
      present.add(k);
      observedMax[k] = Math.max(observedMax[k] || 0, v);
    }
  }
  const fromRubric = (rubricCriteria || []).filter((c) => present.has(c.key));
  const known = new Set(fromRubric.map((c) => c.key));
  const extra = [...present]
    .filter((k) => !known.has(k))
    .map((k) => ({ key: k, label: labelFromKey(k), max_score: Math.max(4, observedMax[k]), weight: 1 }));
  return [...fromRubric, ...extra];
}

export function consensusScores(criteria, raters) {
  const out = {};
  for (const c of criteria) {
    const vals = raters.map((r) => r.scores?.[c.key]).filter(isNum);
    if (vals.length) out[c.key] = mean(vals);
  }
  return out;
}

// Normalized distance of one rater from a reference.
function compareToReference(criteria, scores, reference) {
  const diffs = [];
  const perCriterion = {};
  for (const c of criteria) {
    const f = fractionOf(scores?.[c.key], c.max_score);
    const r = fractionOf(reference?.[c.key], c.max_score);
    if (f === null || r === null) continue;
    const d = (f - r) * 100;
    diffs.push(d);
    perCriterion[c.key] = d;
  }
  if (!diffs.length) return { mae: null, bias: null, similarity: null, perCriterion, n: 0 };
  const mae = mean(diffs.map(Math.abs));
  return { mae, bias: mean(diffs), similarity: 100 - mae, perCriterion, n: diffs.length };
}

function comparePair(criteria, a, b) {
  let exact = 0;
  let adjacent = 0;
  let n = 0;
  const rawDiffs = [];
  const fa = [];
  const fb = [];
  for (const c of criteria) {
    const va = a?.[c.key];
    const vb = b?.[c.key];
    if (!isNum(va) || !isNum(vb)) continue;
    n++;
    const d = Math.abs(va - vb);
    rawDiffs.push(d);
    if (d === 0) exact++;
    if (d <= 1) adjacent++;
    fa.push(fractionOf(va, c.max_score));
    fb.push(fractionOf(vb, c.max_score));
  }
  return {
    n,
    exact,
    adjacent,
    mae: n ? mean(rawDiffs) : null,
    normalizedMae: n ? mean(fa.map((x, i) => Math.abs(x - fb[i]) * 100)) : null,
    r: pearson(fa, fb),
  };
}

/**
 * raters:    [{ id, name, scores }]  — scores is the parsed model JSON
 * reference: scores object to compare against
 */
export function computeStats(criteria, raters, reference) {
  const valid = raters.filter((r) => r.scores);

  const perRater = valid.map((r) => {
    const fractions = criteria
      .map((c) => fractionOf(r.scores[c.key], c.max_score))
      .filter((f) => f !== null)
      .map((f) => f * 100);
    const vsRef = compareToReference(criteria, r.scores, reference);
    return {
      ...r,
      average: averageFraction(r.scores, criteria),
      weighted: weightedFraction(r.scores, criteria),
      // Low spread = the model gives nearly the same score to every criterion.
      differentiation: stdDev(fractions),
      ...vsRef,
      tendency: tendency(vsRef.bias),
    };
  });

  const ranked = perRater.filter((r) => r.mae !== null).sort((a, b) => a.mae - b.mae);
  const closest = ranked[0] || null;
  const tie = ranked.length > 1 && Math.abs(ranked[0].mae - ranked[1].mae) < 0.5;

  const closestPerCriterion = {};
  for (const c of criteria) {
    let best = null;
    for (const r of perRater) {
      const d = r.perCriterion[c.key];
      if (d === undefined) continue;
      if (!best || Math.abs(d) < Math.abs(best.d)) best = { id: r.id, d };
      else if (Math.abs(d) === Math.abs(best.d)) best = { ...best, tie: true };
    }
    if (best) closestPerCriterion[c.key] = best;
  }
  const criteriaWon = {};
  for (const best of Object.values(closestPerCriterion)) {
    if (!best.tie) criteriaWon[best.id] = (criteriaWon[best.id] || 0) + 1;
  }

  const pairs = [];
  for (let i = 0; i < valid.length; i++) {
    for (let j = i + 1; j < valid.length; j++) {
      pairs.push({ a: valid[i], b: valid[j], ...comparePair(criteria, valid[i].scores, valid[j].scores) });
    }
  }

  const perCriterion = criteria.map((c) => {
    const vals = valid.map((r) => r.scores[c.key]).filter(isNum);
    const range = vals.length ? Math.max(...vals) - Math.min(...vals) : null;
    return {
      ...c,
      values: vals,
      range,
      stdDev: vals.length ? stdDev(vals) : null,
      // More than one rubric level apart on a 1–4 scale.
      highDisagreement: range !== null && c.max_score > 1 && range / (c.max_score - 1) > 1 / 3,
      allAgree: vals.length === valid.length && vals.length > 1 && range === 0,
    };
  });

  return {
    perRater,
    closest,
    tie,
    closestPerCriterion,
    criteriaWon,
    pairs,
    perCriterion,
    allAgreeCount: perCriterion.filter((c) => c.allAgree).length,
  };
}

// ---------- Session history (per-browser) ----------
const HISTORY_KEY = 'playground-ai-comparison-history';

export function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveHistory(history) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-200)));
  } catch {
    // Storage unavailable (private mode etc.) — history is a convenience only.
  }
}

// Aggregate logged runs per model name.
export function summarizeHistory(history) {
  const byModel = {};
  for (const run of history) {
    for (const m of run.models || []) {
      const s = (byModel[m.name] ||= { name: m.name, runs: 0, closest: 0, maes: [], biases: [] });
      s.runs++;
      if (run.closest === m.name) s.closest++;
      if (isNum(m.mae)) s.maes.push(m.mae);
      if (isNum(m.bias)) s.biases.push(m.bias);
    }
  }
  return Object.values(byModel)
    .map((s) => {
      const avgBias = mean(s.biases);
      const avgMae = mean(s.maes);
      return {
        name: s.name,
        runs: s.runs,
        closest: s.closest,
        closestRate: s.runs ? s.closest / s.runs : 0,
        avgSimilarity: avgMae === null ? null : 100 - avgMae,
        avgBias,
        tendency: tendency(avgBias),
      };
    })
    .sort((a, b) => b.closestRate - a.closestRate || (b.avgSimilarity ?? 0) - (a.avgSimilarity ?? 0));
}

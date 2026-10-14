import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  testPrompt,
  saveEssay,
  listSavedEssays,
  deleteSavedEssay,
  listKnowledge,
  getActiveRubric,
  evaluateEssay,
} from '../api';
import { useAuth } from '../context/AuthContext';
import LoadingIndicator from './LoadingIndicator';
import toast from 'react-hot-toast';
import {
  DEFAULT_CRITERIA,
  isNum,
  fractionOf,
  averageFraction,
  weightedFraction,
  colorForFraction,
  errorMessage,
} from '../lib/scoring';
import {
  buildCriteria,
  consensusScores,
  othersConsensus,
  computeStats,
  loadHistory,
  saveHistory,
  summarizeHistory,
  BIAS_THRESHOLD,
} from '../lib/aiComparison';
import { generateComparisonReport } from '../lib/comparisonReport';
import { salvageJson } from '../lib/jsonRepair';

// -----------------------------------------------------------------------------
// Shared helpers
// -----------------------------------------------------------------------------
const BAR_COLORS = ['bg-blue-500', 'bg-emerald-500', 'bg-purple-500', 'bg-amber-500', 'bg-pink-500', 'bg-cyan-500', 'bg-orange-500'];
const TEXT_COLORS = ['text-blue-700', 'text-emerald-700', 'text-purple-700', 'text-amber-700', 'text-pink-700', 'text-cyan-700', 'text-orange-700'];

// The three comparison slots. Each slot has a fixed color so a model is
// recognizable across the table, statistics and result panels.
const SLOTS = [
  { id: 'A', defaultProvider: 'deepseek', badge: 'bg-blue-100 text-blue-800 border-blue-200',       dot: 'bg-blue-500' },
  { id: 'B', defaultProvider: 'gemma',    badge: 'bg-emerald-100 text-emerald-800 border-emerald-200', dot: 'bg-emerald-500' },
  { id: 'C', defaultProvider: 'ollama', badge: 'bg-purple-100 text-purple-800 border-purple-200', dot: 'bg-purple-500' },
];
const SLOT_STYLE = Object.fromEntries(SLOTS.map((s) => [s.id, s]));

const PROVIDER_NAMES = {
  deepseek: 'DeepSeek',
  gemma: 'Gemma (Ollama)',
  ollama: 'Llama 3.2 3B (Ollama)',
  llamacpp: 'Llama.cpp (Phi-3.5)',
};

function providerDisplayName(provider, model) {
  const base = PROVIDER_NAMES[provider] || provider;
  return model ? `${base} · ${model}` : base;
}

function fmtPct(v, digits = 0) {
  return isNum(v) ? `${v.toFixed(digits)}%` : '—';
}

function fmtSigned(v, digits = 1, suffix = '') {
  if (!isNum(v)) return '—';
  const s = v.toFixed(digits);
  return `${v > 0 ? '+' : ''}${s}${suffix}`;
}

function fmtScore(v) {
  if (!isNum(v)) return '—';
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}

// Build prompts that ask for the active rubric's criteria and ranges.
function buildPromptsFromRubric(criteria) {
  const scoreSchema = criteria
    .map((c) => `${c.key} (integer 1–${c.max_score}, "${c.label}")`)
    .join(', ');
  const keys = criteria.map((c) => c.key).join(', ');
  const rubricLines = criteria
    .map((c) => {
      const levels = Object.entries(c.levels || {})
        .sort((a, b) => Number(b[0]) - Number(a[0]))
        .map(([lvl, text]) => `    ${lvl}: ${text}`)
        .join('\n');
      return `- ${c.key} — ${c.label} (score 1–${c.max_score})${levels ? `\n${levels}` : ''}`;
    })
    .join('\n');

  const system =
    'You are a strict JSON API. Respond with a SINGLE valid JSON object and NOTHING else. ' +
    'Do not write any text before or after the JSON. Do not wrap it in code fences. ' +
    'The response must be parseable by JSON.parse() with no surrounding text. ' +
    `Schema (exact keys): ${scoreSchema}, ` +
    'feedback (a 3–5 sentence overall summary string), ' +
    `criterion_feedback (an object with exactly these keys: ${keys} — each a 2–3 sentence string explaining that specific score), ` +
    'grammar_issues (an array of objects with "original", "correction", "explanation"), ' +
    'recommendations (an array of objects with "title", "issue", "suggestion"). ' +
    'Only include REAL grammar errors where original !== correction; use [] if there are none. ' +
    'Provide 3–5 recommendations that reference specific parts of the essay. ' +
    "Each criterion has its own range; never exceed a criterion's maximum. " +
    'Do not give all criteria the same score; differentiate based on evidence.';

  const scoreLines = criteria.map((c) => `  "${c.key}": <int 1-${c.max_score}>,`).join('\n');
  const cfLines = criteria.map((c) => `    "${c.key}": "<2-3 sentences explaining this score>"`).join(',\n');

  const user = `Evaluate this essay against the rubric.

RUBRIC:
${rubricLines}

"""
{essay}
"""

Return ONLY a JSON object with this exact shape:

{
${scoreLines}
  "feedback": "<3-5 sentence overall summary>",
  "criterion_feedback": {
${cfLines}
  },
  "grammar_issues": [
    { "original": "<text quoted from the essay>", "correction": "<fixed>", "explanation": "<one sentence>" }
  ],
  "recommendations": [
    { "title": "<short label>", "issue": "<what is weak>", "suggestion": "<concrete fix referencing the essay>" }
  ]
}

Output ONLY the JSON. No text before it, no text after it, no code fences.`;

  return { system, user };
}

// Local models (Ollama, llama.cpp) run on the VPS CPU, where every prompt
// and output token costs time. This prompt asks for the same JSON shape with
// the rubric, but no long example, one-sentence notes and capped lists.
const LOCAL_PROVIDERS = new Set(['gemma', 'ollama', 'llamacpp']);

function buildCompactPrompts(criteria) {
  const rubric = criteria
    .map((c) => {
      const top = c.levels?.[c.max_score] || c.levels?.[String(c.max_score)];
      return `- ${c.key} (1-${c.max_score}): ${c.label}${top ? `. Top score means: ${top}` : ''}`;
    })
    .join('\n');
  const scores = criteria.map((c) => `"${c.key}": <1-${c.max_score}>`).join(', ');
  const notes = criteria.map((c) => `"${c.key}": "<one sentence>"`).join(', ');
  const system =
    'You are an essay grader that replies with one JSON object only. ' +
    'Use the full score range and give different scores to different criteria when the essay warrants it.';
  const user = `Grade the essay with this rubric:
${rubric}

Essay:
"""
{essay}
"""

Reply with JSON only, in this shape:
{${scores}, "feedback": "<2-3 sentences>", "criterion_feedback": {${notes}}, "grammar_issues": [{"original": "<quote>", "correction": "<fix>", "explanation": "<short>"}], "recommendations": [{"title": "<short>", "issue": "<short>", "suggestion": "<short>"}]}

At most 3 grammar_issues (only real errors; [] if none) and 3 recommendations.`;
  return { system, user };
}

// JSON schema matching the compact prompt. Ollama 0.5+ enforces it, so a
// small model cannot drift out of the format (the backend falls back to
// plain JSON mode on older Ollama versions).
function buildJsonSchema(criteria) {
  const keys = criteria.map((c) => c.key);
  const text = { type: 'string' };
  const item = (fields) => ({
    type: 'object',
    properties: Object.fromEntries(fields.map((f) => [f, text])),
    required: fields,
  });
  return {
    type: 'object',
    properties: {
      ...Object.fromEntries(criteria.map((c) => [c.key, { type: 'integer', minimum: 1, maximum: c.max_score }])),
      feedback: text,
      criterion_feedback: {
        type: 'object',
        properties: Object.fromEntries(keys.map((k) => [k, text])),
        required: keys,
      },
      grammar_issues: { type: 'array', maxItems: 3, items: item(['original', 'correction', 'explanation']) },
      recommendations: { type: 'array', maxItems: 3, items: item(['title', 'issue', 'suggestion']) },
    },
    required: [...keys, 'feedback', 'criterion_feedback', 'grammar_issues', 'recommendations'],
  };
}

function isDefaultRubric(criteria) {
  return (
    criteria.length === DEFAULT_CRITERIA.length &&
    DEFAULT_CRITERIA.every((d) => criteria.some((c) => c.key === d.key && c.max_score === d.max_score))
  );
}

function tryParse(result) {
  if (!result || result.model === 'error') return null;
  try {
    const parsed = parseResultPayload(result);
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch {
    return null;
  }
}

// -----------------------------------------------------------------------------
// Robust JSON parser
// -----------------------------------------------------------------------------
function parseResultPayload(res) {
  let text = (res && res.text) || '';
  text = text.replace(/^```json\s*/i, '').replace(/```$/, '').trim();

  try {
    return JSON.parse(text);
  } catch (e) {
    const start = text.indexOf('{');
    if (start === -1) throw e;

    let depth = 0;
    let inString = false;
    let escape = false;
    let end = -1;
    for (let i = start; i < text.length; i++) {
      const c = text[i];
      if (escape) { escape = false; continue; }
      if (c === '\\') { escape = true; continue; }
      if (c === '"') { inString = !inString; continue; }
      if (inString) continue;
      if (c === '{') depth++;
      else if (c === '}') {
        depth--;
        if (depth === 0) { end = i; break; }
      }
    }
    if (end !== -1) {
      try {
        return JSON.parse(text.slice(start, end + 1));
      } catch {
        // fall through to salvage
      }
    }
    // Cut off or malformed: keep the complete parts (e.g. the scores).
    const salvaged = salvageJson(text);
    if (!salvaged) throw e;
    return salvaged.partial ? { ...salvaged.value, _partial: true } : salvaged.value;
  }
}

// -----------------------------------------------------------------------------
// EvaluationDisplay
// -----------------------------------------------------------------------------
function EvaluationDisplay({ result, criteria }) {
    const parsed = tryParse(result);
    const shown = parsed ? criteria.filter((c) => parsed[c.key] !== undefined) : [];

    if (shown.length === 0) {
      if (!parsed) console.error('[Playground] JSON parse failed, raw:', result?.text);
      return (
        <pre className="whitespace-pre-wrap text-sm font-mono text-gray-800">
          {result?.text}
        </pre>
      );
    }

    const avg = averageFraction(parsed, shown);
    const criterionFeedback = parsed.criterion_feedback || {};
    const partial = !!parsed._partial;

    const filteredIssues = Array.isArray(parsed.grammar_issues)
      ? parsed.grammar_issues.filter(
          (i) =>
            i &&
            typeof i.original === 'string' &&
            typeof i.correction === 'string' &&
            i.original.trim().toLowerCase() !==
              i.correction.trim().toLowerCase()
        )
      : [];

    const recommendations = Array.isArray(parsed.recommendations)
      ? parsed.recommendations.filter(
          (r) =>
            r &&
            typeof r.suggestion === 'string' &&
            r.suggestion.trim().length > 0
        )
      : [];

    return (
      <div className="space-y-4">
        {partial && (
          <p className="text-xs bg-amber-50 border border-amber-200 text-amber-900 rounded-lg p-2" role="status">
            ⚠ This model's answer was cut off. Showing the parts that could be read.
          </p>
        )}
        {/* Overall score */}
        <div className="text-center p-3 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg">
          <div className="text-sm text-gray-700">Overall Average</div>
          <div className={`text-3xl font-bold ${colorForFraction(avg)}`}>
            {avg !== null ? `${Math.round(avg * 100)}%` : '—'}
          </div>
        </div>

        {/* Per-criterion bars + per-criterion feedback */}
        <div className="space-y-4">
          {shown.map(({ key, label, max_score }, idx) => {
            const value = parsed[key];
            const percent = isNum(value) ? Math.min(100, (value / max_score) * 100) : 0;
            const note = criterionFeedback[key];
            return (
              <div key={key}>
                <div className="flex justify-between text-sm mb-1">
                  <span className={`font-medium ${TEXT_COLORS[idx % TEXT_COLORS.length]}`}>{label}</span>
                  <span className={`font-bold ${colorForFraction(fractionOf(value, max_score))}`}>
                    {isNum(value) ? `${value}/${max_score}` : '—'}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`${BAR_COLORS[idx % BAR_COLORS.length]} h-2 rounded-full transition-all`}
                    style={{ width: `${percent}%` }}
                  />
                </div>
                {note && (
                  <p className="text-xs text-gray-700 mt-2 leading-relaxed pl-1">
                    {note}
                  </p>
                )}
              </div>
            );
          })}
        </div>

        {/* Overall feedback paragraph */}
        {parsed.feedback && (
          <div className="mt-4">
            <h4 className="font-semibold text-gray-800 mb-2 text-sm uppercase tracking-wide">
              📝 Overall Feedback
            </h4>
            <div className="bg-white p-4 rounded-lg border border-gray-200 text-gray-800 text-sm leading-relaxed whitespace-pre-wrap">
              {parsed.feedback}
            </div>
          </div>
        )}

        {/* Grammar issues */}
        {filteredIssues.length > 0 && (
          <div className="mt-4">
            <h4 className="font-semibold text-gray-800 mb-2 text-sm uppercase tracking-wide">
              ✏️ Grammar & Mechanics — Specific Fixes ({filteredIssues.length})
            </h4>
            <div className="space-y-2">
              {filteredIssues.map((issue, idx) => (
                <div key={idx} className="bg-white p-3 rounded-lg border border-gray-200 text-sm">
                  <div className="flex items-start gap-2 mb-1">
                    <span className="text-red-500 font-bold flex-shrink-0">✗</span>
                    <span className="text-gray-700 line-through">{issue.original}</span>
                  </div>
                  <div className="flex items-start gap-2 mb-1">
                    <span className="text-emerald-600 font-bold flex-shrink-0">✓</span>
                    <span className="text-gray-900 font-medium">{issue.correction}</span>
                  </div>
                  {issue.explanation && (
                    <p className="text-xs text-gray-600 mt-1 pl-5 italic">
                      {issue.explanation}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recommendations */}
        {recommendations.length > 0 && (
          <div className="mt-4">
            <h4 className="font-semibold text-gray-800 mb-2 text-sm uppercase tracking-wide">
              💡 Recommendations ({recommendations.length})
            </h4>
            <div className="space-y-2">
              {recommendations.map((rec, idx) => (
                <div
                  key={idx}
                  className="bg-amber-50 p-3 rounded-lg border border-amber-200 text-sm"
                >
                  <div className="font-semibold text-amber-900 mb-1">
                    {idx + 1}. {rec.title || 'Improvement'}
                  </div>
                  {rec.issue && (
                    <div className="text-gray-700 mb-1">
                      <span className="font-medium text-amber-800">Issue:</span>{' '}
                      {rec.issue}
                    </div>
                  )}
                  {rec.suggestion && (
                    <div className="text-gray-800">
                      <span className="font-medium text-amber-800">Try:</span>{' '}
                      {rec.suggestion}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
}

// -----------------------------------------------------------------------------
// 3-AI scoring difference table
// -----------------------------------------------------------------------------
function SlotBadge({ rater, short = false }) {
  const style = SLOT_STYLE[rater.id];
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded border text-xs font-medium whitespace-nowrap ${style.badge}`}>
      <span className={`w-2 h-2 rounded-full ${style.dot}`} />
      {short ? rater.id : `${rater.id} · ${rater.name}`}
    </span>
  );
}

function diffClass(d, max) {
  if (!isNum(d) || d === 0) return 'text-gray-500';
  const big = Math.abs(d) / Math.max(1, max - 1) > 1 / 3;
  if (d > 0) return big ? 'text-emerald-700 font-bold' : 'text-emerald-600';
  return big ? 'text-red-600 font-bold' : 'text-red-500';
}

function ComparisonTable({ raters, stats, reference, referenceLabel, showReference }) {
  const pairs = stats.pairs;
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
      <h4 className="font-semibold text-gray-800 mb-1">📊 Scoring Difference Table</h4>
      <p className="text-xs text-gray-500 mb-3">
        Each row is one criterion. <strong>Consensus</strong> is the AIs’ average grade and <strong>Spread</strong> how much
        they disagree. Columns like “A−B” show how many points higher AI A graded than AI B (bold = more than one
        rubric level apart). Highlighted rows are where the AIs disagree most.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-600 border-b">
              <th className="text-left font-medium py-2 pr-2">Criterion</th>
              {raters.map((r) => (
                <th key={r.id} className="font-medium px-2 text-center"><SlotBadge rater={r} short /></th>
              ))}
              <th className="font-medium px-2 text-center">Consensus</th>
              <th className="font-medium px-2 text-center">Spread</th>
              {pairs.map((p) => (
                <th key={`${p.a.id}${p.b.id}`} className="font-medium px-2 text-center whitespace-nowrap">
                  {p.a.id}−{p.b.id}
                </th>
              ))}
              {showReference && <th className="font-medium px-2 text-center whitespace-nowrap">{referenceLabel}</th>}
              <th className="font-medium px-2 text-center whitespace-nowrap">Closest</th>
            </tr>
          </thead>
          <tbody>
            {stats.perCriterion.map((c) => {
              const best = stats.closestPerCriterion[c.key];
              const bestRater = best && raters.find((r) => r.id === best.id);
              return (
                <tr
                  key={c.key}
                  className={`border-b border-gray-100 ${c.highDisagreement ? 'bg-amber-50' : ''}`}
                >
                  <td className="py-2 pr-2 text-gray-800">
                    {c.label}
                    <span className="text-xs text-gray-400 ml-1">/{c.max_score}</span>
                    {c.highDisagreement && <span className="ml-1 text-xs text-amber-700" title="Models disagree by more than one level">⚠</span>}
                  </td>
                  {raters.map((r) => (
                    <td key={r.id} className="text-center px-2 font-medium text-gray-800">
                      {fmtScore(r.scores?.[c.key])}
                    </td>
                  ))}
                  <td className="text-center px-2 text-gray-700">{fmtScore(stats.consensus[c.key])}</td>
                  <td className="text-center px-2 text-gray-700">{c.range === null ? '—' : c.range}</td>
                  {pairs.map((p) => {
                    const a = p.a.scores?.[c.key];
                    const b = p.b.scores?.[c.key];
                    const d = isNum(a) && isNum(b) ? a - b : null;
                    return (
                      <td key={`${p.a.id}${p.b.id}`} className={`text-center px-2 ${diffClass(d, c.max_score)}`}>
                        {d === null ? '—' : fmtSigned(d, 0)}
                      </td>
                    );
                  })}
                  {showReference && (
                    <td className="text-center px-2 text-gray-700">{fmtScore(reference?.[c.key])}</td>
                  )}
                  <td className="text-center px-2">
                    {best?.tie ? <span className="text-xs text-gray-500">tie</span> : bestRater ? <SlotBadge rater={bestRater} short /> : '—'}
                  </td>
                </tr>
              );
            })}
            <tr className="font-semibold text-gray-800">
              <td className="py-2 pr-2">Weighted overall</td>
              {stats.perRater.map((r) => (
                <td key={r.id} className="text-center px-2">{fmtPct(r.weighted === null ? null : r.weighted * 100)}</td>
              ))}
              <td colSpan={2 + pairs.length + (showReference ? 1 : 0) + 1} />
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

// -----------------------------------------------------------------------------
// Score comparison: Category | Primary | Compare | Diff
// -----------------------------------------------------------------------------
function ScoreComparison({ criteria, raters, primaryId, compareId, onPrimary, onCompare, onDownload }) {
  const primary = raters.find((r) => r.id === primaryId);
  const compare = raters.find((r) => r.id === compareId);
  const pick = (value, onChange, label) => (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      aria-label={label}
      className="border rounded-lg px-2 py-1 text-sm bg-white"
    >
      {raters.map((r) => (
        <option key={r.id} value={r.id} disabled={!r.scores}>
          {r.id} · {r.name}{r.scores ? '' : ' (no scores)'}
        </option>
      ))}
    </select>
  );
  const wp = primary?.scores ? weightedFraction(primary.scores, criteria) : null;
  const wq = compare?.scores ? weightedFraction(compare.scores, criteria) : null;
  const wDiff = wp !== null && wq !== null ? (wp - wq) * 100 : null;

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <h4 className="font-semibold text-gray-800">📊 Score Comparison</h4>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="text-gray-600">Primary</span>
          {pick(primaryId, onPrimary, 'Primary model')}
          <span className="text-gray-600">vs Compare</span>
          {pick(compareId, onCompare, 'Compare model')}
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-600 border-b">
              <th className="text-left font-medium py-2">Category</th>
              <th className="font-medium text-center">Primary</th>
              <th className="font-medium text-center">Compare</th>
              <th className="font-medium text-center">Diff</th>
            </tr>
          </thead>
          <tbody>
            {criteria.map((c) => {
              const p = primary?.scores?.[c.key];
              const q = compare?.scores?.[c.key];
              const d = isNum(p) && isNum(q) ? p - q : null;
              return (
                <tr key={c.key} className="border-b border-gray-100">
                  <td className="py-2 text-gray-800">
                    {c.label}<span className="text-xs text-gray-400 ml-1">/{c.max_score}</span>
                  </td>
                  <td className="text-center font-medium text-gray-800">{fmtScore(p)}</td>
                  <td className="text-center font-medium text-gray-800">{fmtScore(q)}</td>
                  <td className={`text-center ${diffClass(d, c.max_score)}`}>{d === null ? '—' : fmtSigned(d, 0)}</td>
                </tr>
              );
            })}
            <tr className="font-semibold text-gray-800">
              <td className="py-2">Weighted overall</td>
              <td className="text-center">{wp === null ? '—' : `${Math.round(wp * 100)}%`}</td>
              <td className="text-center">{wq === null ? '—' : `${Math.round(wq * 100)}%`}</td>
              <td className={`text-center ${wDiff === null || Math.round(wDiff) === 0 ? 'text-gray-500' : wDiff > 0 ? 'text-emerald-600' : 'text-red-500'}`}>
                {wDiff === null ? '—' : fmtSigned(wDiff, 0, ' pts')}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 mt-3">
        <p className="text-xs text-gray-500">Diff = Primary − Compare. Green means Primary scored higher.</p>
        <button
          onClick={onDownload}
          className="text-sm bg-gradient-to-r from-sky-500 to-cyan-500 text-white px-3 py-1.5 rounded-lg font-semibold hover:shadow-md transition"
        >
          📄 Save comparison report (PDF)
        </button>
      </div>
    </div>
  );
}

// -----------------------------------------------------------------------------
// Statistics: which AI the scoring belongs closest to
// -----------------------------------------------------------------------------
const REFERENCE_PHRASE = {
  consensus: 'the other models',
  system: 'the system evaluator',
  teacher: 'your scores',
};

function agreementLevel(similarity) {
  if (!isNum(similarity)) return { label: '—', className: 'bg-gray-100 text-gray-600' };
  if (similarity >= 90) return { label: 'Strong agreement', className: 'bg-emerald-100 text-emerald-800' };
  if (similarity >= 75) return { label: 'Moderate agreement', className: 'bg-blue-100 text-blue-800' };
  return { label: 'Weak agreement', className: 'bg-amber-100 text-amber-800' };
}

// Each model's single biggest difference from the reference, in raw points.
function biggestDifferences(stats, criteria) {
  const byKey = Object.fromEntries(criteria.map((c) => [c.key, c]));
  return stats.perRater
    .map((r) => {
      let best = null;
      for (const [key, d] of Object.entries(r.perCriterion)) {
        if (!best || Math.abs(d) > Math.abs(best.d)) best = { key, d };
      }
      if (!best || Math.abs(best.d) < 25) return null; // under about one level on a 1-4 scale
      const c = byKey[best.key];
      return { rater: r, criterion: c, points: (Math.abs(best.d) / 100) * (c.max_score - 1), higher: best.d > 0 };
    })
    .filter(Boolean);
}

function ClosestSummary({ stats, criteria, raters, referenceMode, referenceLabel, pending }) {
  const scored = raters.filter((r) => r.scores);
  const ranked = stats.perRater.filter((r) => isNum(r.mae)).sort((a, b) => a.mae - b.mae);
  const phrase = REFERENCE_PHRASE[referenceMode];

  if (referenceMode === 'consensus' && scored.length < 2) {
    return (
      <div className="rounded-xl border border-dashed border-indigo-200 bg-indigo-50/50 p-4 text-sm text-indigo-900" role="status">
        <div className="font-semibold mb-1">Closest to Consensus</div>
        Waiting for more models: {scored.length} of {raters.length} finished.
        A model is compared with the average of the <em>other</em> models, so the comparison starts once two have scores.
      </div>
    );
  }
  if (!ranked.length) {
    return (
      <p className="text-sm text-gray-500">
        {referenceMode === 'system' ? 'Score the essay with the system evaluator to compare against it.' : 'Enter your scores above to see which AI grades most like you.'}
      </p>
    );
  }

  const closest = ranked[0];
  const differences = biggestDifferences(stats, criteria);

  return (
    <div className="rounded-xl border border-indigo-200 bg-gradient-to-r from-indigo-50 to-purple-50 p-4 space-y-4">
      <div>
        <div className="text-xs uppercase tracking-wide text-indigo-700 mb-1">Closest to {referenceLabel}</div>
        <div className="flex flex-wrap items-center gap-3">
          <SlotBadge rater={closest} />
          <span className="text-2xl font-bold text-gray-900">{fmtPct(closest.similarity, 1)}</span>
          <span className="text-sm text-gray-600">similar to {phrase}</span>
          <span className={`text-xs px-2 py-0.5 rounded-full ${agreementLevel(closest.similarity).className}`}>
            {agreementLevel(closest.similarity).label}
          </span>
        </div>
        {stats.tie && ranked.length > 1 && (
          <p className="text-xs text-amber-800 mt-1">
            {referenceMode === 'consensus' && scored.length === 2
              ? 'With two models, each is compared with the other, so they are always tied. Run a third model to see which one sits closest to the group.'
              : `Near-tie: ${ranked[1].id} is within half a point.`}
          </p>
        )}
      </div>

      <ol className="space-y-2" aria-label="Models ranked by similarity">
        {ranked.map((r, i) => (
          <li key={r.id} className="flex flex-wrap items-center gap-2 text-sm">
            <span className="w-6 text-gray-500 font-semibold">{i + 1}.</span>
            <SlotBadge rater={r} />
            <div className="flex-1 min-w-[120px] h-2 bg-white rounded-full overflow-hidden border border-indigo-100" aria-hidden="true">
              <div className="h-full bg-indigo-500 rounded-full" style={{ width: `${Math.max(0, Math.min(100, r.similarity))}%` }} />
            </div>
            <span className="w-14 text-right font-semibold text-gray-800">{fmtPct(r.similarity, 0)}</span>
            <span className={`text-xs px-2 py-0.5 rounded-full ${r.tendency.className}`}>{r.tendency.label}</span>
          </li>
        ))}
      </ol>

      {differences.length > 0 && (
        <ul className="text-sm text-gray-700 space-y-1 list-disc pl-5">
          {differences.map(({ rater, criterion, points, higher }) => (
            <li key={rater.id}>
              <strong>{rater.id} · {rater.name}</strong> rates <strong>{criterion.label}</strong>{' '}
              {points.toFixed(points < 1 ? 1 : 0)} point{points >= 1.5 ? 's' : ''} {higher ? 'higher' : 'lower'} than {phrase}
              <span className="text-gray-500"> (on 1–{criterion.max_score})</span>.
            </li>
          ))}
        </ul>
      )}

      {pending > 0 && (
        <p className="text-xs text-gray-600" role="status">
          Partial result: {pending} model{pending === 1 ? ' is' : 's are'} still running, so this may change.
        </p>
      )}
    </div>
  );
}

// ---- Plain-language readings of the statistics (for non-technical users) ----

const PLAIN_TERMS = [
  ['Consensus', 'The average grade of the AIs: what they would give if they voted together.'],
  ['Spread', 'How much the AIs disagree on one criterion. Small spread = they gave similar grades.'],
  ['Similarity', 'How close this AI’s grades are to the comparison (100% = exactly the same grades).'],
  ['Avg distance', 'On average, how far this AI’s grades are from the comparison, on a 0–100 scale. Lower is closer.'],
  ['Bias · Lenient · Strict', 'Whether this AI tends to grade higher (lenient) or lower (strict) than the comparison. Balanced means neither.'],
  ['Weighted score', 'The overall grade this AI gave the essay, counting each criterion by its importance in the rubric.'],
  ['Criteria closest', 'On how many criteria this AI was the nearest to the comparison.'],
  ['Differentiation', 'Whether the AI tells strong and weak areas apart. Low means it gave almost the same score to everything.'],
  ['Exact match · Within ±1', 'How often two AIs gave the identical score, or scores at most one level apart.'],
  ['Avg |diff|', 'On average, how many points apart two AIs are on a criterion.'],
  ['Correlation r', 'Whether two AIs agree on which parts of the essay are strongest and weakest. Near 1 = same pattern, near 0 = no clear pattern, below 0 = opposite views. Needs at least 3 criteria.'],
];

function PlainGlossary() {
  return (
    <details className="rounded-lg border border-sky-200 bg-sky-50/60 p-3 text-sm">
      <summary className="cursor-pointer font-semibold text-sky-900">📖 What do these numbers mean? (in plain words)</summary>
      <dl className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-2">
        {PLAIN_TERMS.map(([term, meaning]) => (
          <div key={term}>
            <dt className="font-semibold text-gray-800">{term}</dt>
            <dd className="text-gray-600">{meaning}</dd>
          </div>
        ))}
      </dl>
    </details>
  );
}

function plainCorrelation(r) {
  if (!isNum(r)) return 'not enough criteria to compare strong and weak areas';
  if (r >= 0.7) return 'agree on which areas are strong and weak';
  if (r >= 0.4) return 'partly agree on which areas are strong and weak';
  if (r > -0.1) return 'don’t clearly agree on which areas are strong and weak';
  return 'see the strong and weak areas the opposite way';
}

function plainPair(p) {
  if (!p.n) return '—';
  const within = p.adjacent / p.n;
  const closeness = within >= 0.8 ? 'Usually within one level'
    : within >= 0.5 ? 'Often within one level' : 'Often more than one level apart';
  return `${closeness}; ${plainCorrelation(p.r)}.`;
}

function plainRater(r, referenceMode) {
  const ref = referenceMode === 'teacher' ? 'your grades'
    : referenceMode === 'system' ? 'the system evaluator' : 'the other AIs';
  const sim = r.similarity;
  const near = !isNum(sim) ? 'Grades' : sim >= 90 ? 'Grades almost the same as'
    : sim >= 75 ? 'Grades about the same as' : sim >= 60 ? 'Grades somewhat differently from' : 'Grades quite differently from';
  const lean = r.tendency.label === 'Lenient' ? ', a little more generously'
    : r.tendency.label === 'Strict' ? ', a little more strictly' : '';
  const diff = isNum(r.differentiation) && r.differentiation < 0.5
    ? 'gives almost the same score to every criterion' : 'tells strong and weak areas apart';
  return `${near} ${ref}${lean}; ${diff}.`;
}

function StatisticsPanel({
  stats, raters, criteria, referenceMode, setReferenceMode, referenceLabel,
  teacherRef, setTeacherRef, systemRef, onRunSystem, systemLoading, canRunSystem,
  onLogRun, logged, history, onClearHistory, pending,
}) {
  const { closest } = stats;
  const summary = summarizeHistory(history);

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h4 className="font-semibold text-gray-800">📈 Agreement Statistics</h4>
          <p className="text-xs text-gray-500">
            Distances are on a 0–100 scale per criterion, so criteria with different ranges are comparable.
          </p>
        </div>
        <div className="flex flex-wrap gap-1 bg-gray-100 rounded-lg p-1 text-sm" role="tablist" aria-label="Compare against">
          {[
            { id: 'consensus', label: 'Consensus' },
            { id: 'system', label: 'System evaluator' },
            { id: 'teacher', label: 'Teacher scores' },
          ].map((opt) => (
            <button
              key={opt.id}
              role="tab"
              aria-selected={referenceMode === opt.id}
              onClick={() => setReferenceMode(opt.id)}
              className={`px-3 py-1 rounded-md transition ${
                referenceMode === opt.id ? 'bg-white shadow text-gray-900' : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <PlainGlossary />

      {referenceMode === 'consensus' && (
        <p className="text-xs text-gray-600 bg-gray-50 rounded-lg p-2">
          Each model is compared with the average of the <em>other</em> models (never with itself).
          The closest model is the most “central” grader; an outlier sits far from the rest.
        </p>
      )}

      {referenceMode === 'system' && (
        <div className="text-xs text-gray-600 bg-gray-50 rounded-lg p-2 flex flex-wrap items-center gap-2">
          <span>Reference = this app's own evaluator using the active rubric.</span>
          {systemRef ? (
            <span className="text-emerald-700">✓ Scored</span>
          ) : null}
          <button
            onClick={onRunSystem}
            disabled={systemLoading || !canRunSystem}
            className="ml-auto bg-indigo-600 text-white px-3 py-1 rounded-md hover:bg-indigo-700 disabled:opacity-50"
          >
            {systemLoading ? 'Scoring…' : systemRef ? 'Re-score' : 'Score with system evaluator'}
          </button>
          {!canRunSystem && <span className="w-full text-amber-700">Log in to use the system evaluator.</span>}
        </div>
      )}

      {referenceMode === 'teacher' && (
        <div className="bg-gray-50 rounded-lg p-3">
          <p className="text-xs text-gray-600 mb-2">
            Enter your own scores to see which AI grades most like you.
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
            {criteria.map((c) => (
              <label key={c.key} className="text-xs text-gray-700">
                {c.label} <span className="text-gray-400">(1–{c.max_score})</span>
                <input
                  type="number"
                  min={1}
                  max={c.max_score}
                  value={teacherRef[c.key] ?? ''}
                  onChange={(e) => setTeacherRef((prev) => ({
                    ...prev,
                    [c.key]: e.target.value === '' ? undefined : Math.max(1, Math.min(c.max_score, Number(e.target.value))),
                  }))}
                  className="mt-1 w-full border rounded-md p-1.5 text-sm text-center bg-white"
                />
              </label>
            ))}
          </div>
        </div>
      )}

      <ClosestSummary
        stats={stats}
        criteria={criteria}
        raters={raters}
        referenceMode={referenceMode}
        referenceLabel={referenceLabel}
        pending={pending}
      />

      {/* Per-model cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {stats.perRater.map((r) => (
          <div key={r.id} className={`rounded-xl border p-3 ${closest?.id === r.id ? 'border-indigo-300 ring-1 ring-indigo-200' : 'border-gray-200'}`}>
            <div className="flex items-center justify-between mb-2 gap-2">
              <SlotBadge rater={r} />
              <span className={`text-xs px-2 py-0.5 rounded-full ${r.tendency.className}`}>{r.tendency.label}</span>
            </div>
            <dl className="grid grid-cols-2 gap-y-1 text-sm">
              <dt className="text-gray-500">Similarity</dt>
              <dd className="text-right font-semibold">{fmtPct(r.similarity, 1)}</dd>
              <dt className="text-gray-500">Avg distance</dt>
              <dd className="text-right">{isNum(r.mae) ? `${r.mae.toFixed(1)} pts` : '—'}</dd>
              <dt className="text-gray-500">Bias</dt>
              <dd className="text-right">{fmtSigned(r.bias, 1, ' pts')}</dd>
              <dt className="text-gray-500">Weighted score</dt>
              <dd className="text-right">{fmtPct(isNum(r.weighted) ? r.weighted * 100 : null)}</dd>
              <dt className="text-gray-500">Criteria closest</dt>
              <dd className="text-right">{stats.criteriaWon[r.id] || 0}/{criteria.length}</dd>
              <dt className="text-gray-500" title="Standard deviation of this model's own scores across criteria">Differentiation</dt>
              <dd className="text-right">{r.differentiation.toFixed(1)}</dd>
            </dl>
            <p className="mt-2 text-xs text-gray-700 bg-gray-50 rounded-md p-2">
              <span className="font-semibold">In plain words: </span>{plainRater(r, referenceMode)}
            </p>
          </div>
        ))}
        {raters.filter((r) => !r.scores).map((r) => (
          <div key={r.id} className="rounded-xl border border-dashed border-gray-300 p-3 text-sm text-gray-500 flex items-center gap-2">
            <SlotBadge rater={r} /> no parsable scores
          </div>
        ))}
      </div>
      <p className="text-xs text-gray-500 -mt-2">
        Bias &gt; +{BIAS_THRESHOLD} pts = Lenient, &lt; −{BIAS_THRESHOLD} pts = Strict, otherwise Balanced.
        Low differentiation means a model gives nearly the same score to every criterion.
      </p>

      {/* Pairwise agreement */}
      {stats.pairs.length > 0 && (
        <div>
          <h5 className="text-sm font-semibold text-gray-800 mb-2">Pairwise agreement</h5>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-600 border-b">
                  <th className="text-left font-medium py-1.5">Pair</th>
                  <th className="font-medium text-center">Exact match</th>
                  <th className="font-medium text-center">Within ±1</th>
                  <th className="font-medium text-center">Avg |diff|</th>
                  <th className="font-medium text-center" title="Pearson correlation of normalized scores; needs 3+ criteria and variation">Correlation r</th>
                  <th className="font-medium text-left pl-3">In plain words</th>
                </tr>
              </thead>
              <tbody>
                {stats.pairs.map((p) => (
                  <tr key={`${p.a.id}${p.b.id}`} className="border-b border-gray-100">
                    <td className="py-1.5"><SlotBadge rater={p.a} short /> vs <SlotBadge rater={p.b} short /></td>
                    <td className="text-center">{p.n ? `${p.exact}/${p.n} (${Math.round((p.exact / p.n) * 100)}%)` : '—'}</td>
                    <td className="text-center">{p.n ? `${p.adjacent}/${p.n} (${Math.round((p.adjacent / p.n) * 100)}%)` : '—'}</td>
                    <td className="text-center">{isNum(p.mae) ? `${p.mae.toFixed(2)} pts` : '—'}</td>
                    <td className="text-center">{isNum(p.r) ? p.r.toFixed(2) : '—'}</td>
                    <td className="text-left pl-3 text-gray-700 text-xs">{plainPair(p)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-500 mt-1">
            All models agree exactly on {stats.allAgreeCount}/{criteria.length} criteria.
            {stats.perCriterion.some((c) => c.highDisagreement) && (
              <> Biggest disagreements: {stats.perCriterion.filter((c) => c.highDisagreement).map((c) => c.label).join(', ')}.</>
            )}
          </p>
        </div>
      )}

      {/* Session history */}
      <div className="border-t pt-4">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
          <h5 className="text-sm font-semibold text-gray-800">Across saved runs (this browser)</h5>
          <div className="flex gap-2">
            <button
              onClick={onLogRun}
              disabled={!closest || logged || pending > 0}
              className="text-sm bg-indigo-600 text-white px-3 py-1 rounded-md hover:bg-indigo-700 disabled:opacity-50"
            >
              {logged ? '✓ Run saved' : '📌 Save this run'}
            </button>
            {history.length > 0 && (
              <button onClick={onClearHistory} className="text-sm text-gray-600 hover:text-red-600 px-2">
                Clear
              </button>
            )}
          </div>
        </div>
        {summary.length === 0 ? (
          <p className="text-xs text-gray-500">
            Save runs to build up which model your reference (consensus, system or teacher) most often matches.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-600 border-b">
                  <th className="text-left font-medium py-1.5">Model</th>
                  <th className="font-medium text-center">Runs</th>
                  <th className="font-medium text-center">Closest</th>
                  <th className="font-medium text-center">Avg similarity</th>
                  <th className="font-medium text-center">Avg bias</th>
                  <th className="font-medium text-center">Tendency</th>
                </tr>
              </thead>
              <tbody>
                {summary.map((s, i) => (
                  <tr key={s.name} className="border-b border-gray-100">
                    <td className="py-1.5 text-gray-800">{i === 0 && s.closest > 0 ? '🏆 ' : ''}{s.name}</td>
                    <td className="text-center">{s.runs}</td>
                    <td className="text-center">{s.closest} ({Math.round(s.closestRate * 100)}%)</td>
                    <td className="text-center">{fmtPct(s.avgSimilarity, 1)}</td>
                    <td className="text-center">{fmtSigned(s.avgBias, 1, ' pts')}</td>
                    <td className="text-center">
                      <span className={`text-xs px-2 py-0.5 rounded-full ${s.tendency.className}`}>{s.tendency.label}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

// -----------------------------------------------------------------------------
// Model Information Cards
// -----------------------------------------------------------------------------
const MODEL_INFO = [
  {
    provider: 'deepseek',
    name: 'DeepSeek',
    icon: '🌐',
    color: 'border-blue-300 bg-blue-50',
    advantages: [
      'Free tier with 10M tokens',
      'Strong reasoning and math',
      'Fast API responses (~2-5s)',
    ],
    bestFor: 'Complex analytical essays, quick iterations, cost‑sensitive projects.',
  },
  {
    provider: 'gemma',
    name: 'Ollama (Gemma 2B)',
    icon: '🟢',
    color: 'border-emerald-300 bg-emerald-50',
    advantages: [
      'Lightweight (1.6GB)',
      'Fast on CPU (~10-20s)',
      'Easy to set up',
    ],
    bestFor: 'Quick drafts, low‑resource VPS, simple feedback.',
  },
  {
    provider: 'ollama',
    name: 'Ollama (Llama 3.2 3B)',
    icon: '🦙',
    color: 'border-purple-300 bg-purple-50',
    advantages: [
      '100% local – no API keys',
      'Same model the essay evaluator uses, so it is usually already loaded',
      'Reliable JSON output through Ollama',
    ],
    bestFor: 'Privacy‑sensitive evaluations, offline use, a local second opinion.',
  },
];

// -----------------------------------------------------------------------------
// Main Component
// -----------------------------------------------------------------------------
export default function PromptPlayground() {
  const { user, isAdmin } = useAuth();

  const [showModelInfo, setShowModelInfo] = useState(true);

  const [essayText, setEssayText] = useState('');
  const [savedEssays, setSavedEssays] = useState([]);
  const [selectedEssayId, setSelectedEssayId] = useState('');
  const [selectedEssaySource, setSelectedEssaySource] = useState('');

  const [slots, setSlots] = useState(
    SLOTS.map((s) => ({ id: s.id, provider: s.defaultProvider, model: '' }))
  );
  // { [slotId]: { status: 'loading' | 'done' | 'error', result, error, ms, provider, model } }
  const [runs, setRuns] = useState({});
  const [runId, setRunId] = useState(0);

  const [rubricCriteria, setRubricCriteria] = useState(DEFAULT_CRITERIA);
  const [rubricInfo, setRubricInfo] = useState(null);
  // Once an admin hand-edits the prompts, rubric changes stop overwriting them.
  const promptsEdited = useRef(false);

  const [referenceMode, setReferenceMode] = useState('consensus');
  const [primaryId, setPrimaryId] = useState('A');
  const [compareId, setCompareId] = useState('B');
  const [teacherRef, setTeacherRef] = useState({});
  const [systemRef, setSystemRef] = useState(null);
  const [systemLoading, setSystemLoading] = useState(false);
  const [history, setHistory] = useState(() => loadHistory());
  const [loggedRunId, setLoggedRunId] = useState(null);

  const [systemPrompt, setSystemPrompt] = useState(
    'You are a strict JSON API. Respond with a SINGLE valid JSON object and NOTHING else. ' +
    'Do not write any text before or after the JSON. Do not wrap it in code fences. Do not add notes, comments, or explanations outside the JSON. ' +
    'The response must be parseable by JSON.parse() with no surrounding text. ' +
    'Schema (exact keys): ' +
    'main_statement, organization, evidence, analysis, grammar (each an integer 1, 2, 3, or 4), ' +
    'feedback (a 3–5 sentence overall summary string), ' +
    'criterion_feedback (an object with exactly these keys: main_statement, organization, evidence, analysis, grammar — each a 2–3 sentence string explaining that specific score), ' +
    'grammar_issues (an array of objects), ' +
    'recommendations (an array of objects). ' +
    'Each grammar_issues entry must have exactly these keys: "original" (text quoted from the essay), "correction" (the fixed version), "explanation" (one short sentence). ' +
    'Each recommendations entry must have exactly these keys: "title" (short label), "issue" (what is weak in the essay), "suggestion" (a concrete, actionable fix). ' +
    'RULES for grammar_issues: ' +
    '(a) Only include REAL errors where original !== correction. ' +
    '(b) Never include an entry where original equals correction. ' +
    '(c) Never write "no change needed". ' +
    '(d) If there are no grammar errors, grammar_issues MUST be []. ' +
    'RULES for recommendations: ' +
    '(a) Provide 3–5 recommendations tailored to THIS essay. ' +
    '(b) Each suggestion must reference a specific sentence, paragraph, or claim. ' +
    '(c) Never repeat the same recommendation twice. ' +
    '(d) Never give generic advice like "improve your writing". ' +
    'Use 1 = Beginning, 2 = Developing, 3 = Proficient, 4 = Exemplary. ' +
    'Do not give all criteria the same score; differentiate based on evidence.'
  );

  const [userPromptTemplate, setUserPromptTemplate] = useState(
    `Evaluate this essay:

"""
{essay}
"""

Return ONLY a JSON object with this exact shape:

{
  "main_statement": <int 1-4>,
  "organization":   <int 1-4>,
  "evidence":       <int 1-4>,
  "analysis":       <int 1-4>,
  "grammar":        <int 1-4>,
  "feedback":       "<3-5 sentence overall summary>",
  "criterion_feedback": {
    "main_statement": "<2-3 sentences explaining this score>",
    "organization":   "<2-3 sentences explaining this score>",
    "evidence":       "<2-3 sentences explaining this score>",
    "analysis":       "<2-3 sentences explaining this score>",
    "grammar":        "<2-3 sentences explaining this score>"
  },
  "grammar_issues": [
    {
      "original":    "<text quoted from the essay with an actual error>",
      "correction":  "<corrected version>",
      "explanation": "<one-sentence reason>"
    }
  ],
  "recommendations": [
    {
      "title":      "<short label, e.g. 'Strengthen evidence'>",
      "issue":      "<what is weak in the essay>",
      "suggestion": "<concrete, actionable fix — quote or point to a specific part of the essay>"
    }
  ]
}

CRITICAL RULES:
- Output ONLY the JSON. No text before it, no text after it, no code fences.
- Every grammar_issues entry MUST have original ≠ correction.
- Never include an entry where original equals correction.
- Never write "no change needed" or "correct as is".
- If there are no grammar errors, use "grammar_issues": [].
- Provide 3–5 recommendations. Each must reference a specific part of the essay.
- criterion_feedback must give a SPECIFIC reason for each score, referencing the essay.

Example of a CORRECT response:

{"main_statement": 3, "organization": 4, "evidence": 2, "analysis": 3, "grammar": 3, "feedback": "Your thesis is clear but could be sharper. Organization flows well. Evidence is thin in the second paragraph. Grammar has minor subject-verb agreement issues.", "criterion_feedback": {"main_statement": "The thesis previews both sides but does not take a position, which weakens the argument's focus. Consider stating a clear stance in the introduction.", "organization": "Paragraphing is clear and the essay follows a logical intro-body-conclusion structure. Transitions between paragraphs are smooth.", "evidence": "Most claims are asserted rather than supported. 'Friendship matters' and 'many videos teach things' need concrete examples, statistics, or studies.", "analysis": "The essay lists pros and cons but does not explain why or how these effects occur. Deeper reasoning about causes or implications is missing.", "grammar": "Only minor stylistic weaknesses — no real grammatical errors. Sentences are clear and readable."}, "grammar_issues": [], "recommendations": [{"title": "Strengthen evidence", "issue": "Paragraph 2 makes general claims about accessibility without any concrete data.", "suggestion": "Add a specific statistic or study — e.g., cite a UNESCO report on digital learning access — to ground the claim."}, {"title": "Tighten the conclusion", "issue": "The conclusion restates the intro instead of synthesising the argument.", "suggestion": "End with a forward-looking statement about how schools should adopt technology — connect back to the thesis."}, {"title": "Clarify a key term", "issue": "The phrase 'responsible use' is used twice without definition.", "suggestion": "Define what responsible use means in this context — e.g., verifying sources, limiting screen time, protecting privacy."}]}

Now output the JSON for the essay above. Nothing else.`
  );

  // Load the rubric that applies to this user (their default, the
  // school-wide one, or the built-in one) and build the prompts from it, so
  // nobody has to edit prompts by hand.
  useEffect(() => {
    getActiveRubric()
      .then((data) => {
        if (!Array.isArray(data?.criteria) || !data.criteria.length) return;
        setRubricCriteria(data.criteria);
        setRubricInfo(data.rubric || null);
        // The hand-tuned defaults already match the built-in rubric.
        if (!promptsEdited.current && !isDefaultRubric(data.criteria)) {
          const { system, user: userTemplate } = buildPromptsFromRubric(data.criteria);
          setSystemPrompt(system);
          setUserPromptTemplate(userTemplate);
        }
      })
      .catch(() => {});
  }, [user]);

  useEffect(() => {
    if (user) {
      loadAllEssays();
    } else {
      setSavedEssays([]);
    }
  }, [user]);

  const loadAllEssays = async () => {
    try {
      const [saved, knowledge] = await Promise.all([
        listSavedEssays().catch(() => []),
        listKnowledge().catch(() => []),
      ]);

      const combined = [
        ...saved.map(e => ({
          ...e,
          source: 'saved',
          displayTitle: e.title || `Draft ${e.id}`,
        })),
        ...knowledge.map(e => ({
          ...e,
          source: 'kb',
          displayTitle: e.title || `Evaluation #${e.id}`,
        })),
      ];

      combined.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      setSavedEssays(combined);
    } catch (err) {
      toast.error('Failed to load essays');
    }
  };

  const autoSaveEssay = async () => {
    if (!user || !essayText.trim()) return;
    const now = new Date();
    const autoTitle = `Essay ${now.toLocaleDateString()} ${now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    try {
      await saveEssay(autoTitle, essayText);
      await loadAllEssays();
    } catch (err) {
      console.warn('Auto‑save failed:', err);
    }
  };

  const handleLoadEssay = (e) => {
    const value = e.target.value;
    if (!value) {
      setSelectedEssayId('');
      setSelectedEssaySource('');
      return;
    }
    const [source, idStr] = value.split(':');
    const id = parseInt(idStr);
    const selected = savedEssays.find(e => e.source === source && e.id === id);
    if (selected) {
      setEssayText(selected.essay);
      setSelectedEssayId(id);
      setSelectedEssaySource(source);
      toast.success(`Loaded: ${selected.displayTitle}`);
    }
  };

  const handleDeleteEssay = async () => {
    if (!selectedEssayId || selectedEssaySource !== 'saved') {
      toast.error('Only draft essays can be deleted here.');
      return;
    }
    if (!confirm('Delete this saved essay?')) return;
    try {
      await deleteSavedEssay(selectedEssayId);
      toast.success('Essay deleted');
      setSelectedEssayId('');
      setSelectedEssaySource('');
      setEssayText('');
      loadAllEssays();
    } catch (err) {
      toast.error('Failed to delete');
    }
  };

  const updateSlot = (id, patch) => {
    setSlots((prev) => prev.map((s) => (s.id === id ? { ...s, ...patch } : s)));
  };

  const runSlot = async (slot, essay) => {
    const started = performance.now();
    setRuns((prev) => ({ ...prev, [slot.id]: { status: 'loading', provider: slot.provider, model: slot.model } }));
    // CPU models get the compact prompt unless an admin hand-edited the prompts.
    const compact = LOCAL_PROVIDERS.has(slot.provider) && !promptsEdited.current;
    const prompts = compact
      ? buildCompactPrompts(rubricCriteria)
      : { system: systemPrompt, user: userPromptTemplate };
    try {
      const response = await testPrompt(
        slot.provider,
        prompts.system,
        prompts.user.replace('{essay}', essay),
        slot.model || null,
        true,
        compact ? buildJsonSchema(rubricCriteria) : null,
      );
      const result = response.result;
      const failed = result?.model === 'error';
      setRuns((prev) => ({
        ...prev,
        [slot.id]: {
          status: failed ? 'error' : 'done',
          result,
          error: failed ? result.text : null,
          ms: performance.now() - started,
          stats: result?.stats || null,
          compact,
          provider: slot.provider,
          model: slot.model,
        },
      }));
    } catch (err) {
      setRuns((prev) => ({
        ...prev,
        [slot.id]: { status: 'error', error: errorMessage(err), provider: slot.provider, model: slot.model },
      }));
    }
  };

  const handleRunAll = async () => {
    if (!essayText.trim()) {
      toast.error('Please enter essay text.');
      return;
    }
    if (user) {
      await autoSaveEssay();
    }
    setRunId((n) => n + 1);
    setSystemRef(null);
    // Cloud models (DeepSeek) run first, then the VPS models one at a time,
    // so only one model is ever running and each gets the server to itself.
    const remote = slots.filter((s) => !LOCAL_PROVIDERS.has(s.provider));
    const local = slots.filter((s) => LOCAL_PROVIDERS.has(s.provider));
    setRuns(Object.fromEntries(slots.map((s) => [s.id, { status: 'queued', provider: s.provider, model: s.model }])));
    await Promise.allSettled(remote.map((slot) => runSlot(slot, essayText)));
    for (const slot of local) {
      await runSlot(slot, essayText);
    }
    toast.success('All model runs finished');
  };

  const handleRerun = (slot) => {
    if (!essayText.trim()) {
      toast.error('Please enter essay text.');
      return;
    }
    setRunId((n) => n + 1);
    runSlot(slot, essayText);
  };

  const handleRunSystem = async () => {
    if (!essayText.trim()) {
      toast.error('Please enter essay text.');
      return;
    }
    setSystemLoading(true);
    try {
      const data = await evaluateEssay(essayText, 'analytic');
      setSystemRef(data.scores);
      setLoggedRunId(null);
    } catch (err) {
      toast.error('System evaluation failed: ' + errorMessage(err));
    } finally {
      setSystemLoading(false);
    }
  };

  const applyRubricToPrompts = () => {
    const { system, user: userTemplate } = buildPromptsFromRubric(rubricCriteria);
    setSystemPrompt(system);
    setUserPromptTemplate(userTemplate);
    promptsEdited.current = false;
    toast.success('Prompts rebuilt from your rubric');
  };

  // ---------- Derived comparison data ----------
  const raters = useMemo(() => slots.map((slot) => {
    const run = runs[slot.id];
    return {
      id: slot.id,
      name: providerDisplayName(run?.provider ?? slot.provider, run?.model ?? slot.model),
      scores: run?.status === 'done' ? tryParse(run.result) : null,
    };
  }), [slots, runs]);

  const criteria = useMemo(
    () => buildCriteria(raters.map((r) => r.scores), rubricCriteria),
    [raters, rubricCriteria]
  );

  const consensus = useMemo(() => consensusScores(criteria, raters), [criteria, raters]);

  const reference = referenceMode === 'system' ? systemRef
    : referenceMode === 'teacher' ? teacherRef
    : othersConsensus(criteria, raters);
  const referenceLabel = referenceMode === 'system' ? 'System evaluator'
    : referenceMode === 'teacher' ? 'Teacher'
    : 'Consensus';

  const stats = useMemo(
    () => ({ ...computeStats(criteria, raters, reference), consensus }),
    [criteria, raters, reference, consensus]
  );

  const anyRun = Object.keys(runs).length > 0;
  const pendingCount = Object.values(runs).filter((r) => r.status === 'loading' || r.status === 'queued').length;
  const anyLoading = pendingCount > 0;
  const scoredCount = raters.filter((r) => r.scores).length;
  const logKey = `${runId}:${referenceMode}:${JSON.stringify(referenceMode === 'consensus' ? consensus : reference ?? {})}`;

  const handleLogRun = () => {
    if (!stats.closest) return;
    const entry = {
      at: new Date().toISOString(),
      reference: referenceMode,
      closest: stats.tie ? null : stats.closest.name,
      models: stats.perRater.map((r) => ({ name: r.name, mae: r.mae, bias: r.bias })),
    };
    const next = [...history, entry];
    setHistory(next);
    saveHistory(next);
    setLoggedRunId(logKey);
    toast.success('Run added to statistics');
  };

  const handleDownloadComparison = () => {
    const saved = savedEssays.find((e) => e.source === selectedEssaySource && e.id === selectedEssayId);
    const words = essayText.trim().split(/\s+/).slice(0, 8).join(' ');
    try {
      generateComparisonReport({
        essayTitle: saved?.displayTitle || (words ? `${words}…` : 'Untitled essay'),
        essayText,
        criteria,
        raters,
        primaryId,
        compareId,
        stats,
        reference,
        referenceLabel,
      });
      toast.success('Comparison report downloaded');
    } catch (err) {
      console.error('[comparison report]', err);
      toast.error('Could not create the report: ' + err.message);
    }
  };

  const handleClearHistory = () => {
    if (!confirm('Clear saved comparison statistics?')) return;
    setHistory([]);
    saveHistory([]);
  };

  return (
    <div className="glass-card rounded-2xl shadow-xl p-6 animate-fade-in bg-white/90 backdrop-blur-md">
      <h2 className="text-2xl font-bold text-gray-800 mb-2">🧪 AI Prompt Playground</h2>
      <p className="text-gray-600 mb-4">
        Test and compare different AI models with custom prompts. Essays are saved automatically.
      </p>

      <div className="mb-5 border border-gray-200 rounded-xl overflow-hidden">
        <button
          onClick={() => setShowModelInfo(!showModelInfo)}
          className="w-full flex items-center justify-between p-3 bg-gray-50 hover:bg-gray-100 transition"
        >
          <span className="font-semibold text-gray-700">
            📘 Model Comparison & Recommendations
          </span>
          <span className="text-gray-500 text-sm">
            {showModelInfo ? '▲ Hide' : '▼ Show'}
          </span>
        </button>
        {showModelInfo && (
          <div className="p-4 bg-white grid grid-cols-1 md:grid-cols-3 gap-4">
            {MODEL_INFO.map((model) => (
              <div
                key={model.provider}
                className={`border rounded-xl p-4 ${model.color}`}
              >
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">{model.icon}</span>
                  <h3 className="font-bold text-gray-800">{model.name}</h3>
                </div>
                <ul className="text-sm text-gray-700 space-y-1 mb-3">
                  {model.advantages.map((adv, idx) => (
                    <li key={idx} className="flex items-start gap-1">
                      <span className="text-green-600">✓</span>
                      <span>{adv}</span>
                    </li>
                  ))}
                </ul>
                <p className="text-xs text-gray-600 italic border-t pt-2 mt-1">
                  <strong>Best for:</strong> {model.bestFor}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>

      {user ? (
        <div className="mb-6 p-4 bg-blue-50 rounded-xl border border-blue-200">
          <div className="flex flex-wrap items-end gap-3">
            <div className="flex-1 min-w-[200px]">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                📚 Load Saved Essay
              </label>
              <select
                value={selectedEssayId ? `${selectedEssaySource}:${selectedEssayId}` : ''}
                onChange={handleLoadEssay}
                className="w-full border rounded-lg p-2 bg-white text-gray-800"
              >
                <option value="">-- Select a saved essay --</option>
                {savedEssays.map((essay) => (
                  <option key={`${essay.source}-${essay.id}`} value={`${essay.source}:${essay.id}`}>
                    {essay.source === 'kb' ? '📋' : '📄'} {essay.displayTitle} (
                    {new Date(essay.created_at).toLocaleDateString()})
                  </option>
                ))}
              </select>
            </div>
            {selectedEssayId && (
              <button
                onClick={handleDeleteEssay}
                disabled={selectedEssaySource !== 'saved'}
                className={`px-4 py-2 rounded-lg transition ${
                  selectedEssaySource === 'saved'
                    ? 'bg-red-100 text-red-700 hover:bg-red-200'
                    : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                }`}
              >
                🗑️ Delete
              </button>
            )}
          </div>
          <p className="text-xs text-gray-500 mt-2">
            ✨ Your essays are automatically saved when you run an evaluation. The dropdown shows both drafts and
            evaluated essays from your Knowledge Base.
          </p>
        </div>
      ) : (
        <div className="mb-6 p-4 bg-amber-50 rounded-xl border border-amber-200">
          <p className="text-amber-800 text-sm">
            🔒 <strong>Log in</strong> to save and load your essays.
          </p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: essay + prompts */}
        <div className="space-y-4">
          <div>
            <label className="block font-semibold text-gray-800 mb-1">📄 Essay to Evaluate</label>
            <textarea
              rows="10"
              value={essayText}
              onChange={(e) => setEssayText(e.target.value)}
              className="w-full border rounded-xl p-2 bg-gray-50 text-gray-800"
              placeholder="Paste your essay here..."
            />
          </div>

          <div className="p-3 bg-indigo-50 border border-indigo-200 rounded-xl text-sm text-indigo-900">
            📐 Scoring with <strong>{rubricInfo?.title || 'the built-in rubric'}</strong>:{' '}
            {rubricCriteria.map((c) => `${c.label} (1–${c.max_score})`).join(', ')}.
            <span className="block text-xs text-indigo-700 mt-1">
              Change your default rubric in the 📐 Rubrics tab.
            </span>
          </div>

          {isAdmin && (
            <details className="border border-gray-200 rounded-xl">
              <summary className="cursor-pointer p-3 bg-gray-50 hover:bg-gray-100 font-semibold text-gray-700 rounded-xl">
                ⚙️ Advanced (admin): Edit Prompts
              </summary>
              <div className="p-4 space-y-4">
                <div>
                  <label className="block font-semibold text-gray-800 mb-1">🤖 System Prompt</label>
                  <textarea
                    rows="4"
                    value={systemPrompt}
                    onChange={(e) => { promptsEdited.current = true; setSystemPrompt(e.target.value); }}
                    className="w-full border rounded-xl p-2 bg-gray-50 text-gray-800"
                  />
                </div>

                <div>
                  <label className="block font-semibold text-gray-800 mb-1">
                    📋 User Prompt Template
                    <span className="text-sm font-normal ml-1">(use {'{essay}'})</span>
                  </label>
                  <textarea
                    rows="6"
                    value={userPromptTemplate}
                    onChange={(e) => { promptsEdited.current = true; setUserPromptTemplate(e.target.value); }}
                    className="w-full border rounded-xl p-2 bg-gray-50 font-mono text-sm text-gray-800"
                  />
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <p className="text-xs text-gray-500 flex-1">
                    These prompts apply to all three models. Edit only if you understand the required
                    JSON output format.
                  </p>
                  <button
                    onClick={applyRubricToPrompts}
                    className="text-xs border border-indigo-300 text-indigo-700 px-2 py-1 rounded-md hover:bg-indigo-50"
                  >
                    Rebuild from rubric
                  </button>
                </div>
              </div>
            </details>
          )}
        </div>

        {/* Right: the three models */}
        <div className="space-y-4">
          <div className="bg-blue-50 p-4 rounded-xl border border-blue-200">
            <h3 className="font-semibold text-blue-800 mb-1">🎯 Models to Compare</h3>
            <p className="text-xs text-blue-900/70 mb-3">
              All three receive the same prompts and run in parallel.
            </p>
            <div className="space-y-2">
              {slots.map((slot) => {
                const style = SLOT_STYLE[slot.id];
                const run = runs[slot.id];
                const parsedOk = run?.status === 'done' && tryParse(run.result) !== null;
                return (
                  <div key={slot.id} className="flex flex-wrap items-center gap-2 bg-white rounded-lg border p-2">
                    <span className={`inline-flex items-center justify-center w-7 h-7 rounded-full text-xs font-bold border ${style.badge}`}>
                      {slot.id}
                    </span>
                    <select
                      value={slot.provider}
                      onChange={(e) => updateSlot(slot.id, { provider: e.target.value })}
                      className="flex-1 min-w-[140px] border rounded-lg p-2 bg-white text-gray-800 text-sm"
                      aria-label={`Provider for model ${slot.id}`}
                    >
                      <option value="deepseek">DeepSeek</option>
                      <option value="gemma">Gemma (Ollama)</option>
                      <option value="ollama">Llama 3.2 3B (Ollama)</option>
                    </select>
                    <input
                      type="text"
                      value={slot.model}
                      onChange={(e) => updateSlot(slot.id, { model: e.target.value })}
                      placeholder="Model (optional)"
                      className="w-36 border rounded-lg p-2 bg-white text-gray-800 text-sm"
                      aria-label={`Model name for ${slot.id}`}
                    />
                    <span className="w-20 text-xs text-right">
                      {run?.status === 'queued' && <span className="text-gray-500">queued…</span>}
                      {run?.status === 'loading' && <span className="text-blue-600">running…</span>}
                      {run?.status === 'done' && (
                        <span className={parsedOk ? 'text-emerald-600' : 'text-amber-600'}>
                          {parsedOk ? '✓' : '⚠ not JSON'} {run.ms ? `${(run.ms / 1000).toFixed(0)}s` : ''}
                          {run.stats?.tokens_per_second ? ` · ${run.stats.tokens_per_second} tok/s` : ''}
                        </span>
                      )}
                      {run?.status === 'error' && <span className="text-red-600">✕ failed</span>}
                    </span>
                    <button
                      onClick={() => handleRerun(slot)}
                      disabled={run?.status === 'loading'}
                      className="text-sm text-gray-600 hover:text-gray-900 px-1 disabled:opacity-40"
                      title="Run only this model"
                      aria-label={`Re-run model ${slot.id}`}
                    >
                      ↻
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          <button
            onClick={handleRunAll}
            disabled={anyLoading}
            className="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white py-3 rounded-xl font-semibold shadow-md hover:shadow-lg transition disabled:opacity-50"
          >
            {anyLoading ? 'Running models…' : '🚀 Run All 3 Models'}
          </button>

          {anyRun && (
            <div className="text-sm text-gray-600 bg-gray-50 rounded-xl border p-3">
              {scoredCount} of {slots.length} models returned parsable scores
              {criteria.length > 0 && <> on {criteria.length} criteria</>}.
            </div>
          )}
        </div>
      </div>

      {/* Comparison + statistics */}
      {scoredCount > 0 && criteria.length > 0 && (
        <div className="mt-6 space-y-6">
          <ScoreComparison
            criteria={criteria}
            raters={raters}
            primaryId={primaryId}
            compareId={compareId}
            onPrimary={setPrimaryId}
            onCompare={setCompareId}
            onDownload={handleDownloadComparison}
          />
          <details className="group">
            <summary className="cursor-pointer text-sm font-semibold text-gray-700 mb-2">
              🔎 Detailed table: all three models, consensus and pairwise differences
            </summary>
            <ComparisonTable
              raters={raters}
              stats={stats}
              reference={reference}
              referenceLabel={referenceLabel}
              showReference={referenceMode !== 'consensus'}
            />
          </details>
          <StatisticsPanel
            stats={stats}
            raters={raters}
            criteria={criteria}
            referenceMode={referenceMode}
            setReferenceMode={setReferenceMode}
            referenceLabel={referenceLabel}
            teacherRef={teacherRef}
            setTeacherRef={setTeacherRef}
            systemRef={systemRef}
            onRunSystem={handleRunSystem}
            systemLoading={systemLoading}
            canRunSystem={!!user}
            onLogRun={handleLogRun}
            logged={loggedRunId === logKey}
            pending={pendingCount}
            history={history}
            onClearHistory={handleClearHistory}
          />
        </div>
      )}

      {/* Individual results */}
      {anyRun && (
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-4">
          {raters.map((rater) => {
            const run = runs[rater.id];
            return (
              <div key={rater.id} className="min-w-0">
                <h3 className="font-semibold mb-2"><SlotBadge rater={rater} /></h3>
                <div className="bg-gray-50 rounded-xl border p-4 min-h-[200px] max-h-[600px] overflow-auto">
                  {!run ? (
                    <p className="text-gray-500 text-center mt-16">Not run yet.</p>
                  ) : run.status === 'queued' ? (
                    <p className="text-gray-500 text-center mt-16 text-sm px-4">
                      ⏳ Queued. Models run one at a time so each finishes faster.
                    </p>
                  ) : run.status === 'loading' ? (
                    <LoadingIndicator message={`Running ${rater.name}…`} />
                  ) : run.status === 'error' ? (
                    <p className="text-sm text-red-600 whitespace-pre-wrap">{run.error}</p>
                  ) : (
                    <EvaluationDisplay result={run.result} criteria={criteria.length ? criteria : rubricCriteria} />
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      <div className="mt-6">
        <details className="cursor-pointer group">
          <summary className="font-semibold text-gray-700 flex items-center gap-2">
            <span>🤖 How to Use AI as a Reference (Not a Replacement)</span>
            <span className="text-xs text-gray-400 group-open:hidden">click to expand</span>
          </summary>
          <div className="mt-3 p-4 bg-amber-50 rounded-lg border border-amber-200 text-sm text-gray-700 space-y-2">
            <p><strong>🧠 AI is a tool to assist your judgment, not replace it.</strong></p>
            <ul className="list-disc pl-5 space-y-1">
              <li>Use AI-generated scores and feedback as a <strong>second opinion</strong> to identify potential weaknesses.</li>
              <li>Always review the suggestions critically; your expertise as an educator is paramount.</li>
              <li>Compare results from different models to see variations in emphasis (some may focus more on grammar, others on coherence).</li>
              <li>Use the <strong>Teacher scores</strong> reference to find which model grades most like you.</li>
              <li>When overriding, your feedback becomes part of the learning system, helping future evaluations improve.</li>
            </ul>
            <p className="italic mt-2">💡 Best practice: Use AI feedback to start a conversation with students about their writing, not to give a final grade.</p>
          </div>
        </details>
      </div>
    </div>
  );
}
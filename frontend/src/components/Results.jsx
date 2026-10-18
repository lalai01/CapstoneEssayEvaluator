import React from 'react';
import toast from 'react-hot-toast';
import { saveKnowledge, saveOverride } from '../api';
import { useAuth } from '../context/AuthContext';
import ReportDialog from './ReportDialog';
import AnnotatedEssay from './AnnotatedEssay';
import {
  getCriteria,
  totalScore,
  maxTotal,
  averageFraction,
  weightedFraction,
  fractionOf,
  gradeFromFraction,
  colorForFraction,
  errorMessage,
  satisfactionStyle,
} from '../lib/scoring';

// Override is hidden for now: three AIs already evaluate every essay and the
// user compares them instead. Set to true to bring the buttons back; the
// override dialog and saving code below are kept unchanged.
const SHOW_OVERRIDE = false;

// ============================================================
// Scoring helpers
// ============================================================

// Rubric descriptors for the built-in 1–4 criteria (mirror backend
// ANALYTIC_RUBRIC, condensed). Admin-defined criteria fall back to the
// AI's per-criterion observation.
const RUBRIC_DESCRIPTORS = {
  main_statement: {
    4: "Clear, focused, defensible thesis that directly addresses the prompt.",
    3: "Clear thesis that addresses the prompt, minor weaknesses.",
    2: "Broad, unclear, or partially developed thesis.",
    1: "Thesis absent, unclear, or off-topic.",
  },
  organization: {
    4: "Ideas logically organized with clear progression.",
    3: "Generally well organized, minor weaknesses.",
    2: "Inconsistent organization; some ideas hard to follow.",
    1: "Poorly organized, no clear progression.",
  },
  evidence: {
    4: "Relevant, sufficient evidence that strongly supports the position.",
    3: "Adequate evidence with minor weaknesses.",
    2: "Limited or inconsistently relevant evidence.",
    1: "Little or no relevant evidence.",
  },
  analysis: {
    4: "Thorough explanation and clear connection to the thesis.",
    3: "Adequate explanation with some limitations.",
    2: "Limited explanation or weak connections.",
    1: "Little or no meaningful analysis.",
  },
  grammar: {
    4: "Accurate grammar; errors do not interfere with clarity.",
    3: "Minor errors that do not substantially interfere.",
    2: "Frequent errors that sometimes affect readability.",
    1: "Persistent errors that interfere with meaning.",
  },
};

const CRITERION_IMPROVEMENT_TIPS = {
  main_statement: "State a clear position in your introduction and restate it in your conclusion.",
  organization: "Add topic sentences to each paragraph and use transitions to link ideas.",
  evidence: "Support every claim with a specific example, statistic, or citation.",
  analysis: "After each piece of evidence, explain why it matters and how it supports your thesis.",
  grammar: "Read your essay aloud to catch errors, or run it through a grammar checker.",
};

function pct(f) {
  return f === null ? '—' : `${Math.round(f * 100)}%`;
}

function weightNote(criteria) {
  const weighted = criteria.filter((c) => (c.weight ?? 1) !== 1);
  if (weighted.length === 0) return 'All criteria carry equal weight.';
  return 'Weighted: ' + weighted.map((c) => `${c.label} ${c.weight}×`).join(', ') + '; others 1×.';
}

function filterGrammarIssues(issues) {
  if (!Array.isArray(issues)) return [];
  return issues.filter(
    (i) =>
      i &&
      typeof i.original === 'string' &&
      typeof i.correction === 'string' &&
      i.original.trim().toLowerCase() !== i.correction.trim().toLowerCase()
  );
}

function filterRecommendations(recs) {
  if (!Array.isArray(recs)) return [];
  return recs.filter(
    (r) => r && typeof r.suggestion === 'string' && r.suggestion.trim().length > 0
  );
}

function parseRagContent(text) {
  if (!text) return [];

  let cleaned = text.replace(/^-+\s*Similar past evaluation feedback.*?-+\s*/i, '').trim();
  cleaned = cleaned.replace(/^\[Similar essay feedback\]:\s*/i, '').trim();

  const chunks = cleaned
    .split(/\n\s*\n/)
    .map((c) => c.trim())
    .filter(Boolean);

  if (chunks.length === 0) return [{ title: 'Similar feedback', body: cleaned }];

  return chunks.map((chunk, idx) => {
    const lines = chunk.split('\n').filter((l) => l.trim());
    let title = `Past evaluation ${idx + 1}`;
    let body = chunk;

    if (lines.length > 1 && lines[0].length < 80) {
      title = lines[0].replace(/[:#\-–]+$/, '').trim();
      body = lines.slice(1).join('\n').trim();
    }

    return { title, body };
  });
}

// ============================================================
// AnalyzingLoader
// ============================================================
function AnalyzingLoader() {
  const [step, setStep] = React.useState(0);
  const [elapsed, setElapsed] = React.useState(0);

  const steps = [
    { emoji: '📖', label: 'Reading your essay…' },
    { emoji: '🔍', label: 'Scoring the five criteria…' },
    { emoji: '📝', label: 'Writing detailed feedback…' },
    { emoji: '✏️', label: 'Checking grammar and mechanics…' },
    { emoji: '💡', label: 'Generating recommendations…' },
    { emoji: '⏳', label: 'Almost there…' },
  ];

  React.useEffect(() => {
    const id = setInterval(() => {
      setStep((s) => Math.min(s + 1, steps.length - 1));
    }, 8000);
    return () => clearInterval(id);
  }, [steps.length]);

  React.useEffect(() => {
    const id = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(id);
  }, []);

  const current = steps[step];

  return (
    <div className="glass-card rounded-2xl shadow-xl p-12 bg-white/90 backdrop-blur-md">
      <div className="flex flex-col items-center justify-center">
        <div className="relative w-full max-w-md h-16 mb-8 overflow-hidden">
          <div className="absolute bottom-2 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-blue-300 to-transparent rounded-full" />
          <div className="absolute bottom-4 left-0 animate-walk-across text-4xl">🚶</div>
          <div className="absolute bottom-4 left-0 animate-walk-across text-4xl opacity-20" style={{ animationDelay: '0.15s' }}>🚶</div>
          <div className="absolute bottom-4 left-0 animate-walk-across text-4xl opacity-10" style={{ animationDelay: '0.3s' }}>🚶</div>
        </div>

        <div className="flex items-center gap-3 mb-3">
          <span className="text-3xl animate-pulse">{current.emoji}</span>
          <span className="text-lg font-semibold text-gray-800">{current.label}</span>
        </div>

        <div className="flex items-center gap-2 mb-6">
          {steps.map((_, idx) => (
            <span
              key={idx}
              className={`w-2 h-2 rounded-full transition-all ${
                idx <= step ? 'bg-blue-600 w-6' : 'bg-gray-300'
              }`}
            />
          ))}
        </div>

        <p className="text-sm text-gray-600 text-center">
          This can take <strong>30–60 seconds</strong> on CPU. Thanks for your patience.
        </p>
        <p className="text-xs text-gray-400 mt-1">Elapsed: {elapsed}s</p>

        <div className="w-64 h-1.5 bg-gray-200 rounded-full overflow-hidden mt-6">
          <div className="h-full w-1/3 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full animate-loading-bar" />
        </div>
      </div>
    </div>
  );
}

// ============================================================
// Satisfaction rating (1–10)
// ============================================================
function satisfactionLabel(v) {
  if (v === null) return 'Not rated yet';
  if (v >= 9) return 'Very satisfied';
  if (v >= 7) return 'Satisfied';
  if (v >= 5) return 'Neutral';
  if (v >= 3) return 'Unsatisfied';
  return 'Very unsatisfied';
}

function SatisfactionRating({ value, onChange }) {
  return (
    <div className="bg-gray-50 border border-gray-200 rounded-xl p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-semibold text-gray-700">How satisfied are you with this evaluation?</span>
        <span className={`text-xs font-medium px-2 py-0.5 rounded-full border ${satisfactionStyle(value).chip}`}>
          {value !== null && <strong>{value}/10 · </strong>}
          {satisfactionLabel(value)}
        </span>
      </div>
      <div className="grid grid-cols-10 gap-1" role="radiogroup" aria-label="Satisfaction from 1 to 10">
        {Array.from({ length: 10 }, (_, i) => i + 1).map((n) => {
          const selected = value === n;
          const tone = n <= 3 ? 'bg-red-500' : n <= 6 ? 'bg-amber-500' : 'bg-emerald-500';
          return (
            <button
              key={n}
              type="button"
              role="radio"
              aria-checked={selected}
              onClick={() => onChange(n)}
              className={`py-1.5 rounded-md text-sm font-medium transition ${
                selected
                  ? `${tone} text-white shadow`
                  : value !== null && n < value
                    ? `${tone} text-white opacity-40`
                    : 'bg-white border border-gray-200 text-gray-700 hover:bg-gray-100'
              }`}
            >
              {n}
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ============================================================
// Main Component
// ============================================================
export default function Results({
  scores,
  feedback,
  loading,
  essayText,
  essayTitle,
  evalType,
  showOverrideModal,
  setShowOverrideModal,
  teacherFeedback,
  setTeacherFeedback,
  suggestedChanges,
  setSuggestedChanges,
}) {
  const { user, isTeacher, isAdmin } = useAuth();
  const [teacherScores, setTeacherScores] = React.useState({});
  const [submittingOverride, setSubmittingOverride] = React.useState(false);
  // 1–10 rating of how satisfied the user is with this evaluation.
  const [satisfaction, setSatisfaction] = React.useState(null);
  const [showReport, setShowReport] = React.useState(false);
  // The submitted override (teacher scores + remarks), reused in the report.
  const [submittedOverride, setSubmittedOverride] = React.useState(null);

  // A new evaluation needs a fresh rating.
  React.useEffect(() => {
    setSatisfaction(null);
    setSubmittedOverride(null);
  }, [scores]);

  const aiLabel = scores?.evaluated_by?.label || null;

  const handleSaveToKB = async () => {
    if (!essayText || !scores) {
      toast.error('No evaluation to save.');
      return;
    }
    if (satisfaction === null) {
      toast.error('Please rate your satisfaction (1–10) before saving.');
      return;
    }
    try {
      const payload = {
        title: essayTitle?.trim() || 'Untitled Essay',
        essay: essayText,
        main_statement: evalType === 'holistic' ? null : scores.main_statement,
        organization:   evalType === 'holistic' ? null : scores.organization,
        evidence:       evalType === 'holistic' ? null : scores.evidence,
        analysis:       evalType === 'holistic' ? null : scores.analysis,
        grammar:        evalType === 'holistic' ? null : scores.grammar,
        holistic_score: evalType === 'holistic' ? scores.holistic_score : null,
        level_description: evalType === 'holistic' ? scores.level_description : null,
        criterion_feedback: scores.criterion_feedback || null,
        grammar_issues: scores.grammar_issues || null,
        recommendations: scores.recommendations || null,
        feedback: feedback,
        eval_type: evalType,
        accepted: true,
        satisfaction,
        teacher_feedback: null,
        // Full scores keep admin/teacher-defined criteria and the rubric used.
        scores,
        ai_model: aiLabel,
      };
      await saveKnowledge(payload);
      toast.success(isTeacher
        ? 'Approved and saved to the Knowledge Base'
        : 'Saved to your Knowledge Base (not approved; a teacher must approve evaluations)');
    } catch (err) {
      toast.error('Failed to save: ' + errorMessage(err));
    }
  };

  const handleReject = () => {
    if (!essayText || !scores) {
      toast.error('No evaluation to override.');
      return;
    }
    // Pre-fill the teacher's scores with the AI's so only changes need editing.
    const initial = {};
    if (evalType === 'holistic') {
      initial.holistic_score = scores.holistic_score;
    } else {
      getCriteria(scores).forEach(({ key }) => { initial[key] = scores[key]; });
    }
    setTeacherScores(initial);
    // Remarks are the teacher's own words; the AI feedback is shown separately
    // for reference rather than pre-filled (which saved the AI's text as "remarks").
    setTeacherFeedback(submittedOverride?.remarks || '');
    setSuggestedChanges(submittedOverride?.suggestions || '');
    setShowOverrideModal(true);
  };

  const overrideFields = () => {
    if (evalType === 'holistic') {
      return [{ key: 'holistic_score', label: 'Holistic Score', max_score: 5 }];
    }
    return getCriteria(scores);
  };

  const submitOverride = async () => {
    if (!teacherFeedback.trim()) {
      toast.error('Please write your remarks for this essay.');
      return;
    }
    const cleanedTeacherScores = {};
    for (const { key, label, max_score } of overrideFields()) {
      const v = teacherScores[key];
      if (v === '' || v === undefined || v === null) continue;
      const n = Number(v);
      if (!Number.isInteger(n) || n < 1 || n > max_score) {
        toast.error(`${label} must be a whole number from 1 to ${max_score}.`);
        return;
      }
      cleanedTeacherScores[key] = n;
    }
    setSubmittingOverride(true);
    try {
      await saveOverride({
        title: essayTitle?.trim() || null,
        ai_model: aiLabel,
        original_essay: essayText,
        original_scores: scores,
        teacher_scores: Object.keys(cleanedTeacherScores).length ? cleanedTeacherScores : null,
        teacher_feedback: teacherFeedback.trim(),
        suggested_changes: suggestedChanges.trim() || null,
        accepted: false,
        satisfaction,
      });
      const changed = Object.entries(cleanedTeacherScores).some(([k, v]) => v !== scores[k]);
      setSubmittedOverride({
        teacherScores: changed ? cleanedTeacherScores : null,
        remarks: teacherFeedback.trim(),
        suggestions: suggestedChanges.trim(),
      });
      toast.success('Override saved to the Learning Hub');
      setShowOverrideModal(false);
    } catch (err) {
      toast.error('Failed to save override: ' + errorMessage(err));
    } finally {
      setSubmittingOverride(false);
    }
  };

  // ---------- Loading / Empty ----------
  if (loading) {
    return <AnalyzingLoader />;
  }

  if (!scores) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 text-center animate-fade-in">
        <div className="text-6xl mb-4">📝</div>
        <h3 className="text-xl font-semibold text-gray-700">Ready for evaluation</h3>
        <p className="text-gray-500 mt-2">
          Enter an essay or upload a document, then click Evaluate.
        </p>
      </div>
    );
  }

  // ---------- RAG Section Extraction ----------
  let ragContent = '';
  let mainFeedback = feedback;
  if (feedback && feedback.includes('[RAG_INSIGHTS_START]')) {
    const ragMatch = feedback.match(/\[RAG_INSIGHTS_START\]([\s\S]*?)\[RAG_INSIGHTS_END\]/);
    if (ragMatch) {
      ragContent = ragMatch[1].trim();
      mainFeedback = feedback.replace(
        /\[RAG_INSIGHTS_START\][\s\S]*?\[RAG_INSIGHTS_END\]/,
        ''
      ).trim();
    }
  }

  const criteria = getCriteria(scores);
  const total = totalScore(scores, criteria);
  const possible = maxTotal(criteria);
  const average = averageFraction(scores, criteria);
  const weighted = weightedFraction(scores, criteria);
  const grade = gradeFromFraction(weighted);

  const grammarIssues = filterGrammarIssues(scores?.grammar_issues);
  const recommendations = filterRecommendations(scores?.recommendations);
  const criterionFeedback = scores?.criterion_feedback || {};
  const ragItems = parseRagContent(ragContent);

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="glass-card rounded-2xl shadow-xl p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
          <span className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center text-white text-sm">2</span>
          Evaluation Scores
        </h2>
        {(aiLabel || scores?.rubric?.title) && (
          <div className="flex flex-wrap gap-2 -mt-2 mb-4 text-xs">
            {aiLabel && (
              <span
                className="bg-indigo-50 text-indigo-700 border border-indigo-100 px-2 py-1 rounded-full"
                title={scores?.evaluated_by?.reason || undefined}
              >
                🤖 Scored by {aiLabel}
                {scores?.evaluated_by?.selection === 'auto' && ' · auto-selected'}
              </span>
            )}
            {scores?.rubric?.title && evalType !== 'holistic' && (
              <span className="bg-gray-50 text-gray-700 border border-gray-200 px-2 py-1 rounded-full">
                📐 {scores.rubric.title}
              </span>
            )}
            {scores?.evaluated_by?.reason && (
              <p className="w-full text-gray-500 mt-1">{scores.evaluated_by.reason}</p>
            )}
          </div>
        )}

        {evalType === 'holistic' ? (
          <div className="text-center mb-6">
            <div className="text-sm text-gray-600 mb-2">Holistic Score</div>
            <div className={`text-6xl font-bold ${colorForFraction(fractionOf(scores.holistic_score, 5))}`}>
              {scores.holistic_score}/5
            </div>
            <div className="mt-2 p-3 bg-gray-50 rounded-lg text-gray-700">
              {scores.level_description}
            </div>
          </div>
        ) : (
          <>
            {/* Scoring Result Card */}
            <div className={`rounded-2xl border p-6 mb-6 ${grade.bg}`}>
              <div className="text-sm uppercase tracking-wide text-gray-600 mb-2 text-center">
                Scoring Result
              </div>

              <div className="flex justify-center items-baseline gap-3 mb-2">
                <div className={`text-6xl font-extrabold ${grade.color}`}>
                  {total}
                </div>
                <div className="text-3xl font-bold text-gray-400">
                  / {possible}
                </div>
              </div>

              <div className={`text-2xl font-bold text-center ${grade.color}`}>
                {grade.label}
              </div>

              <div className="mt-4 grid grid-cols-3 gap-3 text-center text-sm">
                <div className="bg-white/70 rounded-lg py-2">
                  <div className="text-gray-600">Average</div>
                  <div className={`font-bold text-lg ${grade.color}`}>
                    {pct(average)}
                  </div>
                </div>
                <div className="bg-white/70 rounded-lg py-2">
                  <div className="text-gray-600">Weighted</div>
                  <div className={`font-bold text-lg ${grade.color}`}>
                    {pct(weighted)}
                  </div>
                </div>
                <div className="bg-white/70 rounded-lg py-2">
                  <div className="text-gray-600">Percentage</div>
                  <div className={`font-bold text-lg ${grade.color}`}>
                    {possible > 0 ? Math.round((total / possible) * 100) : 0}%
                  </div>
                </div>
              </div>

              <div className="mt-3 text-xs text-gray-500 text-center">
                {weightNote(criteria)}
              </div>
            </div>

            {/* Detailed per-criterion breakdown */}
            <div className="space-y-4">
              {criteria.map(({ key, label, max_score }) => {
                const v = scores[key] ?? 0;
                const note = criterionFeedback[key];
                const descriptor = max_score === 4 ? RUBRIC_DESCRIPTORS[key]?.[v] : null;
                const tip = CRITERION_IMPROVEMENT_TIPS[key];
                return (
                  <div key={key} className="bg-blue-50 p-4 rounded-xl border border-blue-100">
                    <div className="flex justify-between items-center mb-2">
                      <div className="text-sm font-semibold text-gray-800">{label}</div>
                      <div className={`text-2xl font-bold ${colorForFraction(fractionOf(v, max_score))}`}>
                        {v}/{max_score}
                      </div>
                    </div>

                    <div className="w-full bg-gray-200 rounded-full h-2 mb-3">
                      <div
                        className="bg-blue-500 h-2 rounded-full transition-all"
                        style={{ width: `${Math.min(100, (v / max_score) * 100)}%` }}
                      />
                    </div>

                    {descriptor && (
                      <p className="text-xs text-gray-600 italic mb-2">
                        {descriptor}
                      </p>
                    )}

                    {note && (
                      <div className="bg-white p-3 rounded-lg border border-blue-200 text-sm text-gray-800 leading-relaxed mb-2">
                        <div className="text-xs font-semibold text-blue-700 mb-1">
                          📝 AI observation
                        </div>
                        {note}
                      </div>
                    )}

                    {v < max_score && tip && (
                      <div className="bg-amber-50 p-2 rounded-lg border border-amber-200 text-xs text-gray-700">
                        <span className="font-semibold text-amber-800">Try:</span> {tip}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </>
        )}

        {user ? (
          <div className="mt-5 space-y-3">
            <SatisfactionRating value={satisfaction} onChange={setSatisfaction} />
            {isTeacher ? (
              <div className="flex gap-3">
                <button
                  onClick={handleSaveToKB}
                  className="flex-1 bg-gradient-to-r from-emerald-500 to-teal-500 text-white py-2 rounded-xl font-semibold shadow-md hover:shadow-lg transition"
                >
                  ✅ Approve & Save to KB
                </button>
                {SHOW_OVERRIDE && (
                  <button
                    onClick={handleReject}
                    className="flex-1 bg-gradient-to-r from-red-500 to-orange-500 text-white py-2 rounded-xl font-semibold shadow-md hover:shadow-lg transition"
                  >
                    ✏️ Reject & Override
                  </button>
                )}
              </div>
            ) : (
              <>
                <div className="flex gap-3">
                  <button
                    onClick={handleSaveToKB}
                    className="flex-1 bg-gradient-to-r from-emerald-500 to-teal-500 text-white py-2 rounded-xl font-semibold shadow-md hover:shadow-lg transition"
                  >
                    💾 Save to My Knowledge Base
                  </button>
                  {SHOW_OVERRIDE && (
                    <button
                      onClick={handleReject}
                      className="flex-1 bg-gradient-to-r from-red-500 to-orange-500 text-white py-2 rounded-xl font-semibold shadow-md hover:shadow-lg transition"
                    >
                      ✏️ Override Scores
                    </button>
                  )}
                </div>
                {SHOW_OVERRIDE && (
                  <p className="text-xs text-gray-600 bg-gray-50 border border-gray-200 rounded-lg p-2">
                    {isAdmin
                      ? 'ℹ️ Your overrides are saved to your Learning Hub. Only teachers’ overrides and approvals teach the AI; to grade as a teacher, add your email as a teacher in the Admin Dashboard.'
                      : 'ℹ️ Disagree with the AI? Override it: your scores and remarks are saved to your Learning Hub. Only teachers’ overrides and approvals teach the AI.'}
                  </p>
                )}
              </>
            )}
            <button
              onClick={() => setShowReport(true)}
              className="w-full bg-gradient-to-r from-sky-500 to-cyan-500 text-white py-2 rounded-xl font-semibold shadow-md hover:shadow-lg transition"
            >
              📄 Generate Evaluation Report
            </button>
          </div>
        ) : (
          <div className="mt-5 space-y-3">
            <button
              onClick={() => setShowReport(true)}
              className="w-full bg-gradient-to-r from-sky-500 to-cyan-500 text-white py-2 rounded-xl font-semibold shadow-md hover:shadow-lg transition"
            >
              📄 Generate Evaluation Report
            </button>
            <div className="p-4 bg-amber-50 border border-amber-200 rounded-xl text-center">
              <p className="text-amber-800 text-sm">
                🔒 <strong>Sign in with Google or Yahoo</strong> to save this evaluation. As a guest it isn&apos;t
                kept, so download the report if you need a copy.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Detailed Feedback + RAG + Grammar + Recommendations */}
      <div className="glass-card rounded-2xl shadow-xl p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">📖 Detailed Feedback</h2>

        <AnnotatedEssay essay={essayText} scores={scores} />

        {/* RAG — now formatted as recommendation-style cards */}
        {ragItems.length > 0 && (
          <details className="mb-4 bg-purple-50 p-3 rounded-lg border border-purple-200">
            <summary className="font-semibold text-purple-800 cursor-pointer">
              📚 Similar Past Evaluations (RAG) — {ragItems.length} retrieved
            </summary>
            <div className="mt-3 space-y-2">
              {ragItems.map((item, idx) => (
                <div
                  key={idx}
                  className="bg-white p-3 rounded-lg border border-purple-200 text-sm"
                >
                  <div className="font-semibold text-purple-900 mb-1">
                    {idx + 1}. {item.title}
                  </div>
                  {item.body && (
                    <div className="text-gray-800 leading-relaxed whitespace-pre-wrap">
                      {item.body}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </details>
        )}

        <div className="bg-gray-50 p-5 rounded-xl border border-gray-100 overflow-y-auto max-h-[400px]">
          <div className="prose max-w-none text-gray-700 whitespace-pre-wrap">
            {mainFeedback}
          </div>
        </div>

        {/* Grammar */}
        {grammarIssues.length > 0 && (
          <div className="mt-5">
            <h3 className="font-semibold text-gray-800 mb-2 text-sm uppercase tracking-wide">
              ✏️ Grammar & Mechanics — Specific Fixes ({grammarIssues.length})
            </h3>
            <div className="space-y-2">
              {grammarIssues.map((issue, idx) => (
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
          <div className="mt-5">
            <h3 className="font-semibold text-gray-800 mb-2 text-sm uppercase tracking-wide">
              💡 Recommendations ({recommendations.length})
            </h3>
            <div className="space-y-2">
              {recommendations.map((rec, idx) => (
                <div key={idx} className="bg-amber-50 p-3 rounded-lg border border-amber-200 text-sm">
                  <div className="font-semibold text-amber-900 mb-1">
                    {idx + 1}. {rec.title || 'Improvement'}
                  </div>
                  {rec.issue && (
                    <div className="text-gray-700 mb-1">
                      <span className="font-medium text-amber-800">Issue:</span> {rec.issue}
                    </div>
                  )}
                  {rec.suggestion && (
                    <div className="text-gray-800">
                      <span className="font-medium text-amber-800">Try:</span> {rec.suggestion}
                    </div>
                  )}
                  {rec.quote && (
                    <div className="text-xs text-violet-800 mt-1 italic">In your essay: “{rec.quote}”</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {showReport && (
        <ReportDialog
          onClose={() => setShowReport(false)}
          defaults={{
            teacherName: user?.user_metadata?.full_name || user?.email || '',
            // The override's remarks and suggestions become the report's teacher comments.
            teacherComments: submittedOverride
              ? [submittedOverride.remarks, submittedOverride.suggestions && `Suggestions: ${submittedOverride.suggestions}`]
                .filter(Boolean).join('\n\n')
              : '',
          }}
          reportData={{
            title: essayTitle,
            essayText,
            feedback: mainFeedback,
            scores,
            evalType,
            teacherScores: submittedOverride?.teacherScores || null,
          }}
        />
      )}

      {/* Override Modal */}
      {showOverrideModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 animate-fade-in">
          <div className="bg-white rounded-2xl p-6 max-w-2xl w-full mx-4 shadow-2xl max-h-[90vh] overflow-y-auto">
            <h3 className="text-2xl font-bold mb-4 text-gray-800">{isTeacher ? 'Teacher Override' : 'Override the AI’s Scores'}</h3>

            <label className="block text-sm font-semibold mb-2">Your Scores</label>
            <div className="mb-4 border rounded-xl overflow-hidden">
              <div className="grid grid-cols-[1fr_auto_auto] gap-x-4 px-3 py-2 bg-gray-50 text-xs font-semibold text-gray-600">
                <span>Criterion</span>
                <span className="text-center w-16">AI</span>
                <span className="text-center w-24">Yours</span>
              </div>
              {overrideFields().map(({ key, label, max_score }) => {
                const entered = teacherScores[key];
                const changed = entered !== '' && entered != null && Number(entered) !== scores[key];
                return (
                  <div key={key} className="grid grid-cols-[1fr_auto_auto] gap-x-4 items-center px-3 py-2 border-t text-sm">
                    <span className="text-gray-800">{label}</span>
                    <span className="text-center w-16 text-gray-500">{scores[key] ?? '—'}/{max_score}</span>
                    <input
                      type="number"
                      min={1}
                      max={max_score}
                      step={1}
                      value={teacherScores[key] ?? ''}
                      onChange={(e) => setTeacherScores((prev) => ({ ...prev, [key]: e.target.value }))}
                      className={`w-24 border rounded-lg p-1.5 text-center focus:ring-2 focus:ring-blue-500 ${
                        changed ? 'border-amber-400 bg-amber-50' : ''
                      }`}
                      aria-label={`Your score for ${label}, 1 to ${max_score}`}
                    />
                  </div>
                );
              })}
            </div>

            {mainFeedback && (
              <details className="mb-4 border rounded-xl bg-gray-50">
                <summary className="cursor-pointer px-3 py-2 text-sm font-semibold text-gray-700">
                  🤖 AI feedback (for reference{aiLabel ? ` · ${aiLabel}` : ''})
                </summary>
                <p className="px-3 pb-3 text-sm text-gray-700 whitespace-pre-wrap max-h-40 overflow-y-auto">
                  {mainFeedback}
                </p>
              </details>
            )}

            <label htmlFor="override-remarks" className="block text-sm font-semibold mb-1">
              {isTeacher ? 'Teacher’s remarks' : 'Your remarks'} <span className="text-red-500">*</span>
            </label>
            <p className="text-xs text-gray-500 mb-1">
              {isTeacher
                ? 'Your own assessment: what the AI got wrong and how you judge the essay. Saved to the Learning Hub and used to improve later evaluations.'
                : 'Why you disagree with the AI. Saved to your Learning Hub for your own reference; only a teacher’s override changes what the AI learns.'}
            </p>
            <textarea
              id="override-remarks"
              rows="4"
              className="w-full border rounded-xl p-3 mb-3 focus:ring-2 focus:ring-blue-500"
              value={teacherFeedback}
              onChange={(e) => setTeacherFeedback(e.target.value)}
              placeholder="e.g., The thesis is clearer than the AI suggests, but the evidence in paragraph 2 is unsupported."
            />
            <label htmlFor="override-suggestions" className="block text-sm font-semibold mb-1">
              Suggestions for the student <span className="font-normal text-gray-400">(optional)</span>
            </label>
            <textarea
              id="override-suggestions"
              rows="3"
              className="w-full border rounded-xl p-3 mb-4 focus:ring-2 focus:ring-blue-500"
              value={suggestedChanges}
              onChange={(e) => setSuggestedChanges(e.target.value)}
              placeholder="e.g., Cite at least two studies in paragraph 2 and explain how each supports your thesis."
            />
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setShowOverrideModal(false)}
                className="px-4 py-2 bg-gray-200 rounded-xl hover:bg-gray-300 transition"
              >
                Cancel
              </button>
              <button
                onClick={submitOverride}
                disabled={submittingOverride}
                className="px-4 py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition disabled:opacity-50"
              >
                {submittingOverride ? 'Saving…' : 'Submit Override'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
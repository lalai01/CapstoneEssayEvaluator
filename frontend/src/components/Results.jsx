import React from 'react';
import toast from 'react-hot-toast';
import { jsPDF } from 'jspdf';
import { saveKnowledge, saveOverride } from '../api';
import { useAuth } from '../context/AuthContext';

// ============================================================
// Scoring helpers
// ============================================================

// Weights used for the weighted score
const CRITERIA_WEIGHTS = {
  main_statement: 1.0,
  organization: 1.0,
  evidence: 1.5,
  analysis: 1.5,
  grammar: 1.0,
};

// Fixed-order list of criteria
const CRITERIA = [
  { key: 'main_statement', label: 'Main Statement' },
  { key: 'organization',   label: 'Organization' },
  { key: 'evidence',       label: 'Evidence' },
  { key: 'analysis',       label: 'Analysis' },
  { key: 'grammar',        label: 'Grammar' },
];

// Total possible points (5 criteria × 4 points each = 20)
const MAX_TOTAL = CRITERIA.length * 4; // 20

// ---------- Total (unweighted) ----------
function getTotalScore(scores) {
  if (!scores) return 0;
  return CRITERIA.reduce((sum, { key }) => {
    const v = scores[key];
    return sum + (typeof v === 'number' ? v : 0);
  }, 0);
}

// ---------- Average (unweighted) ----------
function getAverage(scores) {
  if (!scores) return 0;
  const values = CRITERIA
    .map(({ key }) => scores[key])
    .filter(v => typeof v === 'number');
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

// ---------- Weighted total ----------
function getWeightedTotal(scores) {
  if (!scores) return 0;
  let weightedSum = 0;
  let totalWeight = 0;
  for (const [key, weight] of Object.entries(CRITERIA_WEIGHTS)) {
    const v = scores[key];
    if (typeof v === 'number') {
      weightedSum += v * weight;
      totalWeight += weight;
    }
  }
  return totalWeight > 0 ? weightedSum / totalWeight : 0;
}

// ---------- Grade label from 1–4 scale ----------
function gradeLabel(score) {
  if (score >= 3.5) return { label: 'Excellent', color: 'text-emerald-600', bg: 'bg-emerald-50 border-emerald-200' };
  if (score >= 2.5) return { label: 'Good', color: 'text-blue-600', bg: 'bg-blue-50 border-blue-200' };
  if (score >= 1.5) return { label: 'Developing', color: 'text-amber-600', bg: 'bg-amber-50 border-amber-200' };
  return { label: 'Beginning', color: 'text-red-600', bg: 'bg-red-50 border-red-200' };
}

// ---------- Color for individual 1–4 score ----------
function getScoreColor(value) {
  if (value >= 3.5) return 'text-emerald-600';
  if (value >= 2.5) return 'text-blue-600';
  if (value >= 1.5) return 'text-amber-600';
  return 'text-red-600';
}

// ============================================================
// PDF Export
// ============================================================
function exportEvaluationPDF({ title, essayText, feedback, scores, evalType, userEmail }) {
  const doc = new jsPDF({ unit: 'pt', format: 'a4' });
  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 40;
  let y = margin;

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(18);
  doc.text('Essay Evaluation Report', margin, y);
  y += 24;

  doc.setFontSize(12);
  doc.setFont('helvetica', 'normal');
  doc.text(`Title: ${title || 'Untitled Essay'}`, margin, y); y += 16;
  doc.text(`Evaluator: ${userEmail || 'Anonymous'}`, margin, y); y += 16;
  doc.text(`Evaluation Type: ${evalType}`, margin, y); y += 16;
  doc.text(`Date: ${new Date().toLocaleString()}`, margin, y); y += 24;

  doc.setFont('helvetica', 'bold');
  doc.text('Scores', margin, y); y += 18;
  doc.setFont('helvetica', 'normal');

  if (evalType === 'holistic') {
    doc.text(`Holistic Score: ${scores.holistic_score}/5`, margin, y); y += 16;
    doc.text(`Level: ${scores.level_description || ''}`, margin, y); y += 24;
  } else {
    CRITERIA.forEach(({ key, label }) => {
      const v = scores[key];
      doc.text(`${label}: ${typeof v === 'number' ? v : '—'}/4`, margin, y);
      y += 16;
    });
    y += 8;

    const total = getTotalScore(scores);
    const avg = getAverage(scores);
    const weighted = getWeightedTotal(scores);
    const grade = gradeLabel(weighted);

    doc.setFont('helvetica', 'bold');
    doc.text(`Total: ${total} / ${MAX_TOTAL}`, margin, y); y += 16;
    doc.text(`Average: ${avg.toFixed(2)} / 4`, margin, y); y += 16;
    doc.text(`Weighted: ${weighted.toFixed(2)} / 4 (${grade.label})`, margin, y); y += 24;
    doc.setFont('helvetica', 'normal');
  }

  doc.setFont('helvetica', 'bold');
  doc.text('Feedback', margin, y); y += 18;
  doc.setFont('helvetica', 'normal');
  const feedbackLines = doc.splitTextToSize(feedback || '', pageWidth - margin * 2);
  doc.text(feedbackLines, margin, y);
  y += feedbackLines.length * 14 + 16;

  doc.setFont('helvetica', 'bold');
  doc.text('Essay Text', margin, y); y += 18;
  doc.setFont('helvetica', 'normal');
  const essayLines = doc.splitTextToSize(essayText || '', pageWidth - margin * 2);
  doc.text(essayLines, margin, y);

  doc.save(`evaluation-${(title || 'essay').replace(/\s+/g, '_')}.pdf`);
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
  const { user } = useAuth();

  const handleSaveToKB = async () => {
    if (!essayText || !scores) {
      toast.error('No evaluation to save.');
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
        feedback: feedback,
        eval_type: evalType,
        accepted: true,
        satisfaction: 5,
        teacher_feedback: null,
      };
      await saveKnowledge(payload);
      toast.success('Saved to knowledge base!');
    } catch (err) {
      toast.error('Failed to save: ' + err.message);
    }
  };

  const handleReject = () => {
    if (!essayText || !scores) {
      toast.error('No evaluation to override.');
      return;
    }
    setTeacherFeedback(feedback);
    setSuggestedChanges('');
    setShowOverrideModal(true);
  };

  const submitOverride = async () => {
    if (!teacherFeedback.trim()) {
      toast.error('Please provide your feedback.');
      return;
    }
    try {
      await saveOverride({
        original_essay: essayText,
        original_scores: scores,
        teacher_feedback: teacherFeedback,
        suggested_changes: suggestedChanges,
        accepted: false,
      });
      toast.success('Override saved to learning knowledge base!');
      setShowOverrideModal(false);
    } catch (err) {
      toast.error('Failed to save override: ' + err.message);
    }
  };

  // ---------- Loading / Empty states ----------
  if (loading) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 flex flex-col items-center justify-center">
        <div className="spinner w-12 h-12"></div>
        <p className="mt-4 text-gray-600">Analyzing your essay...</p>
      </div>
    );
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

  // ---------- Precompute analytic summary ----------
  const total = getTotalScore(scores);
  const average = getAverage(scores);
  const weighted = getWeightedTotal(scores);
  const grade = gradeLabel(weighted);

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="glass-card rounded-2xl shadow-xl p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
          <span className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center text-white text-sm">2</span>
          Evaluation Scores
        </h2>

        {/* ============ Holistic ============ */}
        {evalType === 'holistic' ? (
          <div className="text-center mb-6">
            <div className="text-sm text-gray-600 mb-2">Holistic Score</div>
            <div className={`text-6xl font-bold ${getScoreColor(scores.holistic_score)}`}>
              {scores.holistic_score}/5
            </div>
            <div className="mt-2 p-3 bg-gray-50 rounded-lg text-gray-700">
              {scores.level_description}
            </div>
          </div>
        ) : (
          <>
            {/* ============ Scoring Result Card ============ */}
            <div className={`rounded-2xl border p-6 mb-6 ${grade.bg}`}>
              <div className="text-sm uppercase tracking-wide text-gray-600 mb-2 text-center">
                Scoring Result
              </div>

              <div className="flex justify-center items-baseline gap-3 mb-2">
                <div className={`text-6xl font-extrabold ${grade.color}`}>
                  {total}
                </div>
                <div className="text-3xl font-bold text-gray-400">
                  / {MAX_TOTAL}
                </div>
              </div>

              <div className={`text-2xl font-bold text-center ${grade.color}`}>
                {grade.label}
              </div>

              <div className="mt-4 grid grid-cols-3 gap-3 text-center text-sm">
                <div className="bg-white/70 rounded-lg py-2">
                  <div className="text-gray-600">Average</div>
                  <div className={`font-bold text-lg ${grade.color}`}>
                    {average.toFixed(2)}<span className="text-xs text-gray-400">/4</span>
                  </div>
                </div>
                <div className="bg-white/70 rounded-lg py-2">
                  <div className="text-gray-600">Weighted</div>
                  <div className={`font-bold text-lg ${grade.color}`}>
                    {weighted.toFixed(2)}<span className="text-xs text-gray-400">/4</span>
                  </div>
                </div>
                <div className="bg-white/70 rounded-lg py-2">
                  <div className="text-gray-600">Percentage</div>
                  <div className={`font-bold text-lg ${grade.color}`}>
                    {Math.round((total / MAX_TOTAL) * 100)}%
                  </div>
                </div>
              </div>

              <div className="mt-3 text-xs text-gray-500 text-center">
                Weighted average applied: Evidence and Analysis carry 1.5× weight.
              </div>
            </div>

            {/* ============ Individual Criteria ============ */}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-center">
              {CRITERIA.map(({ key, label }) => {
                const v = scores[key] ?? 0;
                return (
                  <div key={key} className="bg-blue-50 p-3 rounded-xl">
                    <div className="text-xs text-gray-600">{label}</div>
                    <div className={`text-3xl font-bold ${getScoreColor(v)}`}>
                      {v}/4
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-1.5 mt-2">
                      <div
                        className="bg-blue-500 h-1.5 rounded-full"
                        style={{ width: `${(v / 4) * 100}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}

        {/* ============ Action Buttons ============ */}
        {user ? (
          <div className="mt-5 space-y-3">
            <div className="flex gap-3">
              <button
                onClick={handleSaveToKB}
                className="flex-1 bg-gradient-to-r from-emerald-500 to-teal-500 text-white py-2 rounded-xl font-semibold shadow-md hover:shadow-lg transition"
              >
                💾 Save to KB
              </button>
              <button
                onClick={handleReject}
                className="flex-1 bg-gradient-to-r from-red-500 to-orange-500 text-white py-2 rounded-xl font-semibold shadow-md hover:shadow-lg transition"
              >
                ✏️ Reject & Override
              </button>
            </div>
            <button
              onClick={() =>
                exportEvaluationPDF({
                  title: essayTitle,
                  essayText,
                  feedback: mainFeedback,
                  scores,
                  evalType,
                  userEmail: user?.email,
                })
              }
              className="w-full bg-gradient-to-r from-sky-500 to-cyan-500 text-white py-2 rounded-xl font-semibold shadow-md hover:shadow-lg transition"
            >
              📄 Download Report (PDF)
            </button>
          </div>
        ) : (
          <div className="mt-5 p-4 bg-amber-50 border border-amber-200 rounded-xl text-center">
            <p className="text-amber-800 text-sm">
              🔒 <strong>Log in with Google or Yahoo</strong> to save this evaluation or provide teacher feedback.
            </p>
          </div>
        )}
      </div>

      {/* ============ Detailed Feedback ============ */}
      <div className="glass-card rounded-2xl shadow-xl p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">📖 Detailed Feedback</h2>

        {ragContent && (
          <details className="mb-4 bg-purple-50 p-3 rounded-lg border border-purple-200">
            <summary className="font-semibold text-purple-800 cursor-pointer">
              📚 Similar Past Evaluations (RAG) – click to expand
            </summary>
            <div className="mt-2 text-sm text-gray-700 whitespace-pre-wrap">{ragContent}</div>
          </details>
        )}

        <div className="bg-gray-50 p-5 rounded-xl border border-gray-100 overflow-y-auto max-h-[400px]">
          <div className="prose max-w-none text-gray-700 whitespace-pre-wrap">
            {mainFeedback}
          </div>
        </div>
      </div>

      {/* ============ Override Modal ============ */}
      {showOverrideModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 animate-fade-in">
          <div className="bg-white rounded-2xl p-6 max-w-2xl w-full mx-4 shadow-2xl max-h-[90vh] overflow-y-auto">
            <h3 className="text-2xl font-bold mb-4 text-gray-800">Teacher Override</h3>
            <label className="block text-sm font-semibold mb-1">Your Feedback (required)</label>
            <textarea
              rows="4"
              className="w-full border rounded-xl p-3 mb-3 focus:ring-2 focus:ring-blue-500"
              value={teacherFeedback}
              onChange={(e) => setTeacherFeedback(e.target.value)}
            />
            <label className="block text-sm font-semibold mb-1">Suggested Changes (optional)</label>
            <textarea
              rows="3"
              className="w-full border rounded-xl p-3 mb-4 focus:ring-2 focus:ring-blue-500"
              value={suggestedChanges}
              onChange={(e) => setSuggestedChanges(e.target.value)}
              placeholder="What specific changes would improve the essay?"
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
                className="px-4 py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition"
              >
                Submit Override
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
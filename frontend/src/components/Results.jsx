import React, { useState } from 'react';
import toast from 'react-hot-toast';
import { saveKnowledge, saveOverride } from '../api';
import { useAuth } from '../context/AuthContext';

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
  const [satisfaction, setSatisfaction] = useState(5); // default, will be overridden dynamically

  // Calculate dynamic satisfaction based on overall score
  const computeSatisfaction = () => {
    if (!scores) return 5;
    if (evalType === 'holistic') {
      // Map holistic 1-5 to 1-10
      return scores.holistic_score * 2;
    } else {
      const avg = (scores.grammar + scores.coherence + scores.content) / 3;
      return Math.round(avg / 10);
    }
  };

  // Update satisfaction whenever scores change
  React.useEffect(() => {
    if (scores) {
      setSatisfaction(computeSatisfaction());
    }
  }, [scores, evalType]);

  const handleSaveToKB = async () => {
    if (!essayText || !scores) {
      toast.error('No evaluation to save.');
      return;
    }
    try {
      const payload = {
        title: essayTitle?.trim() || 'Untitled Essay',
        essay: essayText,
        grammar: evalType === 'holistic' ? 0 : scores.grammar,
        coherence: evalType === 'holistic' ? 0 : scores.coherence,
        content: evalType === 'holistic' ? 0 : scores.content,
        feedback: feedback,
        eval_type: evalType,
        accepted: true,
        satisfaction: satisfaction,
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
        satisfaction: satisfaction,   // ✅ include satisfaction in override
      });
      toast.success('Override saved to learning knowledge base!');
      setShowOverrideModal(false);
    } catch (err) {
      toast.error('Failed to save override: ' + err.message);
    }
  };

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
        <p className="text-gray-500 mt-2">Enter an essay or upload a document, then click Evaluate.</p>
      </div>
    );
  }

  const getScoreColor = (score) => {
    if (score >= 85) return 'text-emerald-600';
    if (score >= 70) return 'text-blue-600';
    return 'text-amber-600';
  };

  const avgScore =
    evalType === 'analytic'
      ? Math.round((scores.grammar + scores.coherence + scores.content) / 3)
      : null;

  // Extract RAG section if present (same as before)
  let ragContent = '';
  let mainFeedback = feedback;
  if (feedback && feedback.includes('[RAG_INSIGHTS_START]')) {
    const ragMatch = feedback.match(/\[RAG_INSIGHTS_START\]([\s\S]*?)\[RAG_INSIGHTS_END\]/);
    if (ragMatch) {
      ragContent = ragMatch[1].trim();
      mainFeedback = feedback.replace(/\[RAG_INSIGHTS_START\][\s\S]*?\[RAG_INSIGHTS_END\]/, '').trim();
    }
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="glass-card rounded-2xl shadow-xl p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
          <span className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center text-white text-sm">2</span>
          Evaluation Scores
        </h2>

        {evalType === 'holistic' ? (
          <div className="text-center mb-6">
            <div className="text-sm text-gray-600 mb-2">Holistic Score</div>
            <div className={`text-6xl font-bold ${getScoreColor(scores.holistic_score * 20)}`}>
              {scores.holistic_score}/5
            </div>
            <div className="mt-2 p-3 bg-gray-50 rounded-lg text-gray-700">
              {scores.level_description}
            </div>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div className="score-card bg-gradient-to-br from-blue-50 to-white p-4 rounded-xl shadow">
                <div className="text-sm text-gray-600 font-medium">Grammar</div>
                <div className={`text-4xl font-bold ${getScoreColor(scores.grammar)}`}>{scores.grammar}</div>
                <div className="w-full bg-gray-200 rounded-full h-1.5 mt-2">
                  <div
                    className="bg-blue-500 h-1.5 rounded-full"
                    style={{ width: `${scores.grammar}%` }}
                  />
                </div>
              </div>
              <div className="score-card bg-gradient-to-br from-emerald-50 to-white p-4 rounded-xl shadow">
                <div className="text-sm text-gray-600 font-medium">Coherence</div>
                <div className={`text-4xl font-bold ${getScoreColor(scores.coherence)}`}>{scores.coherence}</div>
                <div className="w-full bg-gray-200 rounded-full h-1.5 mt-2">
                  <div
                    className="bg-emerald-500 h-1.5 rounded-full"
                    style={{ width: `${scores.coherence}%` }}
                  />
                </div>
              </div>
              <div className="score-card bg-gradient-to-br from-purple-50 to-white p-4 rounded-xl shadow">
                <div className="text-sm text-gray-600 font-medium">Content</div>
                <div className={`text-4xl font-bold ${getScoreColor(scores.content)}`}>{scores.content}</div>
                <div className="w-full bg-gray-200 rounded-full h-1.5 mt-2">
                  <div
                    className="bg-purple-500 h-1.5 rounded-full"
                    style={{ width: `${scores.content}%` }}
                  />
                </div>
              </div>
            </div>
            <div className="mt-5 pt-4 border-t border-gray-100">
              <div className="text-sm text-gray-600">Overall Score</div>
              <div className={`text-3xl font-bold ${getScoreColor(avgScore)}`}>{avgScore}/100</div>
            </div>
          </>
        )}

        {/* Satisfaction display (dynamic) */}
        <div className="mt-4 flex items-center gap-2 text-sm text-gray-600">
          <span>📊 Estimated Satisfaction:</span>
          <span className="font-semibold">{satisfaction}/10</span>
        </div>

        {user ? (
          <div className="flex gap-3 mt-5">
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
        ) : (
          <div className="mt-5 p-4 bg-amber-50 border border-amber-200 rounded-xl text-center">
            <p className="text-amber-800 text-sm">
              🔒 <strong>Log in with Google or Yahoo</strong> to save this evaluation or provide teacher feedback.
            </p>
          </div>
        )}
      </div>

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
          <div className="prose max-w-none text-gray-700 whitespace-pre-wrap">{mainFeedback}</div>
        </div>
      </div>

      {/* Override Modal with Satisfaction Input */}
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

            <label className="block text-sm font-semibold mb-1">
              Satisfaction Score (1–10): <span className="font-normal">{satisfaction}</span>
            </label>
            <input
              type="range"
              min="1"
              max="10"
              value={satisfaction}
              onChange={(e) => setSatisfaction(parseInt(e.target.value))}
              className="w-full mb-4"
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
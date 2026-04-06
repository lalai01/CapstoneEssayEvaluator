import React from 'react';
import toast from 'react-hot-toast';
import { saveKnowledge } from '../api';

export default function Results({ scores, feedback, loading, essayText, evalType }) {
  const handleSave = async () => {
    if (!essayText || !scores) {
      toast.error('No evaluation to save.');
      return;
    }
    try {
      await saveKnowledge({
        essay: essayText,
        grammar: scores.grammar,
        coherence: scores.coherence,
        content: scores.content,
        feedback: feedback,
        eval_type: evalType,
        accepted: true,
        satisfaction: 5,
        teacher_feedback: null
      });
      toast.success('Saved to knowledge base!');
    } catch (err) {
      toast.error('Failed to save: ' + err.message);
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
      <div className="glass-card rounded-2xl shadow-xl p-12 text-center">
        <div className="text-6xl mb-4">📝</div>
        <h3 className="text-xl font-semibold text-gray-700">Ready for evaluation</h3>
        <p className="text-gray-500 mt-2">Enter an essay or upload a document, then click Evaluate.</p>
      </div>
    );
  }

  const avgScore = Math.round((scores.grammar + scores.coherence + scores.content) / 3);
  const getScoreColor = (score) => {
    if (score >= 85) return 'text-emerald-600';
    if (score >= 70) return 'text-blue-600';
    return 'text-orange-600';
  };

  return (
    <div className="space-y-6">
      <div className="glass-card rounded-2xl shadow-xl p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
          <span className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center text-white text-sm">2</span>
          Evaluation Scores
        </h2>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="score-card bg-gradient-to-br from-blue-50 to-white p-4 rounded-xl shadow">
            <div className="text-sm text-gray-600 font-medium">Grammar</div>
            <div className={`text-4xl font-bold ${getScoreColor(scores.grammar)}`}>{scores.grammar}</div>
          </div>
          <div className="score-card bg-gradient-to-br from-emerald-50 to-white p-4 rounded-xl shadow">
            <div className="text-sm text-gray-600 font-medium">Coherence</div>
            <div className={`text-4xl font-bold ${getScoreColor(scores.coherence)}`}>{scores.coherence}</div>
          </div>
          <div className="score-card bg-gradient-to-br from-purple-50 to-white p-4 rounded-xl shadow">
            <div className="text-sm text-gray-600 font-medium">Content</div>
            <div className={`text-4xl font-bold ${getScoreColor(scores.content)}`}>{scores.content}</div>
          </div>
        </div>
        <div className="mt-5 pt-4 border-t border-gray-100">
          <div className="text-sm text-gray-600">Overall Score</div>
          <div className={`text-3xl font-bold ${getScoreColor(avgScore)}`}>{avgScore}/100</div>
        </div>
        <button
          onClick={handleSave}
          className="mt-5 w-full bg-gradient-to-r from-emerald-500 to-teal-500 text-white py-2 rounded-xl font-semibold shadow-md hover:shadow-lg transition"
        >
          💾 Save to Knowledge Base
        </button>
      </div>

      <div className="glass-card rounded-2xl shadow-xl p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">📖 Detailed Feedback</h2>
        <div className="prose max-w-none text-gray-700 whitespace-pre-wrap bg-gray-50 p-5 rounded-xl border border-gray-100">
          {feedback}
        </div>
      </div>
    </div>
  );
}
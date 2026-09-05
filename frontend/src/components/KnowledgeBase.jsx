import React, { useState, useEffect } from 'react';
import { listKnowledge, getRubric, getSuggestions } from '../api';
import { useAuth } from '../context/AuthContext';
import toast from 'react-hot-toast';

export default function KnowledgeBase() {
  const { user } = useAuth();
  const [rubric, setRubric] = useState(null);
  const [suggestions, setSuggestions] = useState(null);
  const [entries, setEntries] = useState([]);
  const [selectedEntry, setSelectedEntry] = useState(null);
  const [showFullEssay, setShowFullEssay] = useState(false);
  const [showFullFeedback, setShowFullFeedback] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (user) {
      loadData();
    } else {
      setLoading(false);
    }
  }, [user]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [rubricData, suggestionsData, entriesData] = await Promise.all([
        getRubric(),
        getSuggestions(),
        listKnowledge()
      ]);
      setRubric(rubricData);
      setSuggestions(suggestionsData);
      setEntries(entriesData);
    } catch (err) {
      toast.error('Failed to load data: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatText = (text) => {
    if (!text) return '';
    let formatted = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    formatted = formatted.replace(/\n/g, '<br />');
    return <span dangerouslySetInnerHTML={{ __html: formatted }} />;
  };

  const getScoreColor = (score) => {
    if (score >= 85) return 'text-emerald-600';
    if (score >= 70) return 'text-blue-600';
    return 'text-amber-600';
  };

  if (!user) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 text-center">
        <div className="text-6xl mb-4">🔒</div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Login Required</h2>
        <p className="text-gray-700">Please log in to view your knowledge base.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 text-center">
        <div className="animate-spin text-4xl mb-4">⏳</div>
        <p className="text-gray-700">Loading your knowledge base...</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 animate-fade-in">
      {/* Left column: List of saved evaluations */}
      <div className="glass-card rounded-2xl shadow-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold text-indigo-700">📚 Saved Evaluations</h2>
          <span className="text-sm bg-blue-100 text-blue-800 px-3 py-1 rounded-full">
            {entries.length} {entries.length === 1 ? 'entry' : 'entries'}
          </span>
        </div>

        {entries.length === 0 ? (
          <div className="text-center py-12">
            <div className="text-5xl mb-3">📭</div>
            <p className="text-gray-700">No saved evaluations yet.</p>
            <p className="text-sm text-gray-600 mt-1">Evaluate an essay and click "Save to KB".</p>
          </div>
        ) : (
          <ul className="space-y-2 max-h-[600px] overflow-y-auto pr-2">
            {entries.map(entry => {
              const avgScore = entry.eval_type === 'holistic'
                ? entry.content * 20  // 1-5 → 20-100
                : Math.round((entry.grammar + entry.coherence + entry.content) / 3);
              return (
                <li
                  key={entry.id}
                  className={`border border-gray-200 rounded-xl p-3 cursor-pointer transition-all hover:shadow-md ${
                    selectedEntry?.id === entry.id
                      ? 'bg-blue-50 border-blue-300 shadow-md'
                      : 'bg-white hover:bg-gray-50'
                  }`}
                  onClick={() => {
                    setSelectedEntry(entry);
                    setShowFullEssay(false);
                    setShowFullFeedback(false);
                  }}
                >
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <div className="font-semibold text-gray-800 truncate">
                        {entry.title || `Essay #${entry.id}`}
                      </div>
                      <div className="text-xs text-gray-600 mt-1">
                        {new Date(entry.created_at).toLocaleDateString(undefined, {
                          month: 'short', day: 'numeric', year: 'numeric'
                        })}
                      </div>
                    </div>
                    <div className={`text-lg font-bold ${getScoreColor(avgScore)}`}>
                      {entry.eval_type === 'holistic' ? `${entry.content}/5` : avgScore}
                    </div>
                  </div>
                  <div className="text-xs text-gray-600 mt-2 line-clamp-2">
                    {entry.essay.substring(0, 100)}...
                  </div>
                  <div className="flex items-center gap-3 mt-2 text-xs">
                    {entry.eval_type === 'analytic' ? (
                      <>
                        <span className="flex items-center gap-1">
                          <span className="w-2 h-2 rounded-full bg-blue-400"></span>
                          <span className="text-gray-700">G:{entry.grammar}</span>
                        </span>
                        <span className="flex items-center gap-1">
                          <span className="w-2 h-2 rounded-full bg-green-400"></span>
                          <span className="text-gray-700">C:{entry.coherence}</span>
                        </span>
                        <span className="flex items-center gap-1">
                          <span className="w-2 h-2 rounded-full bg-purple-400"></span>
                          <span className="text-gray-700">C:{entry.content}</span>
                        </span>
                      </>
                    ) : (
                      <span className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full bg-amber-400"></span>
                        <span className="text-gray-700">Holistic: {entry.content}/5</span>
                      </span>
                    )}
                    <span className="ml-auto text-gray-600">
                      {entry.eval_type === 'holistic' ? '🌟 Holistic' : '📊 Analytic'}
                    </span>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {/* Right column: Details of selected entry */}
      <div className="glass-card rounded-2xl shadow-xl p-6">
        <h2 className="text-2xl font-bold text-indigo-700 mb-4">📋 Evaluation Details</h2>

        {selectedEntry ? (
          <div className="space-y-5 max-h-[600px] overflow-y-auto pr-2">
            {/* Title and Date */}
            <div className="border-b border-gray-200 pb-3">
              <h3 className="text-xl font-semibold text-gray-800">
                {selectedEntry.title || `Essay #${selectedEntry.id}`}
              </h3>
              <p className="text-sm text-gray-600 mt-1">
                Evaluated on {new Date(selectedEntry.created_at).toLocaleString()}
              </p>
            </div>

            {/* Scores - Conditional rendering for holistic vs analytic */}
            {selectedEntry.eval_type === 'holistic' ? (
              <div className="bg-gray-50 p-4 rounded-xl">
                <h4 className="font-semibold text-indigo-600 mb-3">Holistic Score</h4>
                <div className="text-center">
                  <div className={`text-6xl font-bold ${getScoreColor(selectedEntry.content * 20)}`}>
                    {selectedEntry.content}/5
                  </div>
                  <div className="mt-2 p-3 bg-white rounded-lg text-gray-700 border border-gray-200">
                    {selectedEntry.feedback?.split('\n')[0] || 'No description available.'}
                  </div>
                </div>
                <div className="mt-4 pt-3 border-t border-gray-200">
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-gray-800">Overall</span>
                    <span className={`text-xl font-bold ${getScoreColor(selectedEntry.content * 20)}`}>
                      {selectedEntry.content}/5
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-gray-50 p-4 rounded-xl">
                <h4 className="font-semibold text-indigo-600 mb-3">Scores</h4>
                <div className="space-y-3">
                  {['grammar', 'coherence', 'content'].map(cat => {
                    const score = selectedEntry[cat];
                    const colorClass = cat === 'grammar' ? 'blue' : cat === 'coherence' ? 'emerald' : 'purple';
                    return (
                      <div key={cat}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="font-medium text-gray-700 capitalize">{cat}</span>
                          <span className={`font-bold text-${colorClass}-600`}>{score}</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className={`bg-${colorClass}-500 h-2 rounded-full`}
                            style={{ width: `${score}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="mt-4 pt-3 border-t border-gray-200">
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-gray-800">Overall</span>
                    <span className={`text-xl font-bold ${getScoreColor(
                      Math.round((selectedEntry.grammar + selectedEntry.coherence + selectedEntry.content) / 3)
                    )}`}>
                      {Math.round((selectedEntry.grammar + selectedEntry.coherence + selectedEntry.content) / 3)}/100
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* AI Feedback */}
            <div>
              <h4 className="font-semibold text-gray-800 mb-2 flex items-center gap-2">
                <span>🤖 AI Feedback</span>
                {selectedEntry.accepted && (
                  <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">Accepted</span>
                )}
              </h4>
              <div className="bg-gray-50 p-4 rounded-lg text-sm leading-relaxed text-gray-800">
                {showFullFeedback
                  ? formatText(selectedEntry.feedback)
                  : formatText(selectedEntry.feedback.substring(0, 600))}
                {selectedEntry.feedback.length > 600 && (
                  <button
                    onClick={() => setShowFullFeedback(!showFullFeedback)}
                    className="text-blue-600 ml-2 text-xs font-medium hover:underline"
                  >
                    {showFullFeedback ? 'Show less' : 'Show more'}
                  </button>
                )}
              </div>
            </div>

            {/* Teacher Override */}
            {selectedEntry.teacher_feedback && (
              <div>
                <h4 className="font-semibold text-gray-800 mb-2">👩‍🏫 Teacher Override</h4>
                <div className="bg-yellow-50 p-4 rounded-lg text-sm border-l-4 border-yellow-400 text-gray-800">
                  {formatText(selectedEntry.teacher_feedback)}
                </div>
              </div>
            )}

            {/* Essay Text */}
            <div>
              <h4 className="font-semibold text-gray-800 mb-2">📄 Essay Text</h4>
              <div className="bg-gray-50 p-4 rounded-lg text-sm whitespace-pre-wrap font-mono max-h-60 overflow-y-auto text-gray-800">
                {showFullEssay ? selectedEntry.essay : selectedEntry.essay.substring(0, 500)}
                {selectedEntry.essay.length > 500 && (
                  <button
                    onClick={() => setShowFullEssay(!showFullEssay)}
                    className="text-blue-600 ml-2 text-xs font-medium hover:underline"
                  >
                    {showFullEssay ? 'Show less' : 'Show more'}
                  </button>
                )}
              </div>
            </div>

            {/* Metadata Tags */}
            <div className="flex flex-wrap gap-2 pt-2">
              <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded-full">
                {selectedEntry.eval_type === 'holistic' ? '🌟 Holistic' : '📊 Analytic'}
              </span>
              <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded-full">
                Satisfaction: {selectedEntry.satisfaction}/10
              </span>
            </div>
          </div>
        ) : (
          <div className="text-center py-16">
            <div className="text-5xl mb-3">📋</div>
            <p className="text-gray-700">Select an entry from the left to view details.</p>
          </div>
        )}

        {/* Rubric and Suggestions (Collapsible) */}
        <div className="mt-6 pt-4 border-t border-gray-200">
          <details className="cursor-pointer group">
            <summary className="font-semibold text-gray-800 flex items-center gap-2">
              <span>📖 Scoring Rubric</span>
              <span className="text-xs text-gray-600 group-open:hidden">click to expand</span>
            </summary>
            {rubric && (
              <dl className="mt-3 space-y-3 pl-2">
                <div className="bg-blue-50 p-3 rounded-lg">
                  <dt className="font-medium text-blue-700">Grammar</dt>
                  <dd className="text-gray-700 text-sm mt-1">{rubric.grammar}</dd>
                </div>
                <div className="bg-green-50 p-3 rounded-lg">
                  <dt className="font-medium text-green-700">Coherence</dt>
                  <dd className="text-gray-700 text-sm mt-1">{rubric.coherence}</dd>
                </div>
                <div className="bg-purple-50 p-3 rounded-lg">
                  <dt className="font-medium text-purple-700">Content</dt>
                  <dd className="text-gray-700 text-sm mt-1">{rubric.content}</dd>
                </div>
              </dl>
            )}
          </details>

          {suggestions && (
            <details className="cursor-pointer mt-4 group">
              <summary className="font-semibold text-gray-800 flex items-center gap-2">
                <span>💡 How to Use This Tool</span>
                <span className="text-xs text-gray-600 group-open:hidden">click to expand</span>
              </summary>
              <div className="mt-3 space-y-3 pl-2">
                <div className="bg-indigo-50 p-3 rounded-lg">
                  <strong className="text-indigo-700">What it does</strong>
                  <p className="text-sm text-gray-700 mt-1">{suggestions.what}</p>
                </div>
                <div className="bg-indigo-50 p-3 rounded-lg">
                  <strong className="text-indigo-700">When to use</strong>
                  <p className="text-sm text-gray-700 mt-1 whitespace-pre-wrap">{suggestions.when}</p>
                </div>
                <div className="bg-indigo-50 p-3 rounded-lg">
                  <strong className="text-indigo-700">Best results</strong>
                  <p className="text-sm text-gray-700 mt-1 whitespace-pre-wrap">{suggestions.how}</p>
                </div>
              </div>
            </details>
          )}
        </div>
      </div>
    </div>
  );
}
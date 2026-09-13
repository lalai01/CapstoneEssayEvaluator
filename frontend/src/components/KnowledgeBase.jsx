import React, { useState, useEffect } from 'react';
import { listKnowledge, getRubric, getSuggestions } from '../api';
import { useAuth } from '../context/AuthContext';
import toast from 'react-hot-toast';

// ---------- Safe value renderer ----------
function safeRender(value) {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

// ---------- Average of 5 rubric criteria on 1-4 scale ----------
function computeAverage(entry) {
  if (!entry) return null;
  const keys = ['main_statement', 'organization', 'evidence', 'analysis', 'grammar'];
  const values = keys
    .map(k => entry[k])
    .filter(v => typeof v === 'number' && !isNaN(v));
  if (values.length === 0) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

// ---------- Get a single numeric score to display in the list ----------
function getDisplayScore(entry) {
  if (!entry) return { value: '—', isHolistic: false };

  if (entry.eval_type === 'holistic') {
    const hs = entry.holistic_score;
    return {
      value: typeof hs === 'number' ? `${hs}/5` : '—',
      isHolistic: true,
    };
  }

  const avg = computeAverage(entry);
  return {
    value: avg !== null ? `${avg.toFixed(1)}/4` : '—',
    isHolistic: false,
  };
}

// ---------- Color from 1-4 scale ----------
function colorFrom4Scale(score) {
  if (score === null || score === undefined) return 'text-gray-400';
  if (score >= 3.5) return 'text-emerald-600';
  if (score >= 2.5) return 'text-blue-600';
  if (score >= 1.5) return 'text-amber-600';
  return 'text-red-500';
}

// ---------- Color from 1-5 scale (holistic) ----------
function colorFrom5Scale(score) {
  if (score === null || score === undefined) return 'text-gray-400';
  if (score >= 4.5) return 'text-emerald-600';
  if (score >= 3.5) return 'text-blue-600';
  if (score >= 2.5) return 'text-amber-600';
  return 'text-red-500';
}

// ---------- Rubric criterion colors (visual distinction) ----------
const CRITERION_COLORS = {
  main_statement: { dot: 'bg-blue-400',   bar: 'bg-blue-500',   text: 'text-blue-700' },
  organization:   { dot: 'bg-emerald-400', bar: 'bg-emerald-500', text: 'text-emerald-700' },
  evidence:       { dot: 'bg-purple-400', bar: 'bg-purple-500',  text: 'text-purple-700' },
  analysis:       { dot: 'bg-amber-400',  bar: 'bg-amber-500',   text: 'text-amber-700' },
  grammar:        { dot: 'bg-pink-400',   bar: 'bg-pink-500',    text: 'text-pink-700' },
};

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
        listKnowledge(),
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
      {/* ============ Left: entries list ============ */}
      <div className="glass-card rounded-2xl shadow-xl p-6 bg-white/40 backdrop-blur-md">
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
            <p className="text-sm text-gray-600 mt-1">
              Evaluate an essay and click "Save to KB".
            </p>
          </div>
        ) : (
          <ul className="space-y-2 max-h-[600px] overflow-y-auto pr-2">
            {entries.map(entry => {
              const display = getDisplayScore(entry);
              const isHolistic = entry.eval_type === 'holistic';
              const scoreClass = isHolistic
                ? colorFrom5Scale(entry.holistic_score)
                : colorFrom4Scale(computeAverage(entry));

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
                          month: 'short',
                          day: 'numeric',
                          year: 'numeric',
                        })}
                      </div>
                    </div>
                    <div className={`text-lg font-bold ${scoreClass}`}>
                      {display.value}
                    </div>
                  </div>

                  <div className="text-xs text-gray-600 mt-2 line-clamp-2">
                    {(entry.essay || '').substring(0, 100)}...
                  </div>

                  <div className="flex items-center gap-3 mt-2 text-xs">
                    {!isHolistic && (
                      <>
                        <span className="flex items-center gap-1">
                          <span className={`w-2 h-2 rounded-full ${CRITERION_COLORS.main_statement.dot}`}></span>
                          <span className="text-gray-700">MS:{safeRender(entry.main_statement)}</span>
                        </span>
                        <span className="flex items-center gap-1">
                          <span className={`w-2 h-2 rounded-full ${CRITERION_COLORS.organization.dot}`}></span>
                          <span className="text-gray-700">OR:{safeRender(entry.organization)}</span>
                        </span>
                        <span className="flex items-center gap-1">
                          <span className={`w-2 h-2 rounded-full ${CRITERION_COLORS.evidence.dot}`}></span>
                          <span className="text-gray-700">EV:{safeRender(entry.evidence)}</span>
                        </span>
                      </>
                    )}
                    <span className="ml-auto text-gray-600">
                      {isHolistic ? '🌟 Holistic' : '📊 Analytic'}
                    </span>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {/* ============ Right: details ============ */}
      <div className="glass-card rounded-2xl shadow-xl p-6 bg-white/40 backdrop-blur-md">
        <h2 className="text-2xl font-bold text-indigo-700 mb-4">📋 Evaluation Details</h2>

        {selectedEntry ? (
          <div className="space-y-5 max-h-[600px] overflow-y-auto pr-2">
            <div className="border-b border-gray-200 pb-3">
              <h3 className="text-xl font-semibold text-gray-800">
                {selectedEntry.title || `Essay #${selectedEntry.id}`}
              </h3>
              <p className="text-sm text-gray-600 mt-1">
                Evaluated on {new Date(selectedEntry.created_at).toLocaleString()}
              </p>
            </div>

            {selectedEntry.eval_type === 'holistic' ? (
              <div className="bg-gray-50 p-4 rounded-xl text-center">
                <h4 className="font-semibold text-indigo-600 mb-3">Holistic Score</h4>
                <div className={`text-6xl font-bold ${colorFrom5Scale(selectedEntry.holistic_score)}`}>
                  {typeof selectedEntry.holistic_score === 'number'
                    ? `${selectedEntry.holistic_score}/5`
                    : '—'}
                </div>
                <div className="mt-2 p-3 bg-white rounded-lg text-gray-700 border border-gray-200">
                  {selectedEntry.level_description || 'No description available.'}
                </div>
              </div>
            ) : (
              <div className="bg-gray-50 p-4 rounded-xl">
                <h4 className="font-semibold text-indigo-600 mb-3">Scores</h4>
                <div className="space-y-3">
                  {[
                    { key: 'main_statement', label: 'Main Statement' },
                    { key: 'organization',   label: 'Organization' },
                    { key: 'evidence',       label: 'Evidence' },
                    { key: 'analysis',       label: 'Analysis' },
                    { key: 'grammar',        label: 'Grammar' },
                  ].map(({ key, label }) => {
                    const value = selectedEntry[key];
                    const color = CRITERION_COLORS[key];
                    const percent = typeof value === 'number' ? (value / 4) * 100 : 0;
                    return (
                      <div key={key}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className={`font-medium ${color.text}`}>{label}</span>
                          <span className={`font-bold ${colorFrom4Scale(value)}`}>
                            {typeof value === 'number' ? `${value}/4` : '—'}
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className={`${color.bar} h-2 rounded-full transition-all`}
                            style={{ width: `${percent}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Overall average */}
                {(() => {
                  const avg = computeAverage(selectedEntry);
                  return (
                    <div className="mt-4 pt-3 border-t border-gray-200 flex justify-between items-center">
                      <span className="font-medium text-gray-800">Overall Average</span>
                      <span className={`text-xl font-bold ${colorFrom4Scale(avg)}`}>
                        {avg !== null ? `${avg.toFixed(2)}/4` : '—'}
                      </span>
                    </div>
                  );
                })()}
              </div>
            )}

            <div>
              <h4 className="font-semibold text-gray-800 mb-2 flex items-center gap-2">
                <span>🤖 AI Feedback</span>
                {selectedEntry.accepted && (
                  <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">
                    Accepted
                  </span>
                )}
              </h4>
              <div className="bg-gray-50 p-4 rounded-lg text-sm leading-relaxed text-gray-800">
                {showFullFeedback
                  ? formatText(selectedEntry.feedback)
                  : formatText((selectedEntry.feedback || '').substring(0, 600))}
                {(selectedEntry.feedback || '').length > 600 && (
                  <button
                    onClick={() => setShowFullFeedback(!showFullFeedback)}
                    className="text-blue-600 ml-2 text-xs font-medium hover:underline"
                  >
                    {showFullFeedback ? 'Show less' : 'Show more'}
                  </button>
                )}
              </div>
            </div>

            {selectedEntry.teacher_feedback && (
              <div>
                <h4 className="font-semibold text-gray-800 mb-2">👩‍🏫 Teacher Override</h4>
                <div className="bg-yellow-50 p-4 rounded-lg text-sm border-l-4 border-yellow-400 text-gray-800">
                  {formatText(selectedEntry.teacher_feedback)}
                </div>
              </div>
            )}

            <div>
              <h4 className="font-semibold text-gray-800 mb-2">📄 Essay Text</h4>
              <div className="bg-gray-50 p-4 rounded-lg text-sm whitespace-pre-wrap font-mono max-h-60 overflow-y-auto text-gray-800">
                {showFullEssay
                  ? selectedEntry.essay
                  : (selectedEntry.essay || '').substring(0, 500)}
                {(selectedEntry.essay || '').length > 500 && (
                  <button
                    onClick={() => setShowFullEssay(!showFullEssay)}
                    className="text-blue-600 ml-2 text-xs font-medium hover:underline"
                  >
                    {showFullEssay ? 'Show less' : 'Show more'}
                  </button>
                )}
              </div>
            </div>

            <div className="flex flex-wrap gap-2 pt-2">
              <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded-full">
                {selectedEntry.eval_type === 'holistic' ? '🌟 Holistic' : '📊 Analytic'}
              </span>
              <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded-full">
                Satisfaction: {selectedEntry.satisfaction ?? '—'}/10
              </span>
            </div>
          </div>
        ) : (
          <div className="text-center py-16">
            <div className="text-5xl mb-3">📋</div>
            <p className="text-gray-700">Select an entry from the left to view details.</p>
          </div>
        )}

        {/* Rubric */}
        <div className="mt-6 pt-4 border-t border-gray-200">
          <details className="cursor-pointer group">
            <summary className="font-semibold text-gray-800 flex items-center gap-2">
              <span>📖 Scoring Rubric</span>
              <span className="text-xs text-gray-600 group-open:hidden">click to expand</span>
            </summary>

            {rubric && typeof rubric === 'object' && (
              <div className="mt-3 space-y-3 pl-2">
                {Object.entries(rubric).map(([criterion, levels]) => (
                  <div key={criterion} className="bg-gray-50 p-3 rounded-lg">
                    <div
                      className={`font-semibold capitalize mb-2 ${
                        CRITERION_COLORS[criterion]?.text || 'text-indigo-700'
                      }`}
                    >
                      {criterion.replace(/_/g, ' ')}
                    </div>
                    {levels && typeof levels === 'object' ? (
                      <ul className="space-y-1">
                        {[4, 3, 2, 1].map(level => (
                          <li key={level} className="text-sm text-gray-700">
                            <span className="font-bold text-gray-800">{level}:</span>{' '}
                            {safeRender(levels[level])}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-sm text-gray-700">{safeRender(levels)}</p>
                    )}
                  </div>
                ))}
              </div>
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
                  <p className="text-sm text-gray-700 mt-1">{safeRender(suggestions.what)}</p>
                </div>
                <div className="bg-indigo-50 p-3 rounded-lg">
                  <strong className="text-indigo-700">When to use</strong>
                  <p className="text-sm text-gray-700 mt-1 whitespace-pre-wrap">
                    {safeRender(suggestions.when)}
                  </p>
                </div>
                <div className="bg-indigo-50 p-3 rounded-lg">
                  <strong className="text-indigo-700">Best results</strong>
                  <p className="text-sm text-gray-700 mt-1 whitespace-pre-wrap">
                    {safeRender(suggestions.how)}
                  </p>
                </div>
              </div>
            </details>
          )}
        </div>
      </div>
    </div>
  );
}
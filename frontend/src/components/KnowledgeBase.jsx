import React, { useState, useEffect, useMemo } from 'react';
import { listKnowledge, listLearningFeedback, getRubric, getSuggestions } from '../api';
import { useAuth } from '../context/AuthContext';
import LoadingIndicator from './LoadingIndicator';
import EntryFilters, { RichText } from './EntryFilters';
import toast from 'react-hot-toast';
import {
  DEFAULT_CRITERIA,
  getCriteria,
  averageFraction,
  fractionOf,
  colorForFraction,
  errorMessage,
  entryScores,
  entryTitle,
  entryAiLabel,
  entryRubricTitle,
  essayKey,
  matchesSearch,
  countByAi,
  sortEntries,
  isNum,
} from '../lib/scoring';

// ---------- Safe value renderer ----------
function safeRender(value) {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

// Scores and criteria for a saved entry. Newer entries store the full scores
// (any rubric); older ones only have the five built-in 1–4 columns.
function entryScoreView(entry) {
  const full = entryScores(entry);
  if (full && getCriteria(full).length) {
    return { values: full, criteria: getCriteria(full) };
  }
  return { values: entry, criteria: DEFAULT_CRITERIA };
}

function getDisplayScore(entry) {
  if (entry.eval_type === 'holistic') {
    const hs = entry.holistic_score;
    return { text: isNum(hs) ? `${hs}/5` : '—', fraction: fractionOf(hs, 5) };
  }
  const { values, criteria } = entryScoreView(entry);
  const f = averageFraction(values, criteria);
  return { text: f === null ? '—' : `${Math.round(f * 100)}%`, fraction: f };
}

const BAR_COLORS = ['bg-blue-500', 'bg-emerald-500', 'bg-purple-500', 'bg-amber-500', 'bg-pink-500', 'bg-cyan-500', 'bg-orange-500'];
const TEXT_COLORS = ['text-blue-700', 'text-emerald-700', 'text-purple-700', 'text-amber-700', 'text-pink-700', 'text-cyan-700', 'text-orange-700'];

// ---------- Filter helpers for rich fields ----------
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

export default function KnowledgeBase() {
  const { user } = useAuth();
  const [rubric, setRubric] = useState(null);
  const [suggestions, setSuggestions] = useState(null);
  const [entries, setEntries] = useState([]);
  // essayKey → teacher overrides of that essay (from the Learning Hub)
  const [overridesByEssay, setOverridesByEssay] = useState(new Map());
  const [selectedEntry, setSelectedEntry] = useState(null);
  const [showFullEssay, setShowFullEssay] = useState(false);
  const [showFullFeedback, setShowFullFeedback] = useState(false);
  const [loading, setLoading] = useState(true);

  const [search, setSearch] = useState('');
  const [aiFilter, setAiFilter] = useState('all');
  const [sort, setSort] = useState('newest');
  const [status, setStatus] = useState('all');

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
      const [rubricData, suggestionsData, entriesData, overrides] = await Promise.all([
        getRubric(),
        getSuggestions(),
        listKnowledge(),
        listLearningFeedback().catch(() => []),
      ]);
      setRubric(rubricData);
      setSuggestions(suggestionsData);
      setEntries(entriesData);
      const byEssay = new Map();
      for (const o of overrides) {
        const key = essayKey(o.original_essay);
        byEssay.set(key, [...(byEssay.get(key) || []), o]);
      }
      setOverridesByEssay(byEssay);
    } catch (err) {
      toast.error('Failed to load data: ' + errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const overridesOf = (entry) => overridesByEssay.get(essayKey(entry.essay)) || [];

  const aiCounts = useMemo(() => countByAi(entries, entryAiLabel), [entries]);

  const visible = useMemo(() => {
    const filtered = entries.filter((entry) => {
      const overridden = (overridesByEssay.get(essayKey(entry.essay)) || []).length > 0;
      return (
        (aiFilter === 'all' || entryAiLabel(entry) === aiFilter) &&
        (status === 'all' || (status === 'overridden' && overridden) || (status === 'saved-only' && !overridden)) &&
        matchesSearch(search, entryTitle(entry), entry.essay, entry.feedback)
      );
    });
    return sortEntries(filtered, sort, entryTitle);
  }, [entries, overridesByEssay, aiFilter, status, search, sort]);

  if (!user) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 text-center bg-white/90 backdrop-blur-md">
        <div className="text-6xl mb-4">🔒</div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Login Required</h2>
        <p className="text-gray-700">Please log in to view your knowledge base.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 bg-white/90 backdrop-blur-md">
        <LoadingIndicator message="Loading your knowledge base…" />
      </div>
    );
  }

  const selectedView = selectedEntry ? entryScoreView(selectedEntry) : null;
  const selectedOverrides = selectedEntry ? overridesOf(selectedEntry) : [];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 animate-fade-in">
      {/* ============ Left: entries list ============ */}
      <div className="glass-card rounded-2xl shadow-xl p-6 bg-white/90 backdrop-blur-md">
        <div className="flex items-center justify-between mb-1">
          <h2 className="text-2xl font-bold text-indigo-700">📚 Knowledge Base</h2>
          <span className="text-sm bg-blue-100 text-blue-800 px-3 py-1 rounded-full">
            {visible.length}/{entries.length} saved
          </span>
        </div>
        <p className="text-sm text-gray-600 mb-4">
          Evaluations you accepted and saved. Overridden essays are also listed in the Learning Hub.
        </p>

        <EntryFilters
          search={search}
          onSearch={setSearch}
          placeholder="Search title, essay, feedback…"
          aiCounts={aiCounts}
          aiFilter={aiFilter}
          onAiFilter={setAiFilter}
          sort={sort}
          onSort={setSort}
          extraFilters={[
            { id: 'all', label: 'All' },
            { id: 'saved-only', label: '💾 Saved only' },
            { id: 'overridden', label: '✏️ Also overridden' },
          ]}
          activeExtra={status}
          onExtra={setStatus}
        />

        {entries.length === 0 ? (
          <div className="text-center py-12">
            <div className="text-5xl mb-3">📭</div>
            <p className="text-gray-700">No saved evaluations yet.</p>
            <p className="text-sm text-gray-600 mt-1">Evaluate an essay and click "Save to KB".</p>
          </div>
        ) : visible.length === 0 ? (
          <p className="text-gray-500 text-sm">No saved evaluations match these filters.</p>
        ) : (
          <ul className="space-y-2 max-h-[560px] overflow-y-auto pr-2">
            {visible.map((entry) => {
              const display = getDisplayScore(entry);
              const isHolistic = entry.eval_type === 'holistic';
              const overridden = overridesOf(entry).length > 0;
              return (
                <li key={entry.id}>
                  <button
                    className={`w-full text-left border rounded-xl p-3 transition-all hover:shadow-md ${
                      selectedEntry?.id === entry.id
                        ? 'bg-blue-50 border-blue-300 shadow-md'
                        : 'bg-white border-gray-200 hover:bg-gray-50'
                    }`}
                    onClick={() => {
                      setSelectedEntry(entry);
                      setShowFullEssay(false);
                      setShowFullFeedback(false);
                    }}
                  >
                    <div className="flex justify-between items-start gap-2">
                      <div className="flex-1 min-w-0">
                        <div className="font-semibold text-gray-800 truncate">{entryTitle(entry)}</div>
                        <div className="text-xs text-gray-600 mt-1">
                          {new Date(entry.created_at).toLocaleDateString(undefined, {
                            month: 'short', day: 'numeric', year: 'numeric',
                          })}
                          {' · '}
                          {isHolistic ? '🌟 Holistic' : '📊 Analytic'}
                        </div>
                      </div>
                      <div className={`text-lg font-bold ${colorForFraction(display.fraction)}`}>
                        {display.text}
                      </div>
                    </div>
                    <div className="text-xs text-gray-600 mt-2 line-clamp-2">
                      {(entry.essay || '').substring(0, 120)}…
                    </div>
                    <div className="flex flex-wrap gap-1.5 mt-2 text-xs">
                      <span className="bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded-full">🤖 {entryAiLabel(entry)}</span>
                      <span className="bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded-full">💾 Saved</span>
                      {overridden && <span className="bg-amber-50 text-amber-800 px-2 py-0.5 rounded-full">✏️ Overridden</span>}
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {/* ============ Right: details ============ */}
      <div className="glass-card rounded-2xl shadow-xl p-6 bg-white/90 backdrop-blur-md">
        <h2 className="text-2xl font-bold text-indigo-700 mb-4">📋 Evaluation Details</h2>

        {selectedEntry ? (
          <div className="space-y-5 max-h-[680px] overflow-y-auto pr-2">
            <div className="border-b border-gray-200 pb-3">
              <h3 className="text-xl font-semibold text-gray-800">{entryTitle(selectedEntry)}</h3>
              <div className="flex flex-wrap gap-1.5 mt-2 text-xs">
                <span className="bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded-full">🤖 Scored by {entryAiLabel(selectedEntry)}</span>
                {entryRubricTitle(selectedEntry) && (
                  <span className="bg-gray-100 text-gray-700 px-2 py-0.5 rounded-full">📐 {entryRubricTitle(selectedEntry)}</span>
                )}
                <span className="bg-gray-100 text-gray-700 px-2 py-0.5 rounded-full">
                  🕒 {new Date(selectedEntry.created_at).toLocaleString()}
                </span>
                <span className="bg-gray-100 text-gray-700 px-2 py-0.5 rounded-full">
                  Satisfaction {selectedEntry.satisfaction ?? '—'}/10
                </span>
              </div>
            </div>

            {selectedEntry.eval_type === 'holistic' ? (
              <div className="bg-gray-50 p-4 rounded-xl text-center">
                <h4 className="font-semibold text-indigo-600 mb-3">Holistic Score</h4>
                <div className={`text-6xl font-bold ${colorForFraction(fractionOf(selectedEntry.holistic_score, 5))}`}>
                  {isNum(selectedEntry.holistic_score) ? `${selectedEntry.holistic_score}/5` : '—'}
                </div>
                <div className="mt-2 p-3 bg-white rounded-lg text-gray-700 border border-gray-200">
                  {selectedEntry.level_description || 'No description available.'}
                </div>
              </div>
            ) : (
              <div className="bg-gray-50 p-4 rounded-xl">
                <h4 className="font-semibold text-indigo-600 mb-3">Scores</h4>
                <div className="space-y-3">
                  {selectedView.criteria.map(({ key, label, max_score }, idx) => {
                    const value = selectedView.values[key];
                    const percent = isNum(value) ? Math.min(100, (value / max_score) * 100) : 0;
                    const note = selectedEntry.criterion_feedback?.[key];
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
                        {typeof note === 'string' && note && (
                          <p className="text-xs text-gray-700 mt-1 leading-relaxed pl-1">{note}</p>
                        )}
                      </div>
                    );
                  })}
                </div>

                {(() => {
                  const avg = averageFraction(selectedView.values, selectedView.criteria);
                  return (
                    <div className="mt-4 pt-3 border-t border-gray-200 flex justify-between items-center">
                      <span className="font-medium text-gray-800">Overall Average</span>
                      <span className={`text-xl font-bold ${colorForFraction(avg)}`}>
                        {avg !== null ? `${Math.round(avg * 100)}%` : '—'}
                      </span>
                    </div>
                  );
                })()}
              </div>
            )}

            {selectedOverrides.length > 0 && (
              <div>
                <h4 className="font-semibold text-gray-800 mb-2">
                  👩‍🏫 Teacher Remarks ({selectedOverrides.length} override{selectedOverrides.length === 1 ? '' : 's'})
                </h4>
                <div className="space-y-2">
                  {selectedOverrides.map((o) => (
                    <div key={o.id} className="bg-amber-50 p-3 rounded-lg text-sm border-l-4 border-amber-400 text-gray-800">
                      <div className="text-xs text-amber-800 mb-1">{new Date(o.created_at).toLocaleString()}</div>
                      <RichText text={o.teacher_feedback} />
                      {o.suggested_changes && (
                        <p className="mt-2"><span className="font-medium text-amber-900">Suggestions:</span> {o.suggested_changes}</p>
                      )}
                    </div>
                  ))}
                </div>
                <p className="text-xs text-gray-500 mt-1">Score changes are in the 🧠 Learning Hub.</p>
              </div>
            )}

            <div>
              <h4 className="font-semibold text-gray-800 mb-2 flex items-center gap-2">
                <span>🤖 AI Feedback</span>
                {selectedEntry.accepted && (
                  <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">Accepted</span>
                )}
              </h4>
              <div className="bg-gray-50 p-4 rounded-lg text-sm leading-relaxed text-gray-800">
                <RichText
                  text={showFullFeedback ? selectedEntry.feedback : (selectedEntry.feedback || '').substring(0, 600)}
                />
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

            {/* Grammar Issues */}
            {(() => {
              const issues = filterGrammarIssues(selectedEntry.grammar_issues);
              if (issues.length === 0) return null;
              return (
                <div>
                  <h4 className="font-semibold text-gray-800 mb-2">
                    ✏️ Grammar & Mechanics — Specific Fixes ({issues.length})
                  </h4>
                  <div className="space-y-2">
                    {issues.map((issue, idx) => (
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
                          <p className="text-xs text-gray-600 mt-1 pl-5 italic">{issue.explanation}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}

            {/* Recommendations */}
            {(() => {
              const recs = filterRecommendations(selectedEntry.recommendations);
              if (recs.length === 0) return null;
              return (
                <div>
                  <h4 className="font-semibold text-gray-800 mb-2">💡 Recommendations ({recs.length})</h4>
                  <div className="space-y-2">
                    {recs.map((rec, idx) => (
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
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}

            <div>
              <h4 className="font-semibold text-gray-800 mb-2">📄 Essay Text</h4>
              <div className="bg-gray-50 p-4 rounded-lg text-sm whitespace-pre-wrap font-mono max-h-60 overflow-y-auto text-gray-800">
                {showFullEssay ? selectedEntry.essay : (selectedEntry.essay || '').substring(0, 500)}
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
              <span>📖 Built-in Scoring Rubric</span>
              <span className="text-xs text-gray-700 group-open:hidden">click to expand</span>
            </summary>

            {rubric && typeof rubric === 'object' && (
              <div className="mt-3 space-y-3 pl-2">
                {Object.entries(rubric).map(([criterion, levels]) => (
                  <div key={criterion} className="bg-gray-50 p-3 rounded-lg">
                    <div className="font-semibold capitalize mb-2 text-indigo-700">
                      {criterion.replace(/_/g, ' ')}
                    </div>
                    {levels && typeof levels === 'object' ? (
                      <ul className="space-y-1">
                        {[4, 3, 2, 1].map((level) => (
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
                <span className="text-xs text-gray-700 group-open:hidden">click to expand</span>
              </summary>
              <div className="mt-3 space-y-3 pl-2">
                <div className="bg-indigo-50 p-3 rounded-lg">
                  <strong className="text-indigo-700">What it does</strong>
                  <p className="text-sm text-gray-700 mt-1">{safeRender(suggestions.what)}</p>
                </div>
                <div className="bg-indigo-50 p-3 rounded-lg">
                  <strong className="text-indigo-700">When to use</strong>
                  <p className="text-sm text-gray-700 mt-1 whitespace-pre-wrap">{safeRender(suggestions.when)}</p>
                </div>
                <div className="bg-indigo-50 p-3 rounded-lg">
                  <strong className="text-indigo-700">Best results</strong>
                  <p className="text-sm text-gray-700 mt-1 whitespace-pre-wrap">{safeRender(suggestions.how)}</p>
                </div>
              </div>
            </details>
          )}
        </div>
      </div>
    </div>
  );
}

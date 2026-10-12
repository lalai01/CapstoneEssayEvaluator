import React, { useState, useEffect, useMemo } from 'react';
import { listLearningFeedback, listKnowledge } from '../api';
import { useAuth } from '../context/AuthContext';
import LoadingIndicator from './LoadingIndicator';
import EntryFilters, { RichText } from './EntryFilters';
import toast from 'react-hot-toast';
import {
  getCriteria,
  errorMessage,
  entryTitle,
  entryAiLabel,
  entryRubricTitle,
  essayKey,
  matchesSearch,
  countByAi,
  sortEntries,
  isNum,
  satisfactionStyle,
  entryApprovedBy,
} from '../lib/scoring';

function teacherScoresOf(item) {
  return item.teacher_scores || item.original_scores?.teacher_scores || {};
}

function changedCount(item) {
  const original = item.original_scores || {};
  const teacher = teacherScoresOf(item);
  return Object.entries(teacher).filter(([k, v]) => isNum(v) && isNum(original[k]) && v !== original[k]).length;
}

function ScoreComparison({ item }) {
  const original = item.original_scores || {};
  const teacher = teacherScoresOf(item);
  const rows = original.holistic_score !== undefined
    ? [{ key: 'holistic_score', label: 'Holistic Score', max_score: 5 }]
    : getCriteria(original);

  return (
    <table className="w-full text-sm border rounded-lg overflow-hidden">
      <thead className="bg-gray-50 text-gray-600">
        <tr>
          <th className="text-left p-2 font-medium">Criterion</th>
          <th className="p-2 font-medium">AI</th>
          <th className="p-2 font-medium">Teacher</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(({ key, label, max_score }) => {
          const ai = original[key];
          const t = teacher[key];
          const diff = isNum(ai) && isNum(t) ? t - ai : null;
          return (
            <tr key={key} className="border-t">
              <td className="p-2 text-gray-800">{label}</td>
              <td className="p-2 text-center">{ai ?? '—'}/{max_score}</td>
              <td className="p-2 text-center">
                {t ?? '—'}/{max_score}
                {diff !== null && diff !== 0 && (
                  <span className={`ml-1 text-xs ${diff > 0 ? 'text-emerald-600' : 'text-red-500'}`}>
                    ({diff > 0 ? `+${diff}` : diff})
                  </span>
                )}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default function LearningFeedback() {
  const { user, isTeacher } = useAuth();
  const [overrides, setOverrides] = useState([]);
  const [kbEssays, setKbEssays] = useState(new Set());
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showFullEssay, setShowFullEssay] = useState(false);

  const [search, setSearch] = useState('');
  const [aiFilter, setAiFilter] = useState('all');
  const [sort, setSort] = useState('newest');
  const [status, setStatus] = useState('all');

  useEffect(() => {
    if (user) {
      loadOverrides();
    } else {
      setLoading(false);
    }
  }, [user]);

  const loadOverrides = async () => {
    setLoading(true);
    try {
      const [data, kb] = await Promise.all([
        listLearningFeedback(),
        listKnowledge().catch(() => []),
      ]);
      setOverrides(data);
      setKbEssays(new Set(kb.map((e) => essayKey(e.essay))));
    } catch (err) {
      toast.error('Failed to load the Learning Hub: ' + errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const aiCounts = useMemo(() => countByAi(overrides, entryAiLabel), [overrides]);

  const visible = useMemo(() => {
    const filtered = overrides.filter((item) =>
      (aiFilter === 'all' || entryAiLabel(item) === aiFilter) &&
      (status === 'all' ||
        (status === 'scores' && changedCount(item) > 0) ||
        (status === 'in-kb' && kbEssays.has(essayKey(item.original_essay)))) &&
      matchesSearch(search, entryTitle(item), item.original_essay, item.teacher_feedback, item.suggested_changes)
    );
    return sortEntries(filtered, sort, entryTitle);
  }, [overrides, aiFilter, status, search, sort, kbEssays]);

  if (!user) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 text-center">
        <div className="text-6xl mb-4">🔒</div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Login Required</h2>
        <p className="text-gray-600">Please log in to view the Learning Hub.</p>
      </div>
    );
  }

  if (!isTeacher) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 text-center">
        <div className="text-6xl mb-4">👩‍🏫</div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Teachers only</h2>
        <p className="text-gray-600">The Learning Hub lists the overrides you submitted as a teacher.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 bg-white/90 backdrop-blur-md">
        <LoadingIndicator message="Loading the Learning Hub…" />
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 animate-fade-in">
      <div className="glass-card rounded-2xl shadow-xl p-6">
        <div className="flex items-center justify-between mb-1">
          <h2 className="text-2xl font-bold text-gray-800">🧠 Learning Hub</h2>
          <span className="text-sm bg-amber-100 text-amber-800 px-3 py-1 rounded-full">
            {visible.length}/{overrides.length} overrides
          </span>
        </div>
        <p className="text-sm text-gray-600 mb-4">
          Essays where you overrode the AI. Only you can see your overrides; your remarks feed back into future evaluations.
        </p>

        <EntryFilters
          search={search}
          onSearch={setSearch}
          placeholder="Search title, essay, remarks…"
          aiCounts={aiCounts}
          aiFilter={aiFilter}
          onAiFilter={setAiFilter}
          sort={sort}
          onSort={setSort}
          extraFilters={[
            { id: 'all', label: 'All' },
            { id: 'scores', label: 'Scores changed' },
            { id: 'in-kb', label: 'Also in Knowledge Base' },
          ]}
          activeExtra={status}
          onExtra={setStatus}
        />

        {overrides.length === 0 && (
          <p className="text-gray-500">No overrides yet. On an evaluation, click "Reject & Override" and write your remarks.</p>
        )}
        {overrides.length > 0 && visible.length === 0 && (
          <p className="text-gray-500 text-sm">No overrides match these filters.</p>
        )}

        <ul className="space-y-2 max-h-[560px] overflow-y-auto pr-2">
          {visible.map((item) => {
            const changed = changedCount(item);
            const inKb = kbEssays.has(essayKey(item.original_essay));
            return (
              <li key={item.id}>
                <button
                  onClick={() => { setSelected(item); setShowFullEssay(false); }}
                  className={`w-full text-left border rounded-xl p-3 transition hover:shadow-md ${
                    selected?.id === item.id ? 'bg-amber-50 border-amber-300' : 'bg-white border-gray-200 hover:bg-gray-50'
                  }`}
                >
                  <div className="flex justify-between gap-2">
                    <span className="font-semibold text-gray-800 truncate">{entryTitle(item)}</span>
                    <span className="text-xs text-gray-500 shrink-0">
                      {new Date(item.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}
                    </span>
                  </div>
                  {item.teacher_feedback && (
                    <p className="text-xs text-gray-600 mt-1 line-clamp-2">“{item.teacher_feedback}”</p>
                  )}
                  <div className="flex flex-wrap gap-1.5 mt-2 text-xs">
                    <span className="bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded-full">🤖 {entryAiLabel(item)}</span>
                    <span className="bg-amber-50 text-amber-800 px-2 py-0.5 rounded-full">✏️ Overridden</span>
                    {changed > 0 && (
                      <span className="bg-red-50 text-red-700 px-2 py-0.5 rounded-full">
                        {changed} score{changed === 1 ? '' : 's'} changed
                      </span>
                    )}
                    {inKb && <span className="bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded-full">💾 In Knowledge Base</span>}
                    {isNum(item.satisfaction) && (
                      <span className={`px-2 py-0.5 rounded-full border ${satisfactionStyle(item.satisfaction).chip}`}>
                        {item.satisfaction}/10
                      </span>
                    )}
                  </div>
                </button>
              </li>
            );
          })}
        </ul>
      </div>

      <div className="glass-card rounded-2xl shadow-xl p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">📝 Override Details</h2>
        {selected ? (
          <div className="space-y-5 max-h-[680px] overflow-y-auto pr-2">
            <div className="border-b border-gray-200 pb-3">
              <h3 className="text-xl font-semibold text-gray-800">{entryTitle(selected)}</h3>
              <div className="flex flex-wrap gap-1.5 mt-2 text-xs">
                <span className="bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded-full">🤖 Scored by {entryAiLabel(selected)}</span>
                {entryRubricTitle(selected) && (
                  <span className="bg-gray-100 text-gray-700 px-2 py-0.5 rounded-full">📐 {entryRubricTitle(selected)}</span>
                )}
                <span className="bg-gray-100 text-gray-700 px-2 py-0.5 rounded-full">
                  🕒 {new Date(selected.created_at).toLocaleString()}
                </span>
                {selected.satisfaction != null && (
                  <span className={`px-2 py-0.5 rounded-full border ${satisfactionStyle(selected.satisfaction).chip}`}>
                    Satisfaction {selected.satisfaction}/10 · {satisfactionStyle(selected.satisfaction).label}
                  </span>
                )}
                {entryApprovedBy(selected) && (
                  <span className="bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded-full">👩‍🏫 Approved by {entryApprovedBy(selected)}</span>
                )}
              </div>
            </div>

            <div>
              <h4 className="font-semibold text-gray-800 mb-2">Scores: AI vs teacher</h4>
              <ScoreComparison item={selected} />
            </div>

            <div>
              <h4 className="font-semibold text-gray-800 mb-2">👩‍🏫 Teacher's Remarks</h4>
              <div className="bg-amber-50 p-4 rounded-lg text-sm border-l-4 border-amber-400 text-gray-800">
                <RichText text={selected.teacher_feedback || '—'} />
              </div>
            </div>

            {selected.suggested_changes && (
              <div>
                <h4 className="font-semibold text-gray-800 mb-2">💡 Suggestions for the Student</h4>
                <div className="bg-blue-50 p-4 rounded-lg text-sm border-l-4 border-blue-400 text-gray-800">
                  <RichText text={selected.suggested_changes} />
                </div>
              </div>
            )}

            <div>
              <h4 className="font-semibold text-gray-800 mb-2">📄 Essay</h4>
              <div className="bg-gray-50 p-4 rounded-lg text-sm whitespace-pre-wrap max-h-60 overflow-y-auto text-gray-800">
                {showFullEssay ? selected.original_essay : (selected.original_essay || '').slice(0, 500)}
                {(selected.original_essay || '').length > 500 && (
                  <button onClick={() => setShowFullEssay(!showFullEssay)} className="text-blue-600 ml-2 text-xs font-medium hover:underline">
                    {showFullEssay ? 'Show less' : 'Show more'}
                  </button>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="text-gray-500">Select an override on the left to see the teacher's remarks and score changes.</div>
        )}
      </div>
    </div>
  );
}

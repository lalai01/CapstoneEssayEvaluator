import React, { useState, useEffect } from 'react';
import { listKnowledge, getRubric, getSuggestions } from '../api';
import toast from 'react-hot-toast';

export default function KnowledgeBase() {
  const [rubric, setRubric] = useState(null);
  const [suggestions, setSuggestions] = useState(null);
  const [entries, setEntries] = useState([]);
  const [selectedEntry, setSelectedEntry] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
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
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <div className="glass-card rounded-2xl shadow-xl p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">📚 Past Evaluations</h2>
        {entries.length === 0 && <p className="text-gray-500">No entries yet. Evaluate an essay and save it.</p>}
        <ul className="space-y-3 max-h-[600px] overflow-y-auto pr-2">
          {entries.map(entry => (
            <li key={entry.id} className="border-b border-gray-100 pb-3 cursor-pointer hover:bg-gray-50 rounded-lg p-2 transition" onClick={() => setSelectedEntry(entry)}>
              <div className="font-medium text-gray-800">ID {entry.id} – Score {Math.round((entry.grammar+entry.coherence+entry.content)/3)}/100</div>
              <div className="text-xs text-gray-500">{new Date(entry.created_at).toLocaleString()}</div>
            </li>
          ))}
        </ul>
      </div>
      <div className="glass-card rounded-2xl shadow-xl p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">📋 Details & Guide</h2>
        {selectedEntry ? (
          <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
            <div><strong>Scores:</strong> Grammar {selectedEntry.grammar} / Coherence {selectedEntry.coherence} / Content {selectedEntry.content}</div>
            <div><strong>Accepted:</strong> {selectedEntry.accepted ? 'Yes' : 'No'}</div>
            <div><strong>Satisfaction:</strong> {selectedEntry.satisfaction}/10</div>
            <div><strong>Feedback:</strong><br/>{selectedEntry.feedback}</div>
            {selectedEntry.teacher_feedback && <div><strong>Teacher Override:</strong><br/>{selectedEntry.teacher_feedback}</div>}
            <div><strong>Essay preview:</strong><br/>{selectedEntry.essay.substring(0, 300)}...</div>
          </div>
        ) : (
          <div className="text-gray-500">Select an entry from the left to view details.</div>
        )}
        <div className="mt-6 pt-4 border-t">
          <h3 className="font-semibold text-lg">📖 Rubric</h3>
          {rubric && (
            <dl className="mt-2 space-y-2">
              <div><dt className="font-medium">Grammar</dt><dd className="text-gray-600">{rubric.grammar}</dd></div>
              <div><dt className="font-medium">Coherence</dt><dd className="text-gray-600">{rubric.coherence}</dd></div>
              <div><dt className="font-medium">Content</dt><dd className="text-gray-600">{rubric.content}</dd></div>
            </dl>
          )}
        </div>
      </div>
    </div>
  );
}
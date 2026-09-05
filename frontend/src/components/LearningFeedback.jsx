import React, { useState, useEffect } from 'react';
import { listLearningFeedback } from '../api';
import { useAuth } from '../context/AuthContext';
import toast from 'react-hot-toast';

export default function LearningFeedback() {
  const { user } = useAuth();
  const [overrides, setOverrides] = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);

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
      const data = await listLearningFeedback();
      setOverrides(data);
    } catch (err) {
      toast.error('Failed to load learning feedback: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Show login prompt if not authenticated
  if (!user) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 text-center">
        <div className="text-6xl mb-4">🔒</div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Login Required</h2>
        <p className="text-gray-600">Please log in to view the learning feedback knowledge base.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 text-center">
        <div className="animate-spin text-4xl mb-4">⏳</div>
        <p className="text-gray-600">Loading learning feedback...</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 animate-fade-in">
      <div className="glass-card rounded-2xl shadow-xl p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">🧠 Teacher Overrides (Learning KB)</h2>
        {overrides.length === 0 && <p className="text-gray-500">No overrides yet. Reject an evaluation and provide feedback to build this knowledge base.</p>}
        <ul className="space-y-3 max-h-[600px] overflow-y-auto pr-2">
          {overrides.map(item => (
            <li key={item.id} className="border-b border-gray-100 pb-3 cursor-pointer hover:bg-gray-50 rounded-lg p-2 transition" onClick={() => setSelected(item)}>
              <div className="font-medium text-gray-800">Override #{item.id}</div>
              <div className="text-xs text-gray-500">{new Date(item.created_at).toLocaleString()}</div>
            </li>
          ))}
        </ul>
      </div>
      <div className="glass-card rounded-2xl shadow-xl p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">📝 Override Details</h2>
        {selected ? (
          <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
            <div><strong>Original Scores:</strong> Grammar {selected.original_scores?.grammar} / Coherence {selected.original_scores?.coherence} / Content {selected.original_scores?.content}</div>
            <div><strong>Teacher Feedback:</strong><br/>{selected.teacher_feedback}</div>
            {selected.suggested_changes && <div><strong>Suggested Changes:</strong><br/>{selected.suggested_changes}</div>}
            <div><strong>Accepted?</strong> {selected.accepted ? 'Yes' : 'No'}</div>
            <div><strong>Original Essay Preview:</strong><br/>{selected.original_essay?.substring(0, 400)}...</div>
          </div>
        ) : (
          <div className="text-gray-500">Select an override from the left to see details.</div>
        )}
      </div>
    </div>
  );
}
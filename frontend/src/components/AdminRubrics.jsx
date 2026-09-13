import React, { useEffect, useState } from 'react';
import toast from 'react-hot-toast';
import { apiClient } from '../api';
import { useAuth } from '../context/AuthContext';

const EMPTY_CRITERIA = {
  main_statement: { 4: '', 3: '', 2: '', 1: '' },
  organization:   { 4: '', 3: '', 2: '', 1: '' },
  evidence:       { 4: '', 3: '', 2: '', 1: '' },
  analysis:       { 4: '', 3: '', 2: '', 1: '' },
  grammar:        { 4: '', 3: '', 2: '', 1: '' },
};

export default function AdminRubrics() {
  const { user } = useAuth();
  const [rubrics, setRubrics] = useState([]);
  const [editing, setEditing] = useState(null);
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [criteria, setCriteria] = useState(EMPTY_CRITERIA);
  const [loading, setLoading] = useState(false);

  const isAdmin = user?.role === 'admin' || user?.email === 'admin_essay_capstone@gmail.com';

  useEffect(() => {
    if (isAdmin) loadRubrics();
  }, [isAdmin]);

  const loadRubrics = async () => {
    try {
      const { data } = await apiClient.get('/rubrics');
      setRubrics(data);
    } catch (err) {
      toast.error('Failed to load rubrics');
    }
  };

  const startNew = () => {
    setEditing('new');
    setTitle('');
    setDescription('');
    setCriteria(JSON.parse(JSON.stringify(EMPTY_CRITERIA)));
  };

  const startEdit = (r) => {
    setEditing(r.id);
    setTitle(r.title);
    setDescription(r.description || '');
    setCriteria(r.criteria);
  };

  const save = async () => {
    if (!title.trim()) {
      toast.error('Title is required');
      return;
    }
    setLoading(true);
    try {
      const payload = { title, description, criteria, is_active: true };
      if (editing === 'new') {
        await apiClient.post('/rubrics', payload);
      } else {
        await apiClient.put(`/rubrics/${editing}`, payload);
      }
      toast.success('Rubric saved');
      setEditing(null);
      loadRubrics();
    } catch (err) {
      toast.error('Failed to save rubric');
    } finally {
      setLoading(false);
    }
  };

  const remove = async (id) => {
    if (!confirm('Delete this rubric?')) return;
    try {
      await apiClient.delete(`/rubrics/${id}`);
      toast.success('Deleted');
      loadRubrics();
    } catch (err) {
      toast.error('Failed to delete');
    }
  };

  const updateLevel = (criterion, level, value) => {
    setCriteria(prev => ({
      ...prev,
      [criterion]: { ...prev[criterion], [level]: value },
    }));
  };

  if (!isAdmin) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 text-center">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Access Denied</h2>
        <p className="text-gray-600">Administrator access required.</p>
      </div>
    );
  }

  return (
    <div className="glass-card rounded-2xl shadow-xl p-6 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-indigo-700">📐 Rubric Manager</h2>
        <button
          onClick={startNew}
          className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition"
        >
          + New Rubric
        </button>
      </div>

      {!editing && (
        <div className="space-y-3">
          {rubrics.length === 0 && (
            <p className="text-gray-500">No rubrics yet. Click "+ New Rubric" to create one.</p>
          )}
          {rubrics.map(r => (
            <div key={r.id} className="border p-4 rounded-xl flex justify-between items-center bg-white hover:bg-gray-50 transition">
              <div>
                <div className="font-semibold text-gray-800">{r.title}</div>
                <p className="text-sm text-gray-500">{r.description}</p>
              </div>
              <div className="flex gap-3">
                <button onClick={() => startEdit(r)} className="text-blue-600 text-sm hover:underline">Edit</button>
                <button onClick={() => remove(r.id)} className="text-red-600 text-sm hover:underline">Delete</button>
              </div>
            </div>
          ))}
        </div>
      )}

      {editing && (
        <div className="space-y-4">
          <div>
            <label className="block font-medium mb-1">Title</label>
            <input
              value={title}
              onChange={e => setTitle(e.target.value)}
              className="w-full border rounded-lg p-2"
              placeholder="e.g., Argumentative Essay Rubric"
            />
          </div>
          <div>
            <label className="block font-medium mb-1">Description (optional)</label>
            <textarea
              value={description}
              onChange={e => setDescription(e.target.value)}
              className="w-full border rounded-lg p-2"
              rows="2"
              placeholder="Brief description of the rubric purpose"
            />
          </div>

          {Object.entries(criteria).map(([criterion, levels]) => (
            <div key={criterion} className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-semibold capitalize mb-3">
                {criterion.replace('_', ' ')}
              </h4>
              {[4, 3, 2, 1].map(level => (
                <div key={level} className="mb-2">
                  <label className="block text-xs text-gray-500 mb-1">Level {level}</label>
                  <textarea
                    value={levels[level] || ''}
                    onChange={e => updateLevel(criterion, level, e.target.value)}
                    className="w-full border rounded-lg p-2 text-sm"
                    rows="2"
                    placeholder={`Descriptor for level ${level}`}
                  />
                </div>
              ))}
            </div>
          ))}

          <div className="flex gap-2">
            <button
              onClick={save}
              disabled={loading}
              className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 disabled:opacity-50"
            >
              {loading ? 'Saving...' : 'Save Rubric'}
            </button>
            <button
              onClick={() => setEditing(null)}
              className="bg-gray-300 px-4 py-2 rounded-lg hover:bg-gray-400"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
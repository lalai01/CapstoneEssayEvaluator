import React, { useEffect, useState } from 'react';
import toast from 'react-hot-toast';
import { apiClient, getRubric } from '../api';
import { useAuth } from '../context/AuthContext';
import { DEFAULT_CRITERIA, labelFromKey, errorMessage } from '../lib/scoring';

const MIN_SCORE = 2;
const MAX_SCORE = 10;
const DEFAULT_WEIGHTS = Object.fromEntries(DEFAULT_CRITERIA.map((c) => [c.key, c.weight]));

function slugify(text) {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

// Stored criteria (legacy {key: {"4": "..."}} or configurable
// {key: {label, max_score, weight, levels}}) → editable rows.
function toRows(criteria) {
  return Object.entries(criteria || {}).map(([key, value]) => {
    const configurable = value && (value.levels !== undefined || value.max_score !== undefined);
    const levels = { ...(configurable ? value.levels : value) };
    const levelNums = Object.keys(levels).map(Number).filter((n) => !Number.isNaN(n));
    return {
      id: `${key}-${Math.random().toString(36).slice(2)}`,
      key,
      keyTouched: true,
      label: (configurable && value.label) || DEFAULT_CRITERIA.find((c) => c.key === key)?.label || labelFromKey(key),
      max_score: Number((configurable && value.max_score) || Math.max(4, ...levelNums)),
      weight: Number(configurable && value.weight !== undefined ? value.weight : DEFAULT_WEIGHTS[key] ?? 1),
      levels,
    };
  });
}

function toCriteria(rows) {
  const criteria = {};
  for (const r of rows) {
    const levels = {};
    for (let lvl = 1; lvl <= r.max_score; lvl++) {
      const text = (r.levels[lvl] || '').trim();
      if (text) levels[lvl] = text;
    }
    criteria[r.key] = {
      label: r.label.trim(),
      max_score: r.max_score,
      weight: Number(r.weight) || 0,
      levels,
    };
  }
  return criteria;
}

function newRow() {
  return {
    id: `new-${Math.random().toString(36).slice(2)}`,
    key: '',
    keyTouched: false,
    label: '',
    max_score: 4,
    weight: 1,
    levels: {},
  };
}

function RubricCard({ rubric, onActivate, onEdit, onDelete, onCopy }) {
  const summary = toRows(rubric.criteria);
  const activeLabel = rubric.is_system ? 'Active school-wide' : 'My default';
  return (
    <div className="border p-4 rounded-xl flex flex-col sm:flex-row sm:justify-between sm:items-start gap-3 bg-white hover:bg-gray-50 transition">
      <div className="min-w-0">
        <div className="font-semibold text-gray-800 flex flex-wrap items-center gap-2">
          {rubric.title}
          {rubric.is_active && (
            <span className="text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded-full">{activeLabel}</span>
          )}
        </div>
        {rubric.description && <p className="text-sm text-gray-500">{rubric.description}</p>}
        <div className="flex flex-wrap gap-1.5 mt-2">
          {summary.map((c) => (
            <span key={c.id} className="text-xs bg-indigo-50 text-indigo-700 border border-indigo-100 px-2 py-0.5 rounded-full">
              {c.label} · 1–{c.max_score}{c.weight !== 1 ? ` · ${c.weight}×` : ''}
            </span>
          ))}
        </div>
      </div>
      <div className="flex flex-wrap gap-3 shrink-0">
        {rubric.can_edit ? (
          <>
            {!rubric.is_active && (
              <button onClick={() => onActivate(rubric)} className="text-emerald-700 text-sm hover:underline">
                {rubric.is_system ? 'Set active' : 'Make default'}
              </button>
            )}
            <button onClick={() => onEdit(rubric)} className="text-blue-600 text-sm hover:underline">Edit</button>
            <button onClick={() => onDelete(rubric)} className="text-red-600 text-sm hover:underline">Delete</button>
          </>
        ) : (
          <button onClick={() => onCopy(rubric)} className="text-indigo-700 text-sm hover:underline">
            Copy to my rubrics
          </button>
        )}
      </div>
    </div>
  );
}

export default function RubricManager() {
  const { user, isAdmin } = useAuth();
  const [rubrics, setRubrics] = useState([]);
  const [listLoading, setListLoading] = useState(true);
  const [editing, setEditing] = useState(null);       // null | 'new' | rubric id
  const [isSystem, setIsSystem] = useState(false);     // scope of the rubric being edited
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [rows, setRows] = useState([]);
  const [openLevels, setOpenLevels] = useState({});
  const [loading, setLoading] = useState(false);
  const [defaultCriteria, setDefaultCriteria] = useState(null);

  useEffect(() => {
    if (!user) return;
    loadRubrics();
    getRubric().then(setDefaultCriteria).catch(() => setDefaultCriteria(null));
  }, [user]);

  const loadRubrics = async () => {
    setListLoading(true);
    try {
      const { data } = await apiClient.get('/rubrics');
      setRubrics(data);
    } catch (err) {
      toast.error('Failed to load rubrics: ' + errorMessage(err));
    } finally {
      setListLoading(false);
    }
  };

  const openEditor = ({ id, system, rubricTitle, rubricDescription, editorRows }) => {
    setEditing(id);
    setIsSystem(system);
    setTitle(rubricTitle);
    setDescription(rubricDescription);
    setRows(editorRows);
    setOpenLevels({});
  };

  const startNew = (system = false) => {
    openEditor({
      id: 'new',
      system,
      rubricTitle: '',
      rubricDescription: '',
      editorRows: defaultCriteria ? toRows(defaultCriteria) : DEFAULT_CRITERIA.map((c) => ({
        ...newRow(), key: c.key, keyTouched: true, label: c.label, weight: c.weight,
      })),
    });
  };

  const startEdit = (r) => {
    openEditor({
      id: r.id,
      system: r.is_system,
      rubricTitle: r.title,
      rubricDescription: r.description || '',
      editorRows: toRows(r.criteria),
    });
  };

  const startCopy = (r) => {
    openEditor({
      id: 'new',
      system: false,
      rubricTitle: `${r.title} (my copy)`,
      rubricDescription: r.description || '',
      editorRows: toRows(r.criteria),
    });
  };

  const updateRow = (id, patch) => {
    setRows((prev) => prev.map((r) => {
      if (r.id !== id) return r;
      const next = { ...r, ...patch };
      // Auto-derive the key from the label until the teacher edits it directly.
      if (patch.label !== undefined && !r.keyTouched) next.key = slugify(patch.label);
      return next;
    }));
  };

  const updateLevel = (id, level, value) => {
    setRows((prev) => prev.map((r) => (
      r.id === id ? { ...r, levels: { ...r.levels, [level]: value } } : r
    )));
  };

  const moveRow = (index, delta) => {
    setRows((prev) => {
      const next = [...prev];
      const target = index + delta;
      if (target < 0 || target >= next.length) return prev;
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
  };

  const validate = () => {
    if (!title.trim()) return 'Title is required.';
    if (rows.length === 0) return 'Add at least one criterion.';
    const seen = new Set();
    for (const r of rows) {
      if (!r.label.trim()) return 'Every criterion needs a name.';
      if (!/^[a-z0-9_]+$/.test(r.key)) return `"${r.label}" needs a key of lowercase letters, numbers and underscores.`;
      if (seen.has(r.key)) return `Duplicate key "${r.key}".`;
      seen.add(r.key);
      if (!(r.max_score >= MIN_SCORE && r.max_score <= MAX_SCORE)) return `"${r.label}" max score must be ${MIN_SCORE}–${MAX_SCORE}.`;
      if (!(r.weight >= 0)) return `"${r.label}" weight must be 0 or more.`;
    }
    return null;
  };

  const save = async () => {
    const problem = validate();
    if (problem) {
      toast.error(problem);
      return;
    }
    setLoading(true);
    try {
      const payload = { title, description, criteria: toCriteria(rows) };
      if (editing === 'new') {
        // A new rubric becomes the default for its scope.
        await apiClient.post('/rubrics', { ...payload, is_active: true, is_system: isSystem });
        toast.success(isSystem ? 'School-wide rubric saved and activated' : 'Rubric saved as your default');
      } else {
        await apiClient.put(`/rubrics/${editing}`, payload);
        toast.success('Rubric saved');
      }
      setEditing(null);
      loadRubrics();
    } catch (err) {
      toast.error('Failed to save rubric: ' + errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const activate = async (r) => {
    try {
      await apiClient.put(`/rubrics/${r.id}`, { is_active: true });
      toast.success(r.is_system ? `"${r.title}" is now the school-wide rubric` : `"${r.title}" is now your default`);
      loadRubrics();
    } catch (err) {
      toast.error('Failed to activate: ' + errorMessage(err));
    }
  };

  const remove = async (r) => {
    if (!confirm(`Delete "${r.title}"?`)) return;
    try {
      await apiClient.delete(`/rubrics/${r.id}`);
      toast.success('Deleted');
      loadRubrics();
    } catch (err) {
      toast.error('Failed to delete: ' + errorMessage(err));
    }
  };

  if (!user) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 text-center">
        <div className="text-6xl mb-4">🔒</div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Login Required</h2>
        <p className="text-gray-600">Sign in to create and manage your own rubrics.</p>
      </div>
    );
  }

  const totalPoints = rows.reduce((s, r) => s + (Number(r.max_score) || 0), 0);
  const totalWeight = rows.reduce((s, r) => s + (Number(r.weight) || 0), 0);
  const systemRubrics = rubrics.filter((r) => r.is_system);
  const myRubrics = rubrics.filter((r) => !r.is_system);
  const myDefault = myRubrics.find((r) => r.is_active);
  const schoolActive = systemRubrics.find((r) => r.is_active);

  const cardHandlers = { onActivate: activate, onEdit: startEdit, onDelete: remove, onCopy: startCopy };

  return (
    <div className="glass-card rounded-2xl shadow-xl p-6 max-w-6xl mx-auto">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
        <h2 className="text-2xl font-bold text-indigo-700">📐 Rubrics</h2>
        {!editing && (
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => startNew(false)}
              className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition"
            >
              + New Rubric
            </button>
            {isAdmin && (
              <button
                onClick={() => startNew(true)}
                className="border border-indigo-300 text-indigo-700 px-4 py-2 rounded-lg hover:bg-indigo-50 transition"
              >
                + New School-wide Rubric
              </button>
            )}
          </div>
        )}
      </div>

      {!editing && (
        <>
          <div className="text-sm text-gray-700 bg-indigo-50 border border-indigo-100 rounded-lg px-4 py-3 mb-6">
            Your evaluations use{' '}
            <strong>
              {myDefault ? `your rubric “${myDefault.title}”` : schoolActive ? `the school-wide rubric “${schoolActive.title}”` : 'the built-in rubric (5 criteria, 1–4 each)'}
            </strong>{' '}
            unless you pick a different one on the Evaluate page.
          </div>

          {listLoading ? (
            <p className="text-gray-500">Loading rubrics…</p>
          ) : (
            <div className="space-y-8">
              <section>
                <h3 className="font-semibold text-gray-800 mb-1">My rubrics</h3>
                <p className="text-xs text-gray-500 mb-3">
                  Only you can see and use these. Your default overrides the school-wide rubric for your evaluations.
                </p>
                <div className="space-y-3">
                  {myRubrics.length === 0 && (
                    <p className="text-gray-500 text-sm">
                      You haven't created a rubric yet. Click “+ New Rubric”, or copy the school-wide one below.
                    </p>
                  )}
                  {myRubrics.map((r) => <RubricCard key={r.id} rubric={r} {...cardHandlers} />)}
                </div>
              </section>

              <section>
                <h3 className="font-semibold text-gray-800 mb-1">School-wide {isAdmin ? 'rubrics' : 'rubric'}</h3>
                <p className="text-xs text-gray-500 mb-3">
                  {isAdmin
                    ? 'Set by admins. The active one is the default for every teacher without their own default.'
                    : 'Set by the administrator. Used when you have no default rubric of your own.'}
                </p>
                <div className="space-y-3">
                  {systemRubrics.length === 0 && (
                    <p className="text-gray-500 text-sm">No school-wide rubric is set; the built-in rubric is used.</p>
                  )}
                  {systemRubrics.map((r) => <RubricCard key={r.id} rubric={r} {...cardHandlers} />)}
                </div>
              </section>
            </div>
          )}
        </>
      )}

      {editing && (
        <div className="space-y-4">
          <div className={`text-sm rounded-lg px-4 py-2 border ${
            isSystem ? 'bg-amber-50 border-amber-200 text-amber-900' : 'bg-gray-50 border-gray-200 text-gray-700'
          }`}>
            {isSystem
              ? '🏫 School-wide rubric: applies to every teacher who has no default rubric of their own.'
              : '👤 Personal rubric: only you can see and use it.'}
          </div>
          <div>
            <label className="block font-medium mb-1">Title</label>
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="w-full border rounded-lg p-2"
              placeholder="e.g., Argumentative Essay Rubric"
            />
          </div>
          <div>
            <label className="block font-medium mb-1">Description (optional)</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full border rounded-lg p-2"
              rows="2"
              placeholder="Brief description of the rubric purpose"
            />
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 bg-indigo-50 border border-indigo-100 rounded-lg px-4 py-2 text-sm text-indigo-900">
            <span>
              {rows.length} criteria · {totalPoints} total points · weight sum {totalWeight.toFixed(1)}
            </span>
            <span className="text-xs text-indigo-700">
              Weight controls how much a criterion counts toward the weighted score.
            </span>
          </div>

          {rows.map((row, index) => (
            <div key={row.id} className="bg-gray-50 p-4 rounded-lg border">
              <div className="grid grid-cols-1 md:grid-cols-[2fr_1.5fr_auto_auto_auto] gap-3 items-end">
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Criterion name</label>
                  <input
                    value={row.label}
                    onChange={(e) => updateRow(row.id, { label: e.target.value })}
                    className="w-full border rounded-lg p-2 text-sm"
                    placeholder="e.g., Creativity"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Key (used by the AI)</label>
                  <input
                    value={row.key}
                    onChange={(e) => updateRow(row.id, { key: slugify(e.target.value), keyTouched: true })}
                    className="w-full border rounded-lg p-2 text-sm font-mono"
                    placeholder="creativity"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Max score</label>
                  <select
                    value={row.max_score}
                    onChange={(e) => updateRow(row.id, { max_score: Number(e.target.value) })}
                    className="border rounded-lg p-2 text-sm bg-white"
                  >
                    {Array.from({ length: MAX_SCORE - MIN_SCORE + 1 }, (_, i) => MIN_SCORE + i).map((n) => (
                      <option key={n} value={n}>1–{n}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Weight</label>
                  <input
                    type="number"
                    min="0"
                    step="0.5"
                    value={row.weight}
                    onChange={(e) => updateRow(row.id, { weight: e.target.value === '' ? '' : Number(e.target.value) })}
                    className="w-20 border rounded-lg p-2 text-sm"
                  />
                </div>
                <div className="flex gap-1">
                  <button onClick={() => moveRow(index, -1)} disabled={index === 0} className="px-2 py-2 text-gray-500 hover:text-gray-800 disabled:opacity-30" aria-label="Move up">↑</button>
                  <button onClick={() => moveRow(index, 1)} disabled={index === rows.length - 1} className="px-2 py-2 text-gray-500 hover:text-gray-800 disabled:opacity-30" aria-label="Move down">↓</button>
                  <button
                    onClick={() => setRows((prev) => prev.filter((r) => r.id !== row.id))}
                    className="px-2 py-2 text-red-600 hover:text-red-800"
                    aria-label={`Remove ${row.label || 'criterion'}`}
                  >
                    ✕
                  </button>
                </div>
              </div>

              <button
                onClick={() => setOpenLevels((prev) => ({ ...prev, [row.id]: !prev[row.id] }))}
                className="mt-3 text-sm text-indigo-700 hover:underline"
              >
                {openLevels[row.id] ? '▲ Hide' : '▼ Edit'} level descriptors
                <span className="text-gray-500">
                  {' '}({Object.entries(row.levels).filter(([l, t]) => Number(l) <= row.max_score && t?.trim()).length}/{row.max_score} filled)
                </span>
              </button>

              {openLevels[row.id] && (
                <div className="mt-3 space-y-2">
                  <p className="text-xs text-gray-500">
                    Optional. Describe what earns each score; levels left blank use the nearest described level below.
                  </p>
                  {Array.from({ length: row.max_score }, (_, i) => row.max_score - i).map((level) => (
                    <div key={level}>
                      <label className="block text-xs text-gray-500 mb-1">Score {level}</label>
                      <textarea
                        value={row.levels[level] || ''}
                        onChange={(e) => updateLevel(row.id, level, e.target.value)}
                        className="w-full border rounded-lg p-2 text-sm"
                        rows="2"
                        placeholder={`What does a ${level}/${row.max_score} look like?`}
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}

          <button
            onClick={() => setRows((prev) => [...prev, newRow()])}
            className="w-full border-2 border-dashed border-indigo-300 text-indigo-700 rounded-lg py-2 hover:bg-indigo-50 transition"
          >
            + Add criterion
          </button>

          <div className="flex gap-2">
            <button
              onClick={save}
              disabled={loading}
              className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 disabled:opacity-50"
            >
              {loading ? 'Saving...' : editing === 'new' ? (isSystem ? 'Save & Activate' : 'Save & Make Default') : 'Save'}
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

import React, { useEffect, useRef, useState } from 'react';
import toast from 'react-hot-toast';
import { apiClient, getRubric, importRubricFile } from '../api';
import { useAuth } from '../context/AuthContext';
import { DEFAULT_CRITERIA, labelFromKey, errorMessage, slugify } from '../lib/scoring';

const MIN_SCORE = 2;
const MAX_SCORE = 100;
// Up to this max score each score has its own descriptor; above it (points
// rubrics such as 20 per criterion) descriptors are bands, e.g. 18–20.
const PER_SCORE_LIMIT = 10;

// Band levels of a row, best first: [{ from, to, text }]. A level is keyed by
// its lowest score and runs up to just below the next level.
function bandsOf(row) {
  const starts = Object.keys(row.levels).map(Number)
    .filter((n) => n >= 1 && n <= row.max_score && row.levels[n] !== undefined)
    .sort((a, b) => b - a);
  let to = row.max_score;
  return starts.map((from) => {
    const band = { from, to, text: row.levels[from] };
    to = from - 1;
    return band;
  });
}
const DEFAULT_WEIGHTS = Object.fromEntries(DEFAULT_CRITERIA.map((c) => [c.key, c.weight]));


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
    criteria[slugify(r.label)] = {
      label: r.label.trim(),
      max_score: r.max_score,
      weight: Number(r.weight) || 0,
      levels,
    };
  }
  return criteria;
}

function importedRows(criteria, fileName) {
  return criteria.map((c) => ({
    id: `imp-${Math.random().toString(36).slice(2)}`,
    key: '',
    label: c.label,
    max_score: Math.min(MAX_SCORE, Math.max(MIN_SCORE, Number(c.max_score) || 4)),
    weight: Number(c.weight ?? 1),
    levels: Object.fromEntries(Object.entries(c.levels || {}).map(([k, v]) => [Number(k), v])),
    importedFrom: fileName,
  }));
}

const RUBRIC_FILE_TYPES = '.pdf,.xlsx,.csv';
const SOURCE_NAMES = { gemini: 'Gemini', deepseek: 'DeepSeek', table: 'the built-in table reader' };

function newRow() {
  return {
    id: `new-${Math.random().toString(36).slice(2)}`,
    key: '',
    label: '',
    max_score: 4,
    weight: 1,
    levels: {},
  };
}

function RubricCard({ rubric, onActivate, onEdit, onDelete, onCopy, readOnly = false }) {
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
        {rubric.can_edit && !readOnly ? (
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

export default function RubricManager({ scope = 'teacher' }) {
  const school = scope === 'school';
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
  const [importing, setImporting] = useState(false);
  const [importNote, setImportNote] = useState(null);   // { file, source, count, warnings }
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef(null);

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

  // Read a rubric file. Outside the editor it starts a new rubric; inside,
  // the criteria are added to the ones already there (duplicates are marked).
  const importFile = async (file) => {
    if (!file) return;
    if (!/\.(pdf|xlsx|csv)$/i.test(file.name)) {
      toast.error('Choose a PDF, an Excel file (.xlsx) or a CSV.');
      return;
    }
    setImporting(true);
    try {
      const result = await importRubricFile(file);
      const fresh = importedRows(result.criteria, file.name);
      if (editing) {
        setRows((prev) => [...prev, ...fresh]);
        if (!title.trim()) setTitle(result.title);
      } else {
        openEditor({
          id: 'new',
          system: school,
          rubricTitle: result.title,
          rubricDescription: `Imported from ${file.name}`,
          editorRows: fresh,
        });
      }
      setImportNote({ file: file.name, source: result.source, count: fresh.length, warnings: result.warnings || [] });
      toast.success(`Read ${fresh.length} criteria from ${file.name}. Check them, then save.`);
    } catch (err) {
      toast.error('Import failed: ' + errorMessage(err));
    } finally {
      setImporting(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const onDropFile = (e) => {
    e.preventDefault();
    setDragging(false);
    importFile(e.dataTransfer?.files?.[0]);
  };

  const openEditor = ({ id, system, rubricTitle, rubricDescription, editorRows }) => {
    setImportNote(null);
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
        ...newRow(), key: c.key, label: c.label, weight: c.weight,
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
      return { ...r, ...patch };
    }));
  };

  const updateLevel = (id, level, value) => {
    setRows((prev) => prev.map((r) => (
      r.id === id ? { ...r, levels: { ...r.levels, [level]: value } } : r
    )));
  };

  // Bands: move a band's lowest score, add a band, or remove one.
  const moveBand = (row, from, to) => {
    if (!(to >= 1 && to <= row.max_score) || to === from) return;
    if (row.levels[to] !== undefined) {
      toast.error(`Another level already starts at ${to}.`);
      return;
    }
    const { [from]: text, ...rest } = row.levels;
    updateRow(row.id, { levels: { ...rest, [to]: text } });
  };
  const addBand = (row) => {
    const taken = new Set(bandsOf(row).map((b) => b.from));
    const lowest = Math.min(row.max_score + 1, ...taken);
    const start = lowest > 1 ? Math.max(1, Math.floor(lowest / 2)) : [...Array(row.max_score).keys()].map((i) => i + 1).find((n) => !taken.has(n));
    if (!start) return;
    updateRow(row.id, { levels: { ...row.levels, [start]: '' } });
  };
  const removeBand = (row, from) => {
    const { [from]: _removed, ...rest } = row.levels;
    updateRow(row.id, { levels: rest });
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
    const seen = new Map();
    for (const r of rows) {
      if (!r.label.trim()) return 'Every criterion needs a name.';
      const key = slugify(r.label);
      if (!key) return `"${r.label}" needs at least one letter or number in its name.`;
      if (seen.has(key)) return `"${seen.get(key)}" and "${r.label}" are too similar: both become the AI key "${key}". Rename one.`;
      seen.set(key, r.label);
      if (!(Number.isInteger(r.max_score) && r.max_score >= MIN_SCORE && r.max_score <= MAX_SCORE)) return `"${r.label}" max score must be a whole number from ${MIN_SCORE} to ${MAX_SCORE}.`;
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

  if (user && school && !isAdmin) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 text-center">
        <div className="text-6xl mb-4">🛡️</div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Admins only</h2>
        <p className="text-gray-600">The school-wide rubric is managed by administrators.</p>
      </div>
    );
  }

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

  // A criterion "already exists" when an earlier row has the same name (the
  // same AI key), e.g. after importing a file into a rubric that has it.
  const duplicates = new Map();   // row id → the earlier row's name
  {
    const firstByKey = new Map();
    for (const r of rows) {
      const key = slugify(r.label);
      if (!key) continue;
      if (firstByKey.has(key)) duplicates.set(r.id, firstByKey.get(key));
      else firstByKey.set(key, r.label);
    }
  }
  const titleTaken = title.trim()
    ? rubrics.find((r) => r.id !== editing && r.title?.trim().toLowerCase() === title.trim().toLowerCase())
    : null;

  return (
    <div className="glass-card rounded-2xl shadow-xl p-6 max-w-6xl mx-auto">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
        <h2 className="text-2xl font-bold text-indigo-700">{school ? '🏫 School Rubric' : '📐 My Rubrics'}</h2>
        {!editing && (
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={importing}
              className="bg-white border border-indigo-300 text-indigo-700 px-4 py-2 rounded-lg hover:bg-indigo-50 transition disabled:opacity-50"
            >
              {importing ? '⏳ Reading file…' : '📄 Import from PDF/Excel'}
            </button>
            <button
              onClick={() => startNew(school)}
              className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition"
            >
              {school ? '+ New School-wide Rubric' : '+ New Rubric'}
            </button>
          </div>
        )}
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept={RUBRIC_FILE_TYPES}
        className="hidden"
        onChange={(e) => importFile(e.target.files?.[0])}
      />

      {!editing && (
        <>
          <div className="text-sm text-gray-700 bg-indigo-50 border border-indigo-100 rounded-lg px-4 py-3 mb-6">
            {school ? (
              <>
                The active school-wide rubric is the default for everyone who hasn't set their own:{' '}
                <strong>{schoolActive ? `“${schoolActive.title}”` : 'none set, so the built-in rubric (5 criteria, 1–4 each) is used'}</strong>.
                Everyone can view and copy it, but only admins can change it.
              </>
            ) : (
              <>
                Your evaluations use{' '}
                <strong>
                  {myDefault ? `your rubric “${myDefault.title}”` : schoolActive ? `the school-wide rubric “${schoolActive.title}”` : 'the built-in rubric (5 criteria, 1–4 each)'}
                </strong>{' '}
                unless you pick a different one on the Evaluate page.
              </>
            )}
          </div>

          {listLoading ? (
            <p className="text-gray-500">Loading rubrics…</p>
          ) : (
            <div className="space-y-8">
              {!school && (
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
              )}

              <section>
                <h3 className="font-semibold text-gray-800 mb-1">{school ? 'School-wide rubrics' : 'School-wide rubric'}</h3>
                <p className="text-xs text-gray-500 mb-3">
                  {school
                    ? 'Only one is active at a time. Set one active to make it the default for everyone.'
                    : 'Set by the administrator. Used when you have no default rubric of your own. Copy it to start your own.'}
                </p>
                <div className="space-y-3">
                  {systemRubrics.length === 0 && (
                    <p className="text-gray-500 text-sm">No school-wide rubric is set; the built-in rubric is used.</p>
                  )}
                  {systemRubrics.map((r) => <RubricCard key={r.id} rubric={r} {...cardHandlers} readOnly={!school} />)}
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
              ? '🏫 School-wide rubric: applies to everyone who has no default rubric of their own.'
              : '👤 Personal rubric: only you can see and use it.'}
          </div>
          <div>
            <label className="block font-medium mb-1">Title</label>
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className={`w-full border rounded-lg p-2 ${titleTaken ? 'border-amber-400 bg-amber-50' : ''}`}
              placeholder="e.g., Argumentative Essay Rubric"
            />
            {titleTaken && (
              <p className="mt-1 text-xs text-amber-700">⚠️ You already have a rubric called “{titleTaken.title}”. Saving creates a second one; rename it if that isn't what you want.</p>
            )}
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

          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setDragging(false); }}
            onDrop={onDropFile}
            className={`rounded-lg border-2 border-dashed px-4 py-3 text-sm text-center transition ${
              dragging ? 'border-indigo-500 bg-indigo-100' : 'border-indigo-200 bg-white'
            }`}
          >
            {importing ? (
              <span className="text-indigo-700">⏳ Reading the rubric… this can take up to a minute.</span>
            ) : (
              <>
                📄 Add criteria from a rubric file: drag a PDF or Excel file here, or{' '}
                <button type="button" onClick={() => fileInputRef.current?.click()} className="text-indigo-700 underline font-medium">
                  browse
                </button>
                .
              </>
            )}
          </div>

          {importNote && (
            <div className="rounded-lg border border-sky-200 bg-sky-50 px-4 py-2 text-sm text-sky-900">
              Read {importNote.count} criteria from <strong>{importNote.file}</strong> using {SOURCE_NAMES[importNote.source] || importNote.source}.
              Check names, scores and descriptions below; change or remove any, or add your own, then save.
              {importNote.warnings.map((w) => <div key={w} className="text-xs text-amber-800 mt-1">⚠️ {w}</div>)}
            </div>
          )}

          {rows.map((row, index) => (
            <div key={row.id} className={`p-4 rounded-lg border ${duplicates.has(row.id) ? 'bg-amber-50 border-amber-400' : 'bg-gray-50'}`}>
              {(duplicates.has(row.id) || row.importedFrom) && (
                <div className="flex flex-wrap gap-2 mb-2 text-xs">
                  {duplicates.has(row.id) && (
                    <span className="rounded-full bg-amber-100 text-amber-800 px-2 py-0.5 font-medium">
                      ⚠️ Already exists: “{duplicates.get(row.id)}” is above. Rename or remove one.
                    </span>
                  )}
                  {row.importedFrom && (
                    <span className="rounded-full bg-sky-100 text-sky-800 px-2 py-0.5">📄 Imported from {row.importedFrom}</span>
                  )}
                </div>
              )}
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
                  <span className="block text-xs text-gray-500 mb-1">AI key (automatic)</span>
                  <div
                    className="w-full border border-dashed rounded-lg p-2 text-sm font-mono bg-gray-100 text-gray-700 truncate"
                    title="Generated from the criterion name; this is the name the AI scores under."
                  >
                    {slugify(row.label) || '—'}
                  </div>
                  {row.key && slugify(row.label) && row.key !== slugify(row.label) && (
                    <span className="block text-[11px] text-amber-700 mt-0.5">was: {row.key}</span>
                  )}
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Max score</label>
                  <input
                    type="number"
                    min={MIN_SCORE}
                    max={MAX_SCORE}
                    step="1"
                    value={row.max_score}
                    onChange={(e) => updateRow(row.id, { max_score: e.target.value === '' ? '' : Number(e.target.value) })}
                    className="w-20 border rounded-lg p-2 text-sm bg-white"
                    title={`Scored 1 to this number (${MIN_SCORE}–${MAX_SCORE}), e.g. 4 for a 4-level rubric or 20 for 20 points.`}
                  />
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
                  {row.max_score > PER_SCORE_LIMIT
                    ? ` (${bandsOf(row).length} score bands)`
                    : ` (${Object.entries(row.levels).filter(([l, t]) => Number(l) <= row.max_score && t?.trim()).length}/${row.max_score} filled)`}
                </span>
              </button>

              {openLevels[row.id] && row.max_score > PER_SCORE_LIMIT && (
                <div className="mt-3 space-y-2">
                  <p className="text-xs text-gray-500">
                    Scored 1–{row.max_score}. Each band covers a range of points, e.g. Excellent 18–20. Set where a band
                    starts; it runs up to the band above it.
                  </p>
                  {bandsOf(row).map((band) => (
                    <div key={band.from} className="rounded-lg border bg-white p-2">
                      <div className="flex flex-wrap items-center gap-2 text-xs text-gray-600 mb-1">
                        <span>Scores</span>
                        <input
                          type="number"
                          min="1"
                          max={row.max_score}
                          defaultValue={band.from}
                          onBlur={(e) => moveBand(row, band.from, Number(e.target.value))}
                          className="w-16 border rounded p-1 text-sm"
                          aria-label={`Band starting score (currently ${band.from})`}
                        />
                        <span>to {band.to}</span>
                        <button onClick={() => removeBand(row, band.from)} className="ml-auto text-red-600 hover:underline">Remove</button>
                      </div>
                      <textarea
                        value={band.text || ''}
                        onChange={(e) => updateLevel(row.id, band.from, e.target.value)}
                        className="w-full border rounded-lg p-2 text-sm"
                        rows="2"
                        placeholder={`What earns ${band.from}–${band.to} out of ${row.max_score}? e.g. "Excellent: …"`}
                      />
                    </div>
                  ))}
                  <button onClick={() => addBand(row)} className="text-sm text-indigo-700 hover:underline">+ Add score band</button>
                </div>
              )}

              {openLevels[row.id] && row.max_score <= PER_SCORE_LIMIT && (
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

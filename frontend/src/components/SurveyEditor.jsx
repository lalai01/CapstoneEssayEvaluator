import React, { useRef, useState } from 'react';
import toast from 'react-hot-toast';
import {
  addQuestion, addQuestionsBulk, deleteQuestion, deleteSurveyFile, reorderQuestions, updateQuestion, uploadSurveyFile,
} from '../api';
import { SurveyAttachment } from './SurveyQuestion';
import {
  QUESTION_TYPES, copyOf, fileInfo, headingText, isHeading, kindLabel, kindOf, matrixItems, optionsText, storedShape,
  structure, tabsOf,
} from '../lib/surveyKit';
import { errorMessage } from '../lib/scoring';

const ATTACH_TYPES = '.pdf,.doc,.docx,.xls,.xlsx,.csv,.txt,.png,.jpg,.jpeg,.gif,.webp,.ppt,.pptx';
const emptyForm = { id: null, text: '', kind: 'likert', options: '', required: true };

// The admin's survey editor: the survey's sections are tabs. Pick a tab to
// add, edit, move or delete its questions, attach files to it, or rename,
// move, duplicate and delete whole tabs.
export default function SurveyEditor({ survey, questions, reload }) {
  const [current, setCurrent] = useState(0);
  const [form, setForm] = useState(null);           // the question being added/edited
  const [busy, setBusy] = useState(false);
  const fileRef = useRef(null);

  const tabs = tabsOf(questions);
  const tabIndex = Math.min(current, Math.max(0, tabs.length - 1));
  const tab = tabs[tabIndex];
  const allIds = () => structure(questions).map((it) => it.q.id);
  const tabIds = (t) => [...(t.section ? [t.section.id] : []), ...t.items.map((it) => it.q.id)];

  const run = async (label, fn) => {
    setBusy(true);
    try {
      await fn();
      await reload();
    } catch (err) {
      toast.error(`${label} failed: ${errorMessage(err)}`);
    } finally {
      setBusy(false);
    }
  };

  // New items are added at the end, then moved to the end of the current tab.
  const placeInTab = async (newIds) => {
    if (!tab) return;
    const others = allIds().filter((id) => !newIds.includes(id));
    const anchor = tabIds(tab).filter((id) => !newIds.includes(id)).pop();
    const at = anchor === undefined ? others.length : others.indexOf(anchor) + 1;
    await reorderQuestions(survey.id, [...others.slice(0, at), ...newIds, ...others.slice(at)]);
  };

  // ---------- tabs ----------
  const newTab = () => {
    const title = window.prompt('Name of the new tab (e.g. "Essay 2" or "Part V. Feedback"):');
    if (!title?.trim()) return;
    run('Adding the tab', async () => {
      await addQuestionsBulk(survey.id, [{ question: title.trim(), ...storedShape('section', ''), is_required: false }]);
      setCurrent(tabs.length);
    });
  };

  const renameTab = () => {
    if (!tab?.section) return;
    const title = window.prompt('Tab name:', tab.section.question);
    if (title === null || !title.trim()) return;
    const text = window.prompt('Text shown at the top of the tab (optional):', headingText(tab.section));
    if (text === null) return;
    run('Renaming the tab', () => updateQuestion(tab.section.id, {
      question: title.trim(), ...storedShape('section', text), is_required: false,
    }));
  };

  const moveTab = (dir) => {
    const target = tabIndex + dir;
    if (target < 0 || target >= tabs.length) return;
    const groups = tabs.map(tabIds);
    [groups[tabIndex], groups[target]] = [groups[target], groups[tabIndex]];
    run('Moving the tab', async () => {
      await reorderQuestions(survey.id, groups.flat());
      setCurrent(target);
    });
  };

  const duplicateTab = () => {
    const items = [...(tab.section ? [tab.section] : []), ...tab.items.map((it) => it.q)];
    run('Duplicating the tab', async () => {
      // The copy is its own tab, so it always starts with a heading.
      const copies = tab.section
        ? items.map((q, i) => (i === 0 ? { ...copyOf(q), question: `${q.question} (copy)` } : copyOf(q)))
        : [{ question: 'Copy', ...storedShape('section', ''), is_required: false }, ...items.map(copyOf)];
      const { ids } = await addQuestionsBulk(survey.id, copies);
      // Put the copy right after this tab.
      const groups = tabs.map(tabIds);
      groups.splice(tabIndex + 1, 0, ids);
      await reorderQuestions(survey.id, groups.flat().filter((id, i, a) => a.indexOf(id) === i));
      setCurrent(tabIndex + 1);
    });
  };

  const deleteTab = () => {
    const count = tab.items.length;
    if (!window.confirm(`Delete the tab “${tab.section?.question || 'Start'}” and its ${count} item${count === 1 ? '' : 's'}? Answers already given to these questions are kept but no longer shown.`)) return;
    run('Deleting the tab', async () => {
      for (const q of [...tab.items.map((it) => it.q), ...(tab.section ? [tab.section] : [])]) await removeItem(q);
      setCurrent(Math.max(0, tabIndex - 1));
    });
  };

  // ---------- items ----------
  // Deleting an attachment also deletes its file, unless another item (e.g. a
  // duplicated tab) still uses it.
  const removeItem = async (q) => {
    await deleteQuestion(q.id);
    const info = fileInfo(q);
    if (info && !questions.some((o) => o.id !== q.id && fileInfo(o)?.path === info.path)) {
      await deleteSurveyFile(info.path).catch(() => {});
    }
  };

  const moveItem = (id, dir) => {
    const ids = tab.items.map((it) => it.q.id);
    const i = ids.indexOf(id);
    const j = i + dir;
    if (j < 0 || j >= ids.length) return;
    [ids[i], ids[j]] = [ids[j], ids[i]];
    const reordered = tabs.map((t) => (t === tab ? [...(t.section ? [t.section.id] : []), ...ids] : tabIds(t))).flat();
    run('Moving the question', () => reorderQuestions(survey.id, reordered));
  };

  const saveForm = (e) => {
    e.preventDefault();
    if (!form.text.trim()) return;
    const data = {
      survey_id: survey.id,
      question: form.text.trim(),
      ...storedShape(form.kind, form.options),
      is_required: form.kind === 'subsection' ? false : form.required,
      order_number: 0,
    };
    run(form.id ? 'Saving the question' : 'Adding the question', async () => {
      if (form.id) {
        await updateQuestion(form.id, data);
      } else {
        const { id } = await addQuestion(survey.id, data);
        await placeInTab([id]);
      }
      setForm(null);
    });
  };

  const attachFiles = (files) => {
    if (!files?.length) return;
    run('Attaching', async () => {
      const items = [];
      for (const file of files) {
        const up = await uploadSurveyFile(survey.id, file);
        items.push({ question: up.name, question_type: 'text', options: ['::file', up.path, up.name, up.type], is_required: false });
      }
      const { ids } = await addQuestionsBulk(survey.id, items);
      await placeInTab(ids);
      toast.success(`Attached ${items.length} file${items.length === 1 ? '' : 's'} to this tab.`);
    });
    if (fileRef.current) fileRef.current.value = '';
  };

  // Rename the items of every AI grid in this tab at once (e.g. new model names).
  const matrixInTab = tab ? tab.items.filter((it) => kindOf(it.q) === 'matrix') : [];
  const renameModels = () => {
    const now = matrixItems(matrixInTab[0].q).join(', ');
    const next = window.prompt(`AI models rated in the ${matrixInTab.length} grid question(s) of this tab (comma separated):`, now);
    if (next === null) return;
    const shape = storedShape('matrix', next);
    if (shape.options.length < 2) {
      toast.error('Give at least one model name.');
      return;
    }
    run('Renaming the models', async () => {
      for (const it of matrixInTab) await updateQuestion(it.q.id, { options: shape.options });
      toast.success('Model names updated. Answers already given keep the old names.');
    });
  };

  const editForm = (q) => setForm({ id: q.id, text: q.question, kind: kindOf(q), options: optionsText(q), required: q.is_required });

  return (
    <div className="space-y-4">
      {/* Tab bar */}
      <div className="flex flex-wrap items-center gap-1 border-b pb-2">
        {tabs.map((t, i) => (
          <button
            key={t.section?.id ?? 'start'}
            type="button"
            onClick={() => { setCurrent(i); setForm(null); }}
            className={`rounded-t-lg px-3 py-2 text-sm border-b-2 transition ${
              i === tabIndex ? 'border-indigo-600 text-indigo-700 font-semibold bg-indigo-50' : 'border-transparent text-gray-600 hover:text-gray-900'
            }`}
          >
            {t.section?.question || 'Start (before the first tab)'}
            <span className="ml-1 text-xs text-gray-400">{t.items.filter((it) => !isHeading(it.q)).length}</span>
          </button>
        ))}
        <button type="button" onClick={newTab} disabled={busy} className="ml-1 px-3 py-2 text-sm text-indigo-700 hover:underline">+ New tab</button>
      </div>

      {!tab ? (
        <p className="text-gray-500 text-sm">No questions yet. Add a tab, then add questions or files to it.</p>
      ) : (
        <>
          {/* Tab toolbar */}
          <div className="flex flex-wrap gap-2 text-sm">
            <button onClick={() => setForm({ ...emptyForm })} disabled={busy} className="bg-green-600 text-white px-3 py-1.5 rounded-lg hover:bg-green-700 disabled:opacity-50">+ Add question to this tab</button>
            <button onClick={() => fileRef.current?.click()} disabled={busy} className="bg-white border border-indigo-300 text-indigo-700 px-3 py-1.5 rounded-lg hover:bg-indigo-50 disabled:opacity-50">📎 Attach files to this tab</button>
            <input ref={fileRef} type="file" multiple accept={ATTACH_TYPES} className="hidden" onChange={(e) => attachFiles([...e.target.files])} />
            {matrixInTab.length > 0 && (
              <button onClick={renameModels} disabled={busy} className="bg-white border px-3 py-1.5 rounded-lg hover:bg-gray-50 disabled:opacity-50">🔁 Rename AI models in this tab</button>
            )}
            <span className="flex-1" />
            {tab.section && <button onClick={renameTab} disabled={busy} className="bg-white border px-3 py-1.5 rounded-lg hover:bg-gray-50">✏️ Rename tab</button>}
            <button onClick={() => moveTab(-1)} disabled={busy || tabIndex === 0} className="bg-white border px-2 py-1.5 rounded-lg disabled:opacity-30" aria-label="Move tab left">◀</button>
            <button onClick={() => moveTab(1)} disabled={busy || tabIndex === tabs.length - 1} className="bg-white border px-2 py-1.5 rounded-lg disabled:opacity-30" aria-label="Move tab right">▶</button>
            <button onClick={duplicateTab} disabled={busy} className="bg-white border px-3 py-1.5 rounded-lg hover:bg-gray-50">⧉ Duplicate tab</button>
            <button onClick={deleteTab} disabled={busy} className="bg-white border border-red-200 text-red-600 px-3 py-1.5 rounded-lg hover:bg-red-50">🗑 Delete tab</button>
          </div>
          {tab.section && headingText(tab.section) && (
            <p className="text-sm text-gray-600 whitespace-pre-line bg-gray-50 rounded-lg p-3">{headingText(tab.section)}</p>
          )}

          {/* Add / edit form */}
          {form && (
            <form onSubmit={saveForm} className="bg-white border-2 border-indigo-200 p-4 rounded-lg shadow space-y-3">
              <h4 className="font-medium text-gray-800">{form.id ? 'Edit question' : `New question in “${tab.section?.question || 'Start'}”`}</h4>
              <div>
                <label className="block text-sm font-medium mb-1">Question or heading text</label>
                <textarea value={form.text} onChange={(e) => setForm({ ...form, text: e.target.value })} rows="2" required className="w-full border rounded-lg p-2" />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Type</label>
                <select value={form.kind} onChange={(e) => setForm({ ...form, kind: e.target.value })} className="w-full border rounded-lg p-2">
                  {QUESTION_TYPES.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
                </select>
              </div>
              {['radio', 'checkbox', 'matrix'].includes(form.kind) && (
                <div>
                  <label className="block text-sm font-medium mb-1">
                    {form.kind === 'matrix' ? 'AI models (or items) to rate' : 'Options'} <span className="text-gray-400 font-normal">(comma separated)</span>
                  </label>
                  <input value={form.options} onChange={(e) => setForm({ ...form, options: e.target.value })} className="w-full border rounded-lg p-2"
                    placeholder={form.kind === 'matrix' ? 'DeepSeek, Qwen 2.5 (local), GPT-OSS 120B' : 'Yes, No'} />
                </div>
              )}
              {form.kind === 'subsection' && (
                <div>
                  <label className="block text-sm font-medium mb-1">Text under the sub-heading <span className="text-gray-400 font-normal">(optional)</span></label>
                  <textarea value={form.options} onChange={(e) => setForm({ ...form, options: e.target.value })} rows="2" className="w-full border rounded-lg p-2" />
                </div>
              )}
              {form.kind !== 'subsection' && (
                <label className="flex items-center gap-2 text-sm">
                  <input type="checkbox" checked={form.required} onChange={(e) => setForm({ ...form, required: e.target.checked })} /> Required
                </label>
              )}
              <div className="flex gap-2">
                <button type="submit" disabled={busy} className="bg-indigo-600 text-white px-4 py-2 rounded-lg disabled:opacity-50">{form.id ? 'Save' : 'Add'}</button>
                <button type="button" onClick={() => setForm(null)} className="bg-gray-200 px-4 py-2 rounded-lg">Cancel</button>
              </div>
            </form>
          )}

          {/* Items of this tab */}
          <div className="space-y-2">
            {tab.items.length === 0 && <p className="text-gray-500 text-sm">This tab is empty. Add questions or attach files.</p>}
            {tab.items.map((it, i) => (
              <div key={it.q.id} className={`border p-3 rounded-lg bg-white flex gap-3 items-start ${it.kind === 'subsection' ? 'bg-indigo-50/60' : ''}`}>
                <div className="flex-1 min-w-0">
                  {it.kind === 'file' ? <SurveyAttachment q={it.q} compact /> : (
                    <div className={`text-gray-800 ${it.kind === 'subsection' ? 'font-semibold' : ''}`}>
                      {it.number && <span className="text-gray-400 mr-1">{it.number}.</span>}{it.q.question}
                    </div>
                  )}
                  <div className="flex flex-wrap gap-2 mt-1">
                    <span className="text-xs bg-gray-100 px-2 py-0.5 rounded-full">{kindLabel(it.q)}</span>
                    {it.q.is_required && !['subsection', 'file'].includes(it.kind) && <span className="text-xs bg-red-50 text-red-600 px-2 py-0.5 rounded-full">Required</span>}
                    {optionsText(it.q) && it.kind !== 'subsection' && it.kind !== 'file' && <span className="text-xs text-gray-500">{optionsText(it.q)}</span>}
                  </div>
                </div>
                <div className="flex gap-1 shrink-0 text-sm">
                  <button onClick={() => moveItem(it.q.id, -1)} disabled={busy || i === 0} className="px-1.5 text-gray-500 disabled:opacity-30" aria-label="Move up">↑</button>
                  <button onClick={() => moveItem(it.q.id, 1)} disabled={busy || i === tab.items.length - 1} className="px-1.5 text-gray-500 disabled:opacity-30" aria-label="Move down">↓</button>
                  {it.kind !== 'file' && <button onClick={() => editForm(it.q)} disabled={busy} className="px-1.5 text-blue-600 hover:underline">Edit</button>}
                  <button
                    onClick={() => window.confirm(`Delete “${it.q.question}”?`) && run('Deleting', () => removeItem(it.q))}
                    disabled={busy}
                    className="px-1.5 text-red-600 hover:underline"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

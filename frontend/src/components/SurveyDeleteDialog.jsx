import React, { useEffect, useRef, useState } from 'react';
import toast from 'react-hot-toast';
import { deleteSurveyParts, getSurveySummary } from '../api';
import { errorMessage } from '../lib/scoring';

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

// Deleting one or more surveys: the admin ticks what goes. "The whole survey"
// includes its responses and files; "Responses" alone clears answers (e.g.
// test answers) and keeps the survey; "Attached files" removes attachments.
export default function SurveyDeleteDialog({ surveys, onClose, onDone }) {
  const [summaries, setSummaries] = useState({});   // survey id -> counts
  const [parts, setParts] = useState({ survey: true, responses: true, files: true });
  const [working, setWorking] = useState(false);
  const cancelRef = useRef(null);

  useEffect(() => {
    cancelRef.current?.focus();
    const onKey = (e) => { if (e.key === 'Escape' && !working) onClose(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose, working]);

  useEffect(() => {
    let alive = true;
    Promise.all(surveys.map((s) => getSurveySummary(s.id).then((c) => [s.id, c]).catch(() => [s.id, null])))
      .then((rows) => { if (alive) setSummaries(Object.fromEntries(rows)); });
    return () => { alive = false; };
  }, [surveys]);

  const total = (key) => surveys.reduce((n, s) => n + (summaries[s.id]?.[key] || 0), 0);
  const loaded = surveys.every((s) => s.id in summaries);
  const toggle = (key) => setParts((p) => {
    const next = { ...p, [key]: !p[key] };
    if (key === 'survey' && next.survey) Object.assign(next, { responses: true, files: true });
    if (key !== 'survey' && !next[key]) next.survey = false;    // keeping part of it keeps the survey
    return next;
  });
  const nothing = !parts.survey && !parts.responses && !parts.files;

  const summary = parts.survey
    ? `${plural(surveys.length, 'survey')} will be deleted completely: ${plural(total('questions'), 'question')}, ${plural(total('respondents'), 'respondent')}' answers and ${plural(total('files'), 'attached file')}.`
    : [parts.responses && `all answers from ${plural(total('respondents'), 'respondent')}`,
      parts.files && plural(total('files'), 'attached file')].filter(Boolean).join(' and ')
      .replace(/^./, (c) => c.toUpperCase()) + ' will be deleted. The survey and its questions stay.';

  const run = async () => {
    setWorking(true);
    let failed = 0;
    for (const s of surveys) {
      try {
        await deleteSurveyParts(s.id, parts);
      } catch (err) {
        failed += 1;
        toast.error(`“${s.title}”: ${errorMessage(err)}`);
      }
    }
    setWorking(false);
    if (failed < surveys.length) {
      toast.success(parts.survey
        ? `Deleted ${plural(surveys.length - failed, 'survey')}.`
        : 'Deleted the selected parts. The surveys are kept.');
    }
    onDone();
  };

  const option = (key, label, note, count) => (
    <label className={`flex items-start gap-3 rounded-xl border p-3 cursor-pointer transition ${
      parts[key] ? 'border-red-300 bg-red-50' : 'border-gray-200 hover:bg-gray-50'}`}>
      <input type="checkbox" checked={parts[key]} onChange={() => toggle(key)} disabled={working}
        className="mt-0.5 h-4 w-4 accent-red-600" />
      <span>
        <span className="block font-medium text-gray-900">{label} {loaded && count !== undefined && <span className="text-gray-500 font-normal">({count})</span>}</span>
        <span className="block text-xs text-gray-600">{note}</span>
      </span>
    </label>
  );

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 p-4 animate-fade-in"
      onMouseDown={() => !working && onClose()}>
      <div role="alertdialog" aria-modal="true" aria-labelledby="delete-survey-title"
        className="w-full max-w-lg rounded-2xl bg-white p-6 shadow-2xl" onMouseDown={(e) => e.stopPropagation()}>
        <div className="flex items-start gap-4">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-red-100 text-xl text-red-700" aria-hidden="true">🗑</div>
          <div className="min-w-0 flex-1">
            <h3 id="delete-survey-title" className="text-lg font-semibold text-gray-900">
              Delete from {surveys.length === 1 ? `“${surveys[0].title}”` : plural(surveys.length, 'survey')}
            </h3>
            {surveys.length > 1 && (
              <ul className="mt-1 max-h-24 overflow-y-auto text-sm text-gray-600 list-disc pl-5">
                {surveys.map((s) => <li key={s.id}>{s.title}</li>)}
              </ul>
            )}
            <p className="mt-1 text-sm text-gray-600">Tick what to delete. This can’t be undone.</p>
          </div>
        </div>

        <div className="mt-4 space-y-2">
          {option('survey', 'The whole survey', 'Its tabs, questions, all responses and attached files.', undefined)}
          {option('responses', 'Responses', 'Everyone’s answers, e.g. test answers before the real study. Keeps the survey.', plural(total('respondents'), 'respondent'))}
          {option('files', 'Attached files', 'Files attached to its tabs (essays, documents).', plural(total('files'), 'file'))}
        </div>

        <p className={`mt-4 rounded-lg px-3 py-2 text-sm ${nothing ? 'bg-gray-50 text-gray-500' : 'bg-red-50 text-red-800'}`}>
          {nothing ? 'Nothing selected.' : loaded ? summary : 'Counting what will be deleted…'}
        </p>

        <div className="mt-6 flex justify-end gap-3">
          <button ref={cancelRef} type="button" onClick={onClose} disabled={working}
            className="rounded-xl bg-gray-100 px-4 py-2 text-sm font-medium text-gray-800 hover:bg-gray-200 disabled:opacity-50">
            Cancel
          </button>
          <button type="button" onClick={run} disabled={working || nothing}
            className="rounded-xl bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-700 disabled:opacity-50">
            {working ? 'Deleting…' : parts.survey ? `Delete ${plural(surveys.length, 'survey')}` : 'Delete selected'}
          </button>
        </div>
      </div>
    </div>
  );
}

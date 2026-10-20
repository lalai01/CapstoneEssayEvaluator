import React, { useEffect, useState } from 'react';
import toast from 'react-hot-toast';
import { surveyFileUrl } from '../api';
import { LIKERT, fileInfo, headingText, matrixItems, parseMatrix } from '../lib/surveyKit';

const FILE_ICONS = { pdf: '📕', doc: '📘', docx: '📘', xls: '📗', xlsx: '📗', csv: '📗', ppt: '📙', pptx: '📙', txt: '📄' };

// An attached file: images are previewed; anything else opens in a new tab.
// Links are private and fetched when needed (valid for one hour).
export function SurveyAttachment({ q, compact = false }) {
  const { path, name, type } = fileInfo(q);
  const isImage = type.startsWith('image/') || /\.(png|jpe?g|gif|webp)$/i.test(name);
  const [preview, setPreview] = useState(null);
  const [opening, setOpening] = useState(false);

  useEffect(() => {
    if (!isImage || compact) return undefined;
    let alive = true;
    surveyFileUrl(path).then((url) => { if (alive) setPreview(url); }).catch(() => {});
    return () => { alive = false; };
  }, [path, isImage, compact]);

  const open = async () => {
    setOpening(true);
    try {
      window.open(await surveyFileUrl(path), '_blank', 'noopener');
    } catch {
      toast.error('Could not open the file. Please try again.');
    } finally {
      setOpening(false);
    }
  };

  const ext = name.split('.').pop().toLowerCase();
  return (
    <div className={compact ? '' : 'bg-white border rounded-xl p-4 shadow-sm'}>
      {q.question && q.question !== name && !compact && <p className="font-medium text-gray-800 mb-2">{q.question}</p>}
      {preview && <img src={preview} alt={name} className="mb-2 max-h-96 rounded-lg border object-contain" />}
      <div className="flex items-center gap-3">
        <span className="text-2xl" aria-hidden="true">{isImage ? '🖼️' : FILE_ICONS[ext] || '📎'}</span>
        <span className="text-sm text-gray-800 break-all">{name}</span>
        <button type="button" onClick={open} disabled={opening}
          className="ml-auto shrink-0 text-sm bg-indigo-50 text-indigo-700 border border-indigo-200 px-3 py-1 rounded-lg hover:bg-indigo-100 disabled:opacity-50">
          {opening ? 'Opening…' : 'Open'}
        </button>
      </div>
    </div>
  );
}

// Wide enough to show rating tables the way the questionnaire document does.
function useWide(query = '(min-width: 768px)') {
  const [wide, setWide] = useState(() => typeof window !== 'undefined' && window.matchMedia?.(query).matches);
  useEffect(() => {
    const mq = window.matchMedia?.(query);
    if (!mq) return undefined;
    const on = () => setWide(mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, [query]);
  return wide;
}

// A run of rating statements as one table, like the questionnaire document:
// agreement scales -> Statement | 5 … 1; per-AI grids -> Statement | one
// 5 … 1 group per AI model. Narrow screens get the card layout instead.
export function RatingTable({ items, answers, onChange, missingId }) {
  const wide = useWide();
  const isMatrix = items[0].kind === 'matrix';
  const models = isMatrix ? matrixItems(items[0].q) : [null];
  if (!wide) {
    return items.map((item) => (
      <SurveyQuestion key={item.q.id} item={item} value={answers[item.q.id]}
        onChange={(v) => onChange(item.q.id, v)} invalid={missingId === item.q.id} />
    ));
  }
  const valueOf = (q, m) => (isMatrix ? parseMatrix(answers[q.id])[m] : answers[q.id]);
  const set = (q, m, v) => onChange(q.id, isMatrix ? JSON.stringify({ ...parseMatrix(answers[q.id]), [m]: v }) : String(v));
  const cell = 'border border-gray-200 px-1 py-1 text-center';
  const textCell = 'border border-gray-200 py-1 px-3 text-left';   // statements read left-aligned, like the document
  return (
    <div className="overflow-x-auto bg-white rounded-xl border shadow-sm">
      <table className="min-w-full text-sm border-collapse">
        <thead className="bg-gray-50 text-gray-700">
          <tr>
            <th rowSpan={isMatrix ? 2 : 1} className={`${cell} w-10`}>No.</th>
            <th rowSpan={isMatrix ? 2 : 1} className={`${textCell} min-w-[16rem]`}>Statement</th>
            {isMatrix
              ? models.map((m) => <th key={m} colSpan={5} className={`${cell} font-semibold`}>{m}</th>)
              : LIKERT.map((l) => <th key={l.value} className={`${cell} w-12`} title={l.label}>{l.value}<div className="text-[10px] font-normal text-gray-500">{l.short}</div></th>)}
          </tr>
          {isMatrix && (
            <tr>
              {models.flatMap((m) => LIKERT.map((l) => (
                <th key={`${m}-${l.value}`} className={`${cell} w-9 font-normal`} title={`${m}: ${l.label}`}>{l.value}</th>
              )))}
            </tr>
          )}
        </thead>
        <tbody>
          {items.map(({ q, number }) => (
            <tr key={q.id} id={`q-${q.id}`} className={`scroll-mt-24 ${missingId === q.id ? 'bg-red-50' : ''}`}>
              <td className={`${cell} text-gray-500`}>{number}</td>
              <td className={`${textCell} text-gray-800`}>
                {q.question}{q.is_required && <span className="text-red-500 ml-0.5">*</span>}
              </td>
              {models.flatMap((m) => LIKERT.map((l) => (
                <td key={`${m}-${l.value}`} className={`${cell} ${m && l.value === 5 ? 'border-l-2 border-l-gray-300' : ''}`}>
                  <input
                    type="radio"
                    name={`q_${q.id}_${m ?? ''}`}
                    aria-label={`${q.question}${m ? ` – ${m}` : ''}: ${l.label}`}
                    checked={Number(valueOf(q, m)) === l.value}
                    onChange={() => set(q, m, l.value)}
                    className="w-4 h-4 accent-indigo-600 cursor-pointer"
                  />
                </td>
              )))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// One row of 5-4-3-2-1 agreement buttons.
function ScaleButtons({ value, onChange, name }) {
  return (
    <div role="radiogroup" aria-label={name} className="grid grid-cols-5 gap-1.5">
      {LIKERT.map((l) => {
        const selected = Number(value) === l.value;
        return (
          <button
            key={l.value}
            type="button"
            role="radio"
            aria-checked={selected}
            title={`${l.value} – ${l.label}`}
            onClick={() => onChange(String(l.value))}
            className={`rounded-lg border px-1 py-1.5 text-center transition ${
              selected ? 'bg-indigo-600 border-indigo-600 text-white' : 'bg-white border-gray-300 text-gray-700 hover:border-indigo-400'
            }`}
          >
            <div className="text-sm font-semibold leading-none">{l.value}</div>
            <div className={`text-[10px] mt-0.5 ${selected ? 'text-indigo-100' : 'text-gray-500'}`}>{l.short}</div>
          </button>
        );
      })}
    </div>
  );
}

// A heading (section/subsection) or one answerable question of any kind.
export default function SurveyQuestion({ item, value, onChange, invalid }) {
  const { q, kind, number } = item;

  if (kind === 'section') {
    return (
      <div className="pt-4">
        <h3 className="text-xl font-bold text-indigo-800">{q.question}</h3>
        {headingText(q) && <p className="mt-2 text-sm text-gray-700 whitespace-pre-line">{headingText(q)}</p>}
      </div>
    );
  }
  if (kind === 'subsection') {
    return (
      <div className="pt-2">
        <h4 className="text-base font-semibold text-gray-800 border-b border-indigo-200 pb-1">{q.question}</h4>
        {headingText(q) && <p className="mt-1 text-sm text-gray-600 whitespace-pre-line">{headingText(q)}</p>}
      </div>
    );
  }

  if (kind === 'file') return <SurveyAttachment q={q} />;

  const wrap = `scroll-mt-24 bg-white border rounded-xl p-4 shadow-sm ${invalid ? 'border-red-400 ring-2 ring-red-200' : ''}`;
  const label = (
    <p className="font-medium text-gray-800">
      {number && <span className="text-gray-400 mr-1">{number}.</span>}
      {q.question}
      {q.is_required && <span className="text-red-500 ml-1">*</span>}
    </p>
  );

  if (kind === 'matrix') {
    const ratings = parseMatrix(value);
    const set = (item, score) => onChange(JSON.stringify({ ...ratings, [item]: Number(score) }));
    return (
      <div id={`q-${q.id}`} className={wrap}>
        {label}
        <div className="mt-3 space-y-2">
          {matrixItems(q).map((m) => (
            <div key={m} className="grid grid-cols-1 sm:grid-cols-[11rem_1fr] items-center gap-2">
              <span className="text-sm font-medium text-gray-700">{m}</span>
              <ScaleButtons name={`${q.question} – ${m}`} value={ratings[m]} onChange={(v) => set(m, v)} />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (kind === 'likert' || q.question_type === 'rating') {
    return (
      <div id={`q-${q.id}`} className={wrap}>
        {label}
        <div className="mt-3 max-w-md"><ScaleButtons name={q.question} value={value} onChange={onChange} /></div>
      </div>
    );
  }

  if (q.question_type === 'radio' && q.options) {
    return (
      <div id={`q-${q.id}`} className={wrap}>
        {label}
        <div className="mt-2 flex flex-wrap gap-x-5 gap-y-2">
          {q.options.map((opt) => (
            <label key={opt} className="flex items-center gap-2 cursor-pointer">
              <input type="radio" name={`q_${q.id}`} checked={value === opt} onChange={() => onChange(opt)} className="w-4 h-4 text-indigo-600" />
              <span>{opt}</span>
            </label>
          ))}
        </div>
      </div>
    );
  }

  if (q.question_type === 'checkbox' && q.options) {
    const chosen = (value || '').split(',').filter(Boolean);
    const toggle = (opt, on) => onChange((on ? [...chosen, opt] : chosen.filter((o) => o !== opt)).join(','));
    return (
      <div id={`q-${q.id}`} className={wrap}>
        {q.options.length > 1 && label}
        <div className="mt-2 space-y-2">
          {q.options.map((opt) => (
            <label key={opt} className="flex items-start gap-2 cursor-pointer">
              <input type="checkbox" checked={chosen.includes(opt)} onChange={(e) => toggle(opt, e.target.checked)} className="w-4 h-4 mt-1 text-indigo-600" />
              <span className={q.options.length === 1 ? 'font-medium text-gray-800' : ''}>
                {opt}{q.options.length === 1 && q.is_required && <span className="text-red-500 ml-1">*</span>}
              </span>
            </label>
          ))}
        </div>
      </div>
    );
  }

  if (q.question_type === 'essay') {
    return (
      <div id={`q-${q.id}`} className={wrap}>
        {label}
        <textarea value={value || ''} onChange={(e) => onChange(e.target.value)} rows="4"
          placeholder="Write your response…" className="w-full border rounded-lg p-2 mt-2 resize-y" />
      </div>
    );
  }

  return (
    <div id={`q-${q.id}`} className={wrap}>
      {label}
      <input
        type={kind === 'number' ? 'number' : 'text'}
        inputMode={kind === 'number' ? 'numeric' : undefined}
        min={kind === 'number' ? 1 : undefined}
        value={value || ''}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Your answer"
        className={`border rounded-lg p-2 mt-2 ${kind === 'number' ? 'w-32' : 'w-full'}`}
      />
    </div>
  );
}

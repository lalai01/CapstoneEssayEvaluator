import React from 'react';
import { LIKERT, headingText, matrixItems, parseMatrix } from '../lib/surveyKit';

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

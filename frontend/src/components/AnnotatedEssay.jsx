import React, { useMemo, useState } from 'react';
import { ANNOTATION_TYPES, buildAnnotations, segmentText } from '../lib/annotate';

// Background colour for the highest-priority mark in a piece of text; long
// sentences use an underline so they can combine with the others.
const MARK_STYLE = {
  grammar: 'bg-red-100 text-red-900 ring-1 ring-red-200',
  recommendation: 'bg-violet-100 text-violet-900 ring-1 ring-violet-200',
  vague: 'bg-amber-100 text-amber-900',
};
const LONG_STYLE = 'underline decoration-sky-500 decoration-2 underline-offset-4';

const SWATCH = {
  grammar: 'bg-red-100 ring-1 ring-red-300',
  recommendation: 'bg-violet-100 ring-1 ring-violet-300',
  vague: 'bg-amber-100 ring-1 ring-amber-300',
  long: 'border-b-2 border-sky-500',
};

const DETAIL_STYLE = {
  grammar: 'border-red-300 bg-red-50',
  recommendation: 'border-violet-300 bg-violet-50',
  vague: 'border-amber-300 bg-amber-50',
  long: 'border-sky-300 bg-sky-50',
};

export default function AnnotatedEssay({ essay, scores }) {
  const [enabled, setEnabled] = useState({ grammar: true, recommendation: true, vague: true, long: true });
  const [selected, setSelected] = useState(null); // index of the selected piece

  const { annotations, unlocated } = useMemo(() => buildAnnotations(essay, scores), [essay, scores]);
  const counts = useMemo(() => {
    const c = {};
    for (const a of annotations) c[a.type] = (c[a.type] || 0) + 1;
    return c;
  }, [annotations]);
  const pieces = useMemo(
    () => segmentText(String(essay || ''), annotations.filter((a) => enabled[a.type])),
    [essay, annotations, enabled]
  );

  if (!essay) return null;
  const total = annotations.length;
  const active = selected !== null ? pieces[selected] : null;

  return (
    <section className="mb-5" aria-labelledby="annotated-essay-title">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <h3 id="annotated-essay-title" className="font-semibold text-gray-800 text-sm uppercase tracking-wide">
          🖍️ Your essay, highlighted ({total} {total === 1 ? 'spot' : 'spots'} to review)
        </h3>
      </div>

      <div className="flex flex-wrap gap-2 mb-3" role="group" aria-label="Show highlights">
        {Object.entries(ANNOTATION_TYPES).map(([type, info]) => (
          <label
            key={type}
            className={`inline-flex items-center gap-2 text-xs px-2.5 py-1 rounded-full border cursor-pointer select-none ${
              enabled[type] ? 'bg-white border-gray-300 text-gray-800' : 'bg-gray-50 border-gray-200 text-gray-400'
            }`}
          >
            <input
              type="checkbox"
              className="sr-only"
              checked={enabled[type]}
              onChange={() => { setEnabled((e) => ({ ...e, [type]: !e[type] })); setSelected(null); }}
            />
            <span className={`inline-block w-3.5 h-3 rounded-sm ${SWATCH[type]}`} aria-hidden="true" />
            {info.label} ({counts[type] || 0})
          </label>
        ))}
      </div>

      {total === 0 ? (
        <p className="text-sm text-emerald-700 bg-emerald-50 border border-emerald-200 rounded-lg p-3">
          ✓ Nothing flagged. No grammar issues, vague wording or overly long sentences were found.
        </p>
      ) : (
        <>
          <div className="bg-white border border-gray-200 rounded-xl p-4 text-[15px] leading-8 text-gray-800 whitespace-pre-wrap max-h-[420px] overflow-y-auto">
            {pieces.map((piece, i) => {
              if (piece.annotations.length === 0) return <React.Fragment key={i}>{piece.text}</React.Fragment>;
              const top = piece.annotations.find((a) => a.type !== 'long');
              const hasLong = piece.annotations.some((a) => a.type === 'long');
              const label = piece.annotations.map((a) => `${ANNOTATION_TYPES[a.type].label}: ${a.title}`).join('; ');
              return (
                <mark
                  key={i}
                  role="button"
                  tabIndex={0}
                  aria-label={label}
                  title={label}
                  onClick={() => setSelected(i === selected ? null : i)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSelected(i === selected ? null : i); }
                  }}
                  className={`rounded-sm px-0.5 cursor-pointer focus:outline-none focus:ring-2 focus:ring-indigo-400 ${
                    top ? MARK_STYLE[top.type] : 'bg-transparent text-inherit'
                  } ${hasLong ? LONG_STYLE : ''} ${i === selected ? 'ring-2 ring-indigo-500' : ''}`}
                >
                  {piece.text}
                </mark>
              );
            })}
          </div>

          <div className="mt-2 min-h-[2.5rem]" aria-live="polite">
            {active ? (
              <div className="space-y-2">
                {active.annotations.map((a, k) => (
                  <div key={k} className={`text-sm border-l-4 rounded-r-lg p-3 ${DETAIL_STYLE[a.type]}`}>
                    <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">{ANNOTATION_TYPES[a.type].label}</div>
                    <div className="font-semibold text-gray-800">{a.title}</div>
                    {a.detail && <div className="text-gray-700">{a.detail}</div>}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-gray-500">Click or tap a highlight to see what to change.</p>
            )}
          </div>
        </>
      )}

      {unlocated.length > 0 && (
        <p className="text-xs text-gray-500 mt-2">
          {unlocated.length} suggestion{unlocated.length === 1 ? '' : 's'} couldn't be matched to an exact spot in the text.
          They're still listed below.
        </p>
      )}
    </section>
  );
}

import React from 'react';
import { AGREE, LIKERT, SERIES } from '../lib/surveyKit';

// Small, dependency-free charts for survey results. Every mark has a hover
// tooltip and its value is printed, so colour never carries meaning alone.

const pct = (part, whole) => (whole ? Math.round((part / whole) * 100) : 0);

// Donut with a legend that doubles as the data table (count and share).
export function Donut({ data, size = 132 }) {
  const total = data.reduce((s, d) => s + d.value, 0);
  const r = size / 2 - 8;
  const c = size / 2;
  const stroke = 22;
  // Each slice starts where the previous ones end (from 12 o'clock, clockwise);
  // an option keeps its colour slot even when another option has no answers.
  const shown = data.map((d, i) => ({ ...d, color: d.color || SERIES[i % SERIES.length] })).filter((d) => d.value > 0);
  const turn = (v) => (v / total) * Math.PI * 2;
  const arcs = shown.map((d, i) => {
    const start = -Math.PI / 2 + turn(shown.slice(0, i).reduce((s, x) => s + x.value, 0));
    return { ...d, start, end: start + turn(d.value) };
  });
  const point = (a) => [c + r * Math.cos(a), c + r * Math.sin(a)];
  return (
    <div className="flex items-center gap-4">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Pie chart" className="shrink-0">
        <circle cx={c} cy={c} r={r} fill="none" stroke="#f1f0ed" strokeWidth={stroke} />
        {arcs.length === 1 ? (
          <circle cx={c} cy={c} r={r} fill="none" stroke={arcs[0].color} strokeWidth={stroke}>
            <title>{`${arcs[0].label}: ${arcs[0].value} (100%)`}</title>
          </circle>
        ) : arcs.map((a) => {
          const [x1, y1] = point(a.start + 0.012);   // small gap between slices
          const [x2, y2] = point(a.end - 0.012);
          const large = a.end - a.start > Math.PI ? 1 : 0;
          return (
            <path key={a.label} d={`M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`}
              fill="none" stroke={a.color} strokeWidth={stroke}>
              <title>{`${a.label}: ${a.value} (${pct(a.value, total)}%)`}</title>
            </path>
          );
        })}
        <text x={c} y={c - 2} textAnchor="middle" className="fill-gray-800" fontSize="18" fontWeight="700">{total}</text>
        <text x={c} y={c + 14} textAnchor="middle" className="fill-gray-500" fontSize="10">responses</text>
      </svg>
      <ul className="text-sm space-y-1 min-w-0">
        {data.map((d, i) => (
          <li key={d.label} className="flex items-center gap-2">
            <span className="h-3 w-3 rounded-sm shrink-0" style={{ background: d.color || SERIES[i % SERIES.length] }} />
            <span className="text-gray-700 truncate">{d.label}</span>
            <span className="ml-auto pl-2 text-gray-900 font-medium tabular-nums">{d.value}</span>
            <span className="w-10 text-right text-gray-500 tabular-nums">{pct(d.value, total)}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

// Horizontal bars on a fixed scale (1–5 means by default; counts with
// decimals={0}), value printed at the end.
export function MeanBars({ rows, color = SERIES[0], max = 5, decimals = 2 }) {
  const show = (v) => (typeof v === 'number' ? v.toFixed(decimals) : '—');
  return (
    <div className="space-y-2">
      {rows.map((r) => (
        <div key={r.label} className="grid grid-cols-[minmax(7rem,12rem)_1fr_auto] items-center gap-2 text-sm"
          title={`${r.label}: ${show(r.value)}${r.note ? ` ${r.note}` : ''}`}>
          <span className="text-gray-700 truncate">{r.label}</span>
          <div className="h-3 rounded bg-gray-100 overflow-hidden">
            <div className="h-full rounded-r" style={{ width: `${((r.value || 0) / max) * 100}%`, background: r.color || color }} />
          </div>
          <span className="tabular-nums text-gray-900 font-medium w-24 text-right">
            {show(r.value)} {r.note && <span className="text-xs font-normal text-gray-500">{r.note}</span>}
          </span>
        </div>
      ))}
    </div>
  );
}

// Groups (e.g. criteria A, B, C) × series (e.g. AI models): one short bar per
// series inside each group, legend above, values printed.
export function GroupedBars({ groups, series, max = 5 }) {
  return (
    <div>
      <div className="flex flex-wrap gap-4 text-xs text-gray-600 mb-3">
        {series.map((s, i) => (
          <span key={s.name} className="flex items-center gap-1">
            <span className="h-3 w-3 rounded-sm" style={{ background: s.color || SERIES[i] }} /> {s.name}
          </span>
        ))}
      </div>
      <div className="space-y-4">
        {groups.map((g, gi) => (
          <div key={g} className="grid grid-cols-1 sm:grid-cols-[minmax(8rem,14rem)_1fr] gap-2">
            <div className="text-sm font-medium text-gray-800">{g}</div>
            <div className="space-y-1">
              {series.map((s, i) => {
                const v = s.values[gi];
                return (
                  <div key={s.name} className="grid grid-cols-[1fr_3rem] items-center gap-2" title={`${g} · ${s.name}: ${v?.toFixed(2) ?? '—'}`}>
                    <div className="h-3 rounded bg-gray-100 overflow-hidden">
                      <div className="h-full rounded-r" style={{ width: `${((v || 0) / max) * 100}%`, background: s.color || SERIES[i] }} />
                    </div>
                    <span className="text-right text-sm tabular-nums text-gray-900">{v?.toFixed(2) ?? '—'}</span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// 100% stacked bar of answers SD → SA, with 2px gaps; counts in the tooltip.
export function AgreementBar({ counts, n }) {
  return (
    <div className="flex h-3 w-full min-w-[8rem] gap-[2px] rounded overflow-hidden" role="img"
      aria-label={LIKERT.map((l) => `${l.short} ${counts[l.value] || 0}`).join(', ')}>
      {[1, 2, 3, 4, 5].map((v) => (counts[v] ? (
        <div key={v} style={{ width: `${(counts[v] / n) * 100}%`, background: AGREE[v] }}
          title={`${LIKERT.find((l) => l.value === v).label}: ${counts[v]} (${pct(counts[v], n)}%)`} />
      ) : null))}
    </div>
  );
}

export function AgreementLegend() {
  return (
    <div className="flex flex-wrap gap-3 text-xs text-gray-600">
      {[...LIKERT].reverse().map((l) => (
        <span key={l.value} className="flex items-center gap-1">
          <span className="h-3 w-3 rounded-sm" style={{ background: AGREE[l.value] }} /> {l.value} {l.label}
        </span>
      ))}
    </div>
  );
}

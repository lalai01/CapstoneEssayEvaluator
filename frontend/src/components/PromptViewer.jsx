import React, { useMemo, useState } from 'react';
import toast from 'react-hot-toast';
import { EVALUATION_AIS } from '../lib/evaluationAis';

// "Do the three AIs follow one prompt?" Shows the exact prompt each AI
// received for this essay (returned by the backend with its result), checks
// which parts are identical, and highlights the lines that differ from
// DeepSeek's prompt. Every prompt is built by the same function on the
// server: backend/evaluator.py → ai_score_all_criteria (holistic:
// ai_holistic_score).

const REFERENCE = 'deepseek';

// Line-by-line difference (longest common subsequence). Returns rows of
// { type: 'same' | 'added' | 'removed', text } from a's point of view to b.
function diffLines(a, b) {
  const x = a.split('\n');
  const y = b.split('\n');
  const n = x.length;
  const m = y.length;
  const lcs = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i][j] = x[i] === y[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  const rows = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (x[i] === y[j]) { rows.push({ type: 'same', text: x[i] }); i++; j++; }
    else if (lcs[i + 1][j] >= lcs[i][j + 1]) { rows.push({ type: 'removed', text: x[i] }); i++; }
    else { rows.push({ type: 'added', text: y[j] }); j++; }
  }
  while (i < n) rows.push({ type: 'removed', text: x[i++] });
  while (j < m) rows.push({ type: 'added', text: y[j++] });
  return rows;
}

// The prompt's sections (headings in the prompt text), to say which are shared.
const SECTIONS = [
  ['ASSIGNMENT GIVEN BY THE TEACHER', 'Teacher’s assignment (rationale and instructions)'],
  ['RUBRIC', 'Rubric and level descriptors'],
  ['OBSERVED FACTS', 'Observed facts (paragraphs, words, citations, transitions)'],
  ['SCORING RULES', 'Scoring rules'],
  ['HARD RULES', 'Hard rules'],
  ['CALIBRATION EXAMPLES', 'Calibration examples'],
  ['Essay:', 'The essay'],
  ['Return ONLY this JSON object', 'JSON answer format'],
  ['\nRULES:', 'Answer rules'],
];

function sectionText(prompt, heading) {
  const start = prompt.indexOf(heading);
  if (start < 0) return null;
  const next = SECTIONS.map(([h]) => prompt.indexOf(h, start + heading.length)).filter((k) => k > start);
  return prompt.slice(start, next.length ? Math.min(...next) : undefined);
}

export default function PromptViewer({ runs }) {
  const [tab, setTab] = useState(REFERENCE);
  const [showDiff, setShowDiff] = useState(true);

  const prompts = useMemo(() => {
    const out = {};
    for (const ai of EVALUATION_AIS) {
      const p = runs?.[ai.id]?.status === 'done' ? runs[ai.id].scores?.prompt : null;
      if (p?.user) out[ai.id] = p;
    }
    return out;
  }, [runs]);

  if (!runs) return null;
  const available = EVALUATION_AIS.filter((ai) => prompts[ai.id]);
  const running = EVALUATION_AIS.some((ai) => runs[ai.id]?.status === 'running');
  const reference = prompts[REFERENCE] || prompts[available[0]?.id];
  const current = prompts[tab] || reference;
  const currentAi = EVALUATION_AIS.find((ai) => prompts[ai.id] === current);

  const systemSame = available.length > 1 && new Set(available.map((ai) => prompts[ai.id].system)).size === 1;
  const sectionRows = reference
    ? SECTIONS.filter(([h]) => sectionText(reference.user, h) !== null).map(([h, label]) => ({
      label,
      perAi: available.map((ai) => ({ ai, same: sectionText(prompts[ai.id].user, h) === sectionText(reference.user, h) })),
    }))
    : [];
  const diff = current && reference && current !== reference ? diffLines(reference.user, current.user) : null;
  const changed = diff ? diff.filter((r) => r.type !== 'same').length : 0;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(`SYSTEM:\n${current.system}\n\nUSER:\n${current.user}`);
      toast.success('Prompt copied');
    } catch {
      toast.error('Couldn’t copy. Select the text and copy it instead.');
    }
  };

  return (
    <section className="glass-card rounded-2xl shadow-xl p-5 bg-white/95" aria-labelledby="prompt-viewer-title">
      <h2 id="prompt-viewer-title" className="text-lg font-bold text-gray-800">📜 One prompt for all three AIs</h2>
      <p className="text-sm text-gray-600 mt-1">
        Every AI gets its prompt from the same template, built by one function on the server
        (<code className="text-xs bg-gray-100 px-1 rounded">backend/evaluator.py → ai_score_all_criteria</code>),
        with the same rubric, rules and essay, at temperature 0. Below is the exact prompt each AI received for this essay.
      </p>

      {!available.length && (
        <p className="mt-3 text-sm text-gray-500">
          {running ? 'The prompts appear here as each AI finishes…' : 'No prompt to show: the AIs didn’t answer (rule-based scoring doesn’t use a prompt).'}
        </p>
      )}

      {available.length > 0 && (
        <>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[30rem] text-sm">
              <thead>
                <tr className="border-b border-gray-200 text-left text-gray-600">
                  <th className="py-1.5 pr-2 font-semibold">Part of the prompt</th>
                  {available.map((ai) => <th key={ai.id} className="py-1.5 px-2 font-semibold whitespace-nowrap">{ai.icon} {ai.name}</th>)}
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-100">
                  <td className="py-1.5 pr-2">System message (the AI’s role)</td>
                  {available.map((ai) => (
                    <td key={ai.id} className="py-1.5 px-2">{systemSame || available.length === 1 ? '✅ Same' : '⚠️ Differs'}</td>
                  ))}
                </tr>
                {sectionRows.map((row) => (
                  <tr key={row.label} className="border-b border-gray-100">
                    <td className="py-1.5 pr-2">{row.label}</td>
                    {row.perAi.map(({ ai, same }) => (
                      <td key={ai.id} className="py-1.5 px-2">{same ? '✅ Same' : '✂️ Shortened'}</td>
                    ))}
                  </tr>
                ))}
                <tr className="border-b border-gray-100">
                  <td className="py-1.5 pr-2">Essay text read</td>
                  {available.map((ai) => {
                    const p = prompts[ai.id];
                    return (
                      <td key={ai.id} className="py-1.5 px-2 whitespace-nowrap">
                        {p.essay_chars_read >= p.essay_chars_total ? 'All' : `${p.essay_chars_read.toLocaleString()} of ${p.essay_chars_total.toLocaleString()} chars`}
                      </td>
                    );
                  })}
                </tr>
                <tr>
                  <td className="py-1.5 pr-2">Model · temperature</td>
                  {available.map((ai) => (
                    <td key={ai.id} className="py-1.5 px-2 text-xs">{prompts[ai.id].model} · {prompts[ai.id].temperature}</td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>

          {available.some((ai) => prompts[ai.id].variant === 'compact') && (
            <p className="mt-3 rounded-lg border border-violet-200 bg-violet-50 px-3 py-2 text-xs text-violet-900">
              <strong>Why the local AI’s prompt is shorter:</strong> Qwen runs on our server’s CPU, where every word costs
              time. It gets the same prompt with each rubric descriptor cut to its first sentence, the first 4,000
              characters of the essay (about 650 words), and a request for shorter comments. How it scores (rubric,
              facts, scoring rules, hard rules, calibration) is the same.
            </p>
          )}

          <div className="mt-4 flex flex-wrap items-center gap-2" role="tablist" aria-label="Prompt of each AI">
            {available.map((ai) => (
              <button
                key={ai.id}
                type="button"
                role="tab"
                aria-selected={prompts[ai.id] === current}
                onClick={() => setTab(ai.id)}
                className={`rounded-lg border px-3 py-1.5 text-sm ${prompts[ai.id] === current ? 'border-indigo-500 bg-indigo-50 font-semibold text-indigo-800' : 'border-gray-200 bg-white text-gray-700 hover:bg-gray-50'}`}
              >
                {ai.icon} {ai.name}
              </button>
            ))}
            <button type="button" onClick={copy} className="ml-auto text-sm text-indigo-700 underline">Copy this prompt</button>
          </div>

          {current && (
            <div className="mt-3 space-y-3">
              {currentAi && currentAi.id !== REFERENCE && prompts[REFERENCE] && (
                <div className="flex flex-wrap items-center gap-3 text-sm">
                  <span className={changed ? 'text-amber-800' : 'text-emerald-800'}>
                    {changed
                      ? `Compared with DeepSeek’s prompt: ${changed} line${changed === 1 ? '' : 's'} differ.`
                      : '✅ Identical to DeepSeek’s prompt, word for word.'}
                  </span>
                  {changed > 0 && (
                    <label className="flex items-center gap-1.5 text-gray-700">
                      <input type="checkbox" checked={showDiff} onChange={(e) => setShowDiff(e.target.checked)} />
                      Highlight differences
                    </label>
                  )}
                </div>
              )}
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500">System message</h3>
                <pre className="mt-1 whitespace-pre-wrap break-words rounded-lg bg-gray-900 p-3 text-xs text-gray-100">{current.system}</pre>
              </div>
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500">User message (rubric, facts, rules, essay, answer format)</h3>
                <pre className="mt-1 max-h-[28rem] overflow-auto whitespace-pre-wrap break-words rounded-lg bg-gray-900 p-3 text-xs text-gray-100">
                  {diff && showDiff
                    ? diff.map((r, k) => (
                      <div
                        key={k}
                        className={r.type === 'added' ? 'bg-emerald-800/70' : r.type === 'removed' ? 'bg-red-900/60 line-through opacity-80' : ''}
                      >
                        {r.type === 'added' ? '+ ' : r.type === 'removed' ? '− ' : '  '}{r.text || ' '}
                      </div>
                    ))
                    : current.user}
                </pre>
                {diff && showDiff && changed > 0 && (
                  <p className="mt-1 text-xs text-gray-500">
                    <span className="rounded bg-red-100 px-1 text-red-800">− red</span> only in DeepSeek’s prompt ·{' '}
                    <span className="rounded bg-emerald-100 px-1 text-emerald-800">+ green</span> only in {currentAi?.name}’s prompt · the rest is identical.
                  </p>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}

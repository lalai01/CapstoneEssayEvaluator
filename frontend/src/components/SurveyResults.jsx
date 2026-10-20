import React, { useMemo, useState } from 'react';
import toast from 'react-hot-toast';
import { testPrompt } from '../api';
import { AgreementBar, AgreementLegend, Donut, GroupedBars, MeanBars } from './SurveyCharts';
import {
  SERIES, alphaLabel, analyzeSurvey, downloadText, isHeading, kindOf, matrixItems, parseMatrix, questionCode, surveyCsv,
} from '../lib/surveyKit';

const f2 = (v) => (typeof v === 'number' && !Number.isNaN(v) ? v.toFixed(2) : '—');
// Sequential blue for a 1–5 mean in the respondent table (value printed too).
const shade = (v) => (typeof v === 'number' ? { background: `rgba(42, 120, 214, ${0.06 + ((v - 1) / 4) * 0.5})` } : {});

function Card({ title, subtitle, children, actions }) {
  return (
    <section className="bg-white border rounded-2xl p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-2 mb-4">
        <div>
          <h3 className="text-lg font-bold text-gray-800">{title}</h3>
          {subtitle && <p className="text-xs text-gray-500 mt-0.5">{subtitle}</p>}
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}

function Tile({ label, value, note }) {
  return (
    <div className="bg-white border rounded-xl p-4">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-2xl font-bold text-gray-900 tabular-nums">{value}</div>
      {note && <div className="text-xs text-gray-500 mt-0.5">{note}</div>}
    </div>
  );
}

// Plain-text summary of the results, for the AI-written conclusion.
function describe(survey, A) {
  const lines = [`Survey: ${survey.title}`, `Respondents: ${A.respondents.length}`];
  for (const p of A.profile) {
    if (p.type === 'number') lines.push(`${p.q.question}: mean ${f2(p.mean)}, SD ${f2(p.sd)}, range ${p.min}-${p.max}`);
    else lines.push(`${p.q.question}: ${p.counts.map((c) => `${c.label} ${c.value}`).join(', ')}`);
  }
  for (const g of A.matrixGroups) {
    lines.push(`${g.title}: ${A.models.map((m) => `${m} mean ${f2(g.byModel[m].mean)} (${g.byModel[m].interpretation})`).join('; ')}`);
  }
  if (A.models.length) {
    lines.push(`Overall per AI model: ${A.models.map((m) => `${m} ${f2(A.modelOverall[m].mean)} (${A.modelOverall[m].interpretation})`).join('; ')}`);
  }
  for (const g of A.likertGroups) {
    lines.push(`${g.title}: mean ${f2(g.construct.mean)}, SD ${f2(g.construct.sd)} (${g.construct.interpretation}), Cronbach alpha ${f2(g.construct.alpha)}`);
  }
  for (const o of A.openEnded) {
    lines.push(`Open-ended "${o.q.question}": ${o.answers.map((a) => a.text.replace(/\s+/g, ' ').slice(0, 200)).join(' | ')}`);
  }
  return lines.join('\n');
}

export default function SurveyResults({ survey, questions, responses, userNames, onRefresh }) {
  const A = useMemo(() => analyzeSurvey(questions, responses, userNames), [questions, responses, userNames]);
  const [openId, setOpenId] = useState(null);
  const [conclusion, setConclusion] = useState('');
  const [concluding, setConcluding] = useState(false);

  if (!A.respondents.length) {
    return <p className="text-gray-500">No responses yet. Results appear here automatically once people submit the survey.</p>;
  }

  const last = A.respondents.reduce((a, r) => (new Date(r.submitted) > new Date(a) ? r.submitted : a), A.respondents[0].submitted);
  const slug = (survey.title || 'survey').replace(/[^a-z0-9]+/gi, '-').toLowerCase().slice(0, 40);
  const exportCsv = () => {
    const { data } = surveyCsv(A);
    downloadText(`${slug}-responses.csv`, data);
  };
  const exportCodebook = () => downloadText(`${slug}-codebook.csv`, surveyCsv(A).codebook);

  const conclude = async () => {
    setConcluding(true);
    try {
      const r = await testPrompt('groq',
        'You are a research data analyst writing for an academic thesis. Write a concise, formal interpretation of the survey results: describe the respondents, compare the AI models, interpret each UTAUT construct (means use the scale 4.21-5.00 Strongly Agree, 3.41-4.20 Agree, 2.61-3.40 Neutral, 1.81-2.60 Disagree, 1.00-1.80 Strongly Disagree), mention reliability (Cronbach alpha), and summarise the open-ended themes. Do not invent numbers.',
        describe(survey, A));
      setConclusion(r.result?.text || 'Could not generate a conclusion.');
    } catch {
      toast.error('Could not generate the conclusion. Try again.');
    } finally {
      setConcluding(false);
    }
  };

  const ranked = [...A.models].sort((a, b) => (A.modelOverall[b].mean ?? 0) - (A.modelOverall[a].mean ?? 0));
  const modelColor = (m) => SERIES[A.models.indexOf(m) % SERIES.length];
  const parts = [...new Set(A.likertGroups.map((g) => g.part))];
  const open = A.perRespondent.find((r) => r.id === openId);

  return (
    <div className="space-y-6">
      {/* ---------- Overview ---------- */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Tile label="Respondents" value={A.respondents.length} />
        {A.consent && <Tile label="Gave consent" value={`${A.consent.agreed}/${A.consent.n}`} />}
        <Tile label="Latest response" value={new Date(last).toLocaleDateString()} />
        {A.models.length > 0 && (
          <Tile label="Highest-rated AI" value={ranked[0]} note={`mean ${f2(A.modelOverall[ranked[0]].mean)} · ${A.modelOverall[ranked[0]].interpretation}`} />
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        <button onClick={onRefresh} className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-1.5 rounded-lg text-sm">🔄 Refresh</button>
        <button onClick={exportCsv} className="bg-emerald-600 hover:bg-emerald-700 text-white px-3 py-1.5 rounded-lg text-sm">⬇ Responses (CSV for Excel/SPSS)</button>
        <button onClick={exportCodebook} className="bg-white border hover:bg-gray-50 text-gray-700 px-3 py-1.5 rounded-lg text-sm">⬇ Codebook (CSV)</button>
        <button onClick={conclude} disabled={concluding} className="bg-purple-600 hover:bg-purple-700 text-white px-3 py-1.5 rounded-lg text-sm disabled:opacity-50">
          {concluding ? 'Writing…' : '🤖 AI interpretation of the results'}
        </button>
      </div>
      {conclusion && (
        <div className="bg-purple-50 border border-purple-200 rounded-xl p-4">
          <h4 className="font-semibold text-purple-800 mb-2">📘 AI-written interpretation <span className="font-normal text-xs text-purple-700">(check every statement against the tables below before using it)</span></h4>
          <p className="text-sm text-gray-700 whitespace-pre-wrap">{conclusion}</p>
        </div>
      )}

      {/* ---------- Profile ---------- */}
      {A.profile.length > 0 && (
        <Card title="Respondent profile">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {A.profile.map((p) => (
              <div key={p.q.id}>
                <h4 className="text-sm font-semibold text-gray-700 mb-2">{p.q.question}</h4>
                {p.type === 'choice' && <Donut data={p.counts} />}
                {p.type === 'number' && (
                  <div className="grid grid-cols-3 gap-2">
                    <Tile label="Mean" value={f2(p.mean)} note={`SD ${f2(p.sd)}`} />
                    <Tile label="Youngest" value={p.min ?? '—'} />
                    <Tile label="Oldest" value={p.max ?? '—'} />
                  </div>
                )}
                {p.type === 'text' && (
                  <MeanBars decimals={0} max={Math.max(...p.counts.map((c) => c.value), 1)}
                    rows={p.counts.map((c) => ({ label: c.label, value: c.value, note: c.value === 1 ? 'response' : 'responses' }))} />
                )}
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* ---------- AI models (matrix questions) ---------- */}
      {A.matrixGroups.length > 0 && (
        <Card title={A.matrixGroups[0].part || 'Ratings per AI model'}
          subtitle="Mean of 1–5 ratings. SD = standard deviation across respondents; α = Cronbach's alpha (internal consistency).">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-5">
            {ranked.map((m, i) => (
              <div key={m} className="rounded-xl border p-3">
                <div className="flex items-center gap-2 text-sm font-semibold text-gray-800">
                  <span className="h-3 w-3 rounded-sm" style={{ background: modelColor(m) }} /> {m}
                  <span className="ml-auto text-xs text-gray-500">#{i + 1}</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 tabular-nums mt-1">{f2(A.modelOverall[m].mean)}</div>
                <div className="text-xs text-gray-500">SD {f2(A.modelOverall[m].sd)} · {A.modelOverall[m].interpretation}</div>
              </div>
            ))}
          </div>
          <GroupedBars
            groups={[...A.matrixGroups.map((g) => g.title), 'Overall']}
            series={A.models.map((m) => ({
              name: m, color: modelColor(m),
              values: [...A.matrixGroups.map((g) => g.byModel[m].mean), A.modelOverall[m].mean],
            }))}
          />
          <div className="mt-6 overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-gray-500 border-b">
                  <th className="py-2 pr-2">Code</th>
                  <th className="py-2 pr-2">Statement</th>
                  {A.models.map((m) => <th key={m} className="py-2 px-2 text-center whitespace-nowrap">{m}<div className="font-normal">mean (SD)</div></th>)}
                </tr>
              </thead>
              <tbody>
                {A.matrixGroups.map((g) => (
                  <React.Fragment key={g.key}>
                    <tr className="bg-indigo-50/60 font-semibold text-gray-800">
                      <td className="py-2 pr-2" colSpan={2}>{g.title}</td>
                      {A.models.map((m) => (
                        <td key={m} className="py-2 px-2 text-center tabular-nums">
                          {f2(g.byModel[m].mean)} ({f2(g.byModel[m].sd)})
                          <div className="text-xs font-normal text-gray-600">{g.byModel[m].interpretation} · α {f2(g.byModel[m].alpha)}</div>
                        </td>
                      ))}
                    </tr>
                    {g.items.map((it) => (
                      <tr key={it.q.id} className="border-b last:border-0 align-top">
                        <td className="py-2 pr-2 text-gray-500 whitespace-nowrap">{it.code}</td>
                        <td className="py-2 pr-2 text-gray-700">{it.q.question}</td>
                        {A.models.map((m) => (
                          <td key={m} className="py-2 px-2 text-center tabular-nums" title={it.byModel[m].interpretation}>
                            {f2(it.byModel[m].mean)} <span className="text-gray-500">({f2(it.byModel[m].sd)})</span>
                          </td>
                        ))}
                      </tr>
                    ))}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ---------- Likert constructs (e.g. UTAUT) ---------- */}
      {parts.map((part) => {
        const groups = A.likertGroups.filter((g) => g.part === part);
        return (
          <Card key={part || 'ratings'} title={part || 'Ratings'}
            subtitle="Construct mean = average of each respondent's item mean. Interpretation: 4.21–5.00 Strongly Agree · 3.41–4.20 Agree · 2.61–3.40 Neutral · 1.81–2.60 Disagree · 1.00–1.80 Strongly Disagree.">
            <MeanBars rows={groups.map((g) => ({ label: g.title, value: g.construct.mean, note: g.construct.interpretation }))} />
            <div className="mt-5 mb-2"><AgreementLegend /></div>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-gray-500 border-b">
                    <th className="py-2 pr-2">Code</th>
                    <th className="py-2 pr-2">Statement</th>
                    <th className="py-2 px-2 text-right">Mean</th>
                    <th className="py-2 px-2 text-right">SD</th>
                    <th className="py-2 px-2">Interpretation</th>
                    <th className="py-2 pl-2 w-40">Answers</th>
                  </tr>
                </thead>
                <tbody>
                  {groups.map((g) => (
                    <React.Fragment key={g.key}>
                      <tr className="bg-indigo-50/60 font-semibold text-gray-800">
                        <td className="py-2 pr-2" colSpan={2}>{g.title}</td>
                        <td className="py-2 px-2 text-right tabular-nums">{f2(g.construct.mean)}</td>
                        <td className="py-2 px-2 text-right tabular-nums">{f2(g.construct.sd)}</td>
                        <td className="py-2 px-2">{g.construct.interpretation}</td>
                        <td className="py-2 pl-2 text-xs font-normal text-gray-600">α {f2(g.construct.alpha)} · {alphaLabel(g.construct.alpha)}</td>
                      </tr>
                      {g.items.map((it) => (
                        <tr key={it.q.id} className="border-b last:border-0 align-top">
                          <td className="py-2 pr-2 text-gray-500 whitespace-nowrap">{it.code}</td>
                          <td className="py-2 pr-2 text-gray-700">{it.q.question}</td>
                          <td className="py-2 px-2 text-right tabular-nums">{f2(it.stats.mean)}</td>
                          <td className="py-2 px-2 text-right tabular-nums text-gray-600">{f2(it.stats.sd)}</td>
                          <td className="py-2 px-2 text-gray-700">{it.stats.interpretation}</td>
                          <td className="py-2 pl-2"><AgreementBar counts={it.stats.counts} n={it.stats.n} /></td>
                        </tr>
                      ))}
                    </React.Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        );
      })}

      {/* ---------- Compare respondents ---------- */}
      <Card title="Compare respondents" subtitle="Each respondent's average per construct and per AI model (1–5; darker = higher). Click a row to see all of that person's answers.">
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-gray-500 border-b">
                <th className="py-2 pr-2">Code</th>
                <th className="py-2 pr-2">Name</th>
                {A.models.map((m) => <th key={m} className="py-2 px-2 text-center whitespace-nowrap">{m}</th>)}
                {A.likertGroups.map((g) => <th key={g.key} className="py-2 px-2 text-center" title={g.title}>{g.title.replace(/^[A-Z]\.\s*/, '')}</th>)}
              </tr>
            </thead>
            <tbody>
              {A.perRespondent.map((r) => (
                <tr key={r.id} onClick={() => setOpenId(openId === r.id ? null : r.id)}
                  className={`border-b cursor-pointer hover:bg-gray-50 ${openId === r.id ? 'bg-indigo-50' : ''}`}>
                  <td className="py-2 pr-2 text-gray-500">{r.code}</td>
                  <td className="py-2 pr-2 text-gray-800">{r.name}</td>
                  {A.models.map((m) => <td key={m} className="py-2 px-2 text-center tabular-nums" style={shade(r.models[m])}>{f2(r.models[m])}</td>)}
                  {A.likertGroups.map((g) => <td key={g.key} className="py-2 px-2 text-center tabular-nums" style={shade(r.constructs[g.key])}>{f2(r.constructs[g.key])}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {open && (
          <div className="mt-4 rounded-xl border border-indigo-200 bg-indigo-50/40 p-4">
            <div className="flex justify-between items-center mb-2">
              <h4 className="font-semibold text-gray-800">{open.code} · {open.name}</h4>
              <button onClick={() => setOpenId(null)} className="text-sm text-indigo-700 underline">Close</button>
            </div>
            <div className="space-y-1 text-sm max-h-96 overflow-y-auto">
              {A.items.filter((it) => !isHeading(it.q)).map((it) => {
                const a = open.answers[it.q.id];
                return (
                  <div key={it.q.id} className="grid grid-cols-[4rem_1fr] gap-2 border-b border-indigo-100 py-1">
                    <span className="text-gray-500">{questionCode(it)}</span>
                    <span>
                      <span className="text-gray-600">{it.q.question}: </span>
                      <strong className="text-gray-900 whitespace-pre-wrap">
                        {kindOf(it.q) === 'matrix'
                          ? matrixItems(it.q).map((m) => `${m} ${parseMatrix(a)[m] ?? '—'}`).join(' · ')
                          : a || '—'}
                      </strong>
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </Card>

      {/* ---------- Open-ended ---------- */}
      {A.openEnded.length > 0 && (
        <Card title="Open-ended answers">
          <div className="space-y-5">
            {A.openEnded.map((o) => (
              <div key={o.q.id}>
                <h4 className="text-sm font-semibold text-gray-800 mb-2">{o.q.question}</h4>
                {o.answers.length === 0 ? <p className="text-sm text-gray-500">No answers.</p> : (
                  <ul className="space-y-2">
                    {o.answers.map((a) => (
                      <li key={a.code} className="text-sm bg-gray-50 border rounded-lg p-3">
                        <span className="text-xs text-gray-500 mr-2">{a.code}</span>
                        <span className="whitespace-pre-wrap text-gray-800">{a.text}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

// Structured surveys (e.g. the Final Questionnaire) on top of the plain
// survey tables. A question's kind is its question_type, refined by a marker
// in options[0] so no database change is needed:
//   ::section / ::subsection  heading (options[1] = description), no answer
//   ::likert                  1-5 agreement scale (question_type 'rating')
//   ::matrix                  one 1-5 rating per item in options[1..]
//                             (e.g. per AI model); answer = JSON {item: score}
//   ::number                  numeric short answer (e.g. age)
//   ::file                    attachment, no answer: options = ['::file', path, name, type]
// Each section is shown as a tab (page) of the survey.
import { EVALUATION_AIS } from './evaluationAis';

export const LIKERT = [
  { value: 5, short: 'SA', label: 'Strongly Agree' },
  { value: 4, short: 'A', label: 'Agree' },
  { value: 3, short: 'N', label: 'Neutral' },
  { value: 2, short: 'D', label: 'Disagree' },
  { value: 1, short: 'SD', label: 'Strongly Disagree' },
];

// Chart colours. SERIES: validated categorical order (an option or AI model
// keeps its slot). AGREE: diverging red – gray – blue for 1 (SD) … 5 (SA).
export const SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];
export const AGREE = { 1: '#e34948', 2: '#f4a7a3', 3: '#cfccc6', 4: '#86b6ef', 5: '#2a78d6' };

export function kindOf(q) {
  const marker = Array.isArray(q?.options) && typeof q.options[0] === 'string' && q.options[0].startsWith('::')
    ? q.options[0].slice(2) : null;
  return marker || q?.question_type || 'text';
}

export const isHeading = (q) => ['section', 'subsection'].includes(kindOf(q));
// Items with nothing to answer: headings and attachments.
export const isDisplayOnly = (q) => ['section', 'subsection', 'file'].includes(kindOf(q));
export const fileInfo = (q) => (kindOf(q) === 'file'
  ? { path: q.options[1], name: q.options[2] || 'Attachment', type: q.options[3] || '' } : null);

// The survey split into tabs: one per section (items before the first
// section form an untitled first tab). Each tab: { section, items }.
export function tabsOf(questions) {
  const tabs = [];
  for (const item of structure(questions)) {
    if (item.kind === 'section' || !tabs.length) tabs.push({ section: item.kind === 'section' ? item.q : null, items: [] });
    if (item.kind !== 'section') tabs[tabs.length - 1].items.push(item);
  }
  return tabs.filter((t) => t.section || t.items.length);
}
export const headingText = (q) => (isHeading(q) ? q.options?.[1] || '' : '');
export const matrixItems = (q) => (kindOf(q) === 'matrix' ? q.options.slice(1) : []);

export function parseMatrix(answer) {
  try {
    const v = JSON.parse(answer || '{}');
    return v && typeof v === 'object' ? v : {};
  } catch {
    return {};
  }
}

// Editor kinds. Those after 'rating' are stored on the existing question types
// with a marker in options[0], so no database change is needed.
export const QUESTION_TYPES = [
  { value: 'text', label: 'Short Text' },
  { value: 'essay', label: 'Essay (Long Text)' },
  { value: 'radio', label: 'Multiple choice (one answer)' },
  { value: 'checkbox', label: 'Checkboxes (any number)' },
  { value: 'rating', label: 'Rating (1‑5 stars)' },
  { value: 'likert', label: 'Agreement scale (5 SA … 1 SD)' },
  { value: 'matrix', label: 'Agreement scale per AI model (grid)' },
  { value: 'number', label: 'Number (e.g. age)' },
  { value: 'subsection', label: 'Sub-heading (groups items, e.g. a construct)' },
];
export const kindLabel = (q) => (kindOf(q) === 'file' ? 'Attachment'
  : kindOf(q) === 'section' ? 'Tab' : QUESTION_TYPES.find((t) => t.value === kindOf(q))?.label || q.question_type);

// Editor kind + options text -> stored question_type and options.
export function storedShape(kind, optionsText = '') {
  const list = optionsText.split(',').map((x) => x.trim()).filter(Boolean);
  switch (kind) {
    case 'radio': case 'checkbox': return { question_type: kind, options: list };
    case 'rating': return { question_type: 'rating', options: ['1', '2', '3', '4', '5'] };
    case 'likert': return { question_type: 'rating', options: ['::likert'] };
    case 'matrix': return { question_type: 'rating', options: ['::matrix', ...list] };
    case 'number': return { question_type: 'text', options: ['::number'] };
    case 'section': case 'subsection': return { question_type: 'text', options: [`::${kind}`, optionsText.trim()] };
    default: return { question_type: kind, options: null };
  }
}

// The editor's options text for a stored question (reverse of storedShape).
export function optionsText(q) {
  const kind = kindOf(q);
  if (isHeading(q)) return headingText(q);
  if (kind === 'matrix') return matrixItems(q).join(', ');
  return (q.options || []).filter((o) => !String(o).startsWith('::')).join(', ');
}

// A copy of a question for re-inserting (duplicating a tab or a survey).
export const copyOf = (q) => ({ question: q.question, question_type: q.question_type, options: q.options, is_required: q.is_required });

// Is this question answered (enough to submit)?
export function isAnswered(q, value) {
  const kind = kindOf(q);
  if (kind === 'matrix') return matrixItems(q).every((m) => Number(parseMatrix(value)[m]) >= 1);
  if (q.question_type === 'checkbox') return (value || '').split(',').filter(Boolean).length > 0;
  if (kind === 'number') return value !== undefined && value !== '' && Number(value) > 0;
  return value !== undefined && String(value).trim() !== '';
}

// Questions in order, each with the section/subsection it belongs to and its
// number inside that subsection (headings aren't numbered).
export function structure(questions) {
  let section = null;
  let subsection = null;
  let n = 0;
  return [...questions]
    .sort((a, b) => a.order_number - b.order_number)
    .map((q) => {
      const kind = kindOf(q);
      if (kind === 'section') { section = q; subsection = null; n = 0; return { q, kind, section, subsection, number: null }; }
      if (kind === 'subsection') { subsection = q; n = 0; return { q, kind, section, subsection, number: null }; }
      if (kind === 'file') return { q, kind, section, subsection, number: null };
      n += 1;
      return { q, kind, section, subsection, number: n };
    });
}

// ---------- statistics ----------
export const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);

export function sd(xs) {
  if (xs.length < 2) return null;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1));   // sample SD
}

// The usual verbal interpretation of a 5-point Likert mean.
export function interpret(m) {
  if (m === null || m === undefined || Number.isNaN(m)) return '—';
  if (m >= 4.21) return 'Strongly Agree';
  if (m >= 3.41) return 'Agree';
  if (m >= 2.61) return 'Neutral';
  if (m >= 1.81) return 'Disagree';
  return 'Strongly Disagree';
}

// Cronbach's alpha for a set of items, from respondents who answered all of
// them (rows: one array of item scores per respondent). Needs >= 2 items and
// >= 2 respondents; null otherwise or when there is no variance.
export function cronbachAlpha(rows) {
  const complete = rows.filter((r) => r.every((v) => typeof v === 'number'));
  const k = complete[0]?.length || 0;
  if (k < 2 || complete.length < 2) return null;
  const variance = (xs) => {
    const m = mean(xs);
    return xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1);
  };
  const itemVar = Array.from({ length: k }, (_, i) => variance(complete.map((r) => r[i])));
  const totalVar = variance(complete.map((r) => r.reduce((a, b) => a + b, 0)));
  if (!totalVar) return null;
  return (k / (k - 1)) * (1 - itemVar.reduce((a, b) => a + b, 0) / totalVar);
}

export function alphaLabel(a) {
  if (a === null) return 'n/a';
  if (a >= 0.9) return 'Excellent';
  if (a >= 0.8) return 'Good';
  if (a >= 0.7) return 'Acceptable';
  if (a >= 0.6) return 'Questionable';
  if (a >= 0.5) return 'Poor';
  return 'Unacceptable';
}

// ---------- CSV export (one row per respondent) ----------
export function toCsv(rows) {
  const cell = (v) => {
    const s = v === null || v === undefined ? '' : String(v);
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return rows.map((r) => r.map(cell).join(',')).join('\r\n');
}

export function downloadText(filename, text, type = 'text/csv') {
  const blob = new Blob(['﻿' + text], { type: `${type};charset=utf-8` });   // BOM: Excel reads UTF-8
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------- The Final Questionnaire (from "Final Questionnaire.docx") ----------
// Part II rates each AI used by the system; the names follow the Evaluate page.
export const QUESTIONNAIRE_MODELS = EVALUATION_AIS.map((ai) => (ai.local ? `${ai.name} (local)` : ai.name));

const section = (title, description = '') => ({ question: title, question_type: 'text', options: ['::section', description], is_required: false });
const subsection = (title, description = '') => ({ question: title, question_type: 'text', options: ['::subsection', description], is_required: false });
const likert = (text) => ({ question: text, question_type: 'rating', options: ['::likert'], is_required: true });
const matrix = (text, models) => ({ question: text, question_type: 'rating', options: ['::matrix', ...models], is_required: true });

export function finalQuestionnaire(models = QUESTIONNAIRE_MODELS) {
  const modelList = models.join(', ').replace(/, ([^,]*)$/, ', and $1');
  return {
    title: 'Final Questionnaire: Rubric-Based AI Automated Essay Evaluation System',
    description: 'Evaluation of the AI-generated assessments and of the system as a whole (UTAUT). About 15 minutes.',
    questions: [
      section('Confidentiality and Voluntary Participation',
        'The information, materials, and system presented in this evaluation are confidential and are intended solely for academic research purposes. By participating in this study, you agree not to disclose, reproduce, distribute, or share any part of the system, evaluation process, or related information without the prior written consent of the researcher.\n\nYour participation is voluntary. Your responses will be treated confidentially and used only for research purposes. Results will be reported in aggregate form, and personally identifiable information will not be included in the research report.'),
      { question: 'Consent', question_type: 'checkbox', is_required: true,
        options: ['I have read and understood the information above and voluntarily agree to participate in this study.'] },

      section('Part I. Respondent Profile'),
      { question: 'Name', question_type: 'text', options: null, is_required: true },
      { question: 'Age', question_type: 'text', options: ['::number'], is_required: true },
      { question: 'Sex', question_type: 'radio', options: ['Male', 'Female', 'Prefer not to say'], is_required: true },
      { question: 'Teaching Experience', question_type: 'radio', options: ['Less than 5 years', '5–10 years', '11–15 years', 'More than 15 years'], is_required: true },
      { question: 'Department/College', question_type: 'text', options: null, is_required: true },
      { question: 'Previous experience using AI tools for educational or assessment purposes', question_type: 'radio', options: ['Yes', 'No'], is_required: true },

      section('Part II. AI-Generated Assessment Evaluation',
        `You will review four existing college-student argumentative essays. First, evaluate each essay independently using the standardized analytic rubric. After your manual evaluation, the same essays will be processed by ${modelList} using the developed system.\n\nRate each AI model separately based on your overall review of its generated scores, explanations, and feedback across the four essays. Do not rate the models against one another; rate each model according to the extent to which each statement describes its outputs.\n\nResponse scale: 5 – Strongly Agree (SA); 4 – Agree (A); 3 – Neutral (N); 2 – Disagree (D); 1 – Strongly Disagree (SD).`),
      subsection('A. Instructional Clarity and Actionability'),
      matrix('The feedback explicitly pinpoints specific rubric criteria where the student met or failed to meet expectations.', models),
      matrix('The feedback cites direct evidence or quotes from the essay to justify its comments.', models),
      matrix('The feedback provides practical next-step suggestions that can be applied to improve the essay.', models),
      matrix('The feedback is written in language that is clear and understandable for the intended student.', models),
      matrix('The feedback focuses on improvement and learning rather than only pointing out errors.', models),
      subsection('B. Rubric Alignment and Fairness'),
      matrix('The AI-generated assessment explicitly follows the teacher-defined rubric criteria.', models),
      matrix('The qualitative comments are logically consistent with the assigned numerical scores.', models),
      matrix('The assessment applies the rubric criteria in a fair and impartial manner.', models),
      matrix('The assessment evaluates all rubric dimensions proportionally without overemphasizing a single criterion.', models),
      matrix('The assessment is free from apparent bias in its scoring and comments.', models),
      subsection('C. Technical Accuracy and Reliability'),
      matrix("The AI-generated assessment accurately reflects information contained in the student's essay.", models),
      matrix('The AI-generated assessment correctly identifies relevant strengths and weaknesses in the essay.', models),
      matrix('The generated evaluation reflects the complete essay content without ignoring key sections.', models),
      matrix('The AI-generated assessment addresses the required aspects of the essay without significant omissions.', models),
      matrix('The AI-generated assessment is sufficiently consistent and reliable for use as support in essay evaluation.', models),

      section('Part III. Unified Theory of Acceptance and Use of Technology (UTAUT)',
        'Please evaluate your overall experience with the developed Rubric-Based AI Automated Essay Evaluation System. These statements refer to the system as a whole and not to any specific AI model.'),
      subsection('A. Performance Expectancy'),
      likert('Using the system improves my efficiency in evaluating essays.'),
      likert('The system enables me to evaluate essays more quickly.'),
      likert('The system enhances my effectiveness in assessing essays.'),
      likert('I find the system useful in my essay evaluation tasks.'),
      subsection('B. Effort Expectancy'),
      likert('Learning to use the system is easy for me.'),
      likert('Interacting with the system is clear and understandable.'),
      likert('I find the system easy to use.'),
      likert('It is easy for me to become skillful at using the system.'),
      subsection('C. Social Influence'),
      likert('People who influence my teaching would encourage me to use this system.'),
      likert('My colleagues would recommend using this system.'),
      likert('My institution would support the use of this system.'),
      likert('Academic leadership in my department would value my use of this system.'),
      subsection('D. Facilitating Conditions'),
      likert('I have the resources necessary to use the system.'),
      likert('I have the knowledge necessary to use the system.'),
      likert('Technical support would be available when needed.'),
      likert('The system is compatible with my existing essay grading workflow.'),
      subsection('E. Behavioral Intention'),
      likert('I intend to use this system for future essay assessments.'),
      likert('I would continue using this system if it became available for regular use.'),
      likert('I would recommend this system to other educators.'),

      section('Part IV. Open-Ended Questions'),
      { question: 'What features of the AI Essay Evaluation System did you find most helpful?', question_type: 'essay', options: null, is_required: true },
      { question: 'What challenges or difficulties did you encounter while using the system or reviewing its generated assessments?', question_type: 'essay', options: null, is_required: true },
      { question: 'What suggestions do you have for improving the AI Essay Evaluation System?', question_type: 'essay', options: null, is_required: true },
    ],
  };
}

// ---------- Analysis of all responses ----------
const ROMAN = /\bPart\s+([IVX]+)\b/i;
const LETTER = /^([A-Z])\.\s/;

// Short code for a question, e.g. "II.A1" or "III.B3" (tables and CSV).
export function questionCode(item) {
  const part = item.section?.question.match(ROMAN)?.[1] || '';
  const letter = item.subsection?.question.match(LETTER)?.[1] || '';
  return `${part ? `${part}.` : ''}${letter}${item.number ?? ''}` || String(item.q.id);
}

const isNameQuestion = (q) => /^(full\s+)?name$/i.test(q.question.trim());
const isConsent = (q) => q.question_type === 'checkbox' && (q.options || []).length === 1;
const scoreOf = (v) => {
  const n = Number(v);
  return n >= 1 && n <= 5 ? n : null;
};
const present = (xs) => xs.filter((v) => v !== null && v !== undefined);

function stats(values) {
  const xs = present(values);
  const counts = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
  xs.forEach((v) => { counts[v] += 1; });
  const m = mean(xs);
  return { n: xs.length, mean: m, sd: sd(xs), counts, interpretation: interpret(m) };
}

// Mean, SD and alpha over respondents, from one row of item scores each.
function scaleStats(rows) {
  const personMeans = present(rows.map((row) => mean(present(row))));
  const m = mean(personMeans);
  return { n: personMeans.length, mean: m, sd: sd(personMeans), interpretation: interpret(m), alpha: cronbachAlpha(rows) };
}

/**
 * questions: survey_questions rows; responses: survey_responses rows
 * (user_id, question_id, answer, created_at); userNames: {user_id: name}.
 */
export function analyzeSurvey(questions, responses, userNames = {}) {
  const items = structure(questions);
  const byUser = new Map();
  for (const r of [...responses].sort((a, b) => new Date(a.created_at) - new Date(b.created_at))) {
    if (!byUser.has(r.user_id)) byUser.set(r.user_id, { id: r.user_id, submitted: r.created_at, answers: {} });
    byUser.get(r.user_id).answers[r.question_id] = r.answer;
  }
  const nameQ = items.find((it) => !isDisplayOnly(it.q) && isNameQuestion(it.q))?.q;
  const respondents = [...byUser.values()].map((r, i) => ({
    ...r,
    code: `R${i + 1}`,
    name: (nameQ && r.answers[nameQ.id]) || userNames[r.id] || `Respondent ${i + 1}`,
  }));
  const answersTo = (q) => respondents.map((r) => r.answers[q.id]);

  const profile = [];
  const openEnded = [];
  let consent = null;
  const groups = new Map();   // one group of rated items per subsection (or section)

  const groupFor = (item, kind) => {
    const head = item.subsection || item.section;
    const key = `${kind}:${head?.id ?? 'none'}`;
    if (!groups.has(key)) {
      groups.set(key, { key, kind, title: head?.question || 'Ratings', part: item.section?.question || '', items: [] });
    }
    return groups.get(key);
  };

  for (const item of items) {
    const { q, kind } = item;
    if (isDisplayOnly(q)) continue;
    if (isConsent(q)) {
      consent = { q, agreed: answersTo(q).filter((a) => (a || '').trim()).length, n: respondents.length };
    } else if (kind === 'matrix') {
      groupFor(item, 'matrix').items.push({ ...item, code: questionCode(item) });
    } else if (kind === 'likert' || q.question_type === 'rating') {
      groupFor(item, 'likert').items.push({ ...item, code: questionCode(item), stats: stats(answersTo(q).map(scoreOf)) });
    } else if (q.question_type === 'radio' || q.question_type === 'checkbox') {
      const counts = (q.options || []).map((o) => ({ label: o, value: 0 }));
      for (const a of answersTo(q)) {
        for (const opt of (a || '').split(',').map((s) => s.trim()).filter(Boolean)) {
          let row = counts.find((c) => c.label === opt);
          if (!row) { row = { label: opt, value: 0 }; counts.push(row); }
          row.value += 1;
        }
      }
      profile.push({ q, type: 'choice', counts });
    } else if (kind === 'number') {
      const xs = answersTo(q).map(Number).filter((v) => v > 0);
      profile.push({ q, type: 'number', n: xs.length, mean: mean(xs), sd: sd(xs),
        min: xs.length ? Math.min(...xs) : null, max: xs.length ? Math.max(...xs) : null });
    } else if (q.question_type === 'essay') {
      openEnded.push({ q, answers: respondents.filter((r) => (r.answers[q.id] || '').trim())
        .map((r) => ({ code: r.code, name: r.name, text: r.answers[q.id] })) });
    } else if (!isNameQuestion(q)) {
      const counts = new Map();   // short text (e.g. department), grouped ignoring case
      for (const a of answersTo(q)) {
        const t = (a || '').trim();
        if (!t) continue;
        const prev = counts.get(t.toLowerCase());
        counts.set(t.toLowerCase(), { label: prev?.label || t, value: (prev?.value || 0) + 1 });
      }
      profile.push({ q, type: 'text', counts: [...counts.values()].sort((a, b) => b.value - a.value) });
    }
  }

  // Likert constructs (e.g. UTAUT): item stats plus the construct's mean,
  // SD (over respondents' construct averages) and Cronbach's alpha.
  const likertGroups = [...groups.values()].filter((g) => g.kind === 'likert').map((g) => {
    const rows = respondents.map((r) => g.items.map((it) => scoreOf(r.answers[it.q.id])));
    return { ...g, construct: scaleStats(rows) };
  });

  // Matrix groups (Part II): the same statistics separately for each AI model.
  const matrixRaw = [...groups.values()].filter((g) => g.kind === 'matrix');
  const models = [...new Set(matrixRaw.flatMap((g) => g.items.flatMap((it) => matrixItems(it.q))))];
  const scoreFor = (r, it, m) => scoreOf(parseMatrix(r.answers[it.q.id])[m]);
  const matrixGroups = matrixRaw.map((g) => ({
    ...g,
    items: g.items.map((it) => ({
      ...it,
      byModel: Object.fromEntries(models.map((m) => [m, stats(respondents.map((r) => scoreFor(r, it, m)))])),
    })),
    byModel: Object.fromEntries(models.map((m) => [m, scaleStats(respondents.map((r) => g.items.map((it) => scoreFor(r, it, m))))])),
  }));
  const allMatrixItems = matrixRaw.flatMap((g) => g.items);
  const modelOverall = Object.fromEntries(models.map((m) => [m, scaleStats(respondents.map((r) => allMatrixItems.map((it) => scoreFor(r, it, m))))]));

  // One row per respondent: their average per construct and per AI model.
  const perRespondent = respondents.map((r) => ({
    ...r,
    constructs: Object.fromEntries(likertGroups.map((g) => [g.key, mean(present(g.items.map((it) => scoreOf(r.answers[it.q.id]))))])),
    models: Object.fromEntries(models.map((m) => [m, mean(present(allMatrixItems.map((it) => scoreFor(r, it, m))))])),
  }));

  return { items, respondents, profile, consent, likertGroups, matrixGroups, models, modelOverall, openEnded, perRespondent };
}

// Wide CSV (one row per respondent, one column per answer; matrix answers get
// one column per model) and a codebook mapping codes to full statements.
export function surveyCsv(analysis) {
  const cols = [];
  for (const item of analysis.items) {
    const { q, kind } = item;
    if (isDisplayOnly(q)) continue;
    const code = questionCode(item);
    if (kind === 'matrix') {
      matrixItems(q).forEach((m) => cols.push({ head: `${code} [${m}]`, get: (r) => parseMatrix(r.answers[q.id])[m] ?? '', text: q.question }));
    } else {
      const head = isConsent(q) ? 'Consent' : (kind === 'likert' ? code : q.question);
      cols.push({ head, get: (r) => r.answers[q.id] ?? '', text: q.question });
    }
  }
  const data = [
    ['Respondent', 'Submitted', ...cols.map((c) => c.head)],
    ...analysis.respondents.map((r) => [r.code, new Date(r.submitted).toISOString(), ...cols.map((c) => c.get(r))]),
  ];
  const codebook = [['Column', 'Statement'], ...cols.map((c) => [c.head, c.text])];
  return { data: toCsv(data), codebook: toCsv(codebook) };
}

// Evaluation report (PDF) for a single essay.
//
// Layout: header → details → overall result → criterion breakdown table →
// strengths / areas to improve → per-criterion comments → overall feedback →
// grammar fixes → recommendations → teacher comments & sign-off →
// optional essay appendix. Every block checks remaining space, so long
// feedback and essays flow onto new pages instead of running off the page.

import { jsPDF } from 'jspdf';
import {
  getCriteria,
  totalScore,
  maxTotal,
  weightedFraction,
  fractionOf,
  gradeFromFraction,
  isNum,
  holisticDisplay,
} from './scoring';
import { buildAnnotations, segmentText, ANNOTATION_TYPES } from './annotate';
import { analyzeEssay, interpret, ANALYSIS_REFERENCES } from './essayAnalysis';
import { EVALUATION_AIS } from './evaluationAis';

// Footer line on every page of the PDF report (Data Privacy Act of 2012).
const PRIVACY_FOOTER = 'Confidential: may contain personal data protected under RA 10173 (Data Privacy Act of 2012). '
  + "AI scores support, and do not replace, the teacher's judgement.";

export const PAGE = { margin: 48, top: 56, bottom: 56 };
export const COLORS = {
  brand: [67, 56, 202],      // indigo-700
  text: [31, 41, 55],        // gray-800
  muted: [107, 114, 128],    // gray-500
  line: [229, 231, 235],     // gray-200
  fill: [243, 244, 246],     // gray-100
  good: [5, 150, 105],       // emerald-600
  mid: [37, 99, 235],        // blue-600
  warn: [217, 119, 6],       // amber-600
  bad: [220, 38, 38],        // red-600
};

// The built-in PDF fonts only cover Latin-1; map common typography and drop
// anything else (emoji etc.) so nothing renders as garbage.
export function clean(text) {
  return String(text ?? '')
    .replace(/[‘’‚′]/g, "'")
    .replace(/[“”„″]/g, '"')
    .replace(/[–—−]/g, '-')
    .replace(/…/g, '...')
    .replace(/[•●▪]/g, '-')
    .replace(/×/g, 'x')
    // Internal markers: drop the whole RAG block, keep the analysis text.
    .replace(/\[RAG_INSIGHTS_START\][\s\S]*?\[RAG_INSIGHTS_END\]/g, '')
    .replace(/\[(RAG_INSIGHTS|ESSAY_ANALYSIS)_(START|END)\]/g, '')
    .replace(/[^\n\r\t\x20-\x7E\u00A0-\u00FF]/g, '')
    .replace(/[ \t]+\n/g, '\n')
    .trim();
}

export function gradeColor(f) {
  if (f === null) return COLORS.muted;
  if (f >= 0.833) return COLORS.good;
  if (f >= 0.5) return COLORS.mid;
  if (f >= 0.167) return COLORS.warn;
  return COLORS.bad;
}

function pct(f) {
  return f === null || f === undefined ? '-' : `${Math.round(f * 100)}%`;
}

export function safeFileName(text) {
  return (clean(text) || 'essay').replace(/[^a-z0-9]+/gi, '_').replace(/^_|_$/g, '').slice(0, 60) || 'essay';
}

export class ReportWriter {
  constructor() {
    this.doc = new jsPDF({ unit: 'pt', format: 'a4' });
    this.width = this.doc.internal.pageSize.getWidth();
    this.height = this.doc.internal.pageSize.getHeight();
    this.contentWidth = this.width - PAGE.margin * 2;
    this.y = PAGE.top;
  }

  setFont(style = 'normal', size = 10, color = COLORS.text) {
    this.doc.setFont('helvetica', style);
    this.doc.setFontSize(size);
    this.doc.setTextColor(...color);
  }

  ensureSpace(height) {
    if (this.y + height > this.height - PAGE.bottom) {
      this.doc.addPage();
      this.y = PAGE.top;
    }
  }

  gap(h = 10) {
    this.y += h;
  }

  heading(text) {
    this.ensureSpace(40);
    this.gap(6);
    this.setFont('bold', 13, COLORS.brand);
    this.doc.text(clean(text), PAGE.margin, this.y);
    this.y += 6;
    this.doc.setDrawColor(...COLORS.line);
    this.doc.setLineWidth(1);
    this.doc.line(PAGE.margin, this.y, PAGE.margin + this.contentWidth, this.y);
    this.y += 14;
  }

  paragraph(text, { size = 10, style = 'normal', color = COLORS.text, indent = 0, lineGap = 1.35 } = {}) {
    const body = clean(text);
    if (!body) return;
    this.setFont(style, size, color);
    const lineHeight = size * lineGap;
    for (const block of body.split(/\n{2,}/)) {
      const lines = this.doc.splitTextToSize(block.replace(/\n/g, ' '), this.contentWidth - indent);
      for (const line of lines) {
        this.ensureSpace(lineHeight);
        this.doc.text(line, PAGE.margin + indent, this.y);
        this.y += lineHeight;
      }
      this.y += lineHeight * 0.4;
    }
  }

  // Label/value grid, two pairs per row.
  details(pairs) {
    const colWidth = this.contentWidth / 2;
    for (let i = 0; i < pairs.length; i += 2) {
      this.ensureSpace(18);
      pairs.slice(i, i + 2).forEach(([label, value], j) => {
        const x = PAGE.margin + j * colWidth;
        this.setFont('bold', 9, COLORS.muted);
        this.doc.text(clean(label).toUpperCase(), x, this.y);
        this.setFont('normal', 10, COLORS.text);
        const v = this.doc.splitTextToSize(clean(value) || '-', colWidth - 90)[0];
        this.doc.text(v, x + 88, this.y);
      });
      this.y += 18;
    }
  }

  // Simple table with wrapping cells; repeats the header on new pages.
  table(columns, rows) {
    const padding = 6;
    const lineHeight = 12;
    const drawHeader = () => {
      this.ensureSpace(24);
      this.doc.setFillColor(...COLORS.fill);
      this.doc.rect(PAGE.margin, this.y - 12, this.contentWidth, 20, 'F');
      this.setFont('bold', 9, COLORS.muted);
      let x = PAGE.margin;
      for (const col of columns) {
        const tx = col.align === 'right' ? x + col.width - padding : x + padding;
        this.doc.text(clean(col.label), tx, this.y + 1, { align: col.align === 'right' ? 'right' : 'left' });
        x += col.width;
      }
      this.y += 18;
    };

    drawHeader();
    for (const row of rows) {
      this.setFont('normal', 10, COLORS.text);
      const cells = columns.map((col, i) => this.doc.splitTextToSize(clean(row.cells[i]), col.width - padding * 2));
      const height = Math.max(...cells.map((c) => c.length)) * lineHeight + 6;
      if (this.y + height > this.height - PAGE.bottom) {
        this.doc.addPage();
        this.y = PAGE.top;
        drawHeader();
      }
      let x = PAGE.margin;
      columns.forEach((col, i) => {
        const color = row.colors?.[i] || COLORS.text;
        this.setFont(row.bold?.[i] ? 'bold' : 'normal', 10, color);
        cells[i].forEach((line, k) => {
          const tx = col.align === 'right' ? x + col.width - padding : x + padding;
          this.doc.text(line, tx, this.y + k * lineHeight, { align: col.align === 'right' ? 'right' : 'left' });
        });
        x += col.width;
      });
      this.y += height;
      this.doc.setDrawColor(...COLORS.line);
      this.doc.setLineWidth(0.5);
      this.doc.line(PAGE.margin, this.y - 10, PAGE.margin + this.contentWidth, this.y - 10);
    }
    this.y += 4;
  }

  finish(fileName, footer = 'AI Essay Evaluator - evaluation report') {
    const pages = this.doc.getNumberOfPages();
    for (let i = 1; i <= pages; i++) {
      this.doc.setPage(i);
      this.setFont('normal', 8, COLORS.muted);
      this.doc.text(footer, PAGE.margin, this.height - 28);
      this.doc.text(`Page ${i} of ${pages}`, this.width - PAGE.margin, this.height - 28, { align: 'right' });
      // Data Privacy Act notice on every page, above the page footer.
      this.setFont('normal', 7, COLORS.muted);
      const lines = this.doc.splitTextToSize(PRIVACY_FOOTER, this.contentWidth);
      this.doc.text(lines, PAGE.margin, this.height - 28 - 9 * lines.length);
    }
    this.doc.save(fileName);
  }
}

// Per-word version of clean(): same character mapping, but no trimming, so
// essay positions found by annotate.js stay valid.
function cleanInline(text) {
  return String(text ?? '')
    .replace(/[\u2018\u2019\u201A\u2032]/g, "'")
    .replace(/[\u201C\u201D\u201E\u2033]/g, '"')
    .replace(/[\u2013\u2014\u2212]/g, '-')
    .replace(/\u2026/g, '...')
    .replace(/[^\x20-\x7E\u00A0-\u00FF]/g, '');
}

const MARK_FILL = {
  grammar: [254, 226, 226],        // red-100
  recommendation: [237, 233, 254], // violet-100
  vague: [254, 243, 199],          // amber-100
};
const LONG_LINE = [14, 165, 233];  // sky-500
const NUMBERED = new Set(['grammar', 'recommendation']);

// The essay with the same marks as on screen: coloured backgrounds, a blue
// underline for long sentences, and [n] markers explained after the essay.
function drawHighlightedEssay(w, essayText, scores) {
  const { doc } = w;
  const text = String(essayText || '').replace(/\r\n/g, '\n');
  const { annotations, unlocated } = buildAnnotations(text, scores);
  const pieces = segmentText(text, annotations);

  const numbered = annotations
    .filter((a) => NUMBERED.has(a.type))
    .sort((a, b) => a.start - b.start);
  const numberOf = new Map(numbered.map((a, i) => [a, i + 1]));

  // Legend
  w.ensureSpace(30);
  let lx = PAGE.margin;
  w.setFont('normal', 8, COLORS.muted);
  for (const [type, info] of Object.entries(ANNOTATION_TYPES)) {
    if (MARK_FILL[type]) {
      doc.setFillColor(...MARK_FILL[type]);
      doc.rect(lx, w.y - 7, 10, 9, 'F');
    } else {
      doc.setDrawColor(...LONG_LINE);
      doc.setLineWidth(1.2);
      doc.line(lx, w.y + 2, lx + 10, w.y + 2);
    }
    doc.text(info.label, lx + 14, w.y);
    lx += 14 + doc.getTextWidth(info.label) + 16;
  }
  w.y += 20;

  const size = 10.5;
  const lineH = 17;
  const left = PAGE.margin;
  const right = PAGE.margin + w.contentWidth;
  let x = left;
  let y = w.y;
  const newLine = () => {
    x = left;
    y += lineH;
    if (y > w.height - PAGE.bottom) {
      doc.addPage();
      y = PAGE.top;
    }
  };
  const marked = new Set();

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(size);
  for (const piece of pieces) {
    const bg = piece.annotations.find((a) => MARK_FILL[a.type]);
    const long = piece.annotations.some((a) => a.type === 'long');
    for (const tok of piece.text.split(/(\n|[ \t]+)/)) {
      if (!tok) continue;
      if (tok === '\n') { newLine(); continue; }
      if (/^[ \t]+$/.test(tok)) {
        if (x === left) continue;
        const sw = doc.getTextWidth(' ');
        if (x + sw > right) { newLine(); continue; }
        if (bg) { doc.setFillColor(...MARK_FILL[bg.type]); doc.rect(x, y - 11.5, sw, 15, 'F'); }
        if (long) { doc.setDrawColor(...LONG_LINE); doc.setLineWidth(0.9); doc.line(x, y + 2.5, x + sw, y + 2.5); }
        x += sw;
        continue;
      }
      const word = cleanInline(tok);
      if (!word) continue;
      const ww = doc.getTextWidth(word);
      if (x + ww > right && x > left) newLine();
      if (bg) { doc.setFillColor(...MARK_FILL[bg.type]); doc.rect(x - 0.5, y - 11.5, ww + 1, 15, 'F'); }
      doc.setTextColor(...COLORS.text);
      doc.text(word, x, y);
      if (long) { doc.setDrawColor(...LONG_LINE); doc.setLineWidth(0.9); doc.line(x, y + 2.5, x + ww, y + 2.5); }
      x += ww;
    }
    // [n] marker right after each numbered mark ends.
    for (const a of piece.annotations) {
      if (!NUMBERED.has(a.type) || a.end !== piece.end || marked.has(a)) continue;
      marked.add(a);
      const label = `[${numberOf.get(a)}]`;
      doc.setFontSize(7);
      doc.setFont('helvetica', 'bold');
      const lw = doc.getTextWidth(label);
      if (x + lw > right) newLine();
      doc.setTextColor(...(a.type === 'grammar' ? COLORS.bad : [109, 40, 217]));
      doc.text(label, x + 0.5, y - 5);
      x += lw + 1.5;
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(size);
    }
  }
  w.y = y + lineH + 6;

  // What each number means.
  if (numbered.length || unlocated.length) {
    w.heading('Marked Issues');
    numbered.forEach((a) => {
      w.ensureSpace(30);
      const kind = a.type === 'grammar' ? 'Grammar' : 'Improve';
      w.paragraph(`[${numberOf.get(a)}] ${kind}: "${text.slice(a.start, a.end)}"`, {
        style: 'bold', size: 9.5, color: a.type === 'grammar' ? COLORS.bad : [109, 40, 217],
      });
      w.paragraph(`${a.title}${a.detail ? `. ${a.detail}` : ''}`, { indent: 14, size: 9.5 });
    });
    if (unlocated.length) {
      w.paragraph(`${unlocated.length} further suggestion(s) could not be matched to an exact spot in the text; see the Grammar and Recommendations sections.`,
        { size: 8.5, color: COLORS.muted });
    }
  }
  const counts = { vague: 0, long: 0 };
  for (const a of annotations) if (a.type in counts) counts[a.type] += 1;
  if (counts.vague || counts.long) {
    w.paragraph(
      `Also marked: ${counts.vague} vague word(s) (amber) and ${counts.long} sentence(s) over 30 words (blue underline).`,
      { size: 8.5, color: COLORS.muted }
    );
  }
}

// AI-revised reference version of the essay.
function drawRevision(w, revision) {
  const { doc } = w;
  doc.addPage();
  w.y = PAGE.top;
  w.heading('Reference: AI-Revised Version');
  w.paragraph(
    `Generated by ${revision.evaluated_by?.label || 'an AI model'} as an example of how this essay could be improved. ` +
    'It keeps the student\'s position and ideas. Text in [square brackets] marks where the student should add their own ' +
    'evidence. This is a reference for revision, not the student\'s work.',
    { size: 8.5, color: COLORS.muted }
  );
  if (revision.changes?.length) {
    w.paragraph('Main changes', { style: 'bold', size: 10 });
    revision.changes.forEach((c) => w.paragraph(`- ${c}`, { indent: 10, size: 9.5 }));
    w.gap(6);
  }
  w.paragraph(revision.revised_essay, { size: 10, lineGap: 1.5 });
}

/**
 * options: {
 *   title, essayText, feedback, scores, evalType,
 *   studentName, section, teacherName, teacherComments,
 *   teacherScores, includeEssay,
 *   revision   // optional { revised_essay, changes, evaluated_by } from /revise-essay
 * }
 */
// ---------- In-depth analysis ----------
// Rule-based measures of the essay (readability, sentence structure,
// vocabulary, cohesion, evidence), a paragraph-by-paragraph breakdown, the
// three AIs' scores side by side with the human score, and the method.
function drawDeepAnalysis(w, essayText, { scores, aiRuns, humanScores, evalType }) {
  const a = analyzeEssay(essayText);
  if (!a.words) return;
  w.doc.addPage();
  w.y = PAGE.top;
  w.heading('In-Depth Analysis');
  w.paragraph(
    `This section measures the essay itself: ${a.words} words in ${a.sentences} sentences and ${a.paragraphs} `
    + `paragraph${a.paragraphs === 1 ? '' : 's'} (${a.uniqueWords} different words). The figures are computed `
    + 'with standard formulas, not by an AI, so the same essay always gets the same results. "Typical" is the '
    + 'range usually expected in a secondary or college academic essay.',
    { size: 9, color: COLORS.muted },
  );

  // Writing profile
  w.paragraph('1. Writing profile', { style: 'bold' });
  w.table([
    { label: 'Measure', width: 140 },
    { label: 'Value', width: 82 },
    { label: 'Typical', width: 92 },
    { label: 'What it means', width: w.contentWidth - 314 },
  ], interpret(a).map((r) => ({ cells: [r.label, r.value, r.target, r.note], bold: [true] })));

  // Cohesion: transitions by kind
  w.paragraph('2. Cohesion: transitions used', { style: 'bold' });
  const kinds = Object.entries(a.transitions);
  if (kinds.length) {
    for (const [kind, found] of kinds) {
      w.paragraph(`${kind}: ${Object.entries(found).map(([t, n]) => (n > 1 ? `${t} (${n})` : t)).join(', ')}`, { indent: 10, size: 9 });
    }
    const missing = ['Contrast', 'Cause and effect', 'Example', 'Conclusion'].filter((k) => !a.transitions[k]);
    if (missing.length) {
      w.paragraph(`Not used: ${missing.join(', ')} transitions. Adding them would show how the ideas relate.`, { indent: 10, size: 9, color: COLORS.muted });
    }
  } else {
    w.paragraph('No transition words were found. Link paragraphs and ideas with words such as "however", "therefore", "for example" and "in conclusion".', { indent: 10, size: 9 });
  }

  // Word choice
  w.paragraph('3. Word choice and sentence openings', { style: 'bold' });
  w.paragraph(a.repeated.length
    ? `Most repeated content words: ${a.repeated.map(([word, n]) => `${word} (${n})`).join(', ')}. Repeating the topic word is normal; vary the others with synonyms.`
    : 'No content word is overused.', { indent: 10, size: 9 });
  if (a.repeatedOpeners.length) {
    w.paragraph(`Many sentences begin with the same word: ${a.repeatedOpeners.map(([word, n]) => `"${word}" (${n} times)`).join(', ')}. Vary sentence openings.`, { indent: 10, size: 9 });
  }
  if (Object.keys(a.vague).length) {
    w.paragraph(`Vague or informal words: ${Object.entries(a.vague).map(([word, n]) => `${word} (${n})`).join(', ')}.`, { indent: 10, size: 9 });
  }

  // Paragraph by paragraph
  w.paragraph('4. Paragraph-by-paragraph analysis', { style: 'bold' });
  w.table([
    { label: '#', width: 22 },
    { label: 'Role', width: 68 },
    { label: 'Words', width: 42, align: 'right' },
    { label: 'Sent.', width: 36, align: 'right' },
    { label: 'Opening (topic) sentence', width: 170 },
    { label: 'Observations', width: w.contentWidth - 338 },
  ], a.paragraphsInfo.map((p) => ({
    cells: [String(p.n), p.role, String(p.words), String(p.sentences),
      p.opening.length > 140 ? `${p.opening.slice(0, 137)}...` : p.opening, p.notes.join('; ')],
    colors: [null, null, null, null, null, p.notes[0] === 'well developed' ? COLORS.good : COLORS.bad],
  })));

  // The three AIs side by side, with the human score
  const runs = EVALUATION_AIS
    .map((ai) => ({ ai, s: aiRuns?.[ai.id]?.status === 'done' ? aiRuns[ai.id].scores : null }))
    .filter((r) => r.s && r.s.evaluated_by?.provider !== 'heuristic');
  if (runs.length >= 2 && evalType !== 'holistic') {
    const criteria = getCriteria(scores);
    const human = humanScores && Object.values(humanScores).some(isNum) ? humanScores : null;
    w.paragraph('5. The three AI evaluators compared', { style: 'bold' });
    const fixed = 130;
    const cols = runs.length + (human ? 1 : 0) + 1;
    const cw = Math.floor((w.contentWidth - fixed) / cols);
    const spreadOf = (c) => {
      const vals = runs.map(({ s }) => s[c.key]).filter(isNum);
      return vals.length ? Math.max(...vals) - Math.min(...vals) : null;
    };
    w.table([
      { label: 'Criterion', width: fixed },
      ...(human ? [{ label: 'Human', width: cw, align: 'right' }] : []),
      ...runs.map(({ ai }) => ({ label: ai.name, width: cw, align: 'right' })),
      { label: 'Spread', width: w.contentWidth - fixed - cw * (cols - 1), align: 'right' },
    ], [
      ...criteria.map((c) => {
        const spread = spreadOf(c);
        return {
          cells: [
            `${c.label} (/${c.max_score})`,
            ...(human ? [isNum(human[c.key]) ? String(human[c.key]) : '-'] : []),
            ...runs.map(({ s }) => (isNum(s[c.key]) ? String(s[c.key]) : '-')),
            spread === null ? '-' : spread === 0 ? 'agree' : `${spread} pt${spread === 1 ? '' : 's'}`,
          ],
          bold: [true],
          colors: [...Array(cols).fill(null), spread >= 2 ? COLORS.bad : spread === 0 ? COLORS.good : null],
        };
      }),
      {
        cells: [
          'Overall (weighted)',
          ...(human ? [pct(weightedFraction(human, criteria))] : []),
          ...runs.map(({ s }) => pct(weightedFraction(s, criteria))),
          '',
        ],
        bold: Array(cols + 1).fill(true),
      },
    ]);
    const agree = criteria.filter((c) => { const sp = spreadOf(c); return sp !== null && sp <= 1; }).length;
    const samePrompt = runs.every(({ s }) => s.prompt?.system) && new Set(runs.map(({ s }) => s.prompt.system)).size === 1;
    w.paragraph(
      `The AIs agree within one point on ${agree} of ${criteria.length} criteria. `
      + (samePrompt ? 'All of them received the same scoring prompt (same rubric, rules and essay, temperature 0); the local Qwen model gets a shortened version of it. ' : '')
      + "Where the spread is two points or more, the teacher's judgement should decide."
      + (human ? " The human column is the teacher's score, for comparison." : ''),
      { size: 9, color: COLORS.muted },
    );
  }

  // Method and references
  w.paragraph('Method and references', { style: 'bold' });
  w.paragraph(
    'Flesch Reading Ease = 206.835 - 1.015 x (words / sentences) - 84.6 x (syllables / words); '
    + 'Flesch-Kincaid Grade = 0.39 x (words / sentences) + 11.8 x (syllables / words) - 15.59. '
    + 'Vocabulary variety is the moving-average type-token ratio over 50-word windows. Transitions follow the '
    + 'categories of cohesive devices; evidence markers are citations, "according to", statistics, years and '
    + 'quotations. Syllables, passive voice and paragraph roles are estimated automatically and may be '
    + 'approximate, especially for text read from handwriting.',
    { size: 8, color: COLORS.muted },
  );
  for (const ref of ANALYSIS_REFERENCES) w.paragraph(ref, { size: 8, color: COLORS.muted, indent: 10 });
}

export function generateEvaluationReport(options) {
  const {
    title, essayText, feedback, scores, evalType,
    studentName, section, teacherName, teacherComments,
    teacherScores, includeEssay = true, revision = null,
    aiRuns = null, humanScores = null, includeAnalysis = true,
  } = options;

  const w = new ReportWriter();
  const { doc } = w;
  const now = new Date();
  const wordCount = (essayText || '').trim().split(/\s+/).filter(Boolean).length;
  const hasTeacherScores = teacherScores && Object.values(teacherScores).some(isNum);

  // ---------- Header band ----------
  doc.setFillColor(...COLORS.brand);
  doc.rect(0, 0, w.width, 78, 'F');
  w.setFont('bold', 20, [255, 255, 255]);
  doc.text('Essay Evaluation Report', PAGE.margin, 40);
  w.setFont('normal', 10, [224, 231, 255]);
  doc.text(clean(title || 'Untitled essay'), PAGE.margin, 60);
  doc.text(now.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' }),
    w.width - PAGE.margin, 60, { align: 'right' });
  w.y = 108;

  // ---------- Details ----------
  w.details([
    ['Student', studentName || '-'],
    ['Class / section', section || '-'],
    ['Evaluated by', teacherName || '-'],
    ['Date', now.toLocaleString()],
    ['Evaluation', evalType === 'holistic' ? 'Holistic (1-5)' : 'Analytic (rubric)'],
    ['Rubric', evalType === 'holistic' ? 'Holistic scale' : scores?.rubric?.title || 'Built-in analytic rubric'],
    ['Essay length', `${wordCount} words`],
  ]);

  // ---------- Overall result ----------
  w.heading('Overall Result');
  w.ensureSpace(80);
  const boxTop = w.y - 6;
  doc.setDrawColor(...COLORS.line);
  doc.setFillColor(249, 250, 251);
  doc.roundedRect(PAGE.margin, boxTop, w.contentWidth, 64, 6, 6, 'FD');

  if (evalType === 'holistic') {
    const h = holisticDisplay(scores);
    const f = h ? h.fraction : null;
    w.setFont('bold', 26, gradeColor(f));
    doc.text(h ? h.text.replace('/', ' / ') : '-', PAGE.margin + 18, boxTop + 40);
    w.setFont('normal', 10, COLORS.text);
    const desc = doc.splitTextToSize(clean(scores.level_description), w.contentWidth - 150);
    doc.text(desc.slice(0, 3), PAGE.margin + 130, boxTop + 24);
    w.y = boxTop + 84;
  } else {
    const criteria = getCriteria(scores);
    const total = totalScore(scores, criteria);
    const possible = maxTotal(criteria);
    const weighted = weightedFraction(scores, criteria);
    const grade = gradeFromFraction(weighted);
    const stats = [
      ['Total', `${total} / ${possible}`],
      ['Percentage', possible ? `${Math.round((total / possible) * 100)}%` : '-'],
      ['Weighted', pct(weighted)],
      ['Grade', grade.label],
    ];
    const colW = w.contentWidth / stats.length;
    stats.forEach(([label, value], i) => {
      const x = PAGE.margin + i * colW + 18;
      w.setFont('bold', 8, COLORS.muted);
      doc.text(label.toUpperCase(), x, boxTop + 22);
      w.setFont('bold', 18, i === 3 || i === 2 ? gradeColor(weighted) : COLORS.text);
      doc.text(value, x, boxTop + 46);
    });
    w.y = boxTop + 84;

    // ---------- Criterion table ----------
    w.heading('Score Breakdown');
    const columns = [
      { label: 'Criterion', width: hasTeacherScores ? 175 : 235 },
      { label: 'Weight', width: 60, align: 'right' },
      { label: 'AI score', width: 80, align: 'right' },
      ...(hasTeacherScores ? [{ label: 'Teacher', width: 70, align: 'right' }] : []),
      { label: 'Level', width: w.contentWidth - (hasTeacherScores ? 175 + 60 + 80 + 70 : 235 + 60 + 80) },
    ];
    w.table(columns, criteria.map((c) => {
      const v = scores[c.key];
      const f = fractionOf(v, c.max_score);
      const t = teacherScores?.[c.key];
      return {
        cells: [
          c.label,
          `${c.weight ?? 1}x`,
          `${isNum(v) ? v : '-'} / ${c.max_score}`,
          ...(hasTeacherScores ? [isNum(t) ? `${t} / ${c.max_score}` : '-'] : []),
          `${gradeFromFraction(f).label} (${pct(f)})`,
        ],
        colors: [null, null, gradeColor(f), ...(hasTeacherScores ? [null] : []), gradeColor(f)],
        bold: [true, false, true],
      };
    }));

    // ---------- Strengths and areas to improve ----------
    const ranked = criteria
      .map((c) => ({ ...c, f: fractionOf(scores[c.key], c.max_score) }))
      .filter((c) => c.f !== null)
      .sort((a, b) => b.f - a.f);
    if (ranked.length >= 2) {
      const strengths = ranked.filter((c) => c.f >= 0.5).slice(0, 2);
      const focus = [...ranked].reverse().filter((c) => c.f < 1).slice(0, 2);
      w.heading('Summary');
      w.paragraph(`Strengths: ${strengths.length ? strengths.map((c) => c.label).join(', ') : 'none of the criteria reached the upper half of the scale yet'}.`);
      w.paragraph(`Focus next on: ${focus.length ? focus.map((c) => c.label).join(', ') : 'maintaining this level - every criterion is at its maximum'}.`);
    }

    // ---------- Per-criterion comments ----------
    const notes = criteria.filter((c) => clean(scores.criterion_feedback?.[c.key]));
    if (notes.length) {
      w.heading('Criterion Comments');
      for (const c of notes) {
        w.ensureSpace(30);
        w.paragraph(`${c.label} (${scores[c.key]}/${c.max_score})`, { style: 'bold' });
        w.paragraph(scores.criterion_feedback[c.key], { indent: 10 });
      }
    }
  }

  // ---------- Overall feedback ----------
  if (clean(feedback)) {
    w.heading('Feedback');
    w.paragraph(feedback);
  }

  // ---------- Grammar ----------
  const issues = (scores?.grammar_issues || []).filter(
    (i) => i && clean(i.original) && clean(i.correction) &&
      clean(i.original).toLowerCase() !== clean(i.correction).toLowerCase()
  );
  if (issues.length) {
    w.heading(`Grammar and Mechanics (${issues.length})`);
    issues.forEach((issue, idx) => {
      w.ensureSpace(40);
      w.paragraph(`${idx + 1}. "${issue.original}"`, { color: COLORS.bad });
      w.paragraph(`Correction: "${issue.correction}"`, { indent: 12, color: COLORS.good });
      if (issue.explanation) w.paragraph(issue.explanation, { indent: 12, size: 9, color: COLORS.muted });
    });
  }

  // ---------- Recommendations ----------
  const recs = (scores?.recommendations || []).filter((r) => r && clean(r.suggestion));
  if (recs.length) {
    w.heading(`Recommendations (${recs.length})`);
    recs.forEach((rec, idx) => {
      w.ensureSpace(40);
      w.paragraph(`${idx + 1}. ${rec.title || 'Improvement'}`, { style: 'bold' });
      if (rec.issue) w.paragraph(`Issue: ${rec.issue}`, { indent: 12 });
      w.paragraph(`Suggestion: ${rec.suggestion}`, { indent: 12 });
    });
  }

  // ---------- Teacher comments & sign-off ----------
  w.heading("Teacher's Comments");
  if (clean(teacherComments)) {
    w.paragraph(teacherComments);
  } else {
    w.setFont('normal', 10, COLORS.line);
    for (let i = 0; i < 3; i++) {
      w.ensureSpace(22);
      w.y += 14;
      doc.setDrawColor(...COLORS.line);
      doc.line(PAGE.margin, w.y, PAGE.margin + w.contentWidth, w.y);
    }
    w.y += 8;
  }
  w.ensureSpace(60);
  w.y += 30;
  doc.setDrawColor(...COLORS.muted);
  doc.line(PAGE.margin, w.y, PAGE.margin + 200, w.y);
  doc.line(PAGE.margin + w.contentWidth - 140, w.y, PAGE.margin + w.contentWidth, w.y);
  w.setFont('normal', 8, COLORS.muted);
  doc.text(clean(teacherName) || 'Teacher', PAGE.margin, w.y + 12);
  doc.text('Signature / date', PAGE.margin + w.contentWidth - 140, w.y + 12);
  w.y += 24;

  w.paragraph(
    'Scores and feedback were generated with AI assistance and reviewed by the teacher named above. '
    + 'They are a guide for revision, not a substitute for the teacher\'s judgment.',
    { size: 8, color: COLORS.muted }
  );

  // ---------- In-depth analysis ----------
  if (includeAnalysis && clean(essayText)) {
    drawDeepAnalysis(w, essayText, {
      scores, aiRuns, evalType, humanScores: hasTeacherScores ? teacherScores : humanScores,
    });
  }

  // ---------- Essay appendix (highlighted like the screen) ----------
  if (includeEssay && clean(essayText)) {
    doc.addPage();
    w.y = PAGE.top;
    w.heading('Appendix: Essay with Highlights');
    drawHighlightedEssay(w, essayText, scores);
  }

  // ---------- AI-revised reference ----------
  if (revision?.revised_essay) {
    drawRevision(w, revision);
  }

  const namePart = studentName ? `${safeFileName(studentName)}_` : '';
  w.finish(`evaluation_${namePart}${safeFileName(title)}.pdf`);
}

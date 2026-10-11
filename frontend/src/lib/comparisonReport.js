// AI scoring comparison report (PDF) for the Prompt Playground.
//
// Layout: header → run details → Primary vs Compare table (Category,
// Primary, Compare, Diff) → all three models side by side → which model
// is closest to the reference → per-model statistics → pairwise
// agreement → optional essay appendix.

import { ReportWriter, PAGE, COLORS, clean, gradeColor, safeFileName } from './report';
import { fractionOf, weightedFraction, isNum } from './scoring';

function fmt(v) {
  if (!isNum(v)) return '-';
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}

function signed(v, digits = 0) {
  if (!isNum(v)) return '-';
  return `${v > 0 ? '+' : ''}${v.toFixed(digits)}`;
}

function diffColor(d, max) {
  if (!isNum(d) || d === 0) return COLORS.muted;
  const big = Math.abs(d) / Math.max(1, max - 1) > 1 / 3;
  if (d > 0) return big ? COLORS.good : [16, 185, 129];
  return big ? COLORS.bad : [239, 68, 68];
}

function pct(v) {
  return isNum(v) ? `${v.toFixed(1)}%` : '-';
}

/**
 * options: {
 *   essayTitle, essayText, criteria, raters: [{id, name, scores}],
 *   primaryId, compareId, stats (computeStats result + consensus),
 *   reference, referenceLabel, includeEssay
 * }
 */
export function generateComparisonReport(options) {
  const {
    essayTitle, essayText, criteria, raters, primaryId, compareId,
    stats, reference, referenceLabel, includeEssay = true,
  } = options;

  const w = new ReportWriter();
  const { doc } = w;
  const now = new Date();
  const primary = raters.find((r) => r.id === primaryId);
  const compare = raters.find((r) => r.id === compareId);
  const scored = raters.filter((r) => r.scores);
  const wordCount = (essayText || '').trim().split(/\s+/).filter(Boolean).length;

  // ---------- Header ----------
  doc.setFillColor(...COLORS.brand);
  doc.rect(0, 0, w.width, 78, 'F');
  w.setFont('bold', 20, [255, 255, 255]);
  doc.text('AI Scoring Comparison Report', PAGE.margin, 40);
  w.setFont('normal', 10, [224, 231, 255]);
  doc.text(clean(essayTitle || 'Untitled essay'), PAGE.margin, 60);
  doc.text(now.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' }),
    w.width - PAGE.margin, 60, { align: 'right' });
  w.y = 108;

  w.details([
    ['Primary', primary ? `${primary.id} - ${primary.name}` : '-'],
    ['Compare', compare ? `${compare.id} - ${compare.name}` : '-'],
    ['Models scored', `${scored.length} of ${raters.length}`],
    ['Reference', referenceLabel],
    ['Criteria', String(criteria.length)],
    ['Essay length', `${wordCount} words`],
  ]);

  // ---------- Primary vs Compare ----------
  w.heading('Score Comparison');
  w.paragraph('Diff = Primary - Compare. Positive means the primary model scored higher.', { size: 9, color: COLORS.muted });
  const cw = w.contentWidth;
  const rows = criteria.map((c) => {
    const p = primary?.scores?.[c.key];
    const q = compare?.scores?.[c.key];
    const d = isNum(p) && isNum(q) ? p - q : null;
    return {
      cells: [`${c.label} (1-${c.max_score})`, fmt(p), fmt(q), d === null ? '-' : signed(d)],
      colors: [null, gradeColor(fractionOf(p, c.max_score)), gradeColor(fractionOf(q, c.max_score)), diffColor(d, c.max_score)],
      bold: [true, true, true, true],
    };
  });
  const wp = primary?.scores ? weightedFraction(primary.scores, criteria) : null;
  const wq = compare?.scores ? weightedFraction(compare.scores, criteria) : null;
  rows.push({
    cells: ['Weighted overall', wp === null ? '-' : `${Math.round(wp * 100)}%`, wq === null ? '-' : `${Math.round(wq * 100)}%`,
      wp === null || wq === null ? '-' : `${signed((wp - wq) * 100)} pts`],
    bold: [true, true, true, true],
  });
  w.table([
    { label: 'Category', width: cw - 270 },
    { label: `Primary (${primaryId})`, width: 90, align: 'right' },
    { label: `Compare (${compareId})`, width: 90, align: 'right' },
    { label: 'Diff', width: 90, align: 'right' },
  ], rows);

  // ---------- All three ----------
  if (scored.length > 2) {
    w.heading('All Models Side by Side');
    const modelCols = raters.map((r) => ({ label: r.id, width: 50, align: 'right' }));
    w.table(
      [
        { label: 'Category', width: cw - 50 * raters.length - 75 * 2 },
        ...modelCols,
        { label: 'Consensus', width: 75, align: 'right' },
        { label: 'Spread', width: 75, align: 'right' },
      ],
      stats.perCriterion.map((c) => ({
        cells: [
          c.label,
          ...raters.map((r) => fmt(r.scores?.[c.key])),
          fmt(stats.consensus[c.key]),
          c.range === null ? '-' : String(c.range),
        ],
        colors: [c.highDisagreement ? COLORS.warn : null],
        bold: [true],
      }))
    );
    w.paragraph(raters.map((r) => `${r.id} = ${r.name}`).join('   '), { size: 8, color: COLORS.muted });
    const big = stats.perCriterion.filter((c) => c.highDisagreement).map((c) => c.label);
    w.paragraph(
      `All models agree exactly on ${stats.allAgreeCount} of ${criteria.length} criteria.` +
      (big.length ? ` Biggest disagreements (more than one level apart): ${big.join(', ')}.` : ''),
      { size: 9 }
    );
  }

  // ---------- Closest model ----------
  w.heading(`Closest to ${referenceLabel}`);
  if (stats.closest) {
    w.paragraph(
      `${stats.closest.id} - ${stats.closest.name} is closest, at ${pct(stats.closest.similarity)} similarity ` +
      `(average distance ${stats.closest.mae.toFixed(1)} points on a 0-100 scale)` +
      (stats.tie ? ', a near-tie with the runner-up.' : '.'),
      { style: 'bold' }
    );
  } else {
    w.paragraph('No reference scores were available for this comparison.');
  }
  if (referenceLabel !== 'Consensus' && reference) {
    w.paragraph(
      `Reference scores: ${criteria.map((c) => `${c.label} ${fmt(reference[c.key])}`).join(', ')}.`,
      { size: 9, color: COLORS.muted }
    );
  }

  // ---------- Per-model statistics ----------
  if (stats.perRater.length) {
    w.table(
      [
        { label: 'Model', width: cw - 75 * 5 },
        { label: 'Similarity', width: 75, align: 'right' },
        { label: 'Avg dist.', width: 75, align: 'right' },
        { label: 'Bias', width: 75, align: 'right' },
        { label: 'Tendency', width: 75, align: 'right' },
        { label: 'Weighted', width: 75, align: 'right' },
      ],
      stats.perRater.map((r) => ({
        cells: [
          `${r.id} - ${r.name}`,
          pct(r.similarity),
          isNum(r.mae) ? r.mae.toFixed(1) : '-',
          isNum(r.bias) ? signed(r.bias, 1) : '-',
          r.tendency.label,
          isNum(r.weighted) ? `${Math.round(r.weighted * 100)}%` : '-',
        ],
        bold: [stats.closest?.id === r.id],
      }))
    );
    w.paragraph(
      'Distances use a 0-100 scale per criterion so different ranges compare fairly. ' +
      'Bias above +8 = Lenient, below -8 = Strict, otherwise Balanced.',
      { size: 8, color: COLORS.muted }
    );
  }

  // ---------- Pairwise agreement ----------
  if (stats.pairs.length) {
    w.heading('Pairwise Agreement');
    w.table(
      [
        { label: 'Pair', width: cw - 95 * 4 },
        { label: 'Exact match', width: 95, align: 'right' },
        { label: 'Within +/-1', width: 95, align: 'right' },
        { label: 'Avg |diff|', width: 95, align: 'right' },
        { label: 'Correlation', width: 95, align: 'right' },
      ],
      stats.pairs.map((p) => ({
        cells: [
          `${p.a.id} vs ${p.b.id}`,
          p.n ? `${p.exact}/${p.n}` : '-',
          p.n ? `${p.adjacent}/${p.n}` : '-',
          isNum(p.mae) ? p.mae.toFixed(2) : '-',
          isNum(p.r) ? p.r.toFixed(2) : '-',
        ],
      }))
    );
  }

  // ---------- Per-model feedback ----------
  const withFeedback = scored.filter((r) => clean(r.scores.feedback));
  if (withFeedback.length) {
    w.heading('Model Feedback');
    for (const r of withFeedback) {
      w.ensureSpace(40);
      w.paragraph(`${r.id} - ${r.name}`, { style: 'bold' });
      w.paragraph(r.scores.feedback, { indent: 10 });
    }
  }

  if (includeEssay && clean(essayText)) {
    doc.addPage();
    w.y = PAGE.top;
    w.heading('Appendix: Essay Text');
    w.paragraph(essayText, { size: 10, lineGap: 1.5 });
  }

  w.finish(`ai_comparison_${safeFileName(essayTitle)}.pdf`, 'AI Essay Evaluator - AI scoring comparison');
}


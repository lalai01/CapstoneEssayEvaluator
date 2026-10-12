// Find the parts of an essay that need attention, as character ranges.
//
// Sources:
//   grammar        - the AI's grammar_issues ("original" located in the text)
//   recommendation - the passage each AI recommendation quotes ("quote")
//   vague          - weak or vague wording (a fixed word list)
//   long           - sentences over LONG_SENTENCE_WORDS words
//
// Ranges may overlap (a grammar error inside a long sentence), so the text
// is split at every range boundary and each piece lists every annotation
// covering it. Rendering is left to the component.

export const LONG_SENTENCE_WORDS = 30;

export const ANNOTATION_TYPES = {
  grammar: { label: 'Grammar / spelling', priority: 1 },
  recommendation: { label: 'Needs improvement', priority: 2 },
  vague: { label: 'Vague wording', priority: 3 },
  long: { label: `Long sentence (over ${LONG_SENTENCE_WORDS} words)`, priority: 4 },
};

const VAGUE = /\b(very|really|a lot|lots of|things?|stuff|kind of|sort of|nice|pretty much|basically|actually)\b/gi;

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Locate a quoted phrase, ignoring case, spacing and curly quotes.
export function findPhrase(text, phrase, from = 0) {
  const words = String(phrase || '')
    .replace(/[‘’]/g, "'")
    .replace(/[“”]/g, '"')
    .trim()
    .split(/\s+/)
    .filter(Boolean);
  if (words.length === 0) return null;
  const pattern = words
    .map((w) => escapeRegex(w).replace(/'/g, "['‘’]").replace(/"/g, '["“”]'))
    .join('\\s+');
  const re = new RegExp(pattern, 'gi');
  re.lastIndex = from;
  const m = re.exec(text);
  return m ? { start: m.index, end: m.index + m[0].length } : null;
}

export function buildAnnotations(essay, scores) {
  const text = String(essay || '');
  const annotations = [];
  const unlocated = [];

  const issues = Array.isArray(scores?.grammar_issues) ? scores.grammar_issues : [];
  for (const issue of issues) {
    if (!issue || typeof issue.original !== 'string' || typeof issue.correction !== 'string') continue;
    if (issue.original.trim().toLowerCase() === issue.correction.trim().toLowerCase()) continue;
    const at = findPhrase(text, issue.original);
    if (!at) { unlocated.push({ type: 'grammar', ...issue }); continue; }
    annotations.push({
      type: 'grammar', ...at,
      title: `Change to: “${issue.correction}”`,
      detail: issue.explanation || '',
    });
  }

  const recs = Array.isArray(scores?.recommendations) ? scores.recommendations : [];
  for (const rec of recs) {
    if (!rec || typeof rec.quote !== 'string' || !rec.quote.trim()) continue;
    const at = findPhrase(text, rec.quote);
    if (!at) { unlocated.push({ type: 'recommendation', ...rec }); continue; }
    annotations.push({
      type: 'recommendation', ...at,
      title: rec.title || 'Improve this part',
      detail: [rec.issue, rec.suggestion && `Try: ${rec.suggestion}`].filter(Boolean).join(' '),
    });
  }

  let m;
  VAGUE.lastIndex = 0;
  while ((m = VAGUE.exec(text))) {
    annotations.push({
      type: 'vague', start: m.index, end: m.index + m[0].length,
      title: `“${m[0]}” is vague`,
      detail: 'Use a more precise word, or cut it if the sentence works without it.',
    });
  }

  const SENTENCE = /[^.!?\n]+[.!?]*/g;
  while ((m = SENTENCE.exec(text))) {
    const words = m[0].trim().split(/\s+/).filter(Boolean).length;
    if (words > LONG_SENTENCE_WORDS) {
      const lead = m[0].length - m[0].trimStart().length;
      annotations.push({
        type: 'long', start: m.index + lead, end: m.index + m[0].trimEnd().length,
        title: `Long sentence (${words} words)`,
        detail: 'Consider splitting it into two or three shorter sentences.',
      });
    }
  }

  return { annotations, unlocated };
}

// Split text into pieces; each piece lists the annotations covering it,
// highest priority first.
export function segmentText(text, annotations) {
  const bounds = new Set([0, text.length]);
  for (const a of annotations) {
    bounds.add(Math.max(0, Math.min(text.length, a.start)));
    bounds.add(Math.max(0, Math.min(text.length, a.end)));
  }
  const points = [...bounds].sort((x, y) => x - y);
  const pieces = [];
  for (let i = 0; i < points.length - 1; i++) {
    const start = points[i];
    const end = points[i + 1];
    if (end <= start) continue;
    const covering = annotations
      .filter((a) => a.start <= start && a.end >= end)
      .sort((x, y) => ANNOTATION_TYPES[x.type].priority - ANNOTATION_TYPES[y.type].priority);
    pieces.push({ start, end, text: text.slice(start, end), annotations: covering });
  }
  return pieces;
}

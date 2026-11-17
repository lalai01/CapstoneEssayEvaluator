// In-depth, rule-based analysis of an essay's text for the PDF report:
// readability, sentence structure, vocabulary, cohesion, evidence and a
// paragraph-by-paragraph breakdown. Everything is computed in the browser
// from the essay itself (no AI), so the same essay always gets the same
// figures. Formulas and sources are listed in ANALYSIS_REFERENCES.

const STOPWORDS = new Set(`a about above after again against all also am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for from further had has have having
he her here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of
off on once only or other our ours ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were what when where which while
who whom why will with would you your yours yourself yourselves one many much may might must shall us upon within without
however therefore thus also yet still even ever every really get got make made`.split(/\s+/));

// Transition words and phrases (cohesive devices: Halliday & Hasan, 1976).
const TRANSITIONS = {
  Addition: ['additionally', 'furthermore', 'moreover', 'in addition', 'also', 'besides', 'likewise'],
  Contrast: ['however', 'nevertheless', 'nonetheless', 'on the other hand', 'in contrast', 'although', 'whereas', 'yet', 'despite', 'conversely'],
  'Cause and effect': ['therefore', 'thus', 'consequently', 'as a result', 'hence', 'because of this', 'accordingly'],
  Example: ['for example', 'for instance', 'such as', 'specifically', 'to illustrate', 'namely'],
  Sequence: ['first', 'firstly', 'second', 'secondly', 'third', 'finally', 'next', 'then', 'lastly', 'subsequently'],
  Conclusion: ['in conclusion', 'to conclude', 'in summary', 'to sum up', 'overall', 'ultimately', 'in short'],
};

const VAGUE = ['very', 'really', 'thing', 'things', 'stuff', 'a lot', 'lots of', 'kind of', 'sort of', 'basically', 'pretty', 'good', 'bad', 'nice', 'big', 'got', 'etc'];
const EVIDENCE = [
  /\baccording to\b/gi, /\bstud(y|ies) (show|shows|found|suggest)/gi, /\bresearch (shows|suggests|found|indicates)\b/gi,
  /\(\s*[A-Z][A-Za-z.&\s-]+,?\s*\d{4}\s*\)/g, /\b(19|20)\d{2}\b/g, /\d+(\.\d+)?\s?%/g, /\bpercent\b/gi, /"[^"]{12,}"/g,
];

const words = (text) => (text.match(/[A-Za-z][A-Za-z'’-]*/g) || []);

export function syllables(word) {
  const w = word.toLowerCase().replace(/[^a-z]/g, '');
  if (!w) return 0;
  if (w.length <= 3) return 1;
  const trimmed = w.replace(/(?:[^laeiouy]es|ed|[^laeiouy]e)$/, '').replace(/^y/, '');
  const groups = trimmed.match(/[aeiouy]{1,2}/g);
  return Math.max(1, groups ? groups.length : 1);
}

export function splitParagraphs(text) {
  const byBlank = (text || '').split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
  if (byBlank.length > 1) return byBlank;
  return (text || '').split(/\n/).map((p) => p.trim()).filter(Boolean);
}

export function splitSentences(text) {
  return ((text || '').replace(/\s+/g, ' ').match(/[^.!?]+(?:[.!?]+["”’)]*|$)/g) || [])
    .map((s) => s.trim())
    .filter((s) => words(s).length > 0);
}

function countPhrases(lower, list) {
  const found = {};
  for (const phrase of list) {
    const re = new RegExp(`(^|[^a-z])${phrase.replace(/ /g, '\\s+')}(?=[^a-z]|$)`, 'g');
    const n = (lower.match(re) || []).length;
    if (n) found[phrase] = n;
  }
  return found;
}

const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
const sd = (xs) => {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1));
};

// Moving-average type-token ratio over 50-word windows (Covington &
// McFall, 2010): vocabulary variety that, unlike the plain ratio, does not
// fall just because the essay is longer.
function mattr(tokens, window = 50) {
  if (!tokens.length) return 0;
  if (tokens.length <= window) return new Set(tokens).size / tokens.length;
  let total = 0;
  const counts = new Map();
  for (let i = 0; i < window; i++) counts.set(tokens[i], (counts.get(tokens[i]) || 0) + 1);
  total += counts.size / window;
  for (let i = window; i < tokens.length; i++) {
    const out = tokens[i - window];
    counts.set(out, counts.get(out) - 1);
    if (!counts.get(out)) counts.delete(out);
    counts.set(tokens[i], (counts.get(tokens[i]) || 0) + 1);
    total += counts.size / window;
  }
  return total / (tokens.length - window + 1);
}

export function readingEaseBand(score) {
  if (score >= 80) return 'Easy (grade 6 or below)';
  if (score >= 70) return 'Fairly easy (grade 7)';
  if (score >= 60) return 'Plain English (grades 8-9)';
  if (score >= 50) return 'Fairly difficult (grades 10-12)';
  if (score >= 30) return 'Difficult (college)';
  return 'Very difficult (college graduate)';
}

export function analyzeEssay(text) {
  const paragraphs = splitParagraphs(text);
  const sentences = splitSentences(text);
  const allWords = words(text || '');
  const lowerWords = allWords.map((w) => w.toLowerCase().replace(/[’']s$/, ''));
  const lower = ` ${(text || '').toLowerCase().replace(/\s+/g, ' ')} `;
  const nWords = allWords.length;
  const nSentences = Math.max(1, sentences.length);

  const sentLens = sentences.map((s) => words(s).length);
  const syll = allWords.map(syllables);
  const totalSyll = syll.reduce((a, b) => a + b, 0);
  const complex = syll.filter((n) => n >= 3).length;

  // Flesch (1948) Reading Ease and Flesch-Kincaid Grade Level (Kincaid et al., 1975).
  const wps = nWords / nSentences;
  const spw = nWords ? totalSyll / nWords : 0;
  const readingEase = nWords ? 206.835 - 1.015 * wps - 84.6 * spw : 0;
  const gradeLevel = nWords ? 0.39 * wps + 11.8 * spw - 15.59 : 0;

  const transitions = {};
  let transitionTotal = 0;
  for (const [kind, list] of Object.entries(TRANSITIONS)) {
    const found = countPhrases(lower, list);
    const n = Object.values(found).reduce((a, b) => a + b, 0);
    if (n) { transitions[kind] = found; transitionTotal += n; }
  }
  const vague = countPhrases(lower, VAGUE);
  const vagueTotal = Object.values(vague).reduce((a, b) => a + b, 0);
  const evidence = EVIDENCE.reduce((n, re) => n + ((text || '').match(re) || []).length, 0);
  const passive = sentences.filter((s) => /\b(am|is|are|was|were|be|been|being)\s+(\w+ly\s+)?\w+(ed|en)\b/i.test(s)).length;
  const firstPerson = lowerWords.filter((w) => ['i', 'me', 'my', 'mine', 'myself'].includes(w)).length;
  const questions = sentences.filter((s) => s.trim().endsWith('?')).length;

  // Content words used three or more times.
  const freq = new Map();
  for (const w of lowerWords) {
    if (w.length < 4 || STOPWORDS.has(w)) continue;
    freq.set(w, (freq.get(w) || 0) + 1);
  }
  const repeated = [...freq.entries()].filter(([, n]) => n >= 3).sort((a, b) => b[1] - a[1]).slice(0, 8);

  // Sentences that start with the same word.
  const openers = new Map();
  for (const s of sentences) {
    const first = (words(s)[0] || '').toLowerCase();
    if (first) openers.set(first, (openers.get(first) || 0) + 1);
  }
  const repeatedOpeners = [...openers.entries()].filter(([, n]) => n >= 3 && n / nSentences >= 0.2).sort((a, b) => b[1] - a[1]);

  const paraInfo = paragraphs.map((p, i) => {
    const ps = splitSentences(p);
    const pw = words(p).length;
    const pl = ` ${p.toLowerCase()} `;
    const hasTransition = Object.values(TRANSITIONS).flat().some((t) => new RegExp(`(^|[^a-z])${t.replace(/ /g, '\\s+')}(?=[^a-z]|$)`).test(pl));
    const hasEvidence = EVIDENCE.some((re) => { re.lastIndex = 0; return re.test(p); });
    const role = paragraphs.length === 1 ? 'Whole essay'
      : i === 0 ? 'Introduction'
        : i === paragraphs.length - 1 ? 'Conclusion' : `Body ${i}`;
    const notes = [];
    if (pw < 40 && paragraphs.length > 1) notes.push('underdeveloped (under 40 words)');
    if (pw > 220) notes.push('very long; consider splitting');
    if (ps.length === 1 && paragraphs.length > 1) notes.push('single sentence');
    if (role.startsWith('Body') && !hasEvidence) notes.push('no example, data or source');
    if (i > 0 && !hasTransition) notes.push('no transition to the previous idea');
    if (role === 'Conclusion' && !/(in conclusion|to conclude|in summary|to sum up|overall|ultimately|therefore|thus)/.test(pl)) notes.push('conclusion not signalled');
    if (role === 'Introduction' && ps.length < 2) notes.push('thesis may need more context');
    if (!notes.length) notes.push('well developed');
    return {
      n: i + 1, role, words: pw, sentences: ps.length,
      opening: ps[0] || '', avgSentence: ps.length ? pw / ps.length : 0, notes,
    };
  });

  return {
    words: nWords,
    sentences: sentences.length,
    paragraphs: paragraphs.length,
    avgSentence: wps,
    sentenceSd: sd(sentLens),
    longSentences: sentLens.filter((n) => n > 35).length,
    shortSentences: sentLens.filter((n) => n < 6).length,
    longest: sentLens.length ? Math.max(...sentLens) : 0,
    avgWordLength: nWords ? allWords.reduce((a, w) => a + w.length, 0) / nWords : 0,
    complexPct: nWords ? (complex / nWords) * 100 : 0,
    uniqueWords: new Set(lowerWords).size,
    mattr: mattr(lowerWords),
    readingEase,
    gradeLevel,
    transitions,
    transitionTotal,
    transitionKinds: Object.keys(transitions).length,
    vague,
    vagueTotal,
    evidence,
    passive,
    firstPerson,
    questions,
    repeated,
    repeatedOpeners,
    paragraphsInfo: paraInfo,
  };
}

// Plain-language reading of each figure, with the range usually expected
// in a secondary / college academic essay.
export function interpret(a) {
  const per100 = (n) => (a.words ? (n / a.words) * 100 : 0);
  return [
    {
      label: 'Average sentence length', value: `${a.avgSentence.toFixed(1)} words`, target: '15-25 words',
      note: a.avgSentence < 12 ? 'Short, choppy sentences: combine related ideas.'
        : a.avgSentence > 28 ? 'Long sentences: split some for clarity.' : 'Within the usual range.',
    },
    {
      label: 'Sentence variety (std. dev.)', value: `${a.sentenceSd.toFixed(1)} words`, target: 'above 6',
      note: a.sentenceSd < 4 ? 'Sentences are similar in length: vary them for rhythm.' : 'Good mix of short and long sentences.',
    },
    {
      label: 'Very long sentences (> 35 words)', value: String(a.longSentences), target: '0-1',
      note: a.longSentences > 1 ? 'Check these for run-ons and missing punctuation.' : 'None or few.',
    },
    {
      label: 'Flesch Reading Ease', value: a.readingEase.toFixed(0), target: '30-60 (academic)',
      note: readingEaseBand(a.readingEase),
    },
    {
      label: 'Flesch-Kincaid Grade Level', value: a.gradeLevel.toFixed(1), target: '9-13',
      note: a.gradeLevel < 8 ? 'Simpler than typical academic writing.' : a.gradeLevel > 16 ? 'Very dense; may be hard to follow.' : 'Appropriate for academic writing.',
    },
    {
      label: 'Vocabulary variety (MATTR)', value: a.mattr.toFixed(2), target: 'above 0.70',
      note: a.mattr < 0.65 ? 'Many words repeat: use synonyms and more precise terms.' : a.mattr >= 0.75 ? 'Rich, varied vocabulary.' : 'Adequate variety.',
    },
    {
      label: 'Complex words (3+ syllables)', value: `${a.complexPct.toFixed(1)}%`, target: '10-20%',
      note: a.complexPct < 8 ? 'Mostly simple words: add precise academic vocabulary.' : a.complexPct > 25 ? 'Heavy wording: prefer plain words where possible.' : 'Balanced.',
    },
    {
      label: 'Transitions', value: `${a.transitionTotal} (${a.transitionKinds} kinds)`, target: '1+ per paragraph, 3+ kinds',
      note: a.transitionTotal < a.paragraphs ? 'Few links between ideas: add transitions.' : a.transitionKinds < 3 ? 'Use a wider range (contrast, cause, example).' : 'Ideas are well connected.',
    },
    {
      label: 'Evidence markers', value: String(a.evidence), target: '1+ per body paragraph',
      note: a.evidence === 0 ? 'No sources, data or quotations found: support claims with evidence.' : 'Evidence is present; check that each claim is supported.',
    },
    {
      label: 'Passive constructions', value: `${a.passive} sentence${a.passive === 1 ? '' : 's'}`, target: 'under 20% of sentences',
      note: a.sentences && a.passive / a.sentences > 0.2 ? 'Frequent passive voice: prefer active verbs.' : 'Mostly active voice.',
    },
    {
      label: 'Vague words', value: `${a.vagueTotal} (${per100(a.vagueTotal).toFixed(1)} per 100 words)`, target: 'under 1 per 100 words',
      note: per100(a.vagueTotal) >= 1 ? `Replace vague words (${Object.keys(a.vague).slice(0, 4).join(', ')}) with specific ones.` : 'Word choice is specific.',
    },
    {
      label: 'First-person pronouns', value: String(a.firstPerson), target: 'depends on essay type',
      note: per100(a.firstPerson) > 2 ? 'Frequent "I/my": fine for personal essays, limit it in argumentative ones.' : 'Mostly impersonal academic voice.',
    },
  ];
}

export const ANALYSIS_REFERENCES = [
  'Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: The moving-average type-token ratio (MATTR). Journal of Quantitative Linguistics, 17(2), 94-100.',
  'Flesch, R. (1948). A new readability yardstick. Journal of Applied Psychology, 32(3), 221-233.',
  'Halliday, M. A. K., & Hasan, R. (1976). Cohesion in English. Longman.',
  'Kincaid, J. P., Fishburne, R. P., Rogers, R. L., & Chissom, B. S. (1975). Derivation of new readability formulas for Navy enlisted personnel (Research Branch Report 8-75). Naval Technical Training Command.',
];

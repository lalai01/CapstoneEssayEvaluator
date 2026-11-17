// Removes a student's identifying details from an essay before it leaves
// the browser (Data Privacy Act of 2012, RA 10173): the name and other
// details in the paper's heading, the LRN (12-digit Learner Reference
// Number), email addresses, phone numbers, and any names given. The server
// applies the same rules again (backend/privacy.py).

export const REDACTED = '[REDACTED]';
const HEADER_LINES = 8;   // the heading of a school paper is in its first lines

const LABELS = [
  "name", "full\\s*name", "student(?:'s)?\\s*name", "learner(?:'s)?\\s*name", 'pangalan', 'ngalan',
  'grade\\s*(?:level)?\\s*(?:&|and|/)?\\s*section', 'gr\\.?\\s*(?:&|/)?\\s*sec\\.?', 'year\\s*(?:&|and)\\s*section',
  'section', 'seksyon', 'grade', 'grado', 'lrn', 'learner\\s*reference\\s*(?:number|no\\.?)', 'student\\s*(?:id|no\\.?|number)',
  'id\\s*(?:no\\.?|number)', 'school', 'paaralan', 'eskwelahan', 'teacher', 'guro', 'adviser', 'tagapayo',
  'address', 'tirahan', 'birthday', 'birth\\s*date', 'age', 'edad', 'contact\\s*(?:no\\.?|number)', 'parent', 'guardian', 'magulang',
].join('|').replace(/\\s/g, '[ \\t]');

const HEADER_RE = new RegExp(`^([ \\t]*(?:${LABELS})[ \\t]*[:\\-–][ \\t]*)(\\S[^\\n]*?)[ \\t]*$`, 'gim');
const PATTERNS = [
  ['lrn', /(?<!\d)\d{12}(?!\d)/g, '[LRN REDACTED]'],
  ['email', /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g, '[EMAIL REDACTED]'],
  ['phone', /(?<!\d)(?:\+?63|0)9\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)/g, '[PHONE REDACTED]'],
];

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// → { text, findings: [{ type, text }] }
export function redactPersonalInfo(text, names = []) {
  if (!text) return { text: text || '', findings: [] };
  const findings = [];

  const lines = text.split('\n');
  let headEnd = 0;
  let seen = 0;
  for (let i = 0; i < lines.length; i++) {
    headEnd = i + 1;
    if (lines[i].trim()) {
      seen += 1;
      if (seen >= HEADER_LINES) break;
    }
  }
  const head = lines.slice(0, headEnd).join('\n').replace(HEADER_RE, (all, label) => {
    findings.push({ type: 'header', text: all.trim() });
    return `${label}${REDACTED}`;
  });
  let out = [head, ...lines.slice(headEnd)].join('\n');

  for (const [type, re, label] of PATTERNS) {
    out = out.replace(re, (m) => { findings.push({ type, text: m }); return label; });
  }

  const parts = new Set();
  for (const raw of names) {
    const name = (raw || '').trim();
    if (!name) continue;
    parts.add(name);
    for (const p of name.split(/[\s,.]+/)) if (p.length >= 3) parts.add(p);
  }
  for (const part of [...parts].sort((a, b) => b.length - a.length)) {
    const re = new RegExp(`(?<![\\w\\[])${escapeRe(part)}(?![\\w\\]])`, 'gi');
    out = out.replace(re, (m) => { findings.push({ type: 'name', text: m }); return REDACTED; });
  }
  return { text: out, findings };
}

// The Evaluate page's "minor's essay" settings when nothing is ticked.
export const NO_MINOR = { minor: false, guardianConsent: false, localOnly: false, studentName: '' };

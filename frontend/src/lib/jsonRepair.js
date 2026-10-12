// Recover the readable part of a JSON object that was cut off mid-answer
// (small local models sometimes stop early or drift out of the format).
//
// The text is scanned once, recording every point where a complete value
// ends, together with the brackets still open there. Working backwards from
// the last such point, the open brackets are closed and the result is
// parsed; the first candidate that parses is returned. Scores at the start
// of the answer therefore survive even if the end is missing.
//
// Returns { value, partial } or null when nothing usable can be recovered.

export function salvageJson(raw) {
  const text = String(raw || '');
  const start = text.indexOf('{');
  if (start < 0) return null;
  const s = text.slice(start);

  const stack = []; // frames: { type: '{' | '[', expect: 'key' | 'colon' | 'value' | 'comma' }
  const cuts = []; // { end, closers }
  let complete = false;

  const closersNow = () => stack.map((f) => (f.type === '{' ? '}' : ']')).reverse().join('');
  const valueDone = (end) => {
    const top = stack[stack.length - 1];
    if (top) top.expect = 'comma';
    cuts.push({ end, closers: closersNow() });
  };

  let i = 0;
  while (i < s.length) {
    const c = s[i];
    if (c === ' ' || c === '\n' || c === '\r' || c === '\t') { i++; continue; }
    const top = stack[stack.length - 1];

    if (c === '"') {
      let j = i + 1;
      let escaped = false;
      let closed = false;
      for (; j < s.length; j++) {
        const d = s[j];
        if (escaped) { escaped = false; continue; }
        if (d === '\\') { escaped = true; continue; }
        if (d === '"') { closed = true; break; }
      }
      if (!closed) break; // cut off inside a string
      if (top && top.type === '{' && top.expect === 'key') top.expect = 'colon';
      else valueDone(j + 1);
      i = j + 1;
      continue;
    }
    if (c === ':') { if (top) top.expect = 'value'; i++; continue; }
    if (c === ',') { if (top) top.expect = top.type === '{' ? 'key' : 'value'; i++; continue; }
    if (c === '{' || c === '[') {
      stack.push({ type: c, expect: c === '{' ? 'key' : 'value' });
      i++;
      continue;
    }
    if (c === '}' || c === ']') {
      stack.pop();
      valueDone(i + 1);
      i++;
      if (stack.length === 0) { complete = true; break; }
      continue;
    }
    // number, true, false or null: ends at a delimiter
    let j = i;
    while (j < s.length && !/[\s,\]}]/.test(s[j])) j++;
    if (j >= s.length) break; // possibly cut off mid-number
    valueDone(j);
    i = j;
  }

  for (let k = cuts.length - 1; k >= 0; k--) {
    const { end, closers } = cuts[k];
    try {
      const value = JSON.parse(s.slice(0, end) + closers);
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        return { value, partial: !(complete && k === cuts.length - 1) };
      }
    } catch {
      // try an earlier cut point
    }
  }
  return null;
}

import { useEffect, useState } from 'react';

// Drafts: work in progress kept in this browser (localStorage) so a refresh,
// a closed tab or switching pages doesn't lose it. Signing out clears them,
// so the next person on a shared computer starts empty.
const PREFIX = 'essay-evaluator:draft:';

export function readDraft(key, fallback) {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;   // storage blocked (private mode) or a corrupt value
  }
}

export function writeDraft(key, value) {
  try {
    if (value === null || value === undefined || value === '') localStorage.removeItem(PREFIX + key);
    else localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    // Storage full or blocked: the page still works, the draft just isn't kept.
  }
}

export function clearDrafts() {
  try {
    Object.keys(localStorage)
      .filter((k) => k.startsWith(PREFIX))
      .forEach((k) => localStorage.removeItem(k));
  } catch {
    // nothing stored
  }
}

// useState that is saved as a draft under `key`. `restore` adjusts a value
// read back after a refresh (e.g. work that was interrupted).
export default function useDraft(key, initial, restore) {
  const [value, setValue] = useState(() => {
    const saved = readDraft(key, initial);
    return restore && saved !== initial ? restore(saved) : saved;
  });
  useEffect(() => {
    writeDraft(key, value === initial ? null : value);
  }, [key, value, initial]);
  return [value, setValue];
}

import React from 'react';

const READER_NAMES = {
  'gemini-vision': 'Gemini vision (cloud)',
  'trocr-handwriting': 'TrOCR (local, on our server)',
};

const pct = (x) => `${(x * 100).toFixed(1)}%`;

// Each reader's measured accuracy on the pages admins have verified in the
// OCR Labeler: what the reader produced vs. what is really on the page.
export default function ReaderAccuracy({ stats }) {
  if (!stats) return null;
  const readers = stats.readers || [];

  return (
    <section className="glass-card rounded-2xl shadow-xl p-6 bg-white/95" aria-labelledby="reader-accuracy-title">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="reader-accuracy-title" className="block text-xl font-bold text-gray-800">📊 Reader accuracy on verified pages</h2>
        <p className="text-sm text-gray-600">
          Dataset: <strong>{stats.verified_pages}</strong> verified pages ·{' '}
          <strong>{stats.verified_lines}</strong> lines · {stats.pending_pages} waiting for review
        </p>
      </div>

      {readers.length === 0 ? (
        <p className="mt-3 text-sm text-gray-600">
          No verified pages yet. Each page you save below becomes ground truth, and the accuracy of the reader that
          read it appears here.
        </p>
      ) : (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-600 border-b border-gray-200">
                <th className="py-2 pr-4 font-semibold">Reader</th>
                <th className="py-2 pr-4 font-semibold text-right">Pages</th>
                <th className="py-2 pr-4 font-semibold text-right">Words</th>
                <th className="py-2 pr-4 font-semibold text-right" title="Character error rate">Character errors</th>
                <th className="py-2 pr-4 font-semibold text-right" title="Word error rate">Word errors</th>
                <th className="py-2 font-semibold text-right">Word accuracy</th>
              </tr>
            </thead>
            <tbody>
              {readers.map((r) => (
                <tr key={r.engine} className="border-b border-gray-100 last:border-0">
                  <td className="py-2 pr-4 text-gray-800">{READER_NAMES[r.engine] || r.engine}</td>
                  <td className="py-2 pr-4 text-right tabular-nums">{r.pages}</td>
                  <td className="py-2 pr-4 text-right tabular-nums">{r.words.toLocaleString()}</td>
                  <td className="py-2 pr-4 text-right tabular-nums">{pct(r.cer)}</td>
                  <td className="py-2 pr-4 text-right tabular-nums">{pct(r.wer)}</td>
                  <td className="py-2 text-right tabular-nums font-semibold text-gray-800">{pct(Math.max(0, 1 - r.wer))}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-xs text-gray-500">
            Measured on the reader&apos;s own text before any AI correction, against the transcription an admin
            verified. Lower error is better. TrOCR only reads pages when Gemini is unavailable (out of quota or offline).
          </p>
        </div>
      )}
    </section>
  );
}

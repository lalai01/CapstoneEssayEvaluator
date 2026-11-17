import React from 'react';
import { NO_MINOR } from '../lib/redact';

// "This essay was written by a minor": the protections the Data Privacy Act
// (RA 10173) calls for when the student is under 18. Set before uploading
// photos, so they are read and handled accordingly.
//   value: { minor, guardianConsent, localOnly, studentName }

// locked: the user's own account is a minor's (the box stays ticked and the
// consent is the one a teacher confirmed on the account).
export default function MinorProtection({ value, onChange, highlight, locked = false }) {
  const set = (patch) => onChange({ ...value, ...patch });
  return (
    <div className={`mb-5 rounded-xl border p-4 ${value.minor ? 'border-sky-300 bg-sky-50/70' : 'border-gray-200 bg-gray-50'}`}>
      <label className="flex items-start gap-2 text-sm font-semibold text-gray-800">
        <input
          type="checkbox"
          checked={value.minor}
          disabled={locked}
          onChange={(e) => set(e.target.checked ? { minor: true } : { ...NO_MINOR })}
          className="mt-0.5 h-4 w-4"
        />
        <span>
          🧒 The student is a minor (under 18)
          <span className="block text-xs font-normal text-gray-600">
            Tick this before uploading photos. The essay’s personal details are removed before any AI sees it,
            and its photos are not kept.
          </span>
        </span>
      </label>

      {value.minor && (
        <div className="mt-3 space-y-3 border-t border-sky-200 pt-3 text-sm">
          {locked ? (
            <p className="rounded-lg bg-white/70 p-2 text-gray-800">
              ✓ Your account is for a student under 18, and your teacher confirmed your parent’s or guardian’s
              consent. These protections are always on for your essays.
            </p>
          ) : (
          <label className={`flex items-start gap-2 rounded-lg p-2 ${highlight ? 'bg-red-50 ring-2 ring-red-400' : ''}`}>
            <input
              id="guardian-consent"
              type="checkbox"
              checked={value.guardianConsent}
              onChange={(e) => set({ guardianConsent: e.target.checked })}
              className="mt-0.5 h-4 w-4"
            />
            <span className="text-gray-800">
              <strong>Required:</strong> the school has the parent’s or guardian’s consent for this student’s work to
              be evaluated with this system (for example, the consent form signed at enrollment).
            </span>
          </label>
          )}

          <label className="block">
            <span className="text-gray-700">Student’s name to remove <span className="text-gray-400">(optional)</span></span>
            <input
              type="text"
              value={value.studentName}
              onChange={(e) => set({ studentName: e.target.value })}
              placeholder="e.g., Juan Dela Cruz"
              autoComplete="off"
              className="mt-1 w-full rounded-lg border border-gray-300 bg-white p-2"
            />
            <span className="mt-1 block text-xs text-gray-500">
              Removed wherever it appears in the essay. The heading (Name, LRN, Grade &amp; Section, School, Teacher),
              12-digit LRNs, emails and phone numbers are removed automatically. This name is not saved.
            </span>
          </label>

          <label className="flex items-start gap-2">
            <input
              type="checkbox"
              checked={value.localOnly}
              onChange={(e) => set({ localOnly: e.target.checked })}
              className="mt-0.5 h-4 w-4"
            />
            <span className="text-gray-800">
              Keep this essay on our server only
              <span className="block text-xs text-gray-600">
                Photos are read by the local readers instead of Google Gemini, and only Qwen (local) evaluates it:
                nothing is sent to DeepSeek, Groq or Gemini. Reading handwriting is less accurate and slower, and no
                translation is done.
              </span>
            </span>
          </label>

          <p className="text-xs text-gray-600">
            Minors’ essays are deleted automatically after the retention period (about one school year), and you can
            delete one any time in My Reports.
          </p>
        </div>
      )}
    </div>
  );
}

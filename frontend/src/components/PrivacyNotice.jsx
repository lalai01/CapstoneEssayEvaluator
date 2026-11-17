import React, { useEffect, useState } from 'react';
import { saveConsent } from '../lib/privacy';

// Data Privacy Notice, in line with the Philippine Data Privacy Act of 2012
// (Republic Act No. 10173). The full notice is shown on the Home page and in a
// dialog from the Evaluate page; before a first evaluation the user agrees to
// it once per browser.

export function PrivacyNoticeText() {
  return (
    <div className="space-y-3 text-sm text-gray-700">
      <p>
        This system complies with the <strong>Data Privacy Act of 2012 (Republic Act No. 10173)</strong>, its
        Implementing Rules and Regulations, and the issuances of the National Privacy Commission (NPC). We process
        personal data only for a declared, specific and legitimate purpose, following the principles of
        transparency, legitimate purpose and proportionality.
      </p>
      <div>
        <h4 className="font-semibold text-gray-800">What we collect</h4>
        <ul className="mt-1 list-disc pl-5 space-y-0.5">
          <li>The essays you type or upload (text, photos, PDF or Word files) and their titles.</li>
          <li>If you sign in: your name and email address (from Google, Yahoo or your email sign-in), and your role (student, teacher or admin).</li>
          <li>The scores, feedback, overrides, remarks, ratings and survey answers you create in the system.</li>
        </ul>
      </div>
      <div>
        <h4 className="font-semibold text-gray-800">Why we collect it</h4>
        <p className="mt-1">
          Only to evaluate essays, give feedback, keep your reports, let teachers review results, and improve the
          system. Results used in the research study are reported in aggregate, with no names.
        </p>
      </div>
      <div>
        <h4 className="font-semibold text-gray-800">Who processes it</h4>
        <ul className="mt-1 list-disc pl-5 space-y-0.5">
          <li>To be scored, the essay text is sent to <strong>DeepSeek</strong> and <strong>Groq</strong> (GPT-OSS 120B), and to <strong>Google Gemini</strong> when Groq is unavailable or when a photo is read. These services may store or process data outside the Philippines under their own privacy policies.</li>
          <li>The <strong>Qwen 2.5 (local)</strong> result is produced on our own server: for that one, the essay does not leave it.</li>
          <li>Data is stored in <strong>Supabase</strong> (database and sign-in), with access controls so users see only their own records, and is sent over encrypted HTTPS connections.</li>
        </ul>
      </div>
      <div>
        <h4 className="font-semibold text-gray-800">Handwriting samples</h4>
        <p className="mt-1">
          To improve how handwriting is read, photos of pages uploaded by signed-in users may be kept in private
          storage, viewed only by admins through links that expire, and used to train the handwriting reader. Pages
          uploaded by guests, and pages of essays marked as written by a minor, are never kept.
        </p>
      </div>
      <div className="rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-sky-900">
        <h4 className="font-semibold">Essays written by minors (under 18)</h4>
        <p className="mt-1">
          An essay with a student’s name, grade and section is information about the student’s education, which the
          law treats as <strong>sensitive personal information</strong>. A minor cannot give consent alone: the school
          obtains the <strong>parent’s or guardian’s consent</strong> (for example, at enrollment), and the teacher
          confirms it before evaluating. For a minor’s essay the system:
        </p>
        <ul className="mt-1 list-disc pl-5 space-y-0.5">
          <li>removes the name, LRN, grade and section, school, email and phone number before any AI sees the essay;</li>
          <li>never keeps its photos;</li>
          <li>can keep it entirely on our server (local reader and Qwen, no cloud AI) when the teacher chooses;</li>
          <li>deletes it automatically after about one school year; it can also be deleted any time from My Reports.</li>
        </ul>
        <p className="mt-1">
          The school is the personal information controller and this system processes the data on its behalf, under a
          data processing agreement. A data breach is reported to the National Privacy Commission and the affected
          families as the law requires.
        </p>
      </div>
      <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-amber-900">
        <strong>Before you evaluate:</strong> remove the student’s full name, ID number, section and any other
        personal details from the essay text. The <em>Student name</em> you type when downloading a report is put only
        in the PDF on your device; it is not sent to the AIs.
      </div>
      <div>
        <h4 className="font-semibold text-gray-800">How long we keep it</h4>
        <p className="mt-1">
          While your account is active, or until you delete it: any essay and all its results can be deleted from
          My Reports. Minors’ essays are deleted automatically after about one school year. Data no longer needed for
          its purpose is deleted.
        </p>
      </div>
      <div>
        <h4 className="font-semibold text-gray-800">Your rights</h4>
        <p className="mt-1">
          You have the right to be informed, to access, to object, to correct (rectification), to have your data
          erased or blocked, to data portability, to damages, and to file a complaint with the National Privacy
          Commission (privacy.gov.ph). To exercise them, contact the system administrator or your school’s Data
          Protection Officer.
        </p>
      </div>
      <div>
        <h4 className="font-semibold text-gray-800">Disclaimer</h4>
        <p className="mt-1">
          AI scores and feedback are a <strong>decision-support tool</strong> only. They can be wrong and are not
          final grades: the teacher reviews, overrides when needed, and decides the final grade.
        </p>
      </div>
    </div>
  );
}

export function PrivacyDialog({ open, onClose }) {
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 p-4 animate-fade-in" onMouseDown={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="privacy-dialog-title"
        className="w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl bg-white p-6 shadow-2xl"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <h3 id="privacy-dialog-title" className="text-lg font-bold text-gray-900">🔒 Data Privacy Notice</h3>
        <p className="text-xs text-gray-500 mb-3">Data Privacy Act of 2012 (Republic Act No. 10173)</p>
        <PrivacyNoticeText />
        <div className="mt-5 flex justify-end">
          <button type="button" onClick={onClose} className="rounded-xl bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700">
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

// The short notice and consent box shown above the Evaluate button.
export function PrivacyConsent({ agreed, onChange, highlight }) {
  const [open, setOpen] = useState(false);
  const toggle = (value) => {
    saveConsent(value);
    onChange(value);
  };
  return (
    <div className={`mt-5 rounded-xl border p-3 text-xs ${highlight ? 'border-red-400 bg-red-50' : 'border-gray-200 bg-gray-50'}`}>
      <p className="text-gray-700">
        🔒 <strong>Data Privacy (RA 10173):</strong> remove the student’s name and other personal details from the
        essay. The text is sent to DeepSeek and Groq/Gemini (cloud) and to Qwen on our own server. If the student is
        under 18, tick <em>“The student is a minor”</em> above. AI scores support, and do not replace, the teacher’s
        judgement.{' '}
        <button type="button" onClick={() => setOpen(true)} className="font-semibold text-indigo-700 underline">
          Read the full privacy notice
        </button>
      </p>
      <label className="mt-2 flex items-start gap-2 text-gray-800">
        <input
          id="privacy-consent"
          type="checkbox"
          checked={agreed}
          onChange={(e) => toggle(e.target.checked)}
          className="mt-0.5 h-4 w-4"
        />
        <span>I have read the Data Privacy Notice and agree to the processing of this essay for evaluation.</span>
      </label>
      <PrivacyDialog open={open} onClose={() => setOpen(false)} />
    </div>
  );
}

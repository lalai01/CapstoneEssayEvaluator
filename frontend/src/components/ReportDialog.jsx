import React, { useState } from 'react';
import toast from 'react-hot-toast';
import { generateEvaluationReport } from '../lib/report';
import { reviseEssay } from '../api';
import { useAuth } from '../context/AuthContext';
import { errorMessage } from '../lib/scoring';

// Collects the details a printed report needs (who it's for, who evaluated
// it, teacher comments) and generates the PDF.
export default function ReportDialog({ onClose, defaults, reportData }) {
  const [studentName, setStudentName] = useState('');
  const [section, setSection] = useState('');
  const [teacherName, setTeacherName] = useState(defaults.teacherName || '');
  const [teacherComments, setTeacherComments] = useState(defaults.teacherComments || '');
  const { isTeacher } = useAuth();
  const [includeEssay, setIncludeEssay] = useState(true);
  const [includeRevision, setIncludeRevision] = useState(false);
  const [revision, setRevision] = useState(null); // reused if the PDF is downloaded again
  const [step, setStep] = useState(''); // '' | 'revising' | 'pdf'

  const generate = async () => {
    let revised = revision;
    if (includeRevision && isTeacher && !revised) {
      setStep('revising');
      try {
        revised = await reviseEssay(
          reportData.essayText,
          reportData.scores?.grammar_issues || [],
          reportData.scores?.recommendations || [],
        );
        setRevision(revised);
      } catch (err) {
        setStep('');
        toast.error('Could not generate the revised version: ' + errorMessage(err));
        return;
      }
    }
    setStep('pdf');
    try {
      generateEvaluationReport({
        ...reportData,
        studentName: studentName.trim(),
        section: section.trim(),
        teacherName: teacherName.trim(),
        teacherComments: teacherComments.trim(),
        includeEssay,
        revision: includeRevision ? revised : null,
      });
      toast.success('Report downloaded');
      onClose();
    } catch (err) {
      console.error('[report] generation failed', err);
      toast.error('Could not generate the report: ' + err.message);
    } finally {
      setStep('');
    }
  };
  const generating = step !== '';

  const field = 'w-full border rounded-xl p-2.5 focus:ring-2 focus:ring-blue-500 bg-white';

  return (
    <div
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 animate-fade-in p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="report-dialog-title"
    >
      <div className="bg-white rounded-2xl p-6 max-w-lg w-full shadow-2xl max-h-[90vh] overflow-y-auto">
        <h3 id="report-dialog-title" className="text-2xl font-bold text-gray-800 mb-1">📄 Evaluation Report</h3>
        <p className="text-sm text-gray-500 mb-5">
          A printable PDF with the scores, rubric breakdown, feedback and recommendations.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-3">
          <label className="text-sm font-semibold text-gray-700">
            Student name
            <input value={studentName} onChange={(e) => setStudentName(e.target.value)} className={`${field} mt-1 font-normal`} placeholder="e.g., Juan Dela Cruz" />
          </label>
          <label className="text-sm font-semibold text-gray-700">
            Class / section
            <input value={section} onChange={(e) => setSection(e.target.value)} className={`${field} mt-1 font-normal`} placeholder="e.g., Grade 10 - Rizal" />
          </label>
        </div>

        <label className="block text-sm font-semibold text-gray-700 mb-3">
          Evaluated by
          <input value={teacherName} onChange={(e) => setTeacherName(e.target.value)} className={`${field} mt-1 font-normal`} />
        </label>

        <label className="block text-sm font-semibold text-gray-700 mb-3">
          Teacher's comments <span className="font-normal text-gray-400">(optional; left blank, the report has lines to write on)</span>
          <textarea
            rows="4"
            value={teacherComments}
            onChange={(e) => setTeacherComments(e.target.value)}
            className={`${field} mt-1 font-normal`}
          />
        </label>

        <label className="flex items-center gap-2 text-sm text-gray-700 mb-3 cursor-pointer">
          <input type="checkbox" checked={includeEssay} onChange={(e) => setIncludeEssay(e.target.checked)} className="w-4 h-4" />
          Include the essay with the issues highlighted (appendix)
        </label>

        {isTeacher && (
          <label className="flex items-start gap-2 text-sm text-gray-700 mb-5 cursor-pointer">
            <input
              type="checkbox"
              checked={includeRevision}
              onChange={(e) => setIncludeRevision(e.target.checked)}
              className="w-4 h-4 mt-0.5"
            />
            <span>
              Include an AI-revised version as a reference
              <span className="block text-xs text-gray-500">
                Keeps the student's ideas, fixes the marked issues, and leaves [placeholders] where evidence is needed.
                Clearly labelled as AI-generated. Takes up to a minute.
              </span>
            </span>
          </label>
        )}

        {reportData.teacherScores && (
          <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-lg p-2 mb-4">
            Your override scores will appear next to the AI scores.
          </p>
        )}

        <div className="flex justify-end gap-3">
          <button onClick={onClose} className="px-4 py-2 bg-gray-200 rounded-xl hover:bg-gray-300 transition">
            Cancel
          </button>
          <button
            onClick={generate}
            disabled={generating}
            className="px-4 py-2 bg-gradient-to-r from-sky-500 to-cyan-500 text-white rounded-xl font-semibold hover:shadow-lg transition disabled:opacity-50"
          >
            {step === 'revising' ? 'Writing revised version…' : step === 'pdf' ? 'Generating…' : 'Download PDF'}
          </button>
        </div>
      </div>
    </div>
  );
}

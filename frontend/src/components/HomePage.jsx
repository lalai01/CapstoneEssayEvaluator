import React, { useEffect } from 'react';
import RobotAssistant from './RobotAssistant';
import useInstallPrompt from '../lib/installPrompt';

const HIGHLIGHTS = [
  { icon: '✍️', text: 'Reads handwriting in seconds' },
  { icon: '📐', text: 'Holistic & analytic rubrics' },
  { icon: '👩‍🏫', text: 'Teacher in the loop' },
];

const STEPS = [
  {
    n: 1, color: 'bg-blue-100 text-blue-700', title: 'Upload or type',
    text: 'Paste your essay, or upload a photo, scan, PDF or Word file. Handwriting is read with Google Gemini, printed pages with Tesseract, and images that aren’t essays are turned away.',
  },
  {
    n: 2, color: 'bg-emerald-100 text-emerald-700', title: 'AI evaluation',
    text: 'Scored against your teacher’s rubric with DeepSeek or Llama 3.2 (or let Auto pick), with a grammar check by LanguageTool, written feedback and the parts that need work highlighted.',
  },
  {
    n: 3, color: 'bg-violet-100 text-violet-700', title: 'Review & improve',
    text: 'Download a PDF report. Teachers can override scores with remarks, and every review is kept so later evaluations learn from it.',
  },
];

const FEATURES = [
  { icon: '📄', title: 'Handwriting & PDF reading', text: 'Photos, scans, PDFs and Word files. Unreadable words are flagged so you can fix them before scoring.' },
  { icon: '📐', title: 'Teacher-defined rubrics', text: 'Each teacher sets their own criteria and weights; the school can set a default rubric.' },
  { icon: '🖍️', title: 'Highlighted essay & reports', text: 'See exactly which sentences need work, on screen and in the PDF report.' },
  { icon: '✏️', title: 'Teacher override', text: 'Correct a score, add remarks, and approve results; corrections go into the learning hub.' },
  { icon: '🧠', title: 'Learns from past reviews', text: 'Similar, teacher-approved evaluations are retrieved (RAG) to keep feedback consistent.' },
  { icon: '⚖️', title: 'Compare with your scores', text: 'DeepSeek, Llama 3.2 and Groq grade every essay; enter your own scores to see which AI is closest to you.' },
];

export default function HomePage({ setActiveTab }) {
  const { canInstall, install } = useInstallPrompt();
  useEffect(() => {
    const sections = document.querySelectorAll('.scroll-section');
    const observer = new IntersectionObserver(
      (entries, obs) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            // Once visible, stop watching — no more re-triggers, no flicker.
            obs.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15 }
    );
    sections.forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, []);

  return (
    <div className="space-y-10 animate-fade-in relative">
      {/* ============ HERO ============ */}
      <section className="glass-card rounded-3xl shadow-2xl relative z-10 bg-white/95 overflow-hidden">
        <div className="absolute -top-24 -right-24 h-80 w-80 rounded-full bg-sky-200/50 blur-3xl pointer-events-none" aria-hidden="true" />
        <div className="absolute -bottom-24 -left-16 h-72 w-72 rounded-full bg-indigo-200/40 blur-3xl pointer-events-none" aria-hidden="true" />

        <div className="relative flex flex-col lg:flex-row items-center gap-6 p-8 md:p-12">
          <div className="flex-1 text-center lg:text-left">
            <span className="inline-flex items-center gap-2 rounded-full bg-blue-50 border border-blue-100 px-3 py-1 text-xs font-semibold text-blue-700">
              <span className="h-2 w-2 rounded-full bg-sky-500 animate-pulse" aria-hidden="true" />
              Handwritten, PDF, Word or typed
            </span>
            <h1 className="mt-4 text-4xl md:text-6xl font-extrabold tracking-tight gradient-text">
              AI Essay Evaluator
            </h1>
            <p className="mt-4 text-lg md:text-xl text-gray-700 max-w-xl mx-auto lg:mx-0">
              Fair, detailed feedback on English essays in minutes, scored against your teacher’s own rubric.
            </p>

            <ul className="mt-6 flex flex-wrap justify-center lg:justify-start gap-2">
              {HIGHLIGHTS.map((h) => (
                <li key={h.text} className="flex items-center gap-2 rounded-xl bg-white border border-gray-200 px-3 py-2 text-sm text-gray-700 shadow-sm">
                  <span aria-hidden="true">{h.icon}</span>{h.text}
                </li>
              ))}
            </ul>

            <div className="mt-8 flex flex-wrap justify-center lg:justify-start gap-3">
              <button
                onClick={() => setActiveTab('evaluate')}
                className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-6 py-3 rounded-xl font-semibold shadow-lg hover:shadow-xl hover:brightness-110 transition"
              >
                Start evaluating →
              </button>
              <a href="#how-it-works" className="bg-white border border-gray-300 text-gray-800 px-6 py-3 rounded-xl font-semibold hover:bg-gray-50 transition">
                How it works
              </a>
              {canInstall && (
                <button
                  onClick={install}
                  className="bg-white border border-indigo-300 text-indigo-700 px-6 py-3 rounded-xl font-semibold hover:bg-indigo-50 transition"
                >
                  📲 Install the app
                </button>
              )}
            </div>
          </div>

          <div className="flex-1 flex justify-center">
            <div className="relative w-72 h-80 sm:w-80 sm:h-96 md:w-[26rem] md:h-[28rem]">
              <div className="absolute inset-8 rounded-full bg-gradient-to-br from-sky-100 to-indigo-100 blur-2xl" aria-hidden="true" />
              <div className="relative w-full h-full">
                <RobotAssistant />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============ HOW IT WORKS ============ */}
      <section id="how-it-works" className="scroll-section glass-card rounded-3xl shadow-xl p-8 md:p-10 relative z-10 bg-white/95">
        <h2 className="block text-3xl font-bold text-center text-gray-800">How it works</h2>
        <p className="text-center text-gray-600 mt-2">From page to feedback in three steps.</p>
        <ol className="mt-8 grid md:grid-cols-3 gap-6">
          {STEPS.map((s) => (
            <li key={s.n} className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
              <span className={`inline-flex h-10 w-10 items-center justify-center rounded-full font-bold ${s.color}`}>{s.n}</span>
              <h3 className="mt-4 text-lg font-semibold text-gray-800">{s.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-gray-600">{s.text}</p>
            </li>
          ))}
        </ol>
      </section>

      {/* ============ FEATURES ============ */}
      <section className="scroll-section glass-card rounded-3xl shadow-xl p-8 md:p-10 relative z-10 bg-white/95">
        <h2 className="block text-3xl font-bold text-center text-gray-800">What you get</h2>
        <div className="mt-8 grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {FEATURES.map((f) => (
            <div key={f.title} className="rounded-2xl border border-gray-200 bg-white p-5 hover:shadow-lg hover:border-blue-200 transition">
              <div className="text-3xl" aria-hidden="true">{f.icon}</div>
              <h3 className="mt-3 font-semibold text-gray-800">{f.title}</h3>
              <p className="mt-1 text-sm text-gray-600">{f.text}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ============ FOR STUDENTS / TEACHERS ============ */}
      <section className="scroll-section grid md:grid-cols-2 gap-6 relative z-10">
        <div className="glass-card rounded-3xl shadow-xl p-8 bg-white/95">
          <h2 className="text-2xl font-bold text-gray-800">🎓 For students</h2>
          <ul className="mt-4 space-y-3 text-gray-700">
            <li>• Feedback any time, before you hand your essay in.</li>
            <li>• Scores per criterion, so you know exactly what to work on.</li>
            <li>• The weak sentences highlighted, with suggestions to improve them.</li>
          </ul>
        </div>
        <div className="glass-card rounded-3xl shadow-xl p-8 bg-white/95">
          <h2 className="text-2xl font-bold text-gray-800">👩‍🏫 For teachers</h2>
          <ul className="mt-4 space-y-3 text-gray-700">
            <li>• A consistent first pass on every essay, with your own rubric.</li>
            <li>• Override scores and leave remarks; the system learns from them.</li>
            <li>• PDF reports and an AI-revised reference essay to share with students.</li>
          </ul>
        </div>
      </section>

      {/* ============ PRIVACY ============ */}
      <section className="scroll-section glass-card rounded-3xl shadow-xl p-8 border-l-8 border-emerald-500 relative z-10 bg-white/95">
        <div className="flex items-start gap-4">
          <div className="text-4xl" aria-hidden="true">🔒</div>
          <div>
            <h2 className="text-2xl font-bold text-gray-800">Where your essay goes</h2>
            <ul className="mt-3 space-y-2 text-gray-700">
              <li>• Your essays and results are stored in the app’s database (Supabase).</li>
              <li>• Scoring with <strong>DeepSeek</strong> sends the essay text to DeepSeek’s service. Choose <strong>Llama 3.2 (local)</strong> to keep it on our own server.</li>
              <li>• Photos of handwritten essays are read by <strong>Google Gemini</strong>.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* ============ CALL TO ACTION ============ */}
      <section className="scroll-section text-center py-6 relative z-10">
        <button
          onClick={() => setActiveTab('evaluate')}
          className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-8 py-4 rounded-xl font-bold text-lg shadow-xl hover:shadow-2xl hover:brightness-110 transition"
        >
          ✨ Evaluate an essay now
        </button>
      </section>
    </div>
  );
}

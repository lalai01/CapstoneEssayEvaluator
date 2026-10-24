import React, { useEffect, useRef, useState } from 'react';
import toast from 'react-hot-toast';
import { evaluateEssay, evaluateEssayWithRag, listRubrics } from '../api';
import FileUpload from './FileUpload';
import ConfirmDialog from './ConfirmDialog';
import { useAuth } from '../context/AuthContext';
import { errorMessage } from '../lib/scoring';
import useDraft from '../lib/draft';
import { EVALUATION_AIS } from '../lib/evaluationAis';

// An essay is made of parts, evaluated together as one essay: typed or pasted
// text, and the text read from each uploaded page (e.g. the front and back of
// a handwritten essay). Parts can be edited, reordered and removed.
const NO_PARTS = [];
const newId = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
const typedPart = () => ({ id: newId(), kind: 'typed', text: '' });
const wordCount = (text) => text.split(/\s+/).filter(Boolean).length;

export default function EssayInput({ onNewEssay, hasResults, setAiRuns, onChooseAi, setScores, setFeedback, setLoading, setCurrentEssay, setEvalType, setEssayTitle, loading }) {
  const { user, loading: authLoading } = useAuth();
  const [rubrics, setRubrics] = useState([]);
  // Everything typed or chosen here is kept as a draft in this browser, so a
  // refresh or a visit to another page doesn't lose it.
  const [rubricId, setRubricId] = useDraft('rubricId', '');   // '' = the teacher's default
  const [title, setTitle] = useDraft('title', '');
  const [titleError, setTitleError] = useState(false);
  const titleRef = useRef(null);
  const [savedParts, setParts] = useDraft('parts', NO_PARTS);
  const [evalTypeLocal, setEvalTypeLocal] = useDraft('evalType', 'analytic');
  const [useRag, setUseRag] = useDraft('useRag', true);
  const [confirmNew, setConfirmNew] = useState(false);

  // There is always at least one part to type into.
  const [blankPart] = useState(typedPart);
  const parts = savedParts.length ? savedParts : [blankPart];
  const filled = parts.filter((p) => p.text.trim());
  const essayToEvaluate = () => filled.map((p) => p.text.trim()).join('\n\n');
  const essayWords = wordCount(essayToEvaluate());
  const hasDraft = !!(title.trim() || filled.length);

  const updatePart = (id, text) => setParts(parts.map((p) => (p.id === id ? { ...p, text } : p)));
  const removePart = (id) => setParts(parts.filter((p) => p.id !== id));
  const movePart = (index, step) => {
    const next = [...parts];
    [next[index], next[index + step]] = [next[index + step], next[index]];
    setParts(next);
  };
  const addTypedPart = () => setParts([...parts, typedPart()]);

  const startNewEssay = () => {
    setConfirmNew(false);
    setTitle('');
    setParts(NO_PARTS);
    setTitleError(false);
    onNewEssay?.();
    toast.success('Draft cleared. Start your new essay.');
  };

  useEffect(() => {
    if (!user) {
      setRubrics([]);
      // Keep the chosen rubric while the sign-in is still loading after a refresh.
      if (!authLoading) setRubricId('');
      return;
    }
    listRubrics()
      .then((data) => setRubrics(Array.isArray(data) ? data : []))
      .catch(() => setRubrics([]));
  }, [user, authLoading, setRubricId]);

  const myRubrics = rubrics.filter((r) => !r.is_system);
  const schoolRubrics = rubrics.filter((r) => r.is_system && r.is_active);
  const defaultRubric = myRubrics.find((r) => r.is_active) || schoolRubrics[0];

  // Each uploaded file (or page) is added as a new part, after the others; an
  // empty typing box is replaced. Several files arrive one after another, so
  // the list is updated from its latest value.
  const handleExtracted = (data) => {
    const text = (data && typeof data.text === 'string') ? data.text : '';
    if (!text.trim()) {
      toast.error(`${data?.name ? `${data.name}: ` : ''}no text was found. Try a clearer image or PDF.`);
      return;
    }
    const part = {
      id: newId(),
      kind: 'file',
      name: data.name || 'Uploaded file',
      text: text.trim(),
      confidence: typeof data.confidence === 'number' ? data.confidence : null,
      method: data.method || 'unknown',
    };
    setParts((prev) => [...prev.filter((p) => p.text.trim() || p.kind === 'file'), part]);
  };

  // Essays below SHORT_ESSAY_WORDS ask for confirmation; below MIN_WORDS the
  // server can't evaluate them at all.
  const MIN_WORDS = 20;
  const SHORT_ESSAY_WORDS = 150;
  const [shortDialog, setShortDialog] = useState(null); // { words }

  const handleEvaluate = async () => {
    const essay = essayToEvaluate();
    if (!essay) {
      toast.error('Please type the essay or upload its pages.');
      return;
    }
    if (!title.trim()) {
      setTitleError(true);
      titleRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      titleRef.current?.focus({ preventScroll: true });
      toast.error('Please enter a title for your essay.');
      return;
    }
    const words = wordCount(essay);
    if (words < SHORT_ESSAY_WORDS) {
      setShortDialog({ words });
      return;
    }
    await runEvaluation(essay);
  };

  const runEvaluation = async (essay) => {
    setShortDialog(null);
    setLoading(true);
    setCurrentEssay(essay);
    setEvalType(evalTypeLocal);
    setEssayTitle(title.trim());  // Pass title to parent
    const chosenRubric = rubricId ? Number(rubricId) : null;
    const evaluate = useRag ? evaluateEssayWithRag : evaluateEssay;

    // All three AIs evaluate the essay at once. Each card fills in as its AI
    // finishes; the first real AI result is shown straight away (the user can
    // switch), so nobody waits for the slowest model.
    setScores(null);
    setFeedback(null);
    onChooseAi(null);
    setAiRuns(Object.fromEntries(EVALUATION_AIS.map((ai) => [ai.id, { status: 'running' }])));
    let shown = false;
    let fallback = null;   // a rule-based result, shown only if no AI answers

    const runOne = async (ai) => {
      const started = performance.now();
      try {
        const result = await evaluate(essay, evalTypeLocal, chosenRubric, ai.id, true, title.trim());
        if (typeof result.feedback === 'string' && result.feedback.startsWith('⚠️ Invalid Input')) {
          throw new Error(result.feedback.replace('⚠️ Invalid Input:', '').trim());
        }
        const run = { status: 'done', scores: result.scores, feedback: result.feedback, ms: performance.now() - started };
        setAiRuns((prev) => ({ ...prev, [ai.id]: run }));
        const ruleBased = result.scores?.evaluated_by?.provider === 'heuristic';
        if (!shown && !ruleBased) {
          shown = true;
          onChooseAi(ai.id, run);
          setLoading(false);
        } else if (ruleBased && !fallback) {
          fallback = { id: ai.id, run };
        }
        return true;
      } catch (err) {
        setAiRuns((prev) => ({ ...prev, [ai.id]: { status: 'failed', error: errorMessage(err), ms: performance.now() - started } }));
        return false;
      }
    };
    // Cloud AIs (DeepSeek, GPT-OSS) run at once; local models share the server's
    // CPU, where one at a time is faster (no swapping models in and out of memory).
    const [cloud, local] = [EVALUATION_AIS.filter((a) => !a.local), EVALUATION_AIS.filter((a) => a.local)];
    const runLocalInTurn = async () => {
      const results = [];
      for (const ai of local) results.push(await runOne(ai));
      return results;
    };
    const [cloudResults, localResults] = await Promise.all([Promise.all(cloud.map(runOne)), runLocalInTurn()]);
    const outcomes = [...cloudResults, ...localResults];

    if (!shown && fallback) {
      shown = true;
      onChooseAi(fallback.id, fallback.run);
    }
    setLoading(false);
    const answered = outcomes.filter(Boolean).length;
    if (!shown) {
      toast.error('Evaluation failed: none of the three AIs responded. Please try again.');
    } else {
      toast.success(`${answered} of 3 AIs finished${useRag ? ' (with RAG)' : ''}. Pick the result you prefer.`);
    }
  };

  return (
    <div className="space-y-6">
      <div className="glass-card rounded-2xl shadow-xl p-6 animate-slide-up">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
            <span className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm">1</span>
            Input Your Essay
          </h2>
          {(hasDraft || hasResults) && (
            <button
              type="button"
              onClick={() => setConfirmNew(true)}
              disabled={loading}
              className="text-sm border border-gray-300 text-gray-700 px-3 py-1.5 rounded-lg hover:bg-gray-50 transition disabled:opacity-50"
            >
              ＋ Start a new essay
            </button>
          )}
        </div>
        {hasDraft && (
          <p className="-mt-2 mb-4 text-xs text-gray-500">💾 Draft saved in this browser. It stays after a refresh until you start a new essay or sign out.</p>
        )}

        {/* Login notice for anonymous users */}
        {!user && (
          <div className="mb-5 p-4 bg-amber-50 border border-amber-200 rounded-xl">
            <p className="text-amber-800 text-sm flex items-center gap-2">
              <span>🔒</span>
              <span>
                You can evaluate essays as a guest. <strong>Sign in with Google or Yahoo</strong> to save
                evaluations, use your teacher&apos;s rubric and keep evaluating without the hourly guest limit.
              </span>
            </p>
          </div>
        )}

        {/* Title Input (Required) */}
        <div className="mb-5">
          <label htmlFor="essay-title" className="block text-sm font-semibold text-gray-700 mb-2">
            Essay Title <span className="text-red-500">*</span>
          </label>
          <input
            id="essay-title"
            ref={titleRef}
            type="text"
            value={title}
            onChange={(e) => {
              setTitle(e.target.value);
              if (e.target.value.trim()) setTitleError(false);
            }}
            placeholder="e.g., The Role of Technology in Education"
            aria-invalid={titleError}
            aria-describedby={titleError ? 'essay-title-error' : undefined}
            className={`w-full rounded-xl p-3 transition focus:ring-2 focus:border-transparent ${
              titleError
                ? 'border-2 border-red-500 bg-red-50 focus:ring-red-400'
                : 'border border-gray-200 bg-gray-50 focus:ring-blue-500'
            }`}
            required
          />
          {titleError && (
            <p id="essay-title-error" className="mt-1 text-sm font-medium text-red-600" role="alert">
              ⚠️ Please give your essay a title before evaluating.
            </p>
          )}
        </div>

        <div className="mb-5">
          <label className="block text-sm font-semibold text-gray-700 mb-2">Evaluation Type</label>
          <div className="flex flex-wrap items-center gap-4 bg-gray-50 p-2 rounded-lg">
            <div className="flex gap-4">
              <label className="flex items-center gap-2 px-3 py-1 rounded-md cursor-pointer hover:bg-blue-50 transition">
                <input type="radio" value="analytic" checked={evalTypeLocal === 'analytic'} onChange={(e) => setEvalTypeLocal(e.target.value)} className="w-4 h-4 text-blue-600" />
                <span>📊 Analytic</span>
              </label>
              <label className="flex items-center gap-2 px-3 py-1 rounded-md cursor-pointer hover:bg-blue-50 transition">
                <input type="radio" value="holistic" checked={evalTypeLocal === 'holistic'} onChange={(e) => setEvalTypeLocal(e.target.value)} className="w-4 h-4 text-blue-600" />
                <span>🌟 Holistic</span>
              </label>
            </div>
            <div className="border-l border-gray-300 pl-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={useRag} onChange={(e) => setUseRag(e.target.checked)} className="w-4 h-4 text-purple-600" />
                <span className="text-sm font-medium text-gray-700">🧠 Use RAG (retrieve similar past evaluations)</span>
              </label>
            </div>
          </div>
          <p className="text-xs text-gray-600 mt-1">
            {evalTypeLocal === 'holistic'
              ? 'Holistic: one overall score (1–5) for coherence, organization, style and content together. Fastest: a single short AI check.'
              : 'Analytic: a separate score for each rubric criterion (e.g. thesis, organization, evidence, grammar) with detailed feedback. Takes longer.'}
          </p>
          {useRag && (
            <p className="text-xs text-purple-600 mt-1">
              Enhances feedback by including insights from previously saved evaluations.
            </p>
          )}
        </div>

        <div className="mb-5 rounded-xl border border-blue-100 bg-blue-50/60 p-3">
          <p className="text-sm font-semibold text-gray-800">🤖 Three AIs evaluate every essay</p>
          <p className="text-xs text-gray-600 mt-1">
            {EVALUATION_AIS.map((ai) => `${ai.icon} ${ai.name} (${ai.where.toLowerCase()})`).join(' · ')}.
            Their results appear side by side as each one finishes; choose the one you prefer.
          </p>
        </div>


        {user && evalTypeLocal === 'analytic' && (
          <div className="mb-5">
            <label htmlFor="rubric-select" className="block text-sm font-semibold text-gray-700 mb-2">
              Rubric
            </label>
            <select
              id="rubric-select"
              value={rubricId}
              onChange={(e) => setRubricId(e.target.value)}
              className="w-full border border-gray-200 rounded-xl p-3 bg-gray-50 focus:ring-2 focus:ring-blue-500"
            >
              <option value="">
                Default — {defaultRubric ? defaultRubric.title : 'built-in rubric'}
              </option>
              {myRubrics.length > 0 && (
                <optgroup label="My rubrics">
                  {myRubrics.map((r) => <option key={r.id} value={r.id}>{r.title}</option>)}
                </optgroup>
              )}
              {schoolRubrics.length > 0 && (
                <optgroup label="School-wide">
                  {schoolRubrics.map((r) => <option key={r.id} value={r.id}>{r.title}</option>)}
                </optgroup>
              )}
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Create or edit rubrics in the 📐 Rubrics tab.
            </p>
          </div>
        )}

        <div className="mb-5">
          <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
            <span className="block text-sm font-semibold text-gray-700">Essay</span>
            {filled.length > 0 && (
              <span className="text-xs text-gray-500" role="status">
                {parts.length > 1 ? `${filled.length} of ${parts.length} parts · ` : ''}{essayWords} words · evaluated as one essay
              </span>
            )}
          </div>
          <p className="mb-3 text-xs text-gray-600">
            Type, paste or upload the essay. If it continues on another page, or has several parts, add each one:
            all parts are joined in this order and evaluated together as a single essay.
          </p>

          <ol className="space-y-3">
            {parts.map((part, i) => (
              <li key={part.id} className={`rounded-xl border p-3 ${part.kind === 'file' ? 'border-blue-200 bg-blue-50/60' : 'border-gray-200 bg-white/70'}`}>
                <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                  <div className="min-w-0 text-sm">
                    <label htmlFor={`essay-part-${part.id}`} className="font-semibold text-gray-800">
                      {parts.length > 1 ? `Part ${i + 1}` : 'Your essay'}
                    </label>
                    <span className="ml-2 text-xs text-gray-600">
                      {part.kind === 'file'
                        ? `📷 ${part.name}${part.confidence != null ? ` · ${part.confidence.toFixed(0)}% confidence` : ''} · check for misread words`
                        : '✍️ typed'}
                    </span>
                  </div>
                  {parts.length > 1 && (
                    <div className="flex shrink-0 gap-1">
                      <button type="button" onClick={() => movePart(i, -1)} disabled={i === 0 || loading}
                        className="rounded-md border bg-white px-2 py-0.5 text-xs text-gray-700 hover:bg-gray-50 disabled:opacity-40"
                        aria-label={`Move part ${i + 1} up`}>↑</button>
                      <button type="button" onClick={() => movePart(i, 1)} disabled={i === parts.length - 1 || loading}
                        className="rounded-md border bg-white px-2 py-0.5 text-xs text-gray-700 hover:bg-gray-50 disabled:opacity-40"
                        aria-label={`Move part ${i + 1} down`}>↓</button>
                      <button type="button" onClick={() => removePart(part.id)} disabled={loading}
                        className="rounded-md border border-red-200 bg-white px-2 py-0.5 text-xs text-red-600 hover:bg-red-50 disabled:opacity-40"
                        aria-label={`Remove part ${i + 1}`}>✕ Remove</button>
                    </div>
                  )}
                  {parts.length === 1 && part.kind === 'file' && (
                    <button type="button" onClick={() => setParts(NO_PARTS)} disabled={loading}
                      className="rounded-md border border-red-200 bg-white px-2 py-0.5 text-xs text-red-600 hover:bg-red-50">✕ Remove</button>
                  )}
                </div>
                <textarea
                  id={`essay-part-${part.id}`}
                  rows={parts.length > 1 ? 7 : 10}
                  className="w-full rounded-lg border border-gray-200 bg-white p-3 text-sm focus:border-transparent focus:ring-2 focus:ring-blue-500"
                  value={part.text}
                  onChange={(e) => updatePart(part.id, e.target.value)}
                  placeholder={i === 0
                    ? "Enter your essay here... (e.g., 'Education is the most powerful weapon...')"
                    : 'Type or paste the next part of the essay...'}
                />
              </li>
            ))}
          </ol>

          <button type="button" onClick={addTypedPart} disabled={loading}
            className="mt-3 inline-flex items-center gap-1 rounded-lg border border-dashed border-blue-400 px-3 py-2 text-sm font-medium text-blue-700 hover:bg-blue-50 disabled:opacity-50">
            ＋ Add another part
          </button>
        </div>

        <FileUpload onExtracted={handleExtracted} />

        <button
          onClick={handleEvaluate}
          disabled={loading}
          className="btn-primary w-full mt-6 bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-3 px-4 rounded-xl font-semibold shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition"
        >
          {loading ? <div className="spinner mx-auto"></div> : (useRag ? '✨ Evaluate with RAG' : '✨ Evaluate Essay')}
        </button>
      </div>

      <ConfirmDialog
        open={confirmNew}
        tone="warning"
        icon="📝"
        title="Start a new essay?"
        confirmLabel="Clear and start new"
        cancelLabel="Keep my draft"
        onConfirm={startNewEssay}
        onCancel={() => setConfirmNew(false)}
      >
        <p>This clears the title, the essay text and the current results. Save anything you want to keep to the Knowledge Base first.</p>
      </ConfirmDialog>

      <ConfirmDialog
        open={!!shortDialog}
        tone={shortDialog && shortDialog.words < MIN_WORDS ? 'error' : 'warning'}
        icon="✍️"
        title={shortDialog && shortDialog.words < MIN_WORDS ? 'Essay too short to evaluate' : 'This essay is quite short'}
        confirmLabel={shortDialog && shortDialog.words >= MIN_WORDS ? 'Evaluate anyway' : undefined}
        cancelLabel={shortDialog && shortDialog.words < MIN_WORDS ? 'OK, keep writing' : 'Keep writing'}
        onConfirm={() => runEvaluation(essayToEvaluate())}
        onCancel={() => setShortDialog(null)}
      >
        {shortDialog && shortDialog.words < MIN_WORDS ? (
          <p>
            It has <strong>{shortDialog.words} {shortDialog.words === 1 ? 'word' : 'words'}</strong>.
            An essay needs at least {MIN_WORDS} words to be evaluated.
          </p>
        ) : (
          <>
            <p>
              It has <strong>{shortDialog?.words} words</strong>. Essays under {SHORT_ESSAY_WORDS} words usually
              don't show enough of the thesis, evidence and organization, so scores can be less reliable.
            </p>
            <p>Do you want to evaluate it anyway?</p>
          </>
        )}
      </ConfirmDialog>
    </div>
  );
}
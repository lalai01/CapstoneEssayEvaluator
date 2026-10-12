import React, { useEffect, useState } from 'react';
import toast from 'react-hot-toast';
import { evaluateEssay, evaluateEssayWithRag, listRubrics } from '../api';
import FileUpload from './FileUpload';
import { useAuth } from '../context/AuthContext';
import { errorMessage } from '../lib/scoring';

export default function EssayInput({ setScores, setFeedback, setLoading, setCurrentEssay, setEvalType, setEssayTitle, loading }) {
  const { user } = useAuth();
  const [rubrics, setRubrics] = useState([]);
  const [rubricId, setRubricId] = useState('');   // '' = the teacher's default
  const [aiModel, setAiModel] = useState('auto');  // 'auto' | 'deepseek' | 'local'
  const [title, setTitle] = useState('');
  const [text, setText] = useState('');
  const [evalTypeLocal, setEvalTypeLocal] = useState('analytic');
  const [extractedText, setExtractedText] = useState('');
  const [ocrInfo, setOcrInfo] = useState(null);
  const [useRag, setUseRag] = useState(true);

  useEffect(() => {
    if (!user) {
      setRubrics([]);
      setRubricId('');
      return;
    }
    listRubrics()
      .then((data) => setRubrics(Array.isArray(data) ? data : []))
      .catch(() => setRubrics([]));
  }, [user]);

  const myRubrics = rubrics.filter((r) => !r.is_system);
  const schoolRubrics = rubrics.filter((r) => r.is_system && r.is_active);
  const defaultRubric = myRubrics.find((r) => r.is_active) || schoolRubrics[0];

    const handleExtracted = (data) => {
      const text = (data && typeof data.text === 'string') ? data.text : '';
      const confidence = typeof data?.confidence === 'number' ? data.confidence : 0;
      const method = data?.method || 'unknown';

      if (!text.trim()) {
        toast.error('OCR returned no text. Try a clearer image or PDF.');
        return;
      }

      setExtractedText(text);
      setOcrInfo({
        confidence,
        method,
        lowConfidence: !!data.low_confidence,
      });
      toast.success(`OCR complete: ${confidence.toFixed(1)}% confidence`);
    };

  const handleEvaluate = async () => {
    const essay = text.trim() || extractedText.trim();
    if (!essay) {
      toast.error('Please enter essay text or extract text from a file.');
      return;
    }
    if (!title.trim()) {
      toast.error('Please enter a title for your essay.');
      return;
    }
    setLoading(true);
    setCurrentEssay(essay);
    setEvalType(evalTypeLocal);
    setEssayTitle(title.trim());  // Pass title to parent
    const chosenRubric = rubricId ? Number(rubricId) : null;
    try {
      const result = useRag
        ? await evaluateEssayWithRag(essay, evalTypeLocal, chosenRubric, aiModel)
        : await evaluateEssay(essay, evalTypeLocal, chosenRubric, aiModel);
      setScores(result.scores);
      setFeedback(result.feedback);
      toast.success(`Evaluation complete${useRag ? ' (with RAG)' : ''}`);
    } catch (err) {
      toast.error('Evaluation failed: ' + errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="glass-card rounded-2xl shadow-xl p-6 animate-slide-up">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
          <span className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm">1</span>
          Input Your Essay
        </h2>

        {/* Login notice for anonymous users */}
        {!user && (
          <div className="mb-5 p-4 bg-amber-50 border border-amber-200 rounded-xl">
            <p className="text-amber-800 text-sm flex items-center gap-2">
              <span>🔒</span>
              <span>
                <strong>Log in with Google or Yahoo</strong> to save evaluations to your Knowledge Base 
                and enable teacher override features.
              </span>
            </p>
          </div>
        )}

        {/* Title Input (Required) */}
        <div className="mb-5">
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Essay Title <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="e.g., The Role of Technology in Education"
            className="w-full border border-gray-200 rounded-xl p-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition bg-gray-50"
            required
          />
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

        <div className="mb-5">
          <span id="ai-model-label" className="block text-sm font-semibold text-gray-700 mb-2">AI model</span>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2" role="radiogroup" aria-labelledby="ai-model-label">
            {[
              { id: 'auto', title: '✨ Auto (recommended)', hint: 'Picks the best model for this essay' },
              { id: 'deepseek', title: '🌐 DeepSeek', hint: 'Cloud; reads long essays in full' },
              { id: 'local', title: '🦙 Llama 3.2 (local)', hint: 'Private; essay stays on the server' },
            ].map((opt) => (
              <button
                key={opt.id}
                type="button"
                role="radio"
                aria-checked={aiModel === opt.id}
                onClick={() => setAiModel(opt.id)}
                className={`text-left rounded-xl border p-2.5 transition ${
                  aiModel === opt.id ? 'border-blue-500 bg-blue-50 ring-1 ring-blue-300' : 'border-gray-200 bg-gray-50 hover:bg-gray-100'
                }`}
              >
                <div className="text-sm font-semibold text-gray-800">{opt.title}</div>
                <div className="text-xs text-gray-500">{opt.hint}</div>
              </button>
            ))}
          </div>
          {aiModel === 'auto' && (
            <p className="text-xs text-gray-500 mt-1">
              Long essays (over about 650 words) and complex rubrics go to DeepSeek; others are scored locally. The result explains the choice.
            </p>
          )}
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
          <label className="block text-sm font-semibold text-gray-700 mb-2">Type or paste your essay</label>
          <textarea
            rows={10}
            className="w-full border border-gray-200 rounded-xl p-4 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition bg-gray-50"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter your essay here... (e.g., 'Education is the most powerful weapon...')"
          />
        </div>

        <FileUpload onExtracted={handleExtracted} />

        {extractedText && (
          <div className="mt-5 rounded-xl border border-blue-200 bg-blue-50 p-4">
            <div className="flex justify-between items-center mb-2">
              <h3 className="font-semibold text-blue-800">📄 Extracted Text (OCR)</h3>
              <button
                onClick={() => setText(extractedText)}
                className="text-sm bg-blue-600 text-white px-3 py-1 rounded-lg hover:bg-blue-700 transition"
              >
                Use for Evaluation
              </button>
            </div>
            <p className="text-xs text-gray-600 mb-2">Confidence: {ocrInfo?.confidence?.toFixed(1)}% | Method: {ocrInfo?.method}</p>
            <div className="max-h-60 overflow-y-auto text-sm bg-white p-3 rounded border border-blue-100 whitespace-pre-wrap">{extractedText}</div>
          </div>
        )}

        <button
          onClick={handleEvaluate}
          disabled={loading}
          className="btn-primary w-full mt-6 bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-3 px-4 rounded-xl font-semibold shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition"
        >
          {loading ? <div className="spinner mx-auto"></div> : (useRag ? '✨ Evaluate with RAG' : '✨ Evaluate Essay')}
        </button>
      </div>
    </div>
  );
}
import React, { useState } from 'react';
import toast from 'react-hot-toast';
import { uploadFile, evaluateEssay } from '../api';
import FileUpload from './FileUpload';

export default function EssayInput({ setScores, setFeedback, setLoading, setCurrentEssay, setEvalType, loading }) {
  const [text, setText] = useState('');
  const [evalTypeLocal, setEvalTypeLocal] = useState('analytic');
  const [extractedText, setExtractedText] = useState('');
  const [ocrInfo, setOcrInfo] = useState(null);

  const handleExtracted = (data) => {
    setExtractedText(data.text);
    setOcrInfo({ confidence: data.confidence, method: data.method });
    toast.success(`OCR complete: ${data.confidence.toFixed(1)}% confidence`);
  };

  const handleEvaluate = async () => {
    const essay = text.trim() || extractedText.trim();
    if (!essay) {
      toast.error('Please enter essay text or extract text from a file.');
      return;
    }
    setLoading(true);
    setCurrentEssay(essay);
    setEvalType(evalTypeLocal);
    try {
      const result = await evaluateEssay(essay, evalTypeLocal);
      setScores(result.scores);
      setFeedback(result.feedback);
      toast.success('Evaluation complete!');
    } catch (err) {
      toast.error('Evaluation failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="glass-card rounded-2xl shadow-xl p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
          <span className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm">1</span>
          Input Your Essay
        </h2>
        <div className="mb-5">
          <label className="block text-sm font-semibold text-gray-700 mb-2">Evaluation Type</label>
          <div className="flex gap-4 bg-gray-50 p-2 rounded-lg">
            <label className="flex items-center gap-2 px-3 py-1 rounded-md cursor-pointer hover:bg-blue-50 transition">
              <input type="radio" value="analytic" checked={evalTypeLocal === 'analytic'} onChange={(e) => setEvalTypeLocal(e.target.value)} className="w-4 h-4 text-blue-600" />
              <span>📊 Analytic</span>
            </label>
            <label className="flex items-center gap-2 px-3 py-1 rounded-md cursor-pointer hover:bg-blue-50 transition">
              <input type="radio" value="holistic" checked={evalTypeLocal === 'holistic'} onChange={(e) => setEvalTypeLocal(e.target.value)} className="w-4 h-4 text-blue-600" />
              <span>🌟 Holistic</span>
            </label>
          </div>
        </div>
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
            <div className="max-h-40 overflow-y-auto text-sm bg-white p-3 rounded border border-blue-100">{extractedText}</div>
          </div>
        )}
        <button
          onClick={handleEvaluate}
          disabled={loading}
          className="btn-primary w-full mt-6 bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-3 px-4 rounded-xl font-semibold shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? <div className="spinner mx-auto"></div> : '✨ Evaluate Essay'}
        </button>
      </div>
    </div>
  );
}
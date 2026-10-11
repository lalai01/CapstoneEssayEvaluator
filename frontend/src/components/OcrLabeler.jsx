import React, { useEffect, useState, useRef } from 'react';
import toast from 'react-hot-toast';
import { listPendingOcr, submitOcrCorrection } from '../api';
import { useAuth } from '../context/AuthContext';
import LoadingIndicator from './LoadingIndicator';

export default function OcrLabeler() {
  const { isAdmin } = useAuth();

  const [queue, setQueue] = useState([]);
  const [index, setIndex] = useState(0);
  const [correction, setCorrection] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const textareaRef = useRef(null);

  useEffect(() => {
    if (isAdmin) loadQueue();
  }, [isAdmin]);

  const loadQueue = async () => {
    setLoading(true);
    try {
      const data = await listPendingOcr(50);
      setQueue(data);
      setIndex(0);
      setCorrection(data[0]?.raw_ocr_text || '');
    } catch (err) {
      toast.error('Failed to load queue: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const current = queue[index];
    setCorrection(current?.raw_ocr_text || '');
    setTimeout(() => textareaRef.current?.focus(), 100);
  }, [index, queue]);

  const handleSave = async () => {
    const current = queue[index];
    if (!current) return;
    if (!correction.trim()) {
      toast.error('Correction cannot be empty');
      return;
    }

    setSaving(true);
    try {
      await submitOcrCorrection(current.id, correction);
      toast.success(`Saved sample #${current.id}`);

      // Remove from local queue and advance
      const newQueue = queue.filter((q) => q.id !== current.id);
      setQueue(newQueue);
      if (index >= newQueue.length) {
        setIndex(Math.max(0, newQueue.length - 1));
      }
    } catch (err) {
      toast.error('Save failed: ' + err.message);
    } finally {
      setSaving(false);
    }
  };

  const handleSkip = () => {
    if (index < queue.length - 1) {
      setIndex(index + 1);
    } else {
      toast('Reached end of queue');
    }
  };

  const handlePrev = () => {
    if (index > 0) setIndex(index - 1);
  };

  if (!isAdmin) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 text-center bg-white/90 backdrop-blur-md">
        <div className="text-6xl mb-4">🔒</div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          Admin Only
        </h2>
        <p className="text-gray-700">
          You do not have access to the OCR labeling tool.
        </p>
      </div>
    );
  }

    if (loading) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 bg-white/90 backdrop-blur-md">
        <LoadingIndicator message="Loading correction queue…" />
      </div>
    );
  }
  
  if (queue.length === 0) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 text-center bg-white/90 backdrop-blur-md">
        <div className="text-6xl mb-4">✅</div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          Queue is empty
        </h2>
        <p className="text-gray-700 mb-4">
          No pending OCR samples. Upload more handwritten pages to generate
          new samples.
        </p>
        <button
          onClick={loadQueue}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
        >
          Reload Queue
        </button>
      </div>
    );
  }

  const current = queue[index];

  return (
    <div className="glass-card rounded-2xl shadow-xl p-6 bg-white/90 backdrop-blur-md">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold text-indigo-700">
          🏷️ OCR Correction Queue
        </h2>
        <div className="text-sm text-gray-600">
          {index + 1} of {queue.length} · sample #{current.id}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Image */}
        <div>
          <h3 className="font-semibold text-gray-800 mb-2">
            📄 Original Handwriting
          </h3>
          <div className="bg-gray-100 rounded-xl p-3 border border-gray-200">
            <img
              src={current.image_url}
              alt={`Sample ${current.id}`}
              className="w-full h-auto rounded-lg"
              style={{ maxHeight: '600px', objectFit: 'contain' }}
            />
          </div>
        </div>

        {/* Right: Correction */}
        <div className="flex flex-col">
          <h3 className="font-semibold text-gray-800 mb-2">
            ✍️ Correct Transcription
          </h3>
          <div className="text-xs text-gray-500 mb-2">
            Engine: <span className="font-mono">{current.engine}</span>
          </div>
          <textarea
            ref={textareaRef}
            value={correction}
            onChange={(e) => setCorrection(e.target.value)}
            rows={22}
            className="w-full border border-gray-300 rounded-xl p-3 font-mono text-sm text-gray-800 flex-1"
            placeholder="Type the correct transcription here..."
          />
        </div>
      </div>

      <div className="flex items-center justify-between mt-6">
        <div className="flex gap-2">
          <button
            onClick={handlePrev}
            disabled={index === 0}
            className="px-4 py-2 rounded-lg bg-gray-100 text-gray-700 hover:bg-gray-200 disabled:opacity-40"
          >
            ← Previous
          </button>
          <button
            onClick={handleSkip}
            disabled={index >= queue.length - 1}
            className="px-4 py-2 rounded-lg bg-gray-100 text-gray-700 hover:bg-gray-200 disabled:opacity-40"
          >
            Skip →
          </button>
        </div>

        <button
          onClick={handleSave}
          disabled={saving}
          className="px-6 py-3 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold shadow-md hover:shadow-lg disabled:opacity-50"
        >
          {saving ? 'Saving...' : '💾 Save & Next'}
        </button>
      </div>

      <p className="text-xs text-gray-500 mt-3">
        Tip: type exactly what's written — spelling, punctuation, line breaks.
        Do not correct the writer's grammar.
      </p>
    </div>
  );
}
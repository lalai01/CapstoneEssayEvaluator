import React, { useEffect, useState, useRef } from 'react';
import toast from 'react-hot-toast';
import { discardOcrSample, listPendingOcr, submitOcrCorrection } from '../api';
import { useAuth } from '../context/AuthContext';
import { errorMessage } from '../lib/scoring';
import ConfirmDialog from './ConfirmDialog';
import LoadingIndicator from './LoadingIndicator';

export default function OcrLabeler() {
  const { isAdmin } = useAuth();

  const [queue, setQueue] = useState([]);
  const [index, setIndex] = useState(0);
  const [correction, setCorrection] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [confirmDiscard, setConfirmDiscard] = useState(false);
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
      toast.error('Failed to load queue: ' + errorMessage(err));
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
      dropFromQueue(current.id);
    } catch (err) {
      toast.error('Save failed: ' + errorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  // Remove a sample from the local queue and stay on the next one.
  const dropFromQueue = (id) => {
    const newQueue = queue.filter((q) => q.id !== id);
    setQueue(newQueue);
    if (index >= newQueue.length) {
      setIndex(Math.max(0, newQueue.length - 1));
    }
  };

  const handleDiscard = async () => {
    const current = queue[index];
    setConfirmDiscard(false);
    if (!current) return;
    setSaving(true);
    try {
      await discardOcrSample(current.id);
      toast.success(`Discarded sample #${current.id}`);
      dropFromQueue(current.id);
    } catch (err) {
      toast.error('Discard failed: ' + errorMessage(err));
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
  const lineCount = correction.split(/\r?\n/).filter((l) => l.trim()).length;

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
            Prefilled with what <span className="font-mono">{current.engine}</span> read, before any AI
            correction. Fix it so it matches the page exactly.
          </div>
          <textarea
            ref={textareaRef}
            value={correction}
            onChange={(e) => setCorrection(e.target.value)}
            rows={22}
            className="w-full border border-gray-300 rounded-xl p-3 font-mono text-sm text-gray-800 flex-1"
            placeholder="Type the correct transcription here..."
          />
          <p className="text-xs text-gray-600 mt-2">
            <strong>{lineCount}</strong> text {lineCount === 1 ? 'line' : 'lines'} — this should equal the number of
            written lines in the photo, or the sample can't be used for training.
          </p>
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

        <div className="flex gap-2">
        <button
          onClick={() => setConfirmDiscard(true)}
          disabled={saving}
          className="px-4 py-3 rounded-xl bg-white border border-red-200 text-red-600 font-semibold hover:bg-red-50 disabled:opacity-50"
        >
          🗑 Discard
        </button>
        <button
          onClick={handleSave}
          disabled={saving}
          className="px-6 py-3 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold shadow-md hover:shadow-lg disabled:opacity-50"
        >
          {saving ? 'Saving...' : '💾 Save & Next'}
        </button>
        </div>
      </div>

      <p className="text-xs text-gray-500 mt-3">
        Tip: type exactly what's written — spelling, punctuation, and one text line per written line.
        Do not correct the writer's grammar. Leave out crossed-out words.
        Discard photos that aren't handwritten essays or are too blurry to read.
      </p>

      <ConfirmDialog
        open={confirmDiscard}
        tone="error"
        title={`Discard sample #${current.id}?`}
        confirmLabel="Discard"
        cancelLabel="Keep"
        onConfirm={handleDiscard}
        onCancel={() => setConfirmDiscard(false)}
      >
        <p>It will be removed from the queue and won't be used for training. This can't be undone.</p>
      </ConfirmDialog>
    </div>
  );
}
import React, { useState } from 'react';
import toast from 'react-hot-toast';
import { uploadFile, pollOcrStatus } from '../api';
import ConfirmDialog from './ConfirmDialog';
import { errorMessage } from '../lib/scoring';

// The server answers 422 { code: 'not_essay', message, reason } when the file
// isn't an essay (a face, a car, a blank page...).
class NotEssayError extends Error {
  constructor(message, reason) {
    super(message);
    this.reason = reason;
  }
}

function notEssayFrom(err) {
  const detail = err?.response?.data?.detail;
  if (err?.response?.status === 422 && detail && typeof detail === 'object' && detail.code === 'not_essay') {
    return new NotEssayError(detail.message, detail.reason);
  }
  return null;
}

export default function FileUpload({ onExtracted }) {
  const [uploading, setUploading] = useState(false);
  const [fileName, setFileName] = useState('');
  const [progress, setProgress] = useState('');
  const [notEssay, setNotEssay] = useState(null); // { message, reason }

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setUploading(true);
    setProgress('Uploading file...');

    try {
      let result;
      try {
        result = await uploadFile(file);
      } catch (err) {
        throw notEssayFrom(err) || err;
      }

      // If it's a PDF job, poll for completion
      if (result.job_id) {
        setProgress('Processing PDF pages...');
        const POLL_MS = 1500;
        let attempts = 0;
        const maxAttempts = Math.ceil((5 * 60 * 1000) / POLL_MS); // up to 5 minutes
        while (attempts < maxAttempts) {
          await new Promise(r => setTimeout(r, POLL_MS));
          const status = await pollOcrStatus(result.job_id);

          if (status.status === 'completed') {
            const text = (status.text || '').trim();
            if (!text) {
              throw new Error('PDF OCR finished but returned no text.');
            }
            onExtracted({
              text: status.text,
              confidence: status.confidence ?? 0,
              method: `PDF OCR (${status.engine || 'auto'})`,
              low_confidence: !!status.low_confidence,
            });
            toast.success('PDF text extracted');
            break;
          } else if (status.status === 'failed') {
            if (status.code === 'not_essay') throw new NotEssayError(status.error, '');
            throw new Error(status.error || 'OCR failed');
          } else {
            const seconds = Math.round(((attempts + 1) * POLL_MS) / 1000);
            setProgress(
              status.current_engine
                ? `${status.current_engine} done · ${seconds}s`
                : `Reading PDF… ${seconds}s`
            );
          }
          attempts++;
        }
        if (attempts >= maxAttempts) {
          throw new Error('OCR timed out');
        }
      } else {
        // Image or quick response
        onExtracted({
          text: result.text,
          confidence: result.confidence,
          method: result.method,
        });
        toast.success('Text extracted');
      }
    } catch (err) {
      if (err instanceof NotEssayError) {
        setNotEssay({ message: err.message, reason: err.reason });
      } else {
        toast.error('Upload failed: ' + errorMessage(err));
      }
    } finally {
      setUploading(false);
      setProgress('');
      setFileName('');
      e.target.value = ''; // allow choosing the same file again
    }
  };

  return (
    <div className="rounded-xl border-2 border-dashed border-blue-300 bg-blue-50/50 p-6 text-center">
      <input
        type="file"
        accept=".png,.jpg,.jpeg,.bmp,.tiff,.pdf,.docx"
        onChange={handleFileChange}
        disabled={uploading}
        className="hidden"
        id="file-upload-input"
      />
      <label
        htmlFor="file-upload-input"
        className={`cursor-pointer inline-block px-6 py-3 rounded-xl font-semibold shadow-md transition ${
          uploading
            ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
            : 'bg-blue-600 text-white hover:bg-blue-700'
        }`}
      >
        {uploading ? '⏳ Processing...' : '📂 Browse File'}
      </label>
      <p className="text-xs text-gray-600 mt-3">
        Supported: photos or scans (PNG, JPG, BMP, TIFF), PDF and Word (DOCX) of an English essay
      </p>
      {fileName && (
        <p className="text-sm text-gray-700 mt-2">
          Selected: <strong>{fileName}</strong>
        </p>
      )}
      {progress && (
        <p className="text-sm text-blue-700 mt-1" role="status">{progress}</p>
      )}

      <ConfirmDialog
        open={!!notEssay}
        tone="error"
        icon="📄"
        title="Please attach an essay"
        cancelLabel="Choose another file"
        onCancel={() => setNotEssay(null)}
      >
        <p>{notEssay?.message}</p>
        {notEssay?.reason && <p className="text-xs text-gray-500">Details: {notEssay.reason}</p>}
        <p className="text-xs text-gray-500">
          Tip: photograph the page straight on, in good light, with the whole page in the frame.
        </p>
      </ConfirmDialog>
    </div>
  );
}
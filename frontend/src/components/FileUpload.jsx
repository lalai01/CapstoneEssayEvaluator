import React, { useState } from 'react';
import toast from 'react-hot-toast';
import { uploadFile, pollOcrStatus } from '../api';

export default function FileUpload({ onExtracted }) {
  const [uploading, setUploading] = useState(false);
  const [fileName, setFileName] = useState('');
  const [progress, setProgress] = useState('');

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setUploading(true);
    setProgress('Uploading file...');

    try {
      const result = await uploadFile(file);

      // If it's a PDF job, poll for completion
      if (result.job_id) {
        setProgress('Processing PDF pages...');
        let attempts = 0;
        const maxAttempts = 60; // up to 5 minutes
        while (attempts < maxAttempts) {
          await new Promise(r => setTimeout(r, 5000));
          const status = await pollOcrStatus(result.job_id);

          if (status.status === 'completed') {
            onExtracted({
              text: status.text,
              confidence: status.confidence ?? 90,
              method: `PDF OCR (${status.engine || 'auto'})`,
            });
            toast.success('PDF text extracted');
            break;
          } else if (status.status === 'failed') {
            throw new Error(status.error || 'OCR failed');
          } else {
            setProgress(`Processing page (${attempts + 1})...`);
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
      toast.error('Upload failed: ' + err.message);
    } finally {
      setUploading(false);
      setProgress('');
      setFileName('');
    }
  };

  return (
    <div className="rounded-xl border-2 border-dashed border-blue-300 bg-blue-50/50 p-6 text-center">
      <input
        type="file"
        accept=".png,.jpg,.jpeg,.bmp,.tiff,.pdf"
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
        Supported: PNG, JPG, JPEG, BMP, TIFF, PDF
      </p>
      {fileName && (
        <p className="text-sm text-gray-700 mt-2">
          Selected: <strong>{fileName}</strong>
        </p>
      )}
      {progress && (
        <p className="text-sm text-blue-700 mt-1">{progress}</p>
      )}
    </div>
  );
}
import React, { useRef, useState } from 'react';
import { uploadFile, pollOcrStatus } from '../api';
import toast from 'react-hot-toast';

export default function FileUpload({ onExtracted }) {
  const fileInputRef = useRef();
  const [uploading, setUploading] = useState(false);
  const [preview, setPreview] = useState(null);
  const [fileName, setFileName] = useState('');
  const [agentLogs, setAgentLogs] = useState([]);

  const addLog = (message, type = 'info') => {
    setAgentLogs(prev => [...prev, { message, type, timestamp: new Date().toLocaleTimeString() }]);
  };

  const clearLogs = () => setAgentLogs([]);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setFileName(file.name);
    setUploading(true);
    clearLogs();
    addLog(`📁 Received file: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`, 'info');

    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
      addLog(`🖼️ Image preview loaded`, 'success');
    } else if (file.type === 'application/pdf') {
      setPreview('/pdf-icon.png');
      addLog(`📄 PDF document detected`, 'info');
    }

    addLog(`🚀 Sending to OCR server...`, 'info');

    try {
      const result = await uploadFile(file);

      if (result.job_id) {
        addLog(`⏳ Async OCR job started. Job ID: ${result.job_id}`, 'info');
        addLog(`🔄 Polling for results...`, 'info');
        
        let attempts = 0;
        const maxAttempts = 90;
        const interval = setInterval(async () => {
          attempts++;
          if (attempts > maxAttempts) {
            clearInterval(interval);
            addLog(`❌ OCR timed out after 3 minutes.`, 'error');
            setUploading(false);
            toast.error('OCR timed out.');
            return;
          }

          try {
            const status = await pollOcrStatus(result.job_id);
            if (status.status === 'processing') {
              if (status.current_engine) {
                addLog(`🔄 Trying OCR engine: ${status.current_engine.toUpperCase()}...`, 'info');
              } else {
                addLog(`⏳ Still processing... (${attempts}/${maxAttempts})`, 'info');
              }
            } else if (status.status === 'completed') {
              clearInterval(interval);
              addLog(`✅ OCR completed successfully!`, 'success');
              addLog(`📝 Extracted text length: ${status.text.length} characters`, 'success');
              if (status.engine) {
                addLog(`🔍 Final OCR Engine used: ${status.engine.toUpperCase()}`, 'success');
              }
              onExtracted({
                text: status.text,
                confidence: status.confidence || 90,
                method: `PDF OCR (${status.engine || 'unknown'})`,
                engine: status.engine,
              });
              setUploading(false);
            } else if (status.status === 'failed') {
              clearInterval(interval);
              addLog(`❌ OCR failed: ${status.error}`, 'error');
              toast.error('OCR failed: ' + status.error);
              setUploading(false);
            }
          } catch (pollError) {
            clearInterval(interval);
            addLog(`💥 Polling error: ${pollError.message}`, 'error');
            toast.error('OCR polling failed.');
            setUploading(false);
          }
        }, 2000);
      } else {
        addLog(`✨ OCR completed!`, 'success');
        addLog(`📊 Confidence: ${result.confidence?.toFixed(1)}%`, 'success');
        addLog(`🔍 OCR Engine used: ${result.engine?.toUpperCase() || 'unknown'}`, 'success');
        addLog(`📝 Extracted text length: ${result.text.length} characters`, 'success');
        onExtracted(result);
        setUploading(false);
      }
    } catch (err) {
      addLog(`💥 Upload error: ${err.message}`, 'error');
      toast.error('OCR upload failed: ' + err.message);
      setUploading(false);
    }
  };

  return (
    <div className="mb-5">
      <label className="block text-sm font-semibold text-gray-700 mb-2">
        Upload Image or PDF (auto‑engine selection)
      </label>
      <div className="flex items-center gap-3">
        <input
          type="file"
          accept="image/*,application/pdf"
          onChange={handleFileChange}
          ref={fileInputRef}
          className="hidden"
        />
        <button
          type="button"
          onClick={() => fileInputRef.current.click()}
          className="bg-gray-100 hover:bg-gray-200 px-5 py-2 rounded-lg font-medium transition"
        >
          📂 Browse
        </button>
        <span className="text-sm text-gray-500 truncate">{fileName || 'No file selected'}</span>
      </div>

      {agentLogs.length > 0 && (
        <div className="mt-3 rounded-lg border border-gray-300 bg-gray-900 text-gray-100 font-mono text-xs p-3 max-h-48 overflow-y-auto">
          <div className="flex justify-between items-center mb-2 sticky top-0 bg-gray-900 pb-1">
            <span className="font-bold">🤖 OCR Agent Console</span>
            <button onClick={clearLogs} className="text-gray-400 hover:text-white text-xs">Clear</button>
          </div>
          {agentLogs.map((log, idx) => (
            <div key={idx} className={`py-0.5 ${log.type === 'error' ? 'text-red-400' : log.type === 'success' ? 'text-green-400' : 'text-gray-300'}`}>
              <span className="text-gray-500">[{log.timestamp}]</span> {log.message}
            </div>
          ))}
        </div>
      )}

      {uploading && (
        <p className="text-sm text-blue-500 mt-2 animate-pulse">
          Processing OCR (check console for details)...
        </p>
      )}

      {preview && (
        <div className="mt-3 rounded-lg border p-2 bg-white">
          <img src={preview} alt="Preview" className="max-h-40 object-contain mx-auto" />
          <p className="text-xs text-gray-400 text-center mt-1">
            Preview – OCR engine will be chosen automatically based on quality
          </p>
        </div>
      )}
    </div>
  );
}
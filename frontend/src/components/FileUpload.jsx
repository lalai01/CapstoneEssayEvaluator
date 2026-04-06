import React, { useRef, useState } from 'react';
import { uploadFile } from '../api';
import toast from 'react-hot-toast';

export default function FileUpload({ onExtracted }) {
  const fileInputRef = useRef();
  const [uploading, setUploading] = useState(false);
  const [preview, setPreview] = useState(null);
  const [fileName, setFileName] = useState('');

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setFileName(file.name);
    setUploading(true);
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
    } else if (file.type === 'application/pdf') {
      setPreview('/pdf-icon.png');
    }
    try {
      const result = await uploadFile(file);
      onExtracted(result);
    } catch (err) {
      toast.error('OCR failed: ' + err.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="mb-5">
      <label className="block text-sm font-semibold text-gray-700 mb-2">Upload Image or PDF</label>
      <div className="flex items-center gap-3">
        <input type="file" accept="image/*,application/pdf" onChange={handleFileChange} ref={fileInputRef} className="hidden" />
        <button
          type="button"
          onClick={() => fileInputRef.current.click()}
          className="bg-gray-100 hover:bg-gray-200 px-5 py-2 rounded-lg font-medium transition"
        >
          📂 Browse
        </button>
        <span className="text-sm text-gray-500 truncate">{fileName || 'No file selected'}</span>
      </div>
      {uploading && <p className="text-sm text-blue-500 mt-2 animate-pulse">Processing OCR...</p>}
      {preview && (
        <div className="mt-3 rounded-lg border p-2 bg-white">
          <img src={preview} alt="Preview" className="max-h-40 object-contain mx-auto" />
        </div>
      )}
    </div>
  );
}
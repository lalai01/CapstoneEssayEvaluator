import React, { useState, useEffect } from 'react';
import toast, { Toaster } from 'react-hot-toast';
import EssayInput from './components/EssayInput';
import Results from './components/Results';
import KnowledgeBase from './components/KnowledgeBase';

function App() {
  const [scores, setScores] = useState(null);
  const [feedback, setFeedback] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('evaluate');
  const [currentEssay, setCurrentEssay] = useState('');
  const [evalType, setEvalType] = useState('analytic');

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <Toaster position="top-right" />
      <header className="bg-white/80 backdrop-blur-md shadow-lg sticky top-0 z-10 border-b border-gray-200">
        <div className="container mx-auto px-4 py-5">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold gradient-text">AI Essay Evaluator</h1>
              <p className="text-gray-600 text-sm mt-1">Professional feedback • OCR & PDF support</p>
            </div>
            <div className="hidden md:flex gap-2">
              <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-semibold">Powered by Tesseract</span>
              <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-semibold">AI Analysis</span>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="flex gap-2 mb-8 border-b border-gray-200">
          <button
            className={`px-6 py-3 font-medium rounded-t-lg transition-all ${
              activeTab === 'evaluate'
                ? 'bg-white text-blue-600 border-b-2 border-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
            }`}
            onClick={() => setActiveTab('evaluate')}
          >
            ✍️ Evaluate Essay
          </button>
          <button
            className={`px-6 py-3 font-medium rounded-t-lg transition-all ${
              activeTab === 'knowledge'
                ? 'bg-white text-blue-600 border-b-2 border-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
            }`}
            onClick={() => setActiveTab('knowledge')}
          >
            📚 Knowledge Base
          </button>
        </div>

        {activeTab === 'evaluate' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <EssayInput
              setScores={setScores}
              setFeedback={setFeedback}
              setLoading={setLoading}
              setCurrentEssay={setCurrentEssay}
              setEvalType={setEvalType}
              loading={loading}
            />
            <Results
              scores={scores}
              feedback={feedback}
              loading={loading}
              essayText={currentEssay}
              evalType={evalType}
            />
          </div>
        )}
        {activeTab === 'knowledge' && <KnowledgeBase />}
      </div>
    </div>
  );
}

export default App;
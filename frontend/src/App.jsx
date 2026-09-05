import React, { useState } from 'react';
import toast, { Toaster } from 'react-hot-toast';
import EssayInput from './components/EssayInput';
import Results from './components/Results';
import KnowledgeBase from './components/KnowledgeBase';
import PromptPlayground from './components/PromptPlayground';
import LearningFeedback from './components/LearningFeedback';
import HomePage from './components/HomePage';
import RateUs from './components/RateUs';
import AdminSurveys from './components/AdminSurveys';
import UserSurveys from './components/UserSurveys';   // ✅ New component for users
import LoginModal from './components/LoginModal';
import ThreeDBackground from './components/ThreeDBackground';
import { useAuth } from './context/AuthContext';

function App() {
  const [scores, setScores] = useState(null);
  const [feedback, setFeedback] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('home');
  const [currentEssay, setCurrentEssay] = useState('');
  const [essayTitle, setEssayTitle] = useState('');
  const [evalType, setEvalType] = useState('analytic');

  const [showOverrideModal, setShowOverrideModal] = useState(false);
  const [teacherFeedback, setTeacherFeedback] = useState('');
  const [suggestedChanges, setSuggestedChanges] = useState('');

  const [showLoginModal, setShowLoginModal] = useState(false);

  const { user, signOut } = useAuth();

  const isAdmin = user?.role === 'admin' || user?.email === 'admin_essay_capstone@gmail.com';

  // Build the list of tabs dynamically – Surveys tab for all logged‑in users, Admin Surveys only for admins
  const tabs = [
    { id: 'home', label: '🏠 Home' },
    { id: 'evaluate', label: '✍️ Evaluate Essay' },
    { id: 'knowledge', label: '📚 Knowledge Base' },
    { id: 'learning', label: '🧠 Learning KB' },
    { id: 'playground', label: '🧪 AI Playground' },
    { id: 'rateus', label: '⭐ Rate Us' },
    ...(user ? [{ id: 'surveys', label: '📋 Surveys' }] : []),
    ...(isAdmin ? [{ id: 'admin', label: '🔧 Admin Surveys' }] : []),
  ];

  return (
    <div className="relative min-h-screen">
      <ThreeDBackground />

      <div className="relative z-10 min-h-screen">
        <Toaster position="top-right" toastOptions={{ duration: 4000 }} />

        <header className="relative z-20 flex items-center justify-between px-4 md:px-6 py-3 bg-white/20 backdrop-blur-md border-b border-white/30 shadow-sm">
          <div className="flex flex-col">
            <h1 className="text-xl md:text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              AI Essay Evaluator
            </h1>
            <p className="text-xs md:text-sm text-indigo-200 font-medium">
              Your personal writing assistant
            </p>
          </div>

          <div className="flex items-center gap-2">
            {user ? (
              <>
                <span className="text-sm bg-white/70 backdrop-blur-sm px-3 py-1.5 rounded-full hidden sm:inline-block">
                  {user.email}
                </span>
                <button
                  onClick={signOut}
                  className="bg-red-100/80 backdrop-blur-sm text-red-700 px-4 py-1.5 rounded-lg text-sm font-medium hover:bg-red-200 transition-all"
                >
                  Logout
                </button>
              </>
            ) : (
              <button
                onClick={() => setShowLoginModal(true)}
                className="bg-indigo-600/90 backdrop-blur-sm text-white px-4 py-1.5 rounded-lg text-sm font-medium hover:bg-indigo-700 transition-all"
              >
                Sign In
              </button>
            )}
          </div>
        </header>

        <div className="container mx-auto px-4 pt-6">
          <div className="flex flex-wrap gap-2 border-b border-gray-300/40 pb-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                className={`px-5 py-2.5 font-medium rounded-lg transition-all ${
                  activeTab === tab.id
                    ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md'
                    : 'bg-gray-800/60 backdrop-blur-sm text-gray-200 hover:bg-gray-700/80 hover:text-white border border-gray-600/50'
                }`}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        <div className="container mx-auto px-4 py-8 max-w-7xl">
          {activeTab === 'home' && <HomePage setActiveTab={setActiveTab} />}
          {activeTab === 'evaluate' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 animate-fade-in">
              <EssayInput
                setScores={setScores}
                setFeedback={setFeedback}
                setLoading={setLoading}
                setCurrentEssay={setCurrentEssay}
                setEvalType={setEvalType}
                setEssayTitle={setEssayTitle}
                loading={loading}
              />
              <Results
                scores={scores}
                feedback={feedback}
                loading={loading}
                essayText={currentEssay}
                essayTitle={essayTitle}
                evalType={evalType}
                showOverrideModal={showOverrideModal}
                setShowOverrideModal={setShowOverrideModal}
                teacherFeedback={teacherFeedback}
                setTeacherFeedback={setTeacherFeedback}
                suggestedChanges={suggestedChanges}
                setSuggestedChanges={setSuggestedChanges}
              />
            </div>
          )}
          {activeTab === 'knowledge' && <KnowledgeBase />}
          {activeTab === 'learning' && <LearningFeedback />}
          {activeTab === 'playground' && <PromptPlayground />}
          {activeTab === 'rateus' && <RateUs />}
          {activeTab === 'surveys' && <UserSurveys />}
          {activeTab === 'admin' && <AdminSurveys />}
        </div>
      </div>

      <LoginModal isOpen={showLoginModal} onClose={() => setShowLoginModal(false)} />
    </div>
  );
}

export default App;
import React, { useEffect, useState } from 'react';
import { Toaster } from 'react-hot-toast';
import EssayInput from './components/EssayInput';
import Results from './components/Results';
import KnowledgeBase from './components/KnowledgeBase';
import CompareScores from './components/CompareScores';
import LearningFeedback from './components/LearningFeedback';
import HomePage from './components/HomePage';
import RateUs from './components/RateUs';
import AdminSurveys from './components/AdminSurveys';
import UserSurveys from './components/UserSurveys';
import AdminUsers from './components/AdminUsers';
import RubricManager from './components/RubricManager';
import OcrLabeler from './components/OcrLabeler';
import LoginModal from './components/LoginModal';
import ThreeDBackground from './components/ThreeDBackground';
import AiChoice from './components/AiChoice';
import NavBar from './components/NavBar';
import { useAuth } from './context/AuthContext';
import useLeaveWarning from './lib/useLeaveWarning';
import useDraft from './lib/draft';

// Who may open each page. Admins and teachers are separate roles: admins run
// the system, teachers do the grading. One person can hold both roles, and
// then sees both menus. The server enforces the same rules on every request.
//   public  - everyone, signed in or not
//   user    - any signed-in user (students included)
//   teacher - teachers (approvers)
//   admin   - admins
// `path` is the page's own URL, so it can be bookmarked, shared and reloaded.
const PAGES = [
  { id: 'home', path: '/', label: 'Home', icon: 'home', access: 'public', group: 'main' },
  { id: 'evaluate', path: '/evaluate', label: 'Evaluate', icon: 'evaluate', access: 'public', group: 'main' },
  { id: 'compare', path: '/compare-scores', aliases: ['/playground'], label: 'Compare with My Scores', short: 'Compare', icon: 'compare', access: 'public', group: 'main' },
  { id: 'knowledge', path: '/knowledge-base', label: 'Knowledge Base', icon: 'knowledge', access: 'user', group: 'workspace', description: 'Evaluations you saved' },
  { id: 'surveys', path: '/surveys', label: 'Surveys', icon: 'surveys', access: 'user', group: 'main' },
  { id: 'rateus', path: '/rate-us', label: 'Rate Us', icon: 'rateus', access: 'public', group: 'main' },
  { id: 'learning', path: '/learning-hub', label: 'Learning Hub', icon: 'learning', access: 'user', group: 'workspace', description: 'Your overrides, remarks and score changes' },
  { id: 'rubrics', path: '/my-rubrics', label: 'My Rubrics', icon: 'rubrics', access: 'user', group: 'workspace', description: 'Create rubrics and choose your default' },
  { id: 'admins', path: '/admin', label: 'Admin Dashboard', icon: 'dashboard', access: 'admin', group: 'admin', description: 'Teachers, admins and the activity log' },
  { id: 'school-rubric', path: '/school-rubric', label: 'School Rubric', icon: 'school', access: 'admin', group: 'admin', description: 'Default rubric for every teacher' },
  { id: 'admin', path: '/manage-surveys', label: 'Manage Surveys', icon: 'list', access: 'admin', group: 'admin', description: 'Create surveys and read responses' },
  { id: 'ocr-labeler', path: '/ocr-labeler', label: 'OCR Labeler', icon: 'tag', access: 'admin', group: 'admin', description: 'Correct OCR samples for training' },
];

const pageFromPath = (pathname) => {
  const path = pathname.replace(/\/+$/, '') || '/';
  return PAGES.find((p) => p.path === path || p.aliases?.includes(path));   // aliases: old links
};

// Results restored after a refresh: an AI still working when the page was
// reloaded lost its answer, so its card says so instead of spinning forever.
const restoreRuns = (runs) => Object.fromEntries(Object.entries(runs).map(([id, run]) => [id,
  run?.status === 'running'
    ? { status: 'failed', error: 'Interrupted: the page was reloaded while this AI was working. Evaluate again.' }
    : run]));

const MENU_GROUPS = [
  { id: 'workspace', label: 'My Work', caption: 'Your evaluations, overrides and rubrics', icon: 'knowledge' },
  { id: 'teacher', label: 'Teacher', caption: 'Teacher tools', icon: 'teacher' },
  { id: 'admin', label: 'Admin', caption: 'Administration', icon: 'shield' },
];

const ROLE_BADGES = {
  admin: { label: 'Admin', className: 'bg-indigo-100 text-indigo-800' },
  teacher: { label: 'Teacher · Approver', className: 'bg-emerald-100 text-emerald-800' },
  student: { label: 'Student', className: 'bg-slate-100 text-slate-700' },
};

function App() {
  // The latest evaluation is kept as a draft, so a refresh shows it again.
  const [scores, setScores] = useDraft('scores', null);

  // The three AIs evaluate every essay; the user picks whose result to use.
  const [aiRuns, setAiRuns] = useDraft('aiRuns', null, restoreRuns);   // { [aiId]: { status, scores, feedback, ms, error } }
  const [chosenAi, setChosenAi] = useDraft('chosenAi', null);
  const chooseAi = (id, run) => {
    const picked = run || aiRuns?.[id];
    setChosenAi(id);
    if (picked?.scores) {
      setScores(picked.scores);
      setFeedback(picked.feedback);
    }
  };
  const [feedback, setFeedback] = useDraft('feedback', '');
  const [loading, setLoading] = useState(false);
  // An evaluation in progress can't survive a refresh; ask first.
  useLeaveWarning(loading);
  const [currentEssay, setCurrentEssay] = useDraft('evaluatedEssay', '');
  const [essayTitle, setEssayTitle] = useDraft('evaluatedTitle', '');
  const [evalType, setEvalType] = useDraft('evaluatedType', 'analytic');

  const clearResults = () => {
    setAiRuns(null);
    setChosenAi(null);
    setScores(null);
    setFeedback('');
    setCurrentEssay('');
    setEssayTitle('');
  };

  // The page shown follows the URL; Back and Forward move between pages.
  const [activeTab, setActiveTabState] = useState(() => pageFromPath(window.location.pathname)?.id || 'home');
  useEffect(() => {
    // Unknown paths go Home; old links (e.g. /playground) show their new path.
    const page = pageFromPath(window.location.pathname);
    if (page?.path !== window.location.pathname) {
      window.history.replaceState(null, '', (page?.path || '/') + window.location.search + window.location.hash);
    }
    const onPopState = () => setActiveTabState(pageFromPath(window.location.pathname)?.id || 'home');
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);
  const setActiveTab = (id) => {
    const page = PAGES.find((p) => p.id === id);
    if (!page) return;
    if (window.location.pathname !== page.path) window.history.pushState(null, '', page.path);
    setActiveTabState(id);
    window.scrollTo(0, 0);
  };

  const [showOverrideModal, setShowOverrideModal] = useState(false);
  const [teacherFeedback, setTeacherFeedback] = useState('');
  const [suggestedChanges, setSuggestedChanges] = useState('');

  const [showLoginModal, setShowLoginModal] = useState(false);

  const { user, isAdmin, isTeacher, role, signOut, loading: authLoading, roleLoading } = useAuth();

  const canOpen = (access) =>
    access === 'public' ||
    (access === 'user' && !!user) ||
    (access === 'teacher' && isTeacher) ||
    (access === 'admin' && isAdmin);
  const pages = PAGES.filter((p) => canOpen(p.access));
  const activePage = PAGES.find((p) => p.id === activeTab);
  const allowed = pages.some((p) => p.id === activeTab);
  // After a reload the sign-in and role are still loading: wait for them
  // rather than turning a valid link away.
  const waiting = !allowed && (authLoading || roleLoading);
  const currentTab = allowed ? activeTab : null;

  useEffect(() => {
    document.title = activePage && activePage.id !== 'home'
      ? `${activePage.label} · AI Essay Evaluator`
      : 'AI Essay Evaluator';
  }, [activePage]);

  // The chosen AI's full result, shown on Evaluate and, once the user picks
  // an AI there, on Compare with My Scores. onCompare adds the Compare button.
  const resultsPanel = (onCompare) => (
    <Results
      key={chosenAi || 'none'}
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
      onCompare={onCompare}
    />
  );

  // An admin who is also a teacher shows both badges.
  const roleBadges = !role ? [] : [
    ...(isAdmin ? [ROLE_BADGES.admin] : []),
    ...(isTeacher ? [ROLE_BADGES.teacher] : []),
    ...(!isAdmin && !isTeacher ? [ROLE_BADGES.student] : []),
  ];

  return (
    <div className="relative min-h-screen">
      <ThreeDBackground />

      <div className="relative z-10 min-h-screen">
        <Toaster position="top-right" toastOptions={{ duration: 4000 }} />

        <NavBar
          items={pages}
          groups={MENU_GROUPS}
          currentTab={currentTab}
          onSelect={setActiveTab}
          user={user}
          roleBadges={roleBadges}
          onSignIn={() => setShowLoginModal(true)}
          onSignOut={signOut}
        />

        <div className="container mx-auto px-4 py-8 max-w-7xl">
          {waiting && (
            <div className="flex justify-center py-24" aria-label="Loading">
              <div className="spinner" />
            </div>
          )}
          {!allowed && !waiting && activePage && (
            <div className="glass-card rounded-2xl shadow-xl p-8 bg-white/95 max-w-lg mx-auto text-center animate-fade-in">
              <div className="text-4xl" aria-hidden="true">🔒</div>
              <h2 className="text-xl font-bold text-gray-800 mt-2">{activePage.label}</h2>
              <p className="text-gray-600 mt-2">
                {activePage.access === 'user' && !user
                  ? 'Sign in to open this page.'
                  : 'This page is only for administrators.'}
              </p>
              <div className="mt-5 flex justify-center gap-3">
                {activePage.access === 'user' && !user && (
                  <button
                    onClick={() => setShowLoginModal(true)}
                    className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition"
                  >
                    Sign in
                  </button>
                )}
                <button
                  onClick={() => setActiveTab('home')}
                  className="border border-gray-300 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-50 transition"
                >
                  Go to Home
                </button>
              </div>
            </div>
          )}
          {currentTab === 'home' && <HomePage setActiveTab={setActiveTab} />}
          {currentTab === 'evaluate' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 animate-fade-in">
              <EssayInput
                onNewEssay={clearResults}
                hasResults={!!aiRuns}
                setAiRuns={setAiRuns}
                onChooseAi={chooseAi}
                setScores={setScores}
                setFeedback={setFeedback}
                setLoading={setLoading}
                setCurrentEssay={setCurrentEssay}
                setEvalType={setEvalType}
                setEssayTitle={setEssayTitle}
                loading={loading}
              />
              <div className="space-y-6 min-w-0">
              <AiChoice runs={aiRuns} chosen={chosenAi} onChoose={chooseAi} />
              {resultsPanel(() => setActiveTab('compare'))}
              </div>
            </div>
          )}
          {currentTab === 'knowledge' && <KnowledgeBase />}
          {currentTab === 'learning' && <LearningFeedback />}
          {currentTab === 'compare' && (
            <CompareScores
              aiRuns={aiRuns}
              chosenAi={chosenAi}
              onChooseAi={chooseAi}
              essayTitle={essayTitle}
              evaluatedEssay={currentEssay}
              onGoEvaluate={() => setActiveTab('evaluate')}
              resultsPanel={resultsPanel()}
            />
          )}
          {currentTab === 'rateus' && <RateUs />}
          {currentTab === 'surveys' && <UserSurveys />}
          {currentTab === 'admin' && <AdminSurveys />}
          {currentTab === 'rubrics' && <RubricManager scope="teacher" />}
          {currentTab === 'school-rubric' && <RubricManager scope="school" key="school" />}
          {currentTab === 'ocr-labeler' && <OcrLabeler />}
          {currentTab === 'admins' && <AdminUsers />}
        </div>
      </div>

      <LoginModal isOpen={showLoginModal} onClose={() => setShowLoginModal(false)} />
    </div>
  );
}

export default App;
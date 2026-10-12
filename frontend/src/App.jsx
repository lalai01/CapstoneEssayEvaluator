import React, { useState } from 'react';
import { Toaster } from 'react-hot-toast';
import EssayInput from './components/EssayInput';
import Results from './components/Results';
import KnowledgeBase from './components/KnowledgeBase';
import PromptPlayground from './components/PromptPlayground';
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
import NavBar from './components/NavBar';
import { useAuth } from './context/AuthContext';

// Who may open each page. Admins and teachers are separate roles: admins run
// the system, teachers do the grading. One person can hold both roles, and
// then sees both menus. The server enforces the same rules on every request.
//   public  - everyone, signed in or not
//   user    - any signed-in user (students included)
//   teacher - teachers (approvers)
//   admin   - admins
const PAGES = [
  { id: 'home', label: 'Home', icon: 'home', access: 'public', group: 'main' },
  { id: 'evaluate', label: 'Evaluate', icon: 'evaluate', access: 'public', group: 'main' },
  { id: 'playground', label: 'Playground', icon: 'playground', access: 'public', group: 'main' },
  { id: 'knowledge', label: 'Knowledge Base', icon: 'knowledge', access: 'user', group: 'main' },
  { id: 'surveys', label: 'Surveys', icon: 'surveys', access: 'user', group: 'main' },
  { id: 'rateus', label: 'Rate Us', icon: 'rateus', access: 'public', group: 'main' },
  { id: 'learning', label: 'Learning Hub', icon: 'learning', access: 'teacher', group: 'teacher', description: 'Your overrides, remarks and score changes' },
  { id: 'rubrics', label: 'My Rubrics', icon: 'rubrics', access: 'teacher', group: 'teacher', description: 'Create rubrics and choose your default' },
  { id: 'admins', label: 'Admin Dashboard', icon: 'dashboard', access: 'admin', group: 'admin', description: 'Teachers, admins and the activity log' },
  { id: 'school-rubric', label: 'School Rubric', icon: 'school', access: 'admin', group: 'admin', description: 'Default rubric for every teacher' },
  { id: 'admin', label: 'Manage Surveys', icon: 'list', access: 'admin', group: 'admin', description: 'Create surveys and read responses' },
  { id: 'ocr-labeler', label: 'OCR Labeler', icon: 'tag', access: 'admin', group: 'admin', description: 'Correct OCR samples for training' },
];

const MENU_GROUPS = [
  { id: 'teacher', label: 'Teacher', caption: 'Teacher tools', icon: 'teacher' },
  { id: 'admin', label: 'Admin', caption: 'Administration', icon: 'shield' },
];

const ROLE_BADGES = {
  admin: { label: 'Admin', className: 'bg-indigo-100 text-indigo-800' },
  teacher: { label: 'Teacher · Approver', className: 'bg-emerald-100 text-emerald-800' },
  student: { label: 'Student', className: 'bg-slate-100 text-slate-700' },
};

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

  const { user, isAdmin, isTeacher, role, signOut } = useAuth();

  const canOpen = (access) =>
    access === 'public' ||
    (access === 'user' && !!user) ||
    (access === 'teacher' && isTeacher) ||
    (access === 'admin' && isAdmin);
  const pages = PAGES.filter((p) => canOpen(p.access));
  // If the current page is no longer allowed (signed out, role changed),
  // show Home instead of a page the user can't use.
  const currentTab = pages.some((p) => p.id === activeTab) ? activeTab : 'home';

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
          {currentTab === 'home' && <HomePage setActiveTab={setActiveTab} />}
          {currentTab === 'evaluate' && (
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
          {currentTab === 'knowledge' && <KnowledgeBase />}
          {currentTab === 'learning' && <LearningFeedback />}
          {currentTab === 'playground' && <PromptPlayground />}
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
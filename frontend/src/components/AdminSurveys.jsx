import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import {
  createSurvey, listSurveys, updateSurvey, deleteSurvey,
  addQuestionsBulk, listQuestions,
  getSurveyResponses, apiClient,
} from '../api';
import toast from 'react-hot-toast';
import SurveyResults from './SurveyResults';
import SurveyEditor from './SurveyEditor';
import { copyOf, finalQuestionnaire } from '../lib/surveyKit';

export default function AdminSurveys() {
  const { isAdmin } = useAuth();

  // Global state
  const [surveys, setSurveys] = useState([]);
  const [selectedSurvey, setSelectedSurvey] = useState(null);   // null when creating new
  const [questions, setQuestions] = useState([]);
  const [responses, setResponses] = useState([]);
  const [questionsMap, setQuestionsMap] = useState({});
  const [userNames, setUserNames] = useState({});
  const [view, setView] = useState('list');          // 'list' | 'edit' | 'responses'

  // Survey form
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [isActive, setIsActive] = useState(true);
  const [editingSurveyId, setEditingSurveyId] = useState(null);

  // Question form

  const [creatingTemplate, setCreatingTemplate] = useState(false);

  // ---------- Data Loading ----------
  useEffect(() => { if (isAdmin) loadSurveys(); }, [isAdmin]);

  const loadSurveys = async () => {
    try {
      const data = await listSurveys(false);
      setSurveys(data);
    } catch (err) { toast.error('Failed to load surveys'); }
  };

  const loadQuestions = async (surveyId) => {
    try {
      const data = await listQuestions(surveyId);
      setQuestions(data);
      const map = {};
      data.forEach(q => map[q.id] = q.question);
      setQuestionsMap(map);
    } catch (err) { toast.error('Failed to load questions'); }
  };

  const loadResponses = useCallback(async (surveyId) => {
    try {
      const data = await getSurveyResponses(surveyId);
      setResponses(data);
      const userIds = [...new Set(data.map(r => r.user_id).filter(Boolean))];
      if (userIds.length > 0) fetchUserNames(userIds);
    } catch (err) { toast.error('Failed to load responses'); }
  }, []);

  const fetchUserNames = async (userIds) => {
    try {
      const { data } = await apiClient.get(`/admin/user-profiles?ids=${userIds.join(',')}`);
      if (data?.profiles) {
        const map = {};
        data.profiles.forEach(p => map[p.id] = p.full_name || 'Anonymous');
        setUserNames(map);
      }
    } catch (err) {
      console.error('Failed to fetch user names:', err);
    }
  };

  // ---------- Survey CRUD ----------
  const handleSurveySubmit = async (e) => {
    e.preventDefault();
    try {
      if (editingSurveyId) {
        await updateSurvey(editingSurveyId, { title, description, is_active: isActive });
        toast.success('Survey updated');
      } else {
        const res = await createSurvey({ title, description, is_active: isActive });
        // After creation, set selectedSurvey so that questions section appears
        setSelectedSurvey({ id: res.id, title: title, description: description, is_active: isActive });
        toast.success('Survey created');
      }
      loadSurveys();
    } catch (err) { toast.error('Failed'); }
  };

  const handleEditSurvey = (s) => {
    setEditingSurveyId(s.id);
    setTitle(s.title);
    setDescription(s.description || '');
    setIsActive(s.is_active);
    setView('edit');
    setSelectedSurvey(s);
    loadQuestions(s.id);
  };

  const handleDeleteSurvey = async (id) => {
    if (confirm('Delete survey and all questions?')) {
      await deleteSurvey(id);
      toast.success('Survey deleted');
      loadSurveys();
    }
  };

  // Reset everything and go back to list
  const resetSurveyForm = () => {
    setEditingSurveyId(null);
    setTitle('');
    setDescription('');
    setIsActive(true);
    setView('list');
    setSelectedSurvey(null);
    setQuestions([]);
    setResponses([]);
    setQuestionsMap({});
    setUserNames({});
  };

  // Start a fresh new survey
  const startNewSurvey = () => {
    setEditingSurveyId(null);
    setTitle('');
    setDescription('');
    setIsActive(true);
    setSelectedSurvey(null);   // no survey yet
    setQuestions([]);
    setResponses([]);
    setView('edit');
  };

  // Another set: a copy of a survey with all its tabs, questions and files,
  // inactive until it's reviewed. Responses are not copied.
  const duplicateSurvey = async (s) => {
    try {
      const qs = (await listQuestions(s.id)).sort((a, b) => a.order_number - b.order_number);
      const res = await createSurvey({ title: `${s.title} (copy)`, description: s.description, is_active: false });
      if (qs.length) await addQuestionsBulk(res.id, qs.map(copyOf));
      toast.success('Survey copied. Edit the copy, then tick “Active” when it is ready.');
      await loadSurveys();
      handleEditSurvey({ id: res.id, title: `${s.title} (copy)`, description: s.description, is_active: false });
    } catch (err) {
      toast.error('Could not copy the survey: ' + (err?.response?.data?.detail || err.message));
    }
  };

  // Creates the Final Questionnaire (inactive, so it can be reviewed first).
  const createFinalQuestionnaire = async () => {
    if (!confirm('Create the Final Questionnaire survey (about 55 items)? It starts inactive so you can review it first.')) return;
    setCreatingTemplate(true);
    try {
      const t = finalQuestionnaire();
      const res = await createSurvey({ title: t.title, description: t.description, is_active: false });
      await addQuestionsBulk(res.id, t.questions);
      toast.success('Final Questionnaire created. Review it, then tick “Active” to open it to respondents.');
      await loadSurveys();
      handleEditSurvey({ id: res.id, title: t.title, description: t.description, is_active: false });
    } catch (err) {
      toast.error('Could not create the questionnaire: ' + (err?.response?.data?.detail || err.message));
    } finally {
      setCreatingTemplate(false);
    }
  };

  // ---------- Guard ----------
  if (!isAdmin) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 text-center">
        <div className="text-6xl mb-4">🚫</div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Access Denied</h2>
        <p className="text-gray-600">You do not have administrator privileges.</p>
      </div>
    );
  }

  // ---------- Render ----------
  return (
    <div className="glass-card rounded-2xl shadow-xl p-6 max-w-6xl mx-auto">
      {/* -------- HEADER (List view) -------- */}
      {view === 'list' && (
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-indigo-700">📋 Admin Survey Builder</h2>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={createFinalQuestionnaire}
              disabled={creatingTemplate}
              className="bg-white border border-indigo-300 text-indigo-700 px-4 py-2 rounded-lg hover:bg-indigo-50 transition disabled:opacity-50"
            >
              {creatingTemplate ? 'Creating…' : '📋 Create Final Questionnaire'}
            </button>
            <button
              onClick={startNewSurvey}
              className="bg-indigo-600 text-white px-4 py-2 rounded-lg shadow hover:bg-indigo-700 transition"
            >
              + Create New Survey
            </button>
          </div>
        </div>
      )}

      {/* -------- HEADER (Edit / Responses) -------- */}
      {view !== 'list' && (
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold text-indigo-700">📋 Admin Survey Builder</h2>
          <button onClick={() => resetSurveyForm()} className="text-blue-600 hover:underline">
            &larr; Back to Surveys
          </button>
        </div>
      )}

      {/* ========== LIST VIEW ========== */}
      {view === 'list' && (
        <div className="space-y-3">
          {surveys.length === 0 && <p className="text-gray-500">No surveys yet.</p>}
          {surveys.map(s => (
            <div key={s.id} className="border p-4 rounded-xl flex justify-between items-center bg-white hover:bg-gray-50 transition">
              <div>
                <div className="font-semibold text-gray-800">{s.title}</div>
                <p className="text-sm text-gray-500 mt-1">{s.description}</p>
                <span className={`text-xs px-2 py-0.5 rounded-full ${s.is_active ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>
                  {s.is_active ? '🟢 Active' : '⚪ Inactive'}
                </span>
              </div>
              <div className="flex gap-2">
                <button onClick={() => handleEditSurvey(s)} className="text-blue-600 text-sm hover:underline">Edit</button>
                <button onClick={() => handleEditSurvey(s)} className="text-gray-600 text-sm hover:underline">Questions</button>
                <button onClick={() => duplicateSurvey(s)} className="text-gray-600 text-sm hover:underline">Duplicate</button>
                <button onClick={() => { setSelectedSurvey(s); loadQuestions(s.id); loadResponses(s.id); setView('responses'); }} className="text-gray-600 text-sm hover:underline">Responses</button>
                <button onClick={() => handleDeleteSurvey(s.id)} className="text-red-600 text-sm hover:underline">Delete</button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ========== EDIT VIEW ========== */}
      {view === 'edit' && (
        <div>
          {/* Survey Form – always visible when creating or editing */}
          <form onSubmit={handleSurveySubmit} className="bg-gray-50 p-4 rounded-lg mb-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block font-medium mb-1">Survey Title</label>
                <input type="text" value={title} onChange={e => setTitle(e.target.value)} className="w-full border rounded-lg p-2" required />
              </div>
              <div>
                <label className="block font-medium mb-1">Description (optional)</label>
                <textarea value={description} onChange={e => setDescription(e.target.value)} className="w-full border rounded-lg p-2" rows="2" />
              </div>
            </div>
            <div className="flex items-center gap-2 mt-3">
              <input type="checkbox" checked={isActive} onChange={e => setIsActive(e.target.checked)} />
              <span className="text-sm">Active (visible to users)</span>
            </div>
            <button type="submit" className="mt-3 bg-indigo-600 text-white px-4 py-2 rounded-lg">
              {editingSurveyId ? 'Update Survey' : 'Create Survey'}
            </button>
          </form>

          {/* Tabs and questions – only after a survey exists */}
          {selectedSurvey && (
            <SurveyEditor survey={selectedSurvey} questions={questions} reload={() => loadQuestions(selectedSurvey.id)} />
          )}
        </div>
      )}

      {/* ========== RESPONSES VIEW ========== */}
      {view === 'responses' && selectedSurvey && (
        <div>
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-semibold text-gray-800">📊 Results: {selectedSurvey.title}</h3>
          </div>

          <SurveyResults
            survey={selectedSurvey}
            questions={questions}
            responses={responses}
            userNames={userNames}
            onRefresh={() => loadResponses(selectedSurvey.id)}
          />
          {responses.length > 0 && (
            <>
              {/* Raw table */}
              <details className="mt-4">
                <summary className="cursor-pointer font-medium text-gray-700 hover:text-indigo-600">📝 View Individual Responses</summary>
                <div className="mt-3 overflow-x-auto">
                  <table className="min-w-full border text-sm">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="border p-2 text-left">User</th>
                        <th className="border p-2 text-left">Question</th>
                        <th className="border p-2 text-left">Answer</th>
                        <th className="border p-2 text-left">Date</th>
                      </tr>
                    </thead>
                    <tbody>
                      {responses.map((r,i)=>(
                        <tr key={i} className="hover:bg-gray-50">
                          <td className="border p-2 text-xs">{userNames[r.user_id] || r.user_id?.substring(0,8)+'...'}</td>
                          <td className="border p-2">{questionsMap[r.question_id] || `Q#${r.question_id}`}</td>
                          <td className="border p-2 whitespace-pre-wrap">{r.answer}</td>
                          <td className="border p-2 text-xs">{new Date(r.created_at).toLocaleDateString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            </>
          )}
        </div>
      )}
    </div>
  );
}
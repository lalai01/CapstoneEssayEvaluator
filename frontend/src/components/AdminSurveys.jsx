import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import {
  createSurvey, listSurveys, updateSurvey, deleteSurvey,
  addQuestion, listQuestions, updateQuestion, deleteQuestion,
  getSurveyResponses, apiClient, testPrompt
} from '../api';
import toast from 'react-hot-toast';

const QUESTION_TYPES = [
  { value: 'text', label: 'Short Text' },
  { value: 'essay', label: 'Essay (Long Text)' },
  { value: 'radio', label: 'Radio Button' },
  { value: 'checkbox', label: 'Checkbox' },
  { value: 'rating', label: 'Rating (1‑5 stars)' },
];

export default function AdminSurveys() {
  const { user } = useAuth();
  const isAdmin = user?.role === 'admin' || user?.email === 'admin_essay_capstone@gmail.com';

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
  const [showQuestionForm, setShowQuestionForm] = useState(false);
  const [editingQuestion, setEditingQuestion] = useState(null);
  const [qText, setQText] = useState('');
  const [qType, setQType] = useState('text');
  const [qOptions, setQOptions] = useState('');
  const [qRequired, setQRequired] = useState(true);

  // AI Conclusion state
  const [conclusion, setConclusion] = useState('');
  const [conclusionLoading, setConclusionLoading] = useState(false);

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
    setShowQuestionForm(false);
    resetQuestionForm();
    setConclusion('');
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

  // ---------- Question CRUD ----------
  const handleQuestionSubmit = async (e) => {
    e.preventDefault();
    let options = null;
    if (qType === 'radio' || qType === 'checkbox') {
      options = qOptions.split(',').map(s => s.trim()).filter(Boolean);
    } else if (qType === 'rating') {
      options = ['1','2','3','4','5'];
    }
    const qData = {
      survey_id: selectedSurvey.id,
      question: qText,
      question_type: qType,
      options: options,
      is_required: qRequired,
      order_number: 0
    };
    try {
      if (editingQuestion) {
        await updateQuestion(editingQuestion.id, qData);
        toast.success('Question updated');
      } else {
        await addQuestion(selectedSurvey.id, qData);
        toast.success('Question added');
      }
      resetQuestionForm();
      loadQuestions(selectedSurvey.id);
    } catch (err) { toast.error('Failed'); }
  };

  const handleEditQuestion = (q) => {
    setEditingQuestion(q);
    setQText(q.question);
    setQType(q.question_type);
    setQOptions(q.options?.join(', ') || '');
    setQRequired(q.is_required);
    setShowQuestionForm(true);
  };

  const handleDeleteQuestion = async (id) => {
    if (confirm('Delete question?')) {
      await deleteQuestion(id);
      toast.success('Question deleted');
      loadQuestions(selectedSurvey.id);
    }
  };

  const resetQuestionForm = () => {
    setEditingQuestion(null);
    setQText('');
    setQType('text');
    setQOptions('');
    setQRequired(true);
    setShowQuestionForm(false);
  };

  // ---------- Tally & Summary Helpers ----------
  const getQuestionSummary = (questionId, questionType, options) => {
    const answersForQuestion = responses
      .filter(r => r.question_id === questionId)
      .map(r => r.answer);
    const total = answersForQuestion.length;
    if (total === 0) return null;

    if (questionType === 'rating') {
      const numeric = answersForQuestion.map(Number).filter(n => !isNaN(n));
      const avg = numeric.reduce((a, b) => a + b, 0) / numeric.length;
      const dist = {};
      numeric.forEach(v => dist[v] = (dist[v] || 0) + 1);
      return { type: 'rating', total, avg: avg.toFixed(1), distribution: dist };
    }

    if (questionType === 'radio' || questionType === 'text' || questionType === 'essay') {
      const counts = {};
      answersForQuestion.forEach(ans => counts[ans] = (counts[ans] || 0) + 1);
      const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
      return { type: 'choice', total, counts: sorted };
    }

    if (questionType === 'checkbox') {
      const optionCounts = {};
      answersForQuestion.forEach(ans => {
        ans.split(',').map(s => s.trim()).forEach(opt => {
          if (opt) optionCounts[opt] = (optionCounts[opt] || 0) + 1;
        });
      });
      const sorted = Object.entries(optionCounts).sort((a, b) => b[1] - a[1]);
      return { type: 'checkbox', total, counts: sorted };
    }

    return null;
  };

  // ---------- AI Conclusion ----------
  const generateConclusion = async () => {
    if (responses.length === 0) {
      toast.error('No responses to generate a conclusion from.');
      return;
    }
    setConclusionLoading(true);
    try {
      let summary = `Survey: ${selectedSurvey.title}\n`;
      if (selectedSurvey.description) summary += `Description: ${selectedSurvey.description}\n`;
      summary += `Total respondents: ${new Set(responses.map(r => r.user_id)).size}\n\n`;

      questions.forEach(q => {
        const qSummary = getQuestionSummary(q.id, q.question_type, q.options);
        summary += `Question: "${q.question}"\n`;
        if (qSummary) {
          summary += `Response count: ${qSummary.total}\n`;
          if (qSummary.type === 'rating') {
            summary += `Average rating: ${qSummary.avg}/5\n`;
            summary += 'Distribution:\n';
            for (let star = 5; star >= 1; star--) {
              const count = qSummary.distribution[star] || 0;
              summary += `  ${star}★: ${count} (${((count / qSummary.total) * 100).toFixed(0)}%)\n`;
            }
          } else if (qSummary.type === 'choice' || qSummary.type === 'checkbox') {
            summary += 'Choices:\n';
            qSummary.counts.forEach(([val, cnt]) => {
              summary += `  ${val}: ${cnt} (${((cnt / qSummary.total) * 100).toFixed(0)}%)\n`;
            });
          }
        }
        summary += '\n';
      });

      const systemPrompt = "You are a professional data analyst. Generate a concise, insightful conclusion from the survey data. Use formal, friendly language.";
      const userPrompt = `Analyze the following survey results and write a conclusion:\n\n${summary}`;

      const response = await testPrompt('gemma', systemPrompt, userPrompt, 'gemma2:2b');
      setConclusion(response.result?.text || 'Could not generate conclusion.');
    } catch (err) {
      toast.error('Failed to generate conclusion');
      console.error(err);
    } finally {
      setConclusionLoading(false);
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
          <button
            onClick={startNewSurvey}
            className="bg-indigo-600 text-white px-4 py-2 rounded-lg shadow hover:bg-indigo-700 transition"
          >
            + Create New Survey
          </button>
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
                <button onClick={() => { setSelectedSurvey(s); loadQuestions(s.id); setView('edit'); }} className="text-gray-600 text-sm hover:underline">Questions</button>
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

          {/* Questions section – only after a survey exists */}
          {selectedSurvey && (
            <div>
              <button onClick={() => setShowQuestionForm(true)} className="bg-green-600 text-white px-3 py-1 rounded-lg mb-3">
                + Add Question
              </button>

              {showQuestionForm && (
                <div className="bg-white border-2 border-indigo-200 p-4 rounded-lg mb-4 shadow">
                  <h4 className="font-medium text-gray-800 mb-3">{editingQuestion ? 'Edit Question' : 'New Question'}</h4>
                  <form onSubmit={handleQuestionSubmit} className="space-y-3">
                    <div>
                      <label className="block text-sm font-medium mb-1">Question Text</label>
                      <input type="text" value={qText} onChange={e => setQText(e.target.value)} placeholder="e.g., How satisfied are you?" className="w-full border rounded-lg p-2" required />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">Question Type</label>
                      <select value={qType} onChange={e => setQType(e.target.value)} className="w-full border rounded-lg p-2">
                        {QUESTION_TYPES.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
                      </select>
                    </div>
                    {(qType === 'radio' || qType === 'checkbox') && (
                      <div>
                        <label className="block text-sm font-medium mb-1">Options <span className="text-gray-400 font-normal">(comma separated)</span></label>
                        <input type="text" value={qOptions} onChange={e => setQOptions(e.target.value)} placeholder="e.g., Excellent, Good, Average, Poor" className="w-full border rounded-lg p-2" />
                      </div>
                    )}
                    {qType === 'rating' && <div className="bg-yellow-50 p-2 rounded-lg text-sm">⭐ Rating uses 1‑5 stars.</div>}
                    {qType === 'essay' && <div className="bg-blue-50 p-2 rounded-lg text-sm">📝 Essay allows free‑form text.</div>}
                    <div className="flex items-center gap-6">
                      <label className="flex items-center gap-2">
                        <input type="checkbox" checked={qRequired} onChange={e => setQRequired(e.target.checked)} />
                        <span className="text-sm">Required</span>
                      </label>
                    </div>
                    <div className="flex gap-2">
                      <button type="submit" className="bg-indigo-600 text-white px-4 py-2 rounded-lg">{editingQuestion ? 'Update' : 'Add'}</button>
                      <button type="button" onClick={() => setShowQuestionForm(false)} className="bg-gray-300 px-4 py-2 rounded-lg">Cancel</button>
                    </div>
                  </form>
                </div>
              )}

              <div className="space-y-2">
                {questions.length === 0 && !showQuestionForm && <p className="text-gray-500 text-sm">No questions yet.</p>}
                {questions.sort((a, b) => a.order_number - b.order_number).map(q => (
                  <div key={q.id} className="border p-3 rounded-lg flex justify-between items-center bg-white">
                    <div>
                      <div className="font-medium text-gray-800">{q.order_number}. {q.question}</div>
                      <div className="flex gap-2 mt-1">
                        <span className="text-xs bg-gray-100 px-2 py-0.5 rounded-full">{QUESTION_TYPES.find(t=>t.value===q.question_type)?.label}</span>
                        {q.is_required && <span className="text-xs bg-red-50 text-red-600 px-2 py-0.5 rounded-full">Required</span>}
                        {q.options && <span className="text-xs text-gray-500">Options: {Array.isArray(q.options)?q.options.join(', '):q.options}</span>}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <button onClick={() => handleEditQuestion(q)} className="text-blue-600 text-sm hover:underline">Edit</button>
                      <button onClick={() => handleDeleteQuestion(q.id)} className="text-red-600 text-sm hover:underline">Delete</button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ========== RESPONSES VIEW ========== */}
      {view === 'responses' && selectedSurvey && (
        <div>
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-semibold text-gray-800">📊 Responses for: {selectedSurvey.title}</h3>
            <button onClick={() => loadResponses(selectedSurvey.id)} className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-1 rounded-lg text-sm flex items-center gap-1">🔄 Refresh</button>
          </div>
          <p className="text-sm text-gray-500 mb-2">Total submissions: {new Set(responses.map(r => r.user_id)).size}</p>

          {responses.length === 0 ? (
            <p className="text-gray-500">No responses yet.</p>
          ) : (
            <>
              {/* Graphs */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                {questions.map(q => {
                  const summary = getQuestionSummary(q.id, q.question_type, q.options);
                  if (!summary) return null;
                  return (
                    <div key={q.id} className="bg-white border rounded-xl p-4 shadow-sm">
                      <h4 className="font-medium text-gray-800 mb-2 text-sm">{q.question}</h4>
                      <p className="text-xs text-gray-500 mb-2">{summary.total} response{summary.total!==1?'s':''}</p>
                      {summary.type==='rating' && (
                        <div>
                          <div className="text-3xl font-bold text-indigo-600 mb-2">{summary.avg} / 5</div>
                          {[5,4,3,2,1].map(star => {
                            const count = summary.distribution[star] || 0;
                            const pct = ((count/summary.total)*100).toFixed(0);
                            return (
                              <div key={star} className="flex items-center gap-2 text-xs mb-1">
                                <span className="w-8 text-right">{'⭐'.repeat(star)}</span>
                                <div className="flex-1 h-2 bg-gray-200 rounded-full"><div className="bg-yellow-400 h-2 rounded-full" style={{width:`${pct}%`}}/></div>
                                <span className="w-6 text-gray-600">{count}</span>
                                <span className="w-8 text-gray-500">{pct}%</span>
                              </div>
                            );
                          })}
                        </div>
                      )}
                      {(summary.type==='choice'||summary.type==='checkbox') && (
                        <div className="space-y-1">
                          {summary.counts.map(([val,count]) => {
                            const pct = ((count/summary.total)*100).toFixed(0);
                            return (
                              <div key={val} className="flex items-center gap-2 text-xs">
                                <span className="w-32 truncate text-gray-700">{val}</span>
                                <div className="flex-1 h-2 bg-gray-200 rounded-full"><div className="bg-indigo-400 h-2 rounded-full" style={{width:`${pct}%`}}/></div>
                                <span className="w-6 text-gray-600">{count}</span>
                                <span className="w-8 text-gray-500">{pct}%</span>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* AI Conclusion */}
              <div className="flex justify-end mb-4">
                <button
                  onClick={generateConclusion}
                  disabled={conclusionLoading}
                  className="bg-purple-600 text-white px-4 py-2 rounded-lg shadow hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  {conclusionLoading ? <span className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></span> : <span>🤖</span>}
                  {conclusionLoading ? 'Generating...' : 'Generate AI Conclusion'}
                </button>
              </div>

              {conclusion && (
                <div className="bg-purple-50 border border-purple-200 rounded-xl p-4 mb-4">
                  <h4 className="font-semibold text-purple-800 mb-2">📘 AI‑Generated Conclusion</h4>
                  <p className="text-sm text-gray-700 whitespace-pre-wrap">{conclusion}</p>
                </div>
              )}

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
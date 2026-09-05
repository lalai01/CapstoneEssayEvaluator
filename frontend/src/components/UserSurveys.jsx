import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { listSurveys, listQuestions, submitSurveyResponse, apiClient } from '../api';
import toast from 'react-hot-toast';

export default function UserSurveys() {
  const { user } = useAuth();
  const [surveys, setSurveys] = useState([]);              // surveys with questions
  const [selectedSurvey, setSelectedSurvey] = useState(null);
  const [questions, setQuestions] = useState([]);
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [alreadySubmitted, setAlreadySubmitted] = useState(false);
  const [myResponses, setMyResponses] = useState([]);
  const [viewMode, setViewMode] = useState('list');        // 'list', 'answer', 'view-my-responses', 'no-questions'

  useEffect(() => {
    if (user) {
      loadAvailableSurveys();
    }
  }, [user]);

  // Load active surveys and check which ones have questions
  const loadAvailableSurveys = async () => {
    try {
      const allSurveys = await listSurveys(true);   // active only

      // For each survey, try to get its questions (limit 1 just to check existence)
      const surveysWithQuestions = await Promise.all(
        allSurveys.map(async (s) => {
          try {
            const qs = await listQuestions(s.id);
            return qs.length > 0 ? s : null;
          } catch {
            return null;   // skip if questions can't be loaded
          }
        })
      );

      setSurveys(surveysWithQuestions.filter(Boolean));
    } catch (err) {
      toast.error('Failed to load surveys');
    }
  };

  const selectSurvey = async (survey) => {
    setSelectedSurvey(survey);
    setSubmitted(false);
    setAnswers({});
    setAlreadySubmitted(false);
    try {
      // Double-check if the user already submitted
      const { data: check } = await apiClient.get(`/surveys/${survey.id}/my-response`);
      if (check.already_submitted) {
        setAlreadySubmitted(true);
        // Load their existing responses
        const { data: respData } = await apiClient.get(`/surveys/${survey.id}/my-responses`);
        setMyResponses(respData.responses || []);
        setViewMode('view-my-responses');
        return;
      }

      // Load the questions
      const qs = await listQuestions(survey.id);
      if (!qs || qs.length === 0) {
        // No questions (shouldn't happen because we filtered, but just in case)
        setViewMode('no-questions');
        return;
      }
      setQuestions(qs);
      setViewMode('answer');
    } catch (err) {
      toast.error('Failed to load survey');
    }
  };

  const handleAnswerChange = (questionId, value) => {
    setAnswers(prev => ({ ...prev, [questionId]: value }));
  };

  const handleCheckboxChange = (questionId, option, checked) => {
    setAnswers(prev => {
      const current = prev[questionId] ? prev[questionId].split(',').filter(Boolean) : [];
      if (checked) return { ...prev, [questionId]: [...current, option].join(',') };
      else return { ...prev, [questionId]: current.filter(o => o !== option).join(',') };
    });
  };

  const handleSubmit = async () => {
    if (!selectedSurvey) return;
    for (let q of questions) {
      if (q.is_required) {
        const answer = answers[q.id];
        if (answer === undefined || answer === '' || (q.question_type === 'checkbox' && answer.split(',').filter(Boolean).length === 0)) {
          toast.error(`Please answer: ${q.question}`);
          return;
        }
      }
    }
    setSubmitting(true);
    try {
      await submitSurveyResponse(selectedSurvey.id, answers);
      toast.success('Survey submitted! Thank you.');
      setSubmitted(true);
      setAlreadySubmitted(true);
      // Reload responses
      const { data: respData } = await apiClient.get(`/surveys/${selectedSurvey.id}/my-responses`);
      setMyResponses(respData.responses || []);
      setViewMode('view-my-responses');
    } catch (err) {
      toast.error('Submission failed. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  if (!user) {
    return (
      <div className="glass-card rounded-2xl shadow-xl p-12 text-center">
        <div className="text-6xl mb-4">🔒</div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Login Required</h2>
        <p className="text-gray-600">Please log in to participate in surveys.</p>
      </div>
    );
  }

  return (
    <div className="glass-card rounded-2xl shadow-xl p-6 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold text-indigo-700 mb-4">📋 Surveys</h2>

      {/* ----- Survey List (only shows surveys with questions) ----- */}
      {!selectedSurvey && (
        <div className="space-y-3">
          {surveys.length === 0 ? (
            <p className="text-gray-500">No surveys available at the moment.</p>
          ) : (
            surveys.map(s => (
              <div
                key={s.id}
                onClick={() => selectSurvey(s)}
                className="border p-4 rounded-xl cursor-pointer hover:bg-gray-50 transition bg-white"
              >
                <div className="font-semibold text-gray-800">{s.title}</div>
                {s.description && <div className="text-sm text-gray-600 mt-1">{s.description}</div>}
              </div>
            ))
          )}
        </div>
      )}

      {/* ----- Survey Detail ----- */}
      {selectedSurvey && (
        <div>
          <button onClick={() => setSelectedSurvey(null)} className="text-blue-600 mb-4 inline-block hover:underline">
            &larr; Back to surveys
          </button>

          <h3 className="text-xl font-semibold text-gray-800 mb-2">{selectedSurvey.title}</h3>
          {selectedSurvey.description && <p className="text-gray-600 mb-4">{selectedSurvey.description}</p>}

          {/* No questions (shouldn't happen, but handle gracefully) */}
          {viewMode === 'no-questions' && (
            <div className="bg-amber-50 border border-amber-200 p-4 rounded-xl text-center">
              <p className="text-amber-800">This survey is not ready yet. Please check back later.</p>
            </div>
          )}

          {/* Already submitted – show past responses */}
          {viewMode === 'view-my-responses' && (
            <div>
              <div className="bg-blue-50 border border-blue-200 p-4 rounded-xl mb-6">
                <p className="text-blue-800 font-medium">✅ You have already submitted this survey.</p>
              </div>
              <div className="space-y-4">
                <h4 className="font-semibold text-gray-800">Your Responses</h4>
                {myResponses.map((item, idx) => (
                  <div key={idx} className="bg-white border rounded-lg p-4">
                    <p className="font-medium text-gray-800">{item.question}</p>
                    {item.question_type === 'rating' ? (
                      <div className="text-yellow-500 text-xl mt-1">
                        {'⭐'.repeat(parseInt(item.answer))}
                      </div>
                    ) : (
                      <p className="text-gray-700 mt-1 whitespace-pre-wrap">{item.answer}</p>
                    )}
                    <p className="text-xs text-gray-400 mt-2">{new Date(item.created_at).toLocaleString()}</p>
                  </div>
                ))}
              </div>
              <button
                onClick={() => setSelectedSurvey(null)}
                className="mt-4 bg-indigo-600 text-white px-4 py-2 rounded-lg"
              >
                Back to Surveys
              </button>
            </div>
          )}

          {/* Answer form */}
          {viewMode === 'answer' && (
            <div className="space-y-5">
              {questions.sort((a, b) => a.order_number - b.order_number).map(q => (
                <div key={q.id} className="bg-white border rounded-xl p-4 shadow-sm">
                  <label className="font-medium text-gray-800 flex items-start gap-1">
                    <span className="text-gray-400 mr-1">{q.order_number}.</span>
                    {q.question}
                    {q.is_required && <span className="text-red-500 ml-1">*</span>}
                  </label>

                  {q.question_type === 'text' && (
                    <input
                      type="text"
                      value={answers[q.id] || ''}
                      onChange={e => handleAnswerChange(q.id, e.target.value)}
                      placeholder="Your answer"
                      className="w-full border rounded-lg p-2 mt-2"
                    />
                  )}
                  {q.question_type === 'essay' && (
                    <textarea
                      value={answers[q.id] || ''}
                      onChange={e => handleAnswerChange(q.id, e.target.value)}
                      placeholder="Write your response..."
                      rows="4"
                      className="w-full border rounded-lg p-2 mt-2 resize-y"
                    />
                  )}
                  {q.question_type === 'radio' && q.options && (
                    <div className="mt-2 space-y-2">
                      {q.options.map(opt => (
                        <label key={opt} className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="radio"
                            name={`q_${q.id}`}
                            value={opt}
                            checked={answers[q.id] === opt}
                            onChange={() => handleAnswerChange(q.id, opt)}
                            className="w-4 h-4 text-indigo-600"
                          />
                          <span>{opt}</span>
                        </label>
                      ))}
                    </div>
                  )}
                  {q.question_type === 'checkbox' && q.options && (
                    <div className="mt-2 space-y-2">
                      {q.options.map(opt => (
                        <label key={opt} className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            value={opt}
                            checked={answers[q.id]?.split(',').includes(opt) || false}
                            onChange={e => handleCheckboxChange(q.id, opt, e.target.checked)}
                            className="w-4 h-4 text-indigo-600"
                          />
                          <span>{opt}</span>
                        </label>
                      ))}
                    </div>
                  )}
                  {q.question_type === 'rating' && (
                    <div className="mt-2 flex gap-1">
                      {[1, 2, 3, 4, 5].map(star => (
                        <button
                          key={star}
                          type="button"
                          onClick={() => handleAnswerChange(q.id, star.toString())}
                          className={`text-2xl ${
                            parseInt(answers[q.id] || 0) >= star ? 'text-yellow-500' : 'text-gray-300'
                          }`}
                        >
                          ★
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              <button
                onClick={handleSubmit}
                disabled={submitting}
                className="w-full bg-indigo-600 text-white py-3 rounded-xl font-semibold shadow-md hover:bg-indigo-700 disabled:opacity-50 transition"
              >
                {submitting ? 'Submitting...' : 'Submit Answers'}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
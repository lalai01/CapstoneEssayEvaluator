import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { listSurveys, listQuestions, submitSurveyResponse, apiClient } from '../api';
import toast from 'react-hot-toast';
import SurveyQuestion, { RatingTable } from './SurveyQuestion';
import { displayRuns, isAnswered, isDisplayOnly, kindOf, parseMatrix, tabsOf } from '../lib/surveyKit';
import { readDraft, writeDraft } from '../lib/draft';

// Answers in progress are kept per survey, so a long questionnaire survives a refresh.
const draftKey = (surveyId) => `survey-answers:${surveyId}`;

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
  const [missingId, setMissingId] = useState(null);         // first unanswered required question
  const [tab, setTab] = useState(0);                         // current tab (section) while answering

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
    setAnswers(readDraft(draftKey(survey.id), {}));
    setMissingId(null);
    setTab(0);
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
    setAnswers(prev => {
      const next = { ...prev, [questionId]: value };
      if (selectedSurvey) writeDraft(draftKey(selectedSurvey.id), next);
      return next;
    });
    if (missingId === questionId) setMissingId(null);
  };

  const answerable = questions.filter((q) => !isDisplayOnly(q));
  const requiredCount = answerable.filter((q) => q.is_required).length;
  const answeredCount = answerable.filter((q) => q.is_required && isAnswered(q, answers[q.id])).length;
  const tabs = tabsOf(questions);
  const current = Math.min(tab, Math.max(0, tabs.length - 1));
  const missingIn = (t) => t.items.find(({ q }) => !isDisplayOnly(q) && q.is_required && !isAnswered(q, answers[q.id]));

  const goToTab = (i) => {
    setTab(i);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Show the first unanswered required question: switch to its tab, scroll to it.
  const showMissing = (item) => {
    const at = tabs.findIndex((t) => t.items.includes(item));
    setTab(at);
    setMissingId(item.q.id);
    setTimeout(() => document.getElementById(`q-${item.q.id}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 50);
    toast.error(kindOf(item.q) === 'matrix'
      ? 'Please rate every AI model for the highlighted statement.'
      : `Please answer: ${item.q.question}`);
  };

  const handleSubmit = async () => {
    if (!selectedSurvey) return;
    const missing = tabs.map(missingIn).find(Boolean);
    if (missing) {
      showMissing(missing);
      return;
    }
    setSubmitting(true);
    try {
      // Headings aren't questions; only real answers are sent.
      const toSend = Object.fromEntries(answerable
        .filter((q) => isAnswered(q, answers[q.id]))
        .map((q) => [q.id, String(answers[q.id])]));
      await submitSurveyResponse(selectedSurvey.id, toSend);
      writeDraft(draftKey(selectedSurvey.id), null);
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
                    {kindOf(item) === 'matrix' ? (
                      <ul className="mt-1 text-sm text-gray-700">
                        {Object.entries(parseMatrix(item.answer)).map(([m, v]) => <li key={m}>{m}: <strong>{v}</strong> / 5</li>)}
                      </ul>
                    ) : item.question_type === 'rating' && kindOf(item) === 'likert' ? (
                      <p className="text-gray-700 mt-1"><strong>{item.answer}</strong> / 5</p>
                    ) : item.question_type === 'rating' ? (
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
              {requiredCount > 0 && (
                <div className="sticky top-16 z-10 bg-white/95 backdrop-blur rounded-xl border p-3 shadow-sm">
                  <div className="flex justify-between text-xs text-gray-600 mb-1">
                    <span>{answeredCount} of {requiredCount} required answered</span>
                    <span>Saved as a draft on this device</span>
                  </div>
                  <div className="h-2 rounded-full bg-gray-200 overflow-hidden">
                    <div className="h-full bg-indigo-600 transition-all" style={{ width: `${(answeredCount / requiredCount) * 100}%` }} />
                  </div>
                </div>
              )}
              {tabs.length > 1 && (
                <div role="tablist" aria-label="Survey sections" className="flex gap-1 overflow-x-auto pb-1">
                  {tabs.map((t, i) => {
                    const done = !missingIn(t);
                    return (
                      <button
                        key={t.section?.id ?? 'intro'}
                        type="button"
                        role="tab"
                        aria-selected={i === current}
                        onClick={() => goToTab(i)}
                        className={`shrink-0 rounded-lg px-3 py-2 text-sm border transition ${
                          i === current ? 'bg-indigo-600 border-indigo-600 text-white' : 'bg-white border-gray-200 text-gray-700 hover:border-indigo-300'
                        }`}
                      >
                        <span className={`mr-1 ${done ? '' : 'opacity-40'}`}>{done ? '✓' : `${i + 1}.`}</span>
                        {(t.section?.question || 'Start').replace(/^Part\s+[IVX]+\.\s*/i, (m) => m.trim() + ' ')}
                      </button>
                    );
                  })}
                </div>
              )}

              {tabs[current] && (
                <>
                  {tabs[current].section && (
                    <SurveyQuestion item={{ q: tabs[current].section, kind: 'section' }} value={undefined} onChange={() => {}} />
                  )}
                  {displayRuns(tabs[current].items).map((run) => (run.table ? (
                    <RatingTable
                      key={run.items[0].q.id}
                      items={run.items}
                      answers={answers}
                      onChange={handleAnswerChange}
                      missingId={missingId}
                    />
                  ) : (
                    <SurveyQuestion
                      key={run.items[0].q.id}
                      item={run.items[0]}
                      value={answers[run.items[0].q.id]}
                      onChange={(v) => handleAnswerChange(run.items[0].q.id, v)}
                      invalid={missingId === run.items[0].q.id}
                    />
                  )))}
                </>
              )}

              <div className="flex gap-3">
                {current > 0 && (
                  <button type="button" onClick={() => goToTab(current - 1)}
                    className="flex-1 bg-white border border-gray-300 text-gray-700 py-3 rounded-xl font-semibold hover:bg-gray-50 transition">
                    ← Back
                  </button>
                )}
                {current < tabs.length - 1 ? (
                  <button
                    type="button"
                    onClick={() => {
                      const missing = missingIn(tabs[current]);
                      if (missing) showMissing(missing);
                      else goToTab(current + 1);
                    }}
                    className="flex-1 bg-indigo-600 text-white py-3 rounded-xl font-semibold shadow-md hover:bg-indigo-700 transition"
                  >
                    Next →
                  </button>
                ) : (
                  <button
                    onClick={handleSubmit}
                    disabled={submitting}
                    className="flex-1 bg-indigo-600 text-white py-3 rounded-xl font-semibold shadow-md hover:bg-indigo-700 disabled:opacity-50 transition"
                  >
                    {submitting ? 'Submitting...' : 'Submit Answers'}
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
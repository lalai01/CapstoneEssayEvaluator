import axios from 'axios';
import { supabase } from './lib/supabase';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const apiClient = axios.create({ baseURL: API_BASE });   // ✅ now exported

apiClient.interceptors.request.use(
  async (config) => {
    try {
      const { data: { session } } = await supabase.auth.getSession();

      if (session?.access_token) {
        config.headers.Authorization = `Bearer ${session.access_token}`;
        console.log('✅ Token attached to:', config.url);
      } else {
        console.warn('⚠️ No session for:', config.url);
      }

      return config;
    } catch (err) {
      console.error('❌ Interceptor error:', err);
      return config;
    }
  },
  (error) => Promise.reject(error)
);

// ---------- OCR ----------
export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await apiClient.post('/ocr', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  });
  return response.data;
};

export const pollOcrStatus = async (jobId) => {
  const response = await apiClient.get(`/ocr/status/${jobId}`);
  return response.data;
};

// ---------- Saved Essays ----------
export const saveEssay = async (title, essay) => {
  const response = await apiClient.post('/saved-essays', { title, essay });
  return response.data;
};

export const listSavedEssays = async () => {
  const response = await apiClient.get('/saved-essays');
  return response.data;
};

export const deleteSavedEssay = async (id) => {
  const response = await apiClient.delete(`/saved-essays/${id}`);
  return response.data;
};

// ---------- Evaluation ----------
export const evaluateEssay = async (text, evaluationType = 'analytic') => {
  const response = await apiClient.post('/evaluate', { text, evaluation_type: evaluationType });
  return response.data;
};

export const evaluateEssayWithRag = async (text, evaluationType = 'analytic') => {
  const response = await apiClient.post('/evaluate-rag', { text, evaluation_type: evaluationType });
  return response.data;
};

// ---------- Rubric & Suggestions ----------
export const getRubric = async () => {
  const response = await apiClient.get('/rubric');
  return response.data;
};

export const getSuggestions = async () => {
  const response = await apiClient.get('/suggestions');
  return response.data;
};

// ---------- Knowledge Base ----------
export const saveKnowledge = async (entry) => {
  const response = await apiClient.post('/knowledge', entry);
  return response.data;
};

export const listKnowledge = async () => {
  const response = await apiClient.get('/knowledge');
  return response.data;
};

export const getKnowledgeItem = async (id) => {
  const response = await apiClient.get(`/knowledge/${id}`);
  return response.data;
};

// ---------- Teacher Override ----------
export const saveOverride = async (overrideData) => {
  const response = await apiClient.post('/override', overrideData);
  return response.data;
};

export const listLearningFeedback = async () => {
  const response = await apiClient.get('/learning-kb');
  return response.data;
};

// ---------- AI Prompt Testing ----------
export const testPrompt = async (aiProvider, systemPrompt, userPrompt, model = null) => {
  const response = await apiClient.post('/test-prompt', {
    ai_provider: aiProvider,
    system_prompt: systemPrompt,
    user_prompt: userPrompt,
    model: model,
  });
  return response.data;
};

// ---------- Ratings ----------
export const submitRating = async (rating, comment) => {
  const response = await apiClient.post('/ratings', { rating, comment });
  return response.data;
};

export const getRatings = async () => {
  const response = await apiClient.get('/ratings');
  return response.data;
};

export const getRatingSummary = async () => {
  const response = await apiClient.get('/ratings/summary');
  return response.data;
};

// ---------- Surveys ----------
export const createSurvey = async (survey) => {
  const response = await apiClient.post('/surveys', survey);
  return response.data;
};

export const listSurveys = async (activeOnly = true) => {
  const response = await apiClient.get(`/surveys?active_only=${activeOnly}`);
  return response.data;
};

export const updateSurvey = async (id, survey) => {
  const response = await apiClient.put(`/surveys/${id}`, survey);
  return response.data;
};

export const deleteSurvey = async (id) => {
  const response = await apiClient.delete(`/surveys/${id}`);
  return response.data;
};

// 🔥 Proper payload for survey responses
export const submitSurveyResponse = async (surveyId, answers) => {
  const response = await apiClient.post(`/surveys/${surveyId}/respond`, {
    survey_id: surveyId,
    answers: answers,      // e.g., { "1": "text", "2": "choice" }
  });
  return response.data;
};

// ---------- Survey Questions ----------
export const addQuestion = async (surveyId, question) => {
  const response = await apiClient.post(`/surveys/${surveyId}/questions`, question);
  return response.data;
};

export const listQuestions = async (surveyId) => {
  const response = await apiClient.get(`/surveys/${surveyId}/questions`);
  return response.data;
};

export const updateQuestion = async (id, question) => {
  const response = await apiClient.put(`/questions/${id}`, question);
  return response.data;
};

export const deleteQuestion = async (id) => {
  const response = await apiClient.delete(`/questions/${id}`);
  return response.data;
};

// ---------- Survey Responses (Admin) ----------
export const getSurveyResponses = async (surveyId) => {
  const response = await apiClient.get(`/surveys/${surveyId}/responses`);
  return response.data;
};

// ---------- Comments ----------
export const createComment = async (ratingId, parentId, body) => {
  const response = await apiClient.post('/comments', { rating_id: ratingId, parent_id: parentId, body });
  return response.data;
};

export const listComments = async (ratingId) => {
  const response = await apiClient.get(`/comments/${ratingId}`);
  return response.data;
};

// ---------- Reactions ----------
export const toggleReaction = async (commentId, reactionType) => {
  const response = await apiClient.post('/reactions', { comment_id: commentId, reaction_type: reactionType });
  return response.data;
};

// ---------- User's Own Survey Responses (NEW) ----------
export const getMyResponses = async (surveyId) => {
  const response = await apiClient.get(`/surveys/${surveyId}/my-responses`);
  return response.data;
};
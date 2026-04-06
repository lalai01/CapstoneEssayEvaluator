import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL;

export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await axios.post(`${API_BASE}/ocr`, formData);
  return response.data;
};

export const evaluateEssay = async (text, evaluationType = 'analytic') => {
  const response = await axios.post(`${API_BASE}/evaluate`, {
    text,
    evaluation_type: evaluationType,
  });
  return response.data;
};

export const getRubric = async () => {
  const response = await axios.get(`${API_BASE}/rubric`);
  return response.data;
};

export const getSuggestions = async () => {
  const response = await axios.get(`${API_BASE}/suggestions`);
  return response.data;
};

export const saveKnowledge = async (entry) => {
  const response = await axios.post(`${API_BASE}/knowledge`, entry);
  return response.data;
};

export const listKnowledge = async () => {
  const response = await axios.get(`${API_BASE}/knowledge`);
  return response.data;
};
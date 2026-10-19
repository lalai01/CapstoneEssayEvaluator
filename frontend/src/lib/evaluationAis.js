// The three AIs that evaluate every essay, side by side. `id` is the
// ai_model value the backend expects. `local` AIs run on our server's CPU
// (one at a time); cloud AIs run at the same time as everything else.
// `bar` and `chip` colour the AI consistently in statistics and charts.
export const EVALUATION_AIS = [
  { id: 'deepseek', name: 'DeepSeek', where: 'Cloud', icon: '🌐', bar: 'bg-blue-500', chip: 'bg-blue-50 text-blue-800 border-blue-200' },
  { id: 'llama', name: 'Llama 3.2', where: 'Our server', local: true, icon: '🦙', bar: 'bg-violet-500', chip: 'bg-violet-50 text-violet-800 border-violet-200' },
  // Groq (GPT-OSS 120B) replaced the slower local Gemma 2 / Qwen 2.5; when
  // Groq is unavailable the server uses Gemini and the card says so.
  { id: 'groq', name: 'Groq', where: 'Cloud · GPT-OSS 120B', icon: '🚀', bar: 'bg-emerald-500', chip: 'bg-emerald-50 text-emerald-800 border-emerald-200' },
];

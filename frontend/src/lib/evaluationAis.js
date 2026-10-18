// The three AIs that evaluate every essay, side by side. `id` is the
// ai_model value the backend expects.
// `bar` and `chip` colour the AI consistently in statistics and charts.
export const EVALUATION_AIS = [
  { id: 'deepseek', name: 'DeepSeek', where: 'Cloud', icon: '🌐', bar: 'bg-blue-500', chip: 'bg-blue-50 text-blue-800 border-blue-200' },
  { id: 'llama', name: 'Llama 3.2', where: 'Our server', icon: '🦙', bar: 'bg-violet-500', chip: 'bg-violet-50 text-violet-800 border-violet-200' },
  { id: 'gemma', name: 'Gemma 2', where: 'Our server', icon: '💎', bar: 'bg-emerald-500', chip: 'bg-emerald-50 text-emerald-800 border-emerald-200' },
];

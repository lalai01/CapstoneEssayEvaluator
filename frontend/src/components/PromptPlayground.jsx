import React, { useState, useEffect } from 'react';
import {
  testPrompt,
  saveEssay,
  listSavedEssays,
  deleteSavedEssay,
  listKnowledge,
} from '../api';
import { useAuth } from '../context/AuthContext';
import toast from 'react-hot-toast';

function EvaluationDisplay({ result }) {
  try {
    let text = result.text || '';
    text = text.replace(/^```json\s*/i, '').replace(/```$/, '').trim();
    const parsed = JSON.parse(text);
    if (parsed.grammar_score !== undefined || parsed.coherence_score !== undefined || parsed.content_score !== undefined) {
      const avgScore = Math.round((parsed.grammar_score + parsed.coherence_score + parsed.content_score) / 3);
      const getScoreColor = (score) => {
        if (score >= 85) return 'text-emerald-600 bg-emerald-50';
        if (score >= 70) return 'text-blue-600 bg-blue-50';
        return 'text-amber-600 bg-amber-50';
      };
      const ScoreBar = ({ score, label, colorClass }) => (
        <div className="mb-2">
          <div className="flex justify-between text-xs mb-1">
            <span className="font-medium text-gray-700">{label}</span>
            <span className={`font-bold ${colorClass.split(' ')[0]}`}>{score}</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-1.5">
            <div
              className={`h-1.5 rounded-full ${colorClass.split(' ')[0].replace('text', 'bg')}`}
              style={{ width: `${score}%` }}
            />
          </div>
        </div>
      );
      return (
        <div className="space-y-4">
          <div className="text-center p-3 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg">
            <div className="text-sm text-gray-700">Overall Score</div>
            <div className={`text-3xl font-bold ${getScoreColor(avgScore).split(' ')[0]}`}>
              {avgScore}/100
            </div>
          </div>
          <div className="space-y-1">
            <ScoreBar score={parsed.grammar_score} label="Grammar" colorClass={getScoreColor(parsed.grammar_score)} />
            <ScoreBar score={parsed.coherence_score} label="Coherence" colorClass={getScoreColor(parsed.coherence_score)} />
            <ScoreBar score={parsed.content_score} label="Content" colorClass={getScoreColor(parsed.content_score)} />
          </div>
          {parsed.feedback && (
            <div className="mt-4">
              <h4 className="font-semibold text-gray-800 mb-2 text-sm uppercase tracking-wide">
                📝 Detailed Feedback
              </h4>
              <div className="bg-white p-4 rounded-lg border border-gray-200 text-gray-800 text-sm leading-relaxed whitespace-pre-wrap">
                {parsed.feedback}
              </div>
            </div>
          )}
        </div>
      );
    }
    return <pre className="whitespace-pre-wrap text-sm font-mono text-gray-800">{text}</pre>;
  } catch (e) {
    return <pre className="whitespace-pre-wrap text-sm font-mono text-gray-800">{result.text}</pre>;
  }
}

// -----------------------------------------------------------------------------
// Model Information Cards (Collapsible)
// -----------------------------------------------------------------------------
const MODEL_INFO = [
  {
    provider: 'deepseek',
    name: 'DeepSeek',
    icon: '🌐',
    color: 'border-blue-300 bg-blue-50',
    advantages: [
      'Free tier with 10M tokens',
      'Strong reasoning and math',
      'Fast API responses (~2-5s)',
    ],
    bestFor: 'Complex analytical essays, quick iterations, cost‑sensitive projects.',
  },
  {
    provider: 'llamacpp',
    name: 'Llama.cpp (Phi-3.5)',
    icon: '🦙',
    color: 'border-purple-300 bg-purple-50',
    advantages: [
      '100% local – no API keys',
      'Phi-3.5 Mini (3.8B)',
      'Good balance of quality/speed',
    ],
    bestFor: 'Privacy‑sensitive evaluations, offline use, medium‑complexity essays.',
  },
  {
    provider: 'gemma',
    name: 'Ollama (Gemma 2B)',
    icon: '🟢',
    color: 'border-emerald-300 bg-emerald-50',
    advantages: [
      'Lightweight (1.6GB)',
      'Fast on CPU (~10-20s)',
      'Easy to set up',
    ],
    bestFor: 'Quick drafts, low‑resource VPS, simple feedback.',
  },
];

// -----------------------------------------------------------------------------
// Main Component
// -----------------------------------------------------------------------------
export default function PromptPlayground() {
  const { user } = useAuth();

  // UI state
  const [showModelInfo, setShowModelInfo] = useState(true);

  // Essay state
  const [essayText, setEssayText] = useState('');
  const [savedEssays, setSavedEssays] = useState([]);
  const [selectedEssayId, setSelectedEssayId] = useState('');
  const [selectedEssaySource, setSelectedEssaySource] = useState('');

  // Model state
  const [primaryProvider, setPrimaryProvider] = useState('deepseek');
  const [primaryModel, setPrimaryModel] = useState('');
  const [systemPrompt, setSystemPrompt] = useState(
    'You are an expert essay evaluator with years of experience grading academic writing. Your task is to analyze the provided essay and return a JSON object with exactly the following keys: grammar_score (0-100), coherence_score (0-100), content_score (0-100), and feedback (detailed paragraph). Be objective and consistent. Do not include any text outside the JSON object.'
  );
  const [userPromptTemplate, setUserPromptTemplate] = useState(
    `📄 **Essay to Evaluate**
{essay}

---

🎯 **Your Task**
You are an expert essay evaluator. Analyze the essay above and provide a detailed evaluation.

📊 **Required Output Format**
Return a valid JSON object with exactly these keys:

- \`grammar_score\` : integer (0–100) – Grammatical accuracy, sentence structure, punctuation.
- \`coherence_score\` : integer (0–100) – Logical flow, transitions, organization.
- \`content_score\` : integer (0–100) – Depth of analysis, evidence, argument strength.
- \`feedback\` : string – A constructive paragraph with strengths and actionable improvements.

⚠️ Do not include any text outside the JSON object.`
  );

  // Results
  const [primaryResult, setPrimaryResult] = useState(null);
  const [secondaryResult, setSecondaryResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingSecondary, setLoadingSecondary] = useState(false);

  // Comparison
  const [compareProvider, setCompareProvider] = useState('deepseek');
  const [compareModel, setCompareModel] = useState('');
  const [showCompareSelect, setShowCompareSelect] = useState(false);

  // -------------------------------------------------------------------------
  // Load saved essays from BOTH sources
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (user) {
      loadAllEssays();
    } else {
      setSavedEssays([]);
    }
  }, [user]);

  const loadAllEssays = async () => {
    try {
      const [saved, knowledge] = await Promise.all([
        listSavedEssays().catch(() => []),
        listKnowledge().catch(() => [])
      ]);

      const combined = [
        ...saved.map(e => ({
          ...e,
          source: 'saved',
          displayTitle: e.title || `Draft ${e.id}`,
        })),
        ...knowledge.map(e => ({
          ...e,
          source: 'kb',
          displayTitle: e.title || `Evaluation #${e.id}`,
        }))
      ];

      combined.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      setSavedEssays(combined);
    } catch (err) {
      toast.error('Failed to load essays');
    }
  };

  const autoSaveEssay = async () => {
    if (!user || !essayText.trim()) return;
    const now = new Date();
    const autoTitle = `Essay ${now.toLocaleDateString()} ${now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    try {
      await saveEssay(autoTitle, essayText);
      await loadAllEssays();
    } catch (err) {
      console.warn('Auto‑save failed:', err);
    }
  };

  const handleLoadEssay = (e) => {
    const value = e.target.value;
    if (!value) {
      setSelectedEssayId('');
      setSelectedEssaySource('');
      return;
    }
    const [source, idStr] = value.split(':');
    const id = parseInt(idStr);
    const selected = savedEssays.find(e => e.source === source && e.id === id);
    if (selected) {
      setEssayText(selected.essay);
      setSelectedEssayId(id);
      setSelectedEssaySource(source);
      toast.success(`Loaded: ${selected.displayTitle}`);
    }
  };

  const handleDeleteEssay = async () => {
    if (!selectedEssayId || selectedEssaySource !== 'saved') {
      toast.error('Only draft essays can be deleted here.');
      return;
    }
    if (!confirm('Delete this saved essay?')) return;
    try {
      await deleteSavedEssay(selectedEssayId);
      toast.success('Essay deleted');
      setSelectedEssayId('');
      setSelectedEssaySource('');
      setEssayText('');
      loadAllEssays();
    } catch (err) {
      toast.error('Failed to delete');
    }
  };

  const handlePrimaryTest = async () => {
    if (!essayText.trim()) {
      toast.error('Please enter essay text.');
      return;
    }
    if (user) {
      await autoSaveEssay();
    }

    const finalUserPrompt = userPromptTemplate.replace('{essay}', essayText);
    setLoading(true);
    setPrimaryResult(null);
    setSecondaryResult(null);
    setShowCompareSelect(false);
    try {
      const response = await testPrompt(
        primaryProvider,
        systemPrompt,
        finalUserPrompt,
        primaryModel || null
      );
      setPrimaryResult(response.result);
      toast.success(`Primary (${getProviderDisplayName(primaryProvider)}) response received`);
      setShowCompareSelect(true);
    } catch (err) {
      toast.error('Primary test failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCompareTest = async () => {
    if (!essayText.trim()) {
      toast.error('No essay text available.');
      return;
    }
    const finalUserPrompt = userPromptTemplate.replace('{essay}', essayText);
    setLoadingSecondary(true);
    try {
      const response = await testPrompt(
        compareProvider,
        systemPrompt,
        finalUserPrompt,
        compareModel || null
      );
      setSecondaryResult(response.result);
      toast.success(`Comparison (${getProviderDisplayName(compareProvider)}) response received`);
    } catch (err) {
      toast.error('Comparison test failed: ' + err.message);
    } finally {
      setLoadingSecondary(false);
    }
  };

  const getProviderDisplayName = (provider) => {
    const names = {
      deepseek: 'DeepSeek',
      gemma: 'Gemma (Ollama)',
      ollama: 'Ollama',
      llamacpp: 'Llama.cpp (Phi-3.5)',
    };
    return names[provider] || provider;
  };

  return (
    <div className="glass-card rounded-2xl shadow-xl p-6 animate-fade-in">
      <h2 className="text-2xl font-bold text-gray-800 mb-2">🧪 AI Prompt Playground</h2>
      <p className="text-gray-600 mb-4">
        Test and compare different AI models with custom prompts. Essays are saved automatically.
      </p>

      {/* Collapsible Model Information Panel */}
      <div className="mb-5 border border-gray-200 rounded-xl overflow-hidden">
        <button
          onClick={() => setShowModelInfo(!showModelInfo)}
          className="w-full flex items-center justify-between p-3 bg-gray-50 hover:bg-gray-100 transition"
        >
          <span className="font-semibold text-gray-700">
            📘 Model Comparison & Recommendations
          </span>
          <span className="text-gray-500 text-sm">
            {showModelInfo ? '▲ Hide' : '▼ Show'}
          </span>
        </button>
        {showModelInfo && (
          <div className="p-4 bg-white grid grid-cols-1 md:grid-cols-3 gap-4">
            {MODEL_INFO.map((model) => (
              <div
                key={model.provider}
                className={`border rounded-xl p-4 ${model.color}`}
              >
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">{model.icon}</span>
                  <h3 className="font-bold text-gray-800">{model.name}</h3>
                </div>
                <ul className="text-sm text-gray-700 space-y-1 mb-3">
                  {model.advantages.map((adv, idx) => (
                    <li key={idx} className="flex items-start gap-1">
                      <span className="text-green-600">✓</span>
                      <span>{adv}</span>
                    </li>
                  ))}
                </ul>
                <p className="text-xs text-gray-600 italic border-t pt-2 mt-1">
                  <strong>Best for:</strong> {model.bestFor}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Essay Management */}
      {user ? (
        <div className="mb-6 p-4 bg-blue-50 rounded-xl border border-blue-200">
          <div className="flex flex-wrap items-end gap-3">
            <div className="flex-1 min-w-[200px]">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                📚 Load Saved Essay
              </label>
              <select
                value={selectedEssayId ? `${selectedEssaySource}:${selectedEssayId}` : ''}
                onChange={handleLoadEssay}
                className="w-full border rounded-lg p-2 bg-white text-gray-800"
              >
                <option value="">-- Select a saved essay --</option>
                {savedEssays.map((essay) => (
                  <option key={`${essay.source}-${essay.id}`} value={`${essay.source}:${essay.id}`}>
                    {essay.source === 'kb' ? '📋' : '📄'} {essay.displayTitle} (
                    {new Date(essay.created_at).toLocaleDateString()})
                  </option>
                ))}
              </select>
            </div>
            {selectedEssayId && (
              <button
                onClick={handleDeleteEssay}
                disabled={selectedEssaySource !== 'saved'}
                className={`px-4 py-2 rounded-lg transition ${
                  selectedEssaySource === 'saved'
                    ? 'bg-red-100 text-red-700 hover:bg-red-200'
                    : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                }`}
                title={
                  selectedEssaySource === 'kb'
                    ? 'Knowledge Base entries cannot be deleted from here'
                    : 'Delete draft'
                }
              >
                🗑️ Delete
              </button>
            )}
          </div>
          <p className="text-xs text-gray-500 mt-2">
            ✨ Your essays are automatically saved when you run an evaluation. The dropdown shows both drafts and
            evaluated essays from your Knowledge Base.
          </p>
        </div>
      ) : (
        <div className="mb-6 p-4 bg-amber-50 rounded-xl border border-amber-200">
          <p className="text-amber-800 text-sm">
            🔒 <strong>Log in</strong> to save and load your essays.
          </p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Controls */}
        <div className="space-y-4">
          <div className="bg-blue-50 p-4 rounded-xl border border-blue-200">
            <h3 className="font-semibold text-blue-800 mb-2">🎯 Primary Model</h3>
            <div className="grid grid-cols-2 gap-3">
              <select
                value={primaryProvider}
                onChange={(e) => setPrimaryProvider(e.target.value)}
                className="border rounded-lg p-2 bg-white text-gray-800"
              >
                <option value="deepseek">DeepSeek</option>
                <option value="gemma">Gemma (Ollama)</option>
                <option value="llamacpp">Llama.cpp (Phi-3.5)</option>
              </select>
              <input
                type="text"
                value={primaryModel}
                onChange={(e) => setPrimaryModel(e.target.value)}
                placeholder="Model (optional)"
                className="border rounded-lg p-2 bg-white text-gray-800"
              />
            </div>
          </div>

          <div>
            <label className="block font-semibold text-gray-800 mb-1">🤖 System Prompt</label>
            <textarea
              rows="3"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              className="w-full border rounded-xl p-2 bg-gray-50 text-gray-800"
            />
          </div>

          <div>
            <label className="block font-semibold text-gray-800 mb-1">
              📋 User Prompt Template
              <span className="text-sm font-normal ml-1">(use {'{essay}'})</span>
            </label>
            <textarea
              rows="5"
              value={userPromptTemplate}
              onChange={(e) => setUserPromptTemplate(e.target.value)}
              className="w-full border rounded-xl p-2 bg-gray-50 font-mono text-sm text-gray-800"
            />
          </div>

          <div>
            <label className="block font-semibold text-gray-800 mb-1">📄 Essay to Evaluate</label>
            <textarea
              rows="8"
              value={essayText}
              onChange={(e) => setEssayText(e.target.value)}
              className="w-full border rounded-xl p-2 bg-gray-50 text-gray-800"
              placeholder="Paste your essay here..."
            />
          </div>

          <button
            onClick={handlePrimaryTest}
            disabled={loading}
            className="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white py-3 rounded-xl font-semibold shadow-md hover:shadow-lg transition disabled:opacity-50"
          >
            {loading ? <div className="spinner mx-auto w-5 h-5"></div> : '🚀 Run Primary Evaluation'}
          </button>

          {showCompareSelect && primaryResult && (
            <div className="mt-4 p-4 bg-amber-50 rounded-xl border border-amber-200">
              <h3 className="font-semibold text-amber-800 mb-2">🔍 Get Second Opinion</h3>
              <div className="flex gap-2">
                <select
                  value={compareProvider}
                  onChange={(e) => setCompareProvider(e.target.value)}
                  className="flex-1 border rounded-lg p-2 bg-white text-gray-800"
                >
                  <option value="deepseek">DeepSeek</option>
                  <option value="gemma">Gemma (Ollama)</option>
                  <option value="llamacpp">Llama.cpp (Phi-3.5)</option>
                </select>
                <input
                  type="text"
                  value={compareModel}
                  onChange={(e) => setCompareModel(e.target.value)}
                  placeholder="Model"
                  className="w-32 border rounded-lg p-2 bg-white text-gray-800"
                />
                <button
                  onClick={handleCompareTest}
                  disabled={loadingSecondary}
                  className="bg-amber-600 text-white px-4 py-2 rounded-lg hover:bg-amber-700 transition disabled:opacity-50"
                >
                  {loadingSecondary ? '...' : 'Compare'}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Right: Results */}
        <div className="space-y-4">
          <div>
            <h3 className="font-semibold text-lg mb-2 flex items-center gap-2">
              <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm">
                Primary: {getProviderDisplayName(primaryProvider)}
              </span>
            </h3>
            <div className="bg-gray-50 rounded-xl border p-4 min-h-[200px] max-h-[400px] overflow-auto">
              {primaryResult ? (
                <EvaluationDisplay result={primaryResult} />
              ) : (
                <p className="text-gray-500 text-center mt-16">Run primary evaluation to see results.</p>
              )}
            </div>
          </div>

          {secondaryResult && (
            <div>
              <h3 className="font-semibold text-lg mb-2 flex items-center gap-2">
                <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-sm">
                  Comparison: {getProviderDisplayName(compareProvider)}
                </span>
              </h3>
              <div className="bg-gray-50 rounded-xl border p-4 min-h-[200px] max-h-[400px] overflow-auto">
                <EvaluationDisplay result={secondaryResult} />
              </div>
            </div>
          )}

          {primaryResult && secondaryResult && (() => {
            try {
              const parseResult = (res) => {
                let text = res.text || '';
                text = text.replace(/^```json\s*/i, '').replace(/```$/, '').trim();
                return JSON.parse(text);
              };
              const primary = parseResult(primaryResult);
              const secondary = parseResult(secondaryResult);
              if (primary.grammar_score && secondary.grammar_score) {
                const diffColor = (val) =>
                  val > 0 ? 'text-green-600' : val < 0 ? 'text-red-500' : 'text-gray-600';
                return (
                  <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
                    <h4 className="font-semibold text-gray-800 mb-3 flex items-center gap-2">
                      <span>📊 Score Comparison</span>
                      <span className="text-xs font-normal text-gray-600">
                        ({getProviderDisplayName(primaryProvider)} vs {getProviderDisplayName(compareProvider)})
                      </span>
                    </h4>
                    <div className="space-y-2">
                      <div className="grid grid-cols-4 gap-2 text-sm">
                        <div className="font-medium text-gray-700">Category</div>
                        <div className="font-medium text-gray-700 text-center">Primary</div>
                        <div className="font-medium text-gray-700 text-center">Compare</div>
                        <div className="font-medium text-gray-700 text-center">Diff</div>
                      </div>
                      {['grammar', 'coherence', 'content'].map((cat) => {
                        const pScore = primary[`${cat}_score`];
                        const sScore = secondary[`${cat}_score`];
                        const difference = pScore - sScore;
                        return (
                          <div key={cat} className="grid grid-cols-4 gap-2 text-sm border-t border-gray-100 pt-2">
                            <div className="text-gray-800 capitalize">{cat}</div>
                            <div className="text-center font-medium text-gray-800">{pScore}</div>
                            <div className="text-center font-medium text-gray-800">{sScore}</div>
                            <div className={`text-center font-medium ${diffColor(difference)}`}>
                              {difference > 0 ? `+${difference}` : difference}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              }
            } catch (e) {
              return null;
            }
            return null;
          })()}

          {/* AI Reference Guide Panel */}
          <div className="mt-6">
            <details className="cursor-pointer group">
              <summary className="font-semibold text-gray-700 flex items-center gap-2">
                <span>🤖 How to Use AI as a Reference (Not a Replacement)</span>
                <span className="text-xs text-gray-400 group-open:hidden">click to expand</span>
              </summary>
              <div className="mt-3 p-4 bg-amber-50 rounded-lg border border-amber-200 text-sm text-gray-700 space-y-2">
                <p><strong>🧠 AI is a tool to assist your judgment, not replace it.</strong></p>
                <ul className="list-disc pl-5 space-y-1">
                  <li>Use AI-generated scores and feedback as a <strong>second opinion</strong> to identify potential weaknesses.</li>
                  <li>Always review the suggestions critically; your expertise as an educator is paramount.</li>
                  <li>Compare results from different models to see variations in emphasis (some may focus more on grammar, others on coherence).</li>
                  <li>When overriding, your feedback becomes part of the learning system, helping future evaluations improve.</li>
                </ul>
                <p className="italic mt-2">💡 Best practice: Use AI feedback to start a conversation with students about their writing, not to give a final grade.</p>
              </div>
            </details>
          </div>
        </div>
      </div>
    </div>
  );
}
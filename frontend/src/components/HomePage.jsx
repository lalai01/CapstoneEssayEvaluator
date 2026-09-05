import React, { useEffect } from 'react';
import RobotAssistant from './RobotAssistant';

export default function HomePage({ setActiveTab }) {
  useEffect(() => {
    const sections = document.querySelectorAll('.scroll-section');
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
          } else {
            // Remove class to allow fade-in again when scrolling back up
            entry.target.classList.remove('visible');
          }
        });
      },
      { threshold: 0.2 }
    );
    sections.forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, []);

  return (
    <div className="space-y-12 animate-fade-in relative overflow-hidden">
      {/* Animated background shapes */}
      <div className="absolute inset-0 -z-10 overflow-hidden">
        <div className="floating-shape shape-1"></div>
        <div className="floating-shape shape-2"></div>
        <div className="floating-shape shape-3"></div>
        <div className="floating-shape shape-4"></div>
      </div>

      {/* Hero Section – with Robot and professional border + color fonts */}
      <div className="glass-card rounded-3xl shadow-2xl p-8 md:p-12 relative z-10 bg-white/40 backdrop-blur-md border-gradient">
        <div className="flex flex-col lg:flex-row items-center gap-8">
          {/* Left: Text Content with professional border container */}
          <div className="flex-1 text-center lg:text-left">
            <div className="hero-text-container p-6 rounded-2xl border border-gray-200/60 bg-white/30 shadow-inner">
              <h1 className="text-4xl md:text-6xl font-bold gradient-text mb-4">
                AI Essay Evaluator
              </h1>
              <p className="subtitle-text text-xl text-gray-800 max-w-3xl font-medium">
                Get instant, professional feedback on your essays.
              </p>
              <p className="paragraph-text text-gray-700 mt-3 leading-relaxed">
                Upload handwritten notes, PDFs, or type directly – our AI helps you improve writing skills with dynamic analysis and teacher‑in‑the‑loop learning.
              </p>
            </div>
            <div className="mt-8 flex justify-center lg:justify-start gap-4">
              <button
                onClick={() => setActiveTab('evaluate')}
                className="bg-blue-600 text-white px-6 py-3 rounded-xl font-semibold shadow-lg hover:shadow-xl transition"
              >
                Start Evaluating →
              </button>
              <a href="#how-it-works" className="bg-gray-200 text-gray-800 px-6 py-3 rounded-xl font-semibold hover:bg-gray-300 transition">
                Learn More
              </a>
            </div>
          </div>

          {/* Right: Robot Assistant – bigger and slightly lower */}
          <div className="flex-1 flex justify-center mt-8 lg:mt-12">
            <div className="w-80 h-80 md:w-96 md:h-96">
              <RobotAssistant />
            </div>
          </div>
        </div>
      </div>

      {/* How It Works – 3‑column grid (unchanged, but keep fade-in) */}
      <div id="how-it-works" className="scroll-section glass-card rounded-3xl shadow-xl p-8 relative z-10 bg-white/40 backdrop-blur-md">
        <h2 className="text-3xl font-bold text-center mb-8 text-gray-800">How It Works</h2>
        <div className="grid md:grid-cols-3 gap-8">
          <div className="text-center p-4">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4 text-2xl">1️⃣</div>
            <h3 className="text-xl font-semibold mb-2 text-gray-800">Upload or Type</h3>
            <p className="text-gray-700">Paste your essay, upload an image, or a PDF. Our OCR automatically extracts text.</p>
          </div>
          <div className="text-center p-4">
            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4 text-2xl">2️⃣</div>
            <h3 className="text-xl font-semibold mb-2 text-gray-800">AI Evaluation</h3>
            <p className="text-gray-700">Get scores for grammar, coherence, and content, plus detailed, actionable feedback.</p>
          </div>
          <div className="text-center p-4">
            <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4 text-2xl">3️⃣</div>
            <h3 className="text-xl font-semibold mb-2 text-gray-800">Learn & Improve</h3>
            <p className="text-gray-700">Teachers can override results – the system learns from corrections to give better feedback over time.</p>
          </div>
        </div>
      </div>

      {/* Key Features – 4‑column grid */}
      <div className="scroll-section glass-card rounded-3xl shadow-xl p-8 relative z-10 bg-white/40 backdrop-blur-md">
        <h2 className="text-3xl font-bold text-center mb-8 text-gray-800">Key Features</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-white/50 p-5 rounded-xl hover:shadow-lg transition">
            <div className="text-3xl mb-3">📄</div>
            <h3 className="font-semibold text-gray-800">OCR & PDF Support</h3>
            <p className="text-sm text-gray-700 mt-1">Extract text from images and PDFs using Google Vision or Tesseract.</p>
          </div>
          <div className="bg-white/50 p-5 rounded-xl hover:shadow-lg transition">
            <div className="text-3xl mb-3">🧠</div>
            <h3 className="font-semibold text-gray-800">RAG (Retrieval Augmented)</h3>
            <p className="text-sm text-gray-700 mt-1">Uses past evaluations to enrich feedback – gets smarter over time.</p>
          </div>
          <div className="bg-white/50 p-5 rounded-xl hover:shadow-lg transition">
            <div className="text-3xl mb-3">✏️</div>
            <h3 className="font-semibold text-gray-800">Teacher Override</h3>
            <p className="text-sm text-gray-700 mt-1">Reject AI results and add corrections; stored in a learning knowledge base.</p>
          </div>
          <div className="bg-white/50 p-5 rounded-xl hover:shadow-lg transition">
            <div className="text-3xl mb-3">🤖</div>
            <h3 className="font-semibold text-gray-800">Multi‑AI Playground</h3>
            <p className="text-sm text-gray-700 mt-1">Test GPT, DeepSeek, Gemma with custom prompts – compare results.</p>
          </div>
        </div>
      </div>

      {/* Why This App Matters – 2‑column grid */}
      <div className="scroll-section glass-card rounded-3xl shadow-xl p-8 relative z-10 bg-white/40 backdrop-blur-md">
        <h2 className="text-3xl font-bold text-center mb-8 text-gray-800">🌟 Why This App Matters</h2>
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h3 className="text-xl font-semibold mb-2 text-gray-800">For Students</h3>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Get instant, unbiased feedback on your writing – any time, any place.</li>
              <li>Understand your strengths and weaknesses with detailed scores (grammar, coherence, content).</li>
              <li>Learn from actionable suggestions and sample improvements tailored to your essay.</li>
              <li>Practice for standardized tests (IELTS, TOEFL, GRE) with realistic evaluation.</li>
            </ul>
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-2 text-gray-800">For Teachers & Educators</h3>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Save hours of grading – let AI handle the first pass, then override when needed.</li>
              <li>Store student essays and your corrections in a central knowledge base.</li>
              <li>The system learns from your overrides, improving feedback over time (RAG).</li>
              <li>Test different AI models (GPT, DeepSeek, Gemma) to find the best evaluator for your class.</li>
            </ul>
          </div>
        </div>
        <div className="mt-8 p-5 bg-blue-50 rounded-xl">
          <p className="text-center text-gray-700 italic">
            "Education is not the filling of a pail, but the lighting of a fire." – Our AI helps ignite that fire by providing fast, consistent, and personalised writing feedback.
          </p>
        </div>
      </div>

      {/* NLP Methodology & Knowledge Base – 2‑column grid */}
      <div className="scroll-section glass-card rounded-3xl shadow-xl p-8 relative z-10 border-l-8 border-purple-500 bg-white/40 backdrop-blur-md">
        <h2 className="text-3xl font-bold text-center mb-8 text-gray-800">🧠 NLP Methodology & Knowledge Base</h2>
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h3 className="text-xl font-semibold mb-3 flex items-center gap-2 text-gray-800">
              <span className="text-2xl">📐</span> NLP Techniques Used
            </h3>
            <ul className="space-y-3 text-gray-700">
              <li className="bg-gray-50 p-3 rounded-lg">
                <span className="font-bold text-blue-600">Grammar:</span> Sentence length, punctuation patterns, vocabulary richness – deterministic heuristics, no LLM hallucinations.
              </li>
              <li className="bg-gray-50 p-3 rounded-lg">
                <span className="font-bold text-green-600">Coherence:</span> Transition word counting, paragraph detection, presence of intro/conclusion keywords.
              </li>
              <li className="bg-gray-50 p-3 rounded-lg">
                <span className="font-bold text-purple-600">Content:</span> Evidence word spotting ("example", "research"), word count thresholds, vocabulary diversity.
              </li>
            </ul>
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-3 flex items-center gap-2 text-gray-800">
              <span className="text-2xl">📚</span> Knowledge Base (Dataset)
            </h3>
            <p className="text-gray-700 mb-3">
              Every saved evaluation is stored in <strong>Supabase (table: knowledge_base)</strong>. This creates a growing <strong>proprietary dataset</strong> of real essays with teacher‑approved scores and feedback.
            </p>
            <div className="bg-yellow-50 p-3 rounded-lg mb-3">
              <p className="text-sm font-medium text-yellow-800">⚠️ If the knowledge base is empty:</p>
              <p className="text-sm text-yellow-700">The system still works – RAG simply does nothing. Scores remain rule‑based and reliable.</p>
            </div>
            <p className="text-gray-700">
              <span className="font-bold">Hallucination prevention:</span> Scores are deterministic; RAG only retrieves <em>real past evaluations</em> – the AI never invents feedback. Teacher overrides further refine the dataset.
            </p>
          </div>
        </div>
        <div className="mt-8 p-5 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl border border-indigo-200">
          <h3 className="text-xl font-bold text-gray-800 mb-2 flex items-center gap-2">
            <span className="text-2xl">💡</span> Recommendation: Fine‑Tune on Your Dataset
          </h3>
          <p className="text-gray-700 mb-2">
            Your growing knowledge base is ideal for <strong>fine‑tuning a small LLM</strong> (e.g., Llama 3.2 3B, Mistral 7B) to:
          </p>
          <ul className="list-disc list-inside text-gray-700 space-y-1 ml-2">
            <li>Predict grammar, coherence, content scores directly from essays.</li>
            <li>Replace rule‑based heuristics with a model that adapts to your writing style.</li>
            <li>Continuously improve as you collect more teacher overrides.</li>
          </ul>
          <p className="text-sm text-gray-500 mt-3 italic">
            Without a dataset, fine‑tuning is impossible – but our system already provides a solid rule‑based evaluation. With your dataset, the model becomes even more accurate and personalised.
          </p>
          <div className="mt-3 p-2 bg-white rounded border border-gray-200">
            <p className="text-xs font-mono text-gray-500">Example dataset export query:</p>
            <pre className="text-xs bg-gray-100 p-2 rounded mt-1 overflow-x-auto">
SELECT essay, grammar, coherence, content, teacher_feedback FROM knowledge_base WHERE accepted = true;
            </pre>
          </div>
        </div>
      </div>

      {/* Data Safety */}
      <div className="scroll-section glass-card rounded-3xl shadow-xl p-8 border-l-8 border-green-500 relative z-10 bg-white/40 backdrop-blur-md">
        <div className="flex items-start gap-4">
          <div className="text-4xl">🔒</div>
          <div>
            <h2 className="text-2xl font-bold text-gray-800">Your Data is Safe</h2>
            <p className="text-gray-700 mt-2">
              We take privacy seriously. All essays are stored securely in Supabase (encrypted at rest). 
              Teacher overrides and knowledge base are used only to improve the AI – never shared with third parties. 
              You can delete your data at any time. OCR processing uses Google Cloud Vision (with billing enabled) but images are not retained after processing.
            </p>
            <div className="mt-4 flex gap-4 text-sm">
              <span className="bg-green-100 text-green-700 px-3 py-1 rounded-full">✅ GDPR Compliant</span>
              <span className="bg-green-100 text-green-700 px-3 py-1 rounded-full">✅ Data Encrypted</span>
              <span className="bg-green-100 text-green-700 px-3 py-1 rounded-full">✅ No Third‑Party Sharing</span>
            </div>
          </div>
        </div>
      </div>

      {/* Call to Action */}
      <div className="scroll-section text-center py-8 relative z-10">
        <button
          onClick={() => setActiveTab('evaluate')}
          className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-8 py-4 rounded-xl font-bold text-lg shadow-xl hover:shadow-2xl transition"
        >
          ✨ Try It Now – Evaluate Your Essay
        </button>
      </div>
    </div>
  );
}
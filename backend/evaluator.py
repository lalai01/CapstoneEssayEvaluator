import re
import requests
import os
import time
import language_tool_python
from langdetect import detect, DetectorFactory
from rag import get_similar_essay_context

DetectorFactory.seed = 0

# ---------- Safe LanguageTool Initialization with Retry ----------
tool = None
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

for attempt in range(1, MAX_RETRIES + 1):
    try:
        tool = language_tool_python.LanguageTool('en-US')
        print(f"✅ LanguageTool initialized successfully (attempt {attempt})")
        break
    except Exception as e:
        print(f"⚠️ LanguageTool initialization attempt {attempt} failed: {e}")
        if attempt < MAX_RETRIES:
            print(f"   Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
        else:
            print("❌ LanguageTool could not be initialized. Grammar checking will be disabled.")
            tool = None

# ---------- Holistic Rubric (unchanged) ----------
HOLISTIC_RUBRIC = {
    5: "🌟 Excellent (5/5) – Clear thesis, strong organization, compelling arguments, and virtually no errors. The essay demonstrates mastery of the topic.",
    4: "👍 Good (4/5) – Clear main idea, well-organized, with minor errors that do not impede understanding. Arguments are solid but could be more developed.",
    3: "📝 Satisfactory (3/5) – Understandable but with some weaknesses in clarity, organization, or support. Several grammatical issues or vague points.",
    2: "⚠️ Needs Improvement (2/5) – Unclear thesis, disorganized, or lacking sufficient evidence. Frequent errors make reading difficult.",
    1: "❌ Poor (1/5) – Hard to follow, no clear structure, many errors. The essay fails to address the topic adequately."
}

def is_valid_essay(text):
    words = text.split()
    if len(words) < 20:
        return False, "Essay too short. Please enter at least 20 words."
    try:
        lang = detect(text)
        if lang != 'en':
            return False, f"Essay must be in English (detected: {lang})."
    except Exception:
        return False, "Could not detect language. Please enter English text."
    vowel_pattern = re.compile(r'[aeiou]', re.IGNORECASE)
    real_word_count = sum(1 for w in words if len(w) > 2 and vowel_pattern.search(w))
    if real_word_count < 5:
        return False, "Input does not appear to be real English text."
    return True, None

def analyze_essay_content(essay_text):
    words = essay_text.split()
    word_count = len(words)
    sentences = max(1, len(re.findall(r'[.!?]+', essay_text)))
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    avg_sentence_length = word_count / sentences
    unique_words = len(set(word.lower() for word in words))
    vocabulary_richness = (unique_words / word_count) if word_count > 0 else 0
    transition_words = ['however', 'therefore', 'consequently', 'furthermore', 'moreover',
                       'nevertheless', 'subsequently', 'additionally', 'in conclusion']
    transition_count = sum(1 for word in words if word.lower() in transition_words)
    return {
        'word_count': word_count,
        'sentence_count': sentences,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length,
        'vocabulary_richness': vocabulary_richness,
        'transition_count': transition_count
    }

def find_long_sentences(essay_text, threshold=25):
    sentences = re.split(r'(?<=[.!?])\s+', essay_text)
    long_sentences = []
    for sent in sentences:
        words = sent.split()
        if len(words) > threshold:
            long_sentences.append((sent.strip(), len(words)))
    return long_sentences

def find_vague_words(essay_text):
    vague = ['good', 'bad', 'nice', 'thing', 'stuff', 'very', 'really', 'quite', 'pretty']
    found = {}
    for word in vague:
        count = len(re.findall(rf'\b{word}\b', essay_text, re.IGNORECASE))
        if count:
            found[word] = count
    return found

def check_grammar_with_nlp(text):
    """Return a list of grammar errors with suggestions and context.
       Returns empty list if LanguageTool is unavailable."""
    if tool is None:
        return []
    try:
        matches = tool.check(text)
    except Exception as e:
        print(f"LanguageTool check failed: {e}")
        return []
    errors = []
    for match in matches[:5]:
        if match.replacements:
            suggestion = match.replacements[0]
        else:
            suggestion = "Review this part."
        errors.append({
            'message': match.message,
            'suggestion': suggestion,
            'context': match.context,
            'offset': match.offset,
            'length': match.errorLength
        })
    return errors

def calculate_analytic_scores(essay_text, analysis):
    grammar = 85
    if analysis['avg_sentence_length'] > 25 or analysis['avg_sentence_length'] < 8:
        grammar -= 5
    if analysis['vocabulary_richness'] < 0.4:
        grammar -= 5
    elif analysis['vocabulary_richness'] > 0.7:
        grammar += 3
    if re.search(r'\s+[,.!?]', essay_text) or re.search(r'[,.!?]{2,}', essay_text):
        grammar -= 5
    grammar_errors = check_grammar_with_nlp(essay_text)
    grammar -= min(10, len(grammar_errors) * 2)
    grammar = max(60, min(98, grammar))

    coherence = 80
    paragraphs = essay_text.split('\n\n')
    if len(paragraphs) > 1:
        coherence += 5
    if analysis['transition_count'] > 3:
        coherence += 5
    elif analysis['transition_count'] == 0:
        coherence -= 5
    lower_text = essay_text.lower()
    if any(w in lower_text[:500] for w in ['introduction', 'first', 'begin', 'purpose']):
        coherence += 3
    if any(w in lower_text[-500:] for w in ['conclusion', 'summary', 'finally', 'in conclusion']):
        coherence += 3
    coherence = max(60, min(98, coherence))

    content = 75
    if analysis['word_count'] > 500:
        content += 10
    elif analysis['word_count'] > 300:
        content += 5
    elif analysis['word_count'] < 100:
        content -= 10
    if analysis['vocabulary_richness'] > 0.6:
        content += 5
    evidence_words = ['example', 'for instance', 'such as', 'because', 'research', 'study', 'data']
    evidence_count = sum(1 for w in evidence_words if w in lower_text)
    content += min(10, evidence_count * 2)
    content = max(60, min(98, content))

    return {"grammar": grammar, "coherence": coherence, "content": content}

def calculate_holistic_score(essay_text, analysis):
    analytic = calculate_analytic_scores(essay_text, analysis)
    avg = (analytic['grammar'] + analytic['coherence'] + analytic['content']) / 3
    if avg >= 90:
        return 5
    elif avg >= 80:
        return 4
    elif avg >= 70:
        return 3
    elif avg >= 60:
        return 2
    else:
        return 1

def get_paragraph_number(text, offset):
    paragraphs = text.split('\n\n')
    char_count = 0
    for i, para in enumerate(paragraphs):
        char_count += len(para) + 2
        if offset < char_count:
            return i + 1
    return 1

def generate_rule_based_analytic_feedback(essay_text, scores, analysis, rag_context=""):
    feedback = []
    if rag_context:
        feedback.append("[RAG_INSIGHTS_START]")
        feedback.append(rag_context)
        feedback.append("[RAG_INSIGHTS_END]")

    feedback.append("[ESSAY_ANALYSIS_START]")
    feedback.append(f"Your essay contains {analysis['word_count']} words across {analysis['sentence_count']} sentences.")
    feedback.append(f"Average sentence length: {analysis['avg_sentence_length']:.1f} words.")
    feedback.append("")

    feedback.append(f"📝 GRAMMAR ANALYSIS (Score: {scores['grammar']})")
    grammar_errors = check_grammar_with_nlp(essay_text)
    if grammar_errors:
        feedback.append(f"- Found {len(grammar_errors)} grammar issues:")
        for i, err in enumerate(grammar_errors[:3]):
            para_num = get_paragraph_number(essay_text, err['offset'])
            feedback.append(f"  {i+1}. In paragraph {para_num}: {err['message']}")
            feedback.append(f"     Suggestion: {err['suggestion']}")
    else:
        feedback.append("- No major grammar issues detected.")
    
    long_sents = find_long_sentences(essay_text)
    if long_sents:
        sent, length = long_sents[0]
        para_num = get_paragraph_number(essay_text, essay_text.find(sent[:30]))
        feedback.append(f"- In paragraph {para_num}, a long sentence ({length} words) may be hard to follow. Consider splitting it.")
    
    if analysis['vocabulary_richness'] < 0.45:
        feedback.append("- Vocabulary could be more varied. Try using synonyms or more precise terms.")
    feedback.append("")

    feedback.append(f"🔄 COHERENCE ANALYSIS (Score: {scores['coherence']})")
    paragraphs = essay_text.split('\n\n')
    if len(paragraphs) > 1:
        feedback.append(f"- Your essay has {len(paragraphs)} paragraphs, good for organization.")
    else:
        feedback.append("- Consider breaking your essay into distinct paragraphs.")
    
    for i, para in enumerate(paragraphs):
        if len(para.split()) < 20 and i > 0 and i < len(paragraphs)-1:
            feedback.append(f"- Paragraph {i+1} is quite short. Consider developing it further.")
    
    if analysis['transition_count'] < 2:
        feedback.append("- Add transition words like 'Furthermore' or 'However' to improve flow.")
    else:
        feedback.append("- Good use of transition words.")
    feedback.append("")

    feedback.append(f"📚 CONTENT ANALYSIS (Score: {scores['content']})")
    evidence_count = sum(1 for w in ['example', 'for instance', 'such as', 'because', 'research'] if w in essay_text.lower())
    if evidence_count < 2:
        feedback.append("- Include at least one concrete example to strengthen your argument.")
    else:
        feedback.append(f"- Good use of evidence ({evidence_count} instances).")
    
    if len(paragraphs) >= 2:
        intro_words = ['introduction', 'first', 'begin', 'purpose', 'this essay']
        if not any(w in paragraphs[0].lower() for w in intro_words):
            feedback.append("- Your first paragraph could more clearly introduce the topic.")
        conc_words = ['conclusion', 'summary', 'finally', 'in conclusion', 'overall']
        if not any(w in paragraphs[-1].lower() for w in conc_words):
            feedback.append("- Your final paragraph could be strengthened with a concluding statement.")

    return "\n".join(feedback)

def generate_rule_based_holistic_feedback(essay_text, holistic_score, analysis, rag_context=""):
    feedback = []
    if rag_context:
        feedback.append("[RAG_INSIGHTS_START]")
        feedback.append(rag_context)
        feedback.append("[RAG_INSIGHTS_END]")

    feedback.append(f"🌟 Holistic Score: {holistic_score}/5")
    feedback.append(HOLISTIC_RUBRIC[holistic_score])
    feedback.append("")

    feedback.append("📊 Essay Statistics")
    feedback.append(f"• Words: {analysis['word_count']} | Sentences: {analysis['sentence_count']}")
    feedback.append(f"• Avg sentence length: {analysis['avg_sentence_length']:.1f} words")
    feedback.append(f"• Vocabulary richness: {analysis['vocabulary_richness']:.2f}")
    feedback.append(f"• Transition words used: {analysis['transition_count']}")
    feedback.append("")

    feedback.append("💡 Specific Areas to Improve")
    
    grammar_errors = check_grammar_with_nlp(essay_text)
    if grammar_errors:
        err = grammar_errors[0]
        para_num = get_paragraph_number(essay_text, err['offset'])
        feedback.append(f"• Grammar issue in paragraph {para_num}: {err['message']}")
        feedback.append(f"  ✅ Suggested correction: {err['suggestion']}")
    
    long_sents = find_long_sentences(essay_text)
    if long_sents:
        sent, length = long_sents[0]
        para_num = get_paragraph_number(essay_text, essay_text.find(sent[:30]))
        feedback.append(f"• Long sentence in paragraph {para_num} ({length} words): \"{sent[:100]}...\"")
        feedback.append("  ✅ Try breaking it into shorter sentences.")
    else:
        feedback.append("• Sentence lengths are generally well-balanced.")
    
    if analysis['transition_count'] < 2:
        feedback.append("• Add more transition words (e.g., 'Furthermore', 'However', 'Therefore').")
    else:
        feedback.append("• Good use of transition words.")
    
    vague_found = find_vague_words(essay_text)
    if vague_found:
        feedback.append(f"• Consider replacing vague words like {', '.join(list(vague_found.keys())[:3])} with more precise vocabulary.")
    
    paragraphs = essay_text.split('\n\n')
    if len(paragraphs) < 2:
        feedback.append("• Break your essay into distinct paragraphs (introduction, body, conclusion).")
    
    lower_text = essay_text.lower()
    evidence_count = sum(1 for w in ['example', 'for instance', 'such as', 'because', 'research', 'study', 'data'] if w in lower_text)
    if evidence_count < 2:
        feedback.append("• Include specific examples or evidence to strengthen your arguments.")
    
    feedback.append("")
    if holistic_score >= 4:
        feedback.append("✨ Overall, this is a strong essay. Focus on refining word choice and adding more nuanced examples.")
    elif holistic_score >= 3:
        feedback.append("📝 This essay shows competence. Work on deeper analysis and clearer organization.")
    else:
        feedback.append("⚠️ This essay needs significant revision. Start by clarifying your main thesis and organizing your thoughts.")

    return "\n".join(feedback)

def enhance_feedback_with_ai(essay_text, scores, analysis, rule_feedback):
    """Use local Ollama (Gemma) to rewrite feedback into warmer, more natural language."""
    ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")
    model = "gemma2:2b"
    
    if 'holistic_score' in scores:
        score_info = f"Holistic Score: {scores['holistic_score']}/5\nDescription: {scores.get('level_description', '')}"
    else:
        score_info = f"Grammar: {scores.get('grammar', 'N/A')}\nCoherence: {scores.get('coherence', 'N/A')}\nContent: {scores.get('content', 'N/A')}"
    
    system_msg = "You are an expert writing coach who provides warm, encouraging, and actionable feedback to students."
    user_msg = f"""Rewrite the following technical analysis into a single, natural feedback paragraph. Preserve all scores, specific issues, paragraph numbers, and actionable suggestions. Use a supportive tone.

Essay excerpt: {essay_text[:1000]}...

Scores: {score_info}

Technical Analysis: {rule_feedback[:2000]}

Write only the final feedback paragraph:"""
    
    try:
        response = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                "stream": False,
                "options": {"temperature": 0.7}
            },
            timeout=45
        )
        if response.status_code == 200:
            data = response.json()
            enhanced = data.get("message", {}).get("content", "").strip()
            if enhanced:
                if "[RAG_INSIGHTS_START]" in rule_feedback:
                    rag_part = rule_feedback.split("[RAG_INSIGHTS_START]")[1].split("[RAG_INSIGHTS_END]")[0]
                    return f"[RAG_INSIGHTS_START]\n{rag_part}\n[RAG_INSIGHTS_END]\n\n{enhanced}"
                return enhanced
    except Exception as e:
        print(f"AI enhancement failed, using rule-based feedback: {e}")
    
    return rule_feedback

def evaluate_essay(essay_text, evaluation_type="analytic", use_rag=True):
    """Main evaluation function. Returns scores and AI-enhanced feedback."""
    is_valid, error_msg = is_valid_essay(essay_text)
    if not is_valid:
        if evaluation_type == "holistic":
            return {"holistic_score": 0, "level_description": error_msg}, f"⚠️ Invalid Input: {error_msg}"
        else:
            return {"grammar": 0, "coherence": 0, "content": 0}, f"⚠️ Invalid Input: {error_msg}"

    analysis = analyze_essay_content(essay_text)
    rag_context = ""
    if use_rag:
        try:
            rag_context = get_similar_essay_context(essay_text)
        except Exception as e:
            print(f"RAG error: {e}")

    if evaluation_type == "holistic":
        score = calculate_holistic_score(essay_text, analysis)
        rule_feedback = generate_rule_based_holistic_feedback(essay_text, score, analysis, rag_context)
        scores = {"holistic_score": score, "level_description": HOLISTIC_RUBRIC[score]}
    else:
        scores = calculate_analytic_scores(essay_text, analysis)
        rule_feedback = generate_rule_based_analytic_feedback(essay_text, scores, analysis, rag_context)

    feedback = enhance_feedback_with_ai(essay_text, scores, analysis, rule_feedback)
    return scores, feedback

RUBRIC = {
    "grammar": "Correctness of sentence structure, punctuation, spelling, and tense consistency.",
    "coherence": "Logical flow of ideas, use of transition words, paragraph organization, and clarity.",
    "content": "Depth of argument, relevance to topic, use of evidence, originality, and conclusion strength."
}

SUGGESTION_GUIDE = {
    "what": "This AI evaluates essays by dynamically analyzing your actual writing, providing personalized feedback based on your specific content.",
    "when": """Use this tool when you:
• Need constructive feedback on essay drafts
• Want to improve your writing skills
• Prepare for standardized tests (IELTS, TOEFL, GRE)
• Require consistent grading for multiple essays
• Want to digitize handwritten essays for evaluation""",
    "how": """How to get the best results:
1. Input Methods:
   - Type or paste your essay directly
   - Upload an image (JPG, PNG, etc.) of handwritten/printed essay
   - Upload a PDF file (all pages will be processed)
   
2. For Best OCR Results:
   - Ensure good lighting when photographing
   - Use clear, legible handwriting
   - Avoid shadows and glare
   - For PDFs, ensure they are text-based or high-quality scans
   
3. Review Process:
   - Check extracted text for accuracy
   - Make manual corrections if needed
   - Then click Evaluate for personalized feedback"""
}
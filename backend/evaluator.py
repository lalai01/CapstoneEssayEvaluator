import re
import requests
import os
import language_tool_python
from langdetect import detect, DetectorFactory
from rag import get_similar_essay_context

DetectorFactory.seed = 0
tool = None
try:
    tool = language_tool_python.LanguageTool('en-US')
    print("✅ LanguageTool initialized successfully.")
except Exception as e:
    print(f"⚠️ LanguageTool initialization failed: {e}. Grammar checking will be limited to heuristics.")


ANALYTIC_RUBRIC = {
    "main_statement": {
        4: "Presents a clear, focused, and defensible thesis that directly addresses the prompt and establishes a strong focus or position.",
        3: "Presents a clear thesis that addresses the prompt, with minor weaknesses in focus or development.",
        2: "Presents a thesis, but it is broad, unclear, partially developed, or only partly addresses the prompt.",
        1: "Thesis is absent, unclear, or does not meaningfully address the prompt.",
    },
    "organization": {
        4: "Ideas are logically and coherently organized; paragraphs and transitions create a clear progression of the argument.",
        3: "Ideas are generally well organized, with minor weaknesses in sequencing, paragraphing, or transitions.",
        2: "Organization is inconsistent; some ideas or paragraphs are difficult to follow or insufficiently connected.",
        1: "Ideas are poorly organized and lack a clear progression, making the argument difficult to follow.",
    },
    "evidence": {
        4: "Provides relevant, sufficient, and appropriate evidence that strongly supports the central position.",
        3: "Provides generally relevant and adequate evidence that supports the central position, with minor weaknesses.",
        2: "Provides limited, weak, insufficient, or inconsistently relevant evidence.",
        1: "Provides little or no relevant evidence to support the position.",
    },
    "analysis": {
        4: "Thoroughly explains and interprets evidence and clearly connects it to claims and the thesis through logical reasoning.",
        3: "Adequately explains evidence and generally connects it to the argument, with some limitations in depth or reasoning.",
        2: "Provides limited explanation or weak connections between evidence, claims, and the thesis.",
        1: "Provides little or no meaningful analysis; evidence is mainly listed, repeated, or left unexplained.",
    },
    "grammar": {
        4: "Uses consistently accurate grammar, sentence structure, spelling, punctuation, capitalization, and word choice; errors do not interfere with clarity.",
        3: "Contains minor grammatical or mechanical errors that do not substantially interfere with clarity.",
        2: "Contains frequent errors that sometimes affect readability or clarity.",
        1: "Contains persistent or serious errors that interfere with meaning and readability.",
    },
}

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
    """Return a list of grammar errors with suggestions and context."""
    if tool is None:
        return []
    matches = tool.check(text)
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
    """
    Returns 1–4 scores for each rubric criterion:
    main_statement, organization, evidence, analysis, grammar.
    """
    lower = essay_text.lower()
    paragraphs = essay_text.split("\n\n")
    transitions = analysis["transition_count"]
    vocab = analysis["vocabulary_richness"]
    avg_sl = analysis["avg_sentence_length"]
    word_count = analysis["word_count"]

    # --- Main Statement / Thesis ---
    thesis_signals = ["this essay", "i will argue", "the purpose", "the thesis",
                      "in this paper", "this paper will", "this essay will"]
    first_para = paragraphs[0].lower() if paragraphs else lower
    has_signal = any(s in first_para for s in thesis_signals)
    if has_signal and word_count >= 250:
        main_statement = 4
    elif has_signal and word_count >= 120:
        main_statement = 3
    elif word_count >= 150:
        main_statement = 2
    else:
        main_statement = 1

    # --- Organization ---
    organization = 4
    if len(paragraphs) < 2:
        organization = 2
    elif len(paragraphs) < 3:
        organization = 3
    if transitions == 0:
        organization = min(organization, 2)
    elif transitions < 3:
        organization = min(organization, 3)

    # --- Evidence ---
    evidence_terms = ["example", "for instance", "such as", "because",
                      "research", "study", "data", "according to"]
    evidence_count = sum(1 for w in evidence_terms if w in lower)
    if evidence_count >= 3:
        evidence = 4
    elif evidence_count == 2:
        evidence = 3
    elif evidence_count == 1:
        evidence = 2
    else:
        evidence = 1

    # --- Analysis ---
    if word_count >= 400 and evidence_count >= 3 and transitions >= 3:
        analysis_score = 4
    elif word_count >= 250 and evidence_count >= 2:
        analysis_score = 3
    elif word_count >= 120:
        analysis_score = 2
    else:
        analysis_score = 1

    # --- Grammar and Mechanics ---
    grammar = 4
    if avg_sl > 30 or avg_sl < 6:
        grammar = 2
    elif avg_sl > 25 or avg_sl < 8:
        grammar = 3
    if vocab < 0.35:
        grammar = min(grammar, 2)
    if re.search(r'\s+[,.!?]', essay_text) or re.search(r'[,.!?]{2,}', essay_text):
        grammar = min(grammar, 3)
    grammar_errors = check_grammar_with_nlp(essay_text)
    if len(grammar_errors) >= 4:
        grammar = 2
    elif len(grammar_errors) >= 2 and grammar == 4:
        grammar = 3

    return {
        "main_statement": main_statement,
        "organization": organization,
        "evidence": evidence,
        "analysis": analysis_score,
        "grammar": grammar,
    }

def calculate_holistic_score(essay_text, analysis):
    analytic = calculate_analytic_scores(essay_text, analysis)
    avg = sum(analytic.values()) / len(analytic)   # 1.0 – 4.0
    if avg >= 3.6:
        return 5
    elif avg >= 3.0:
        return 4
    elif avg >= 2.3:
        return 3
    elif avg >= 1.6:
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

def generate_specific_suggestions(essay_text, analysis, scores):
    suggestions = []
    long_sents = find_long_sentences(essay_text)
    if long_sents:
        sent, length = long_sents[0]
        truncated = sent[:150] + "..." if len(sent) > 150 else sent
        suggestions.append({
            "title": "Long sentence detected",
            "original": truncated,
            "suggestion": "Break this into shorter sentences. Example: 'Education is the cornerstone of personal and societal development. It empowers individuals with knowledge and critical thinking skills.'"
        })
    if analysis['transition_count'] < 2:
        suggestions.append({
            "title": "Add transitions",
            "original": "Limited use of transition words.",
            "suggestion": "Add words like 'Furthermore', 'However', or 'For example' to connect ideas."
        })
    vague_found = find_vague_words(essay_text)
    if 'very' in vague_found or 'really' in vague_found:
        suggestions.append({
            "title": "Stronger vocabulary",
            "original": f"Uses weak modifiers: {', '.join([k for k in vague_found])}",
            "suggestion": "Replace 'very important' with 'crucial' or 'essential' for stronger impact."
        })
    return suggestions

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

    feedback.append(f"🧠 MAIN STATEMENT / THESIS (Score: {scores['main_statement']}/4)")
    feedback.append(f"- {ANALYTIC_RUBRIC['main_statement'][scores['main_statement']]}")
    feedback.append("")

    feedback.append(f"🧱 ORGANIZATION (Score: {scores['organization']}/4)")
    feedback.append(f"- {ANALYTIC_RUBRIC['organization'][scores['organization']]}")
    feedback.append("")

    feedback.append(f"📚 EVIDENCE (Score: {scores['evidence']}/4)")
    feedback.append(f"- {ANALYTIC_RUBRIC['evidence'][scores['evidence']]}")
    feedback.append("")

    feedback.append(f"🔍 ANALYSIS (Score: {scores['analysis']}/4)")
    feedback.append(f"- {ANALYTIC_RUBRIC['analysis'][scores['analysis']]}")
    feedback.append("")

    feedback.append(f"📝 GRAMMAR & MECHANICS (Score: {scores['grammar']}/4)")
    feedback.append(f"- {ANALYTIC_RUBRIC['grammar'][scores['grammar']]}")

    grammar_errors = check_grammar_with_nlp(essay_text)
    if grammar_errors:
        feedback.append(f"  • Detected {len(grammar_errors)} grammar issues; top: {grammar_errors[0]['message']}")

    specific = generate_specific_suggestions(essay_text, analysis, scores)
    if specific:
        feedback.append("")
        feedback.append("✨ SPECIFIC IMPROVEMENTS YOU CAN MAKE")
        for i, s in enumerate(specific[:2]):
            feedback.append(f"{i+1}. {s['title']}: {s['suggestion']}")

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
    is_valid, error_msg = is_valid_essay(essay_text)
    if not is_valid:
        empty_scores = (
            {"holistic_score": 0, "level_description": error_msg}
            if evaluation_type == "holistic"
            else {"main_statement": 0, "organization": 0,
                  "evidence": 0, "analysis": 0, "grammar": 0}
        )
        return empty_scores, f"⚠️ Invalid Input: {error_msg}"

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
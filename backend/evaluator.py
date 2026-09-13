# backend/evaluator.py
import re
import json
import os
import requests
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


# ---------- Holistic Rubric (5-point scale) ----------
HOLISTIC_RUBRIC = {
    5: "🌟 Excellent (5/5) – Clear thesis, strong organization, compelling arguments, and virtually no errors. The essay demonstrates mastery of the topic.",
    4: "👍 Good (4/5) – Clear main idea, well-organized, with minor errors that do not impede understanding. Arguments are solid but could be more developed.",
    3: "📝 Satisfactory (3/5) – Understandable but with some weaknesses in clarity, organization, or support. Several grammatical issues or vague points.",
    2: "⚠️ Needs Improvement (2/5) – Unclear thesis, disorganized, or lacking sufficient evidence. Frequent errors make reading difficult.",
    1: "❌ Poor (1/5) – Hard to follow, no clear structure, many errors. The essay fails to address the topic adequately."
}


# ---------- Analytic Rubric (4-point scale) ----------
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

# ---------- Essay Validation ----------
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

# ---------- Content Analysis ----------
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

# ---------- Helper: Sentence & Word Scans ----------
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

def _essay_facts(essay_text):
    """
    Compute ground-truth facts about the essay so the LLM
    doesn't have to guess them.
    """
    paragraphs = [p for p in essay_text.split("\n\n") if p.strip()]
    word_count = len(essay_text.split())
    sentences = max(1, len(re.findall(r'[.!?]+', essay_text)))

    # Detect explicit citations
    cited_patterns = [
        r"according to [A-Z]",
        r"research (?:by|from) [A-Z]",
        r"studies (?:by|from) [A-Z]",
        r"report (?:by|from) [A-Z]",
        r"survey (?:by|from) [A-Z]",
        r"\b(?:19|20)\d{2}\b",
    ]
    citation_count = 0
    for pattern in cited_patterns:
        citation_count += len(re.findall(pattern, essay_text))

    # Count transitions (phrase-aware)
    transition_words = [
        "however", "therefore", "consequently", "furthermore",
        "moreover", "nevertheless", "subsequently", "additionally",
        "first", "second", "third", "finally", "in addition",
        "for example", "for instance", "in conclusion", "as a result",
        "on the other hand", "in other words", "more importantly",
    ]
    lower_text = essay_text.lower()
    transition_count = sum(1 for phrase in transition_words if phrase in lower_text)

    return {
        "paragraph_count": len(paragraphs),
        "word_count": word_count,
        "sentence_count": sentences,
        "citation_count": citation_count,
        "transition_count": transition_count,
    }

# ---------- AI-Based Scoring (with Heuristic Fallback) ----------
def ai_score_all_criteria(essay_text, model="llama3.2:3b"):
    """
    Ask the local LLM (via Ollama) to score the essay on all 5 rubric criteria.
    Uses observed ground-truth facts to prevent the model from guessing.
    """
    ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")

    rubric_block = ""
    for criterion, levels in ANALYTIC_RUBRIC.items():
        rubric_block += f"\n[{criterion}]\n"
        for level in sorted(levels, reverse=True):
            rubric_block += f"  {level}: {levels[level]}\n"

    facts = _essay_facts(essay_text)

    system_msg = (
        "You are a strict, experienced essay examiner. "
        "You score decisively and do not default to middle values. "
        "You use the full 1-4 range based on concrete evidence in the essay. "
        "You differentiate scores across criteria when warranted. "
        "You return ONLY a JSON object and never add explanations."
    )

    user_msg = f"""Carefully evaluate the essay below against the rubric.

RUBRIC (use these exact descriptors to justify scores):

{rubric_block}

OBSERVED FACTS (do NOT re-count; use these as anchors):
- Paragraph count: {facts['paragraph_count']}
- Word count: {facts['word_count']}
- Sentence count: {facts['sentence_count']}
- Explicit citations detected: {facts['citation_count']}
- Transition words used: {facts['transition_count']}

SCORING RULES:
- Base each score on the rubric wording above, not on personal preference.
- Do NOT give the same score to every criterion; differentiate them.
- Use 4 only when the essay clearly satisfies the top-level descriptor.
- Use 3 for solid but imperfect work.
- Use 2 when the descriptor for 2 clearly applies.
- Use 1 only when the criterion is essentially missing or broken.
- If a criterion is strong and another is weak, the scores MUST differ.

HARD RULES (must be obeyed):
- If paragraph_count >= 3 AND transition_count >= 3, organization MUST be 4.
- If citation_count >= 2, evidence MUST be 4.
- If the thesis appears in the first paragraph AND is restated or
  reinforced in the conclusion, main_statement MUST be 4.

CALIBRATION EXAMPLES:

Example A - Essay with a clear thesis, three body paragraphs, and 2+ cited
sources:
  main_statement: 4
  organization: 4
  evidence: 4
  analysis: 3 or 4
  grammar: 3 or 4

Example B - Short single-paragraph essay with a thesis but only one example
and no cited source:
  main_statement: 3
  organization: 2
  evidence: 2
  analysis: 2
  grammar: 3 or 4

Example C - Multi-paragraph essay with a clear thesis but no concrete evidence:
  main_statement: 3
  organization: 3
  evidence: 1 or 2
  analysis: 2
  grammar: 3

Example D - Disorganized essay with no thesis and no evidence:
  main_statement: 1
  organization: 1
  evidence: 1
  analysis: 1
  grammar: 2

Essay:
\"\"\"
{essay_text[:4000]}
\"\"\"

Return ONLY this JSON object (no explanation, no markdown):
{{
  "main_statement": <int 1-4>,
  "organization": <int 1-4>,
  "evidence": <int 1-4>,
  "analysis": <int 1-4>,
  "grammar": <int 1-4>
}}
"""

    try:
        response = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=60,
        )
        if response.status_code != 200:
            print(f"Ollama returned status {response.status_code}")
            return None

        content = response.json().get("message", {}).get("content", "").strip()

        if content.startswith("```"):
            content = content.strip("`").replace("json", "", 1).strip()

        parsed = json.loads(content)
        required = ["main_statement", "organization", "evidence", "analysis", "grammar"]
        for key in required:
            if key not in parsed:
                print(f"Missing key in AI response: {key}")
                return None
            score = int(parsed[key])
            if score < 1 or score > 4:
                print(f"Out-of-range score for {key}: {score}")
                return None
            parsed[key] = score

        return {k: parsed[k] for k in required}

    except json.JSONDecodeError as e:
        print(f"AI scoring JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"AI scoring failed: {e}")
        return None


def _heuristic_scores(essay_text, analysis):
    """Rule-based fallback aligned with the 4-point rubric."""
    lower = essay_text.lower()
    paragraphs = [p for p in essay_text.split("\n\n") if p.strip()]
    para_count = len(paragraphs)
    word_count = analysis["word_count"]
    transitions = analysis["transition_count"]
    vocab = analysis["vocabulary_richness"]
    avg_sl = analysis["avg_sentence_length"]

    # --- Main Statement / Thesis ---
    thesis_signals = [
        "this essay", "i will argue", "the purpose of this",
        "this paper will", "the thesis", "in this paper",
        "i believe", "i contend", "the central argument",
    ]
    argument_markers = ["should", "must", "is important", "is crucial",
                        "is essential", "ought to", "is necessary"]
    first_para = paragraphs[0].lower() if paragraphs else lower
    has_signal = any(s in first_para for s in thesis_signals)
    has_argument = any(m in first_para for m in argument_markers)

    if has_signal and word_count >= 250:
        main_statement = 4
    elif has_signal or has_argument:
        main_statement = 3
    elif word_count >= 150:
        main_statement = 2
    else:
        main_statement = 1

    # --- Organization ---
    intro_markers = ["introduction", "first", "begin", "purpose"]
    conclusion_markers = ["conclusion", "summary", "finally",
                          "in conclusion", "to summarize", "overall"]
    has_intro = any(m in lower[:400] for m in intro_markers)
    has_conclusion = any(m in lower[-400:] for m in conclusion_markers)

    if para_count >= 3 and transitions >= 4:
        organization = 4
    elif para_count >= 2 and (has_intro or has_conclusion or transitions >= 2):
        organization = 4
    elif para_count >= 2 or transitions >= 2:
        organization = 3
    elif transitions >= 1:
        organization = 2
    else:
        organization = 1

    # --- Evidence ---
    evidence_markers = [
        "for example", "for instance", "such as", "according to",
        "research shows", "studies show", "data", "statistics",
        "evidence", "report", "survey",
    ]
    evidence_count = sum(1 for m in evidence_markers if m in lower)

    if evidence_count >= 3 and word_count >= 300:
        evidence = 4
    elif evidence_count >= 2 or (evidence_count >= 1 and word_count >= 300):
        evidence = 3
    elif evidence_count >= 1:
        evidence = 2
    else:
        evidence = 1

    # --- Analysis ---
    reasoning_markers = [
        "because", "therefore", "thus", "hence", "as a result",
        "this shows", "this means", "this demonstrates", "which means",
        "consequently", "in other words", "specifically",
    ]
    reasoning_count = sum(1 for m in reasoning_markers if m in lower)

    if reasoning_count >= 3 and word_count >= 300:
        analysis_score = 4
    elif reasoning_count >= 2 and word_count >= 200:
        analysis_score = 3
    elif reasoning_count >= 1:
        analysis_score = 2
    else:
        analysis_score = 1

    # --- Grammar and Mechanics ---
    grammar_errors = check_grammar_with_nlp(essay_text)
    err_count = len(grammar_errors)
    punctuation_issue = bool(
        re.search(r'\s+[,.!?]', essay_text) or re.search(r'[,.!?]{2,}', essay_text)
    )

    if (err_count <= 1 and not punctuation_issue
            and 8 <= avg_sl <= 25 and vocab >= 0.45):
        grammar = 4
    elif err_count <= 3 and not punctuation_issue:
        grammar = 3
    elif err_count <= 6:
        grammar = 2
    else:
        grammar = 1

    return {
        "main_statement": main_statement,
        "organization": organization,
        "evidence": evidence,
        "analysis": analysis_score,
        "grammar": grammar,
    }


def calculate_analytic_scores(essay_text, analysis):
    """
    Try AI scoring first (via Ollama). Fall back to heuristics if the LLM fails.
    """
    ai = ai_score_all_criteria(essay_text)
    if ai is not None:
        print("✅ AI scoring succeeded")
        return ai
    print("⚠️ AI scoring failed, using heuristics")
    return _heuristic_scores(essay_text, analysis)


# ---------- Holistic Score (mapped from rubric average) ----------
def calculate_holistic_score(essay_text, analysis):
    analytic = calculate_analytic_scores(essay_text, analysis)
    avg = sum(analytic.values()) / len(analytic)  # 1.0 – 4.0
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


# ---------- Paragraph Detection ----------
def get_paragraph_number(text, offset):
    paragraphs = text.split('\n\n')
    char_count = 0
    for i, para in enumerate(paragraphs):
        char_count += len(para) + 2
        if offset < char_count:
            return i + 1
    return 1


# ---------- Suggestions ----------
def generate_specific_suggestions(essay_text, analysis, scores):
    suggestions = []

    long_sents = find_long_sentences(essay_text)
    if long_sents:
        sent, length = long_sents[0]
        truncated = sent[:150] + "..." if len(sent) > 150 else sent
        suggestions.append({
            "title": "Long sentence detected",
            "original": truncated,
            "suggestion": "Break this into shorter sentences for better readability."
        })

    if analysis['transition_count'] == 0:
        suggestions.append({
            "title": "Add transitions",
            "original": "No transition words detected.",
            "suggestion": "Add words like 'Furthermore', 'However', or 'For example' to connect ideas."
        })

    vague_found = find_vague_words(essay_text)
    if 'very' in vague_found or 'really' in vague_found:
        suggestions.append({
            "title": "Stronger vocabulary",
            "original": f"Uses weak modifiers: {', '.join(vague_found.keys())}",
            "suggestion": "Replace 'very important' with 'crucial' or 'essential' for stronger impact."
        })

    return suggestions


# ---------- Feedback Generation ----------
def generate_rule_based_analytic_feedback(essay_text, scores, analysis, rag_context=""):
    feedback = []

    if rag_context:
        feedback.append("[RAG_INSIGHTS_START]")
        feedback.append(rag_context)
        feedback.append("[RAG_INSIGHTS_END]")

    feedback.append("[ESSAY_ANALYSIS_START]")
    feedback.append(f"Your essay contains {analysis['word_count']} words across "
                    f"{analysis['sentence_count']} sentences.")
    feedback.append(f"Average sentence length: {analysis['avg_sentence_length']:.1f} words.")
    feedback.append("")

    feedback.append(f"🧠 MAIN STATEMENT / THESIS (Score: {scores['main_statement']}/4)")
    feedback.append(f"- {ANALYTIC_RUBRIC['main_statement'][scores['main_statement']]}")
    if scores['main_statement'] < 4:
        feedback.append("- 💡 Strengthen this by clearly stating your position in the first paragraph "
                        "(e.g., 'This essay argues that…').")
    feedback.append("")

    feedback.append(f"🧱 ORGANIZATION (Score: {scores['organization']}/4)")
    feedback.append(f"- {ANALYTIC_RUBRIC['organization'][scores['organization']]}")
    if scores['organization'] < 4:
        para_count = len([p for p in essay_text.split('\n\n') if p.strip()])
        if para_count < 3:
            feedback.append("- 💡 Break your argument into at least three paragraphs: introduction, body, conclusion.")
        if analysis['transition_count'] == 0:
            feedback.append("- 💡 Use transitions (e.g., 'Furthermore', 'Therefore', 'In conclusion').")
    feedback.append("")
    
    feedback.append(f"📚 EVIDENCE (Score: {scores['evidence']}/4)")
    feedback.append(f"- {ANALYTIC_RUBRIC['evidence'][scores['evidence']]}")
    if scores['evidence'] < 4:
        feedback.append("- 💡 Include at least one more concrete example, statistic, or reference "
                        "(e.g., 'According to a 2023 study…').")
    feedback.append("")

    feedback.append(f"🔍 ANALYSIS (Score: {scores['analysis']}/4)")
    feedback.append(f"- {ANALYTIC_RUBRIC['analysis'][scores['analysis']]}")
    if scores['analysis'] < 4:
        feedback.append("- 💡 After each piece of evidence, add a sentence that explains what it means "
                        "and how it supports your thesis.")
    feedback.append("")

    feedback.append(f"📝 GRAMMAR & MECHANICS (Score: {scores['grammar']}/4)")
    feedback.append(f"- {ANALYTIC_RUBRIC['grammar'][scores['grammar']]}")
    grammar_errors = check_grammar_with_nlp(essay_text)
    if grammar_errors:
        feedback.append(f"  • Top issue detected: {grammar_errors[0]['message']}")
        if grammar_errors[0].get('suggestion'):
            feedback.append(f"    Suggested fix: {grammar_errors[0]['suggestion']}")
    feedback.append("")

    specific = generate_specific_suggestions(essay_text, analysis, scores)
    if specific:
        feedback.append("✨ SPECIFIC IMPROVEMENTS YOU CAN MAKE")
        for i, s in enumerate(specific[:3]):
            feedback.append(f"{i+1}. {s['title']}: {s['suggestion']}")
        feedback.append("")

    avg = sum(scores.values()) / len(scores)
    if avg >= 3.5:
        feedback.append("✅ Overall: This is a strong essay. Focus on refining details to reach the top of the rubric.")
    elif avg >= 2.5:
        feedback.append("📈 Overall: This essay meets expectations. Work on deepening analysis and evidence.")
    else:
        feedback.append("⚠️ Overall: This essay needs significant revision. Start by strengthening the thesis "
                        "and adding supporting evidence.")

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


def enhance_feedback_with_ai(essay_text, scores, analysis, rule_feedback, facts=None):
    ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")
    model = "llama3.2:3b"   

    if facts is None:
        facts = _essay_facts(essay_text)

    if 'holistic_score' in scores:
        score_info = f"Holistic Score: {scores['holistic_score']}/5\nDescription: {scores.get('level_description', '')}"
    else:
        score_info = (
            f"Main Statement: {scores.get('main_statement', 'N/A')}/4\n"
            f"Organization: {scores.get('organization', 'N/A')}/4\n"
            f"Evidence: {scores.get('evidence', 'N/A')}/4\n"
            f"Analysis: {scores.get('analysis', 'N/A')}/4\n"
            f"Grammar: {scores.get('grammar', 'N/A')}/4"
        )

    facts_block = (
        f"- Paragraph count: {facts['paragraph_count']}\n"
        f"- Word count: {facts['word_count']}\n"
        f"- Transition count: {facts['transition_count']}\n"
        f"- Citation count: {facts['citation_count']}"
    )

    system_msg = (
        "You are an expert writing coach. You rewrite technical evaluations "
        "into warm, encouraging, and actionable feedback for students. "
        "You MUST NOT add suggestions that contradict the OBSERVED FACTS. "
        "You MUST NOT invent issues that are not present in the technical analysis. "
        "You output only the final feedback paragraph."
    )

    user_msg = f"""Rewrite the technical analysis below into a single, natural
feedback paragraph for a student. Preserve all scores, specific issues, and
suggestions. Use a supportive tone.

OBSERVED FACTS — DO NOT CONTRADICT:
{facts_block}

Rules you MUST obey:
- If transition_count > 0, do NOT advise adding transitions.
- If paragraph_count >= 2, do NOT advise breaking the essay into paragraphs.
- If citation_count >= 2, do NOT advise adding more examples.
- Only mention issues that appear in the TECHNICAL ANALYSIS below.
- Do NOT invent new suggestions not present in the technical analysis.
- Keep the response under 120 words.

Essay excerpt:
{essay_text[:1000]}

Scores:
{score_info}

Technical Analysis:
{rule_feedback[:2000]}

Write only the final feedback paragraph:"""

    try:
        response = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
                "options": {"temperature": 0.4},
            },
            timeout=60,
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


# ---------- Main Entry Point ----------
def evaluate_essay(essay_text, evaluation_type="analytic", use_rag=True):
    is_valid, error_msg = is_valid_essay(essay_text)
    if not is_valid:
        if evaluation_type == "holistic":
            return {"holistic_score": 0, "level_description": error_msg}, f"⚠️ Invalid Input: {error_msg}"
        return {
            "main_statement": 0, "organization": 0,
            "evidence": 0, "analysis": 0, "grammar": 0,
        }, f"⚠️ Invalid Input: {error_msg}"

    analysis = analyze_essay_content(essay_text)
    facts = _essay_facts(essay_text)  

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

    feedback = enhance_feedback_with_ai(essay_text, scores, analysis, rule_feedback, facts) 
    return scores, feedback


# ---------- Static Rubric & Suggestion Guide (for UI) ----------
RUBRIC = {
    "main_statement": ANALYTIC_RUBRIC["main_statement"],
    "organization": ANALYTIC_RUBRIC["organization"],
    "evidence": ANALYTIC_RUBRIC["evidence"],
    "analysis": ANALYTIC_RUBRIC["analysis"],
    "grammar": ANALYTIC_RUBRIC["grammar"],
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
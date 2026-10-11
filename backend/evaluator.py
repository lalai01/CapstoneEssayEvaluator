# backend/evaluator.py
import re
import json
import os
import difflib
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


def _accept_correction(raw_text, corrected, source):
    """
    Shared guard for LLM OCR correction.
    Returns the corrected text if it passes both guards, else None.

    Guards:
      - length ratio between 0.85 and 1.15 (rejects summarization and padding)
      - difflib similarity ratio >= 0.85 (rejects rewrites)
    """
    if not corrected or not corrected.strip():
        print(f"[{source}] rejected: empty output")
        return None

    length_ratio = len(corrected) / max(1, len(raw_text))
    if not (0.85 <= length_ratio <= 1.15):
        print(f"[{source}] rejected: length ratio {length_ratio:.2f} "
              f"({len(raw_text)} -> {len(corrected)})")
        return None

    similarity = difflib.SequenceMatcher(
        None, raw_text.lower(), corrected.lower()
    ).ratio()
    if similarity < 0.85:
        print(f"[{source}] rejected: similarity {similarity:.2f} < 0.85 "
              f"(it rewrote too much)")
        return None

    print(f"[{source}] accepted: {len(raw_text)} -> {len(corrected)} chars, "
          f"similarity {similarity:.2f}")
    return corrected


HOLISTIC_RUBRIC = {
    5: "🌟 Excellent (5/5) – Clear thesis, strong organization, compelling arguments, and virtually no errors. The essay demonstrates mastery of the topic.",
    4: "👍 Good (4/5) – Clear main idea, well-organized, with minor errors that do not impede understanding. Arguments are solid but could be more developed.",
    3: "📝 Satisfactory (3/5) – Understandable but with some weaknesses in clarity, organization, or support. Several grammatical issues or vague points.",
    2: "⚠️ Needs Improvement (2/5) – Unclear thesis, disorganized, or lacking sufficient evidence. Frequent errors make reading difficult.",
    1: "❌ Poor (1/5) – Hard to follow, no clear structure, many errors. The essay fails to address the topic adequately."
}

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


DEFAULT_LABELS = {
    "main_statement": "Main Statement",
    "organization": "Organization",
    "evidence": "Evidence",
    "analysis": "Analysis",
    "grammar": "Grammar",
}

DEFAULT_WEIGHTS = {
    "main_statement": 1.0,
    "organization": 1.0,
    "evidence": 1.5,
    "analysis": 1.5,
    "grammar": 1.0,
}

MAX_SCORE_LIMIT = 10


def _to_number(value, default, cast=float):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def normalize_rubric(raw):
    """
    Convert any stored rubric into the canonical shape:

        {key: {"label": str, "max_score": int, "weight": float,
               "levels": {int: descriptor}}}

    Accepts both the legacy shape ({key: {"4": "...", "3": "..."}}) and the
    admin-configurable shape ({key: {"label", "max_score", "weight", "levels"}}).
    """
    if not isinstance(raw, dict) or not raw:
        raw = ANALYTIC_RUBRIC

    normalized = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue

        if "levels" in value or "max_score" in value:
            levels_raw = value.get("levels") or {}
            label = value.get("label") or DEFAULT_LABELS.get(key) or key.replace("_", " ").title()
            weight = _to_number(value.get("weight"), DEFAULT_WEIGHTS.get(key, 1.0))
            max_score = _to_number(value.get("max_score"), None, int)
        else:
            levels_raw = value
            label = DEFAULT_LABELS.get(key) or key.replace("_", " ").title()
            weight = DEFAULT_WEIGHTS.get(key, 1.0)
            max_score = None

        levels = {}
        for level, text in (levels_raw or {}).items():
            lvl = _to_number(level, None, int)
            if lvl is not None and isinstance(text, str) and text.strip():
                levels[lvl] = text.strip()

        if max_score is None:
            max_score = max(levels) if levels else 4
        max_score = max(2, min(MAX_SCORE_LIMIT, max_score))
        levels = {lvl: text for lvl, text in levels.items() if 1 <= lvl <= max_score}

        normalized[key] = {
            "label": label,
            "max_score": max_score,
            "weight": max(0.0, weight),
            "levels": levels,
        }

    return normalized or normalize_rubric(ANALYTIC_RUBRIC)


def rubric_meta(rubric):
    """Criteria list sent to the frontend alongside scores."""
    return [
        {"key": key, "label": c["label"], "max_score": c["max_score"], "weight": c["weight"]}
        for key, c in normalize_rubric(rubric).items()
    ]


def descriptor_for(criterion, score):
    """Descriptor for the exact level, else the nearest defined level below it."""
    levels = criterion.get("levels") or {}
    for lvl in range(int(score), 0, -1):
        if lvl in levels:
            return levels[lvl]
    return ""


def get_active_rubric(user_id=None):
    """
    The rubric admins configured in the Rubric Manager applies system-wide:
    the most recently created active rubric wins. Falls back to the built-in
    analytic rubric. `user_id` is kept for call-site compatibility.
    """
    try:
        from supabase_client import supabase, supabase_admin
        client = supabase_admin or supabase
        if not client:
            return ANALYTIC_RUBRIC

        result = client.table("rubrics") \
            .select("criteria") \
            .eq("is_active", True) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()

        if result.data and result.data[0].get("criteria"):
            return result.data[0]["criteria"]
    except Exception as e:
        print(f"Rubric fetch failed: {e}")

    return ANALYTIC_RUBRIC


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
    paragraphs = [p for p in essay_text.split("\n\n") if p.strip()]
    word_count = len(essay_text.split())
    sentences = max(1, len(re.findall(r'[.!?]+', essay_text)))

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


def _is_default_rubric_shape(criteria):
    return set(criteria) == set(ANALYTIC_RUBRIC) and all(
        c["max_score"] == 4 for c in criteria.values()
    )


def ai_score_all_criteria(essay_text, rubric=None, model="llama3.2:3b"):
    criteria = normalize_rubric(rubric)

    ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")

    rubric_block = ""
    for key, c in criteria.items():
        rubric_block += f"\n[{key}] {c['label']} — score 1 to {c['max_score']}\n"
        for level in sorted(c["levels"], reverse=True):
            rubric_block += f"  {level}: {c['levels'][level]}\n"

    facts = _essay_facts(essay_text)

    system_msg = (
        "You are a strict, experienced essay examiner and JSON API. "
        "You score decisively and do not default to middle values. "
        "You use each criterion's full score range based on concrete evidence in the essay. "
        "You differentiate scores across criteria when warranted. "
        "You return ONLY a single valid JSON object and nothing else — no prose, no markdown, no code fences."
    )

    hard_rules = []
    if "organization" in criteria:
        hard_rules.append(
            f"- If paragraph_count >= 3 AND transition_count >= 3, organization MUST be "
            f"{criteria['organization']['max_score']}."
        )
    if "evidence" in criteria:
        hard_rules.append(
            f"- If citation_count >= 2, evidence MUST be {criteria['evidence']['max_score']}."
        )
    if "main_statement" in criteria:
        hard_rules.append(
            "- If the thesis appears in the first paragraph AND is restated or\n"
            "  reinforced in the conclusion, main_statement MUST be "
            f"{criteria['main_statement']['max_score']}."
        )
    hard_rules_block = (
        "HARD RULES (must be obeyed):\n" + "\n".join(hard_rules) + "\n"
        if hard_rules else ""
    )

    calibration_block = """CALIBRATION EXAMPLES:

Example A - Essay with a clear thesis, three body paragraphs, and 2+ cited
sources:
  main_statement: 4, organization: 4, evidence: 4, analysis: 3 or 4, grammar: 3 or 4

Example B - Short single-paragraph essay with a thesis but only one example
and no cited source:
  main_statement: 3, organization: 2, evidence: 2, analysis: 2, grammar: 3 or 4

Example C - Multi-paragraph essay with a clear thesis but no concrete evidence:
  main_statement: 3, organization: 3, evidence: 1 or 2, analysis: 2, grammar: 3

Example D - Disorganized essay with no thesis and no evidence:
  main_statement: 1, organization: 1, evidence: 1, analysis: 1, grammar: 2
""" if _is_default_rubric_shape(criteria) else ""

    score_lines = "\n".join(
        f'  "{key}": <int 1-{c["max_score"]}>,' for key, c in criteria.items()
    )
    feedback_lines = ",\n".join(
        f'    "{key}": "<2-3 sentences explaining this score>"' for key in criteria
    )

    user_msg = f"""Carefully evaluate the essay below against the rubric.

RUBRIC (use these exact descriptors to justify scores; each criterion has its own range):

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
- Use a criterion's maximum only when the essay clearly satisfies its top-level descriptor.
- Use 1 only when the criterion is essentially missing or broken.
- Never exceed a criterion's maximum score.
- If a criterion is strong and another is weak, the scores MUST differ.

{hard_rules_block}
{calibration_block}
Essay:
\"\"\"
{essay_text[:4000]}
\"\"\"

Return ONLY this JSON object (no explanation, no markdown):

{{
{score_lines}
  "criterion_feedback": {{
{feedback_lines}
  }},
  "grammar_issues": [
    {{
      "original":    "<text quoted from the essay with an actual error>",
      "correction":  "<corrected version>",
      "explanation": "<one-sentence reason>"
    }}
  ],
  "recommendations": [
    {{
      "title":      "<short label>",
      "issue":      "<what is weak in the essay>",
      "suggestion": "<concrete, actionable fix that references a specific part of the essay>"
    }}
  ]
}}

RULES:
- If there are no grammar errors, use "grammar_issues": [].
- Never include a grammar_issues entry where original equals correction.
- Provide 3–5 recommendations. Each must reference a specific part of the essay.
- Every criterion_feedback value must be a plain string of 2-3 sentences.
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
            timeout=120,
        )
        if response.status_code != 200:
            print(f"Ollama returned status {response.status_code}")
            return None

        content = response.json().get("message", {}).get("content", "").strip()

        if content.startswith("```"):
            content = content.strip("`").replace("json", "", 1).strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            if start == -1:
                raise
            depth = 0
            in_string = False
            escape = False
            end = -1
            for i in range(start, len(content)):
                c = content[i]
                if escape:
                    escape = False
                    continue
                if c == "\\":
                    escape = True
                    continue
                if c == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end == -1:
                raise
            parsed = json.loads(content[start:end + 1])

        result = {}
        for key, c in criteria.items():
            if key not in parsed:
                print(f"Missing key in AI response: {key}")
                return None
            score = int(parsed[key])
            if score < 1 or score > c["max_score"]:
                print(f"Out-of-range score for {key}: {score} (max {c['max_score']})")
                return None
            result[key] = score

        cf = parsed.get("criterion_feedback")
        result["criterion_feedback"] = cf if isinstance(cf, dict) else {}

        gi = parsed.get("grammar_issues")
        result["grammar_issues"] = gi if isinstance(gi, list) else []

        recs = parsed.get("recommendations")
        result["recommendations"] = recs if isinstance(recs, list) else []

        result["criteria_meta"] = rubric_meta(criteria)
        return result

    except json.JSONDecodeError as e:
        print(f"AI scoring JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"AI scoring failed: {e}")
        return None


def _heuristic_scores(essay_text, analysis):
    lower = essay_text.lower()
    paragraphs = [p for p in essay_text.split("\n\n") if p.strip()]
    para_count = len(paragraphs)
    word_count = analysis["word_count"]
    transitions = analysis["transition_count"]
    vocab = analysis["vocabulary_richness"]
    avg_sl = analysis["avg_sentence_length"]

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
        "criterion_feedback": {},
        "grammar_issues": [],
        "recommendations": [],
    }


def _rescale(score, from_max, to_max):
    """Map a 1..from_max score onto 1..to_max."""
    if from_max <= 1:
        return 1
    return max(1, min(to_max, round(1 + (score - 1) * (to_max - 1) / (from_max - 1))))


def _heuristic_scores_for_rubric(essay_text, analysis, criteria):
    """
    Heuristics only know the five built-in criteria (1-4). Known criteria are
    rescaled to the admin's range; custom criteria get the scaled mean.
    """
    base = _heuristic_scores(essay_text, analysis)
    base_numeric = [base[k] for k in DEFAULT_LABELS]
    base_mean = sum(base_numeric) / len(base_numeric)

    result = {}
    for key, c in criteria.items():
        raw = base.get(key, base_mean)
        result[key] = _rescale(raw, 4, c["max_score"])
    result["criterion_feedback"] = {}
    result["grammar_issues"] = []
    result["recommendations"] = []
    result["criteria_meta"] = rubric_meta(criteria)
    return result


def calculate_analytic_scores(essay_text, analysis, rubric=None):
    criteria = normalize_rubric(rubric)
    ai = ai_score_all_criteria(essay_text, rubric=criteria)
    if ai is not None:
        print("✅ AI scoring succeeded")
        return ai
    print("⚠️ AI scoring failed, using heuristics")
    return _heuristic_scores_for_rubric(essay_text, analysis, criteria)


def weighted_fraction(scores, rubric=None):
    """Weighted share of the maximum (0.0–1.0) across the rubric's criteria."""
    criteria = normalize_rubric(rubric)
    weighted_sum = total_weight = 0.0
    for key, c in criteria.items():
        v = scores.get(key)
        if isinstance(v, (int, float)) and c["max_score"] > 1:
            weighted_sum += c["weight"] * (v - 1) / (c["max_score"] - 1)
            total_weight += c["weight"]
    return weighted_sum / total_weight if total_weight else 0.0


def calculate_holistic_score(essay_text, analysis, rubric=None):
    analytic = calculate_analytic_scores(essay_text, analysis, rubric=rubric)
    if not any(isinstance(v, (int, float)) for v in analytic.values()):
        return 3
    avg = 1 + 3 * weighted_fraction(analytic, rubric)  # 1.0 – 4.0 equivalent
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


CRITERION_ICONS = {
    "main_statement": "🧠",
    "organization": "🧱",
    "evidence": "📚",
    "analysis": "🔍",
    "grammar": "📝",
}


def _criterion_tips(key, essay_text, analysis):
    if key == "main_statement":
        return ["Strengthen this by clearly stating your position in the first paragraph "
                "(e.g., 'This essay argues that…')."]
    if key == "organization":
        tips = []
        para_count = len([p for p in essay_text.split('\n\n') if p.strip()])
        if para_count < 3:
            tips.append("Break your argument into at least three paragraphs: introduction, body, conclusion.")
        if analysis['transition_count'] == 0:
            tips.append("Use transitions (e.g., 'Furthermore', 'Therefore', 'In conclusion').")
        return tips
    if key == "evidence":
        return ["Include at least one more concrete example, statistic, or reference "
                "(e.g., 'According to a 2023 study…')."]
    if key == "analysis":
        return ["After each piece of evidence, add a sentence that explains what it means "
                "and how it supports your thesis."]
    return []


def generate_rule_based_analytic_feedback(essay_text, scores, analysis, rag_context="", rubric=None):
    criteria = normalize_rubric(rubric)
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

    for key, c in criteria.items():
        score = scores.get(key)
        if not isinstance(score, (int, float)):
            continue
        icon = CRITERION_ICONS.get(key, "📌")
        feedback.append(f"{icon} {c['label'].upper()} (Score: {score}/{c['max_score']})")
        descriptor = descriptor_for(c, score)
        if descriptor:
            feedback.append(f"- {descriptor}")
        if score < c["max_score"]:
            for tip in _criterion_tips(key, essay_text, analysis):
                feedback.append(f"- 💡 {tip}")
        if key == "grammar":
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

    avg = 1 + 3 * weighted_fraction(scores, criteria)  # 1.0 – 4.0 equivalent
    if avg >= 3.75:
        feedback.append("✅ Overall: This essay is at the top of the rubric across all criteria. "
                        "For further polish, consider refining transitions or adding a counter-argument.")
    elif avg >= 3.25:
        feedback.append("✅ Overall: This is a strong essay. Focus on refining the weakest criterion to reach the top of the rubric.")
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


def ai_correct_ocr_text(raw_text, provider="deepseek"):
    """
    Correct character-level OCR misreads in printed-text OCR output.

    This function is BOUNDED. It must not rewrite the text.

    Rules:
      - Only replace tokens that are clearly non-words or visual misreads.
      - Preserve the writer's spelling, grammar, and punctuation exactly.
      - Preserve line breaks and blank lines.
      - Reject the correction if length ratio or similarity fail the guard.

    provider: "deepseek" (default) or "ollama"
    """
    if not raw_text or len(raw_text.strip()) < 10:
        return raw_text

    system_msg = (
        "You are a transcription corrector for a printed-text OCR system. "
        "The OCR engine produced a rough read of a printed document. "
        "Your ONLY job is to fix characters the OCR clearly misread.\n"
        "\n"
        "HARD RULES:\n"
        "1. Output the SAME WORDS that were printed. Do NOT correct grammar.\n"
        "2. Do NOT fix spelling, punctuation, or sentence structure.\n"
        "3. Do NOT add words that aren't there. Do NOT remove words that are.\n"
        "4. Do NOT paraphrase or change synonyms.\n"
        "5. Only replace a token when it is clearly a non-word or visual misread.\n"
        "6. If a token is already a real English word, leave it exactly as-is.\n"
        "7. Preserve every line break and blank line.\n"
        "8. Return only the corrected text. No commentary, no preamble."
    )

    user_msg = (
        "OCR output:\n"
        f"\"\"\"\n{raw_text[:4000]}\n\"\"\"\n\n"
        "Corrected transcription (same words, only fix clear misreads):"
    )

    if provider == "deepseek":
        try:
            from ai_models import call_deepseek
            result = call_deepseek(system_msg, user_msg, model="deepseek-chat")
            corrected = (result.get("text") or "").strip()
            accepted = _accept_correction(raw_text, corrected, "ocr-correct")
            if accepted:
                return accepted
        except Exception as e:
            print(f"[ocr-correct] deepseek failed, falling back to ollama: {e}")

    ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")
    try:
        response = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": "gemma2:2b",
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=45,
        )
        if response.status_code == 200:
            corrected = response.json().get(
                "message", {}
            ).get("content", "").strip()
            accepted = _accept_correction(raw_text, corrected, "ocr-correct")
            if accepted:
                return accepted
    except Exception as e:
        print(f"[ocr-correct] ollama failed: {e}")

    return raw_text


def ai_correct_handwriting_text(raw_text, provider="deepseek"):
    """
    Correct character-level OCR misreads in handwriting (TrOCR) output.

    This function is BOUNDED. It must not rewrite the text.

    Rules:
      - Only replace tokens that are clearly non-words or visual misreads.
      - Preserve the writer's spelling, grammar, and punctuation exactly.
      - Preserve line breaks and blank lines.
      - Reject the correction if length ratio or similarity fail the guard.

    provider: "deepseek" (default) or "ollama"
    """
    if not raw_text or len(raw_text.strip()) < 20:
        return raw_text

    print(f"[handwriting-correct] calling with {len(raw_text)} chars, "
          f"provider={provider}")

    system_msg = (
        "You are a transcription corrector for a handwritten OCR system. "
        "The OCR engine produced a rough read of a handwritten page. "
        "Your ONLY job is to fix characters the OCR clearly misread.\n"
        "\n"
        "HARD RULES:\n"
        "1. Output the SAME WORDS the writer actually wrote. Do NOT correct grammar.\n"
        "   If the writer wrote 'become worsen', output 'become worsen'.\n"
        "2. Do NOT fix spelling, punctuation, or sentence structure.\n"
        "3. Do NOT add words that aren't there. Do NOT remove words that are.\n"
        "4. Do NOT paraphrase. Do NOT summarize. Do NOT change synonyms.\n"
        "5. Only replace a token when it is clearly a non-word or visual misread.\n"
        "6. If a token is already a real English word, leave it exactly as-is.\n"
        "7. Preserve every line break and blank line.\n"
        "8. Return only the corrected text. No commentary, no preamble."
    )

    user_msg = (
        "OCR output:\n"
        f"\"\"\"\n{raw_text[:4000]}\n\"\"\"\n\n"
        "Corrected transcription (same words, only fix clear misreads):"
    )

    if provider == "deepseek":
        try:
            from ai_models import call_deepseek
            result = call_deepseek(system_msg, user_msg, model="deepseek-chat")
            corrected = (result.get("text") or "").strip()
            accepted = _accept_correction(
                raw_text, corrected, "handwriting-correct"
            )
            if accepted:
                return accepted
        except Exception as e:
            print(f"[handwriting-correct] deepseek failed, falling back to "
                  f"ollama: {e}")

    ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")
    try:
        response = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": "llama3.2:3b",
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=90,
        )
        print(f"[handwriting-correct] ollama status={response.status_code}")
        if response.status_code == 200:
            corrected = response.json().get(
                "message", {}
            ).get("content", "").strip()
            accepted = _accept_correction(
                raw_text, corrected, "handwriting-correct"
            )
            if accepted:
                return accepted
    except Exception as e:
        print(f"[handwriting-correct] ollama failed: {e}")

    print("[handwriting-correct] returning raw (no change)")
    return raw_text


def enhance_feedback_with_ai(essay_text, scores, analysis, rule_feedback, facts=None, rubric=None):
    ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")
    model = "llama3.2:3b"

    if facts is None:
        facts = _essay_facts(essay_text)

    if 'holistic_score' in scores:
        score_info = f"Holistic Score: {scores['holistic_score']}/5\nDescription: {scores.get('level_description', '')}"
    else:
        score_info = "\n".join(
            f"{c['label']}: {scores.get(key, 'N/A')}/{c['max_score']}"
            for key, c in normalize_rubric(rubric).items()
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


def evaluate_essay(essay_text, evaluation_type="analytic", use_rag=True, rubric=None):
    is_valid, error_msg = is_valid_essay(essay_text)

    if not is_valid:
        if evaluation_type == "holistic":
            return (
                {"holistic_score": 0, "level_description": error_msg},
                f"⚠️ Invalid Input: {error_msg}",
            )
        return (
            {
                "main_statement": 0,
                "organization": 0,
                "evidence": 0,
                "analysis": 0,
                "grammar": 0,
                "criterion_feedback": {},
                "grammar_issues": [],
                "recommendations": [],
            },
            f"⚠️ Invalid Input: {error_msg}",
        )

    analysis = analyze_essay_content(essay_text)
    facts = _essay_facts(essay_text)

    rag_context = ""
    if use_rag:
        try:
            rag_context = get_similar_essay_context(essay_text)
        except Exception as e:
            print(f"RAG error: {e}")

    if evaluation_type == "holistic":
        score = calculate_holistic_score(essay_text, analysis, rubric=rubric)
        scores = {
            "holistic_score": score,
            "level_description": HOLISTIC_RUBRIC[score],
        }
        rule_feedback = generate_rule_based_holistic_feedback(
            essay_text, score, analysis, rag_context
        )
    else:
        scores = calculate_analytic_scores(essay_text, analysis, rubric=rubric)
        rule_feedback = generate_rule_based_analytic_feedback(
            essay_text, scores, analysis, rag_context, rubric=rubric
        )

    feedback = enhance_feedback_with_ai(
        essay_text, scores, analysis, rule_feedback, facts, rubric=rubric
    )

    return scores, feedback


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
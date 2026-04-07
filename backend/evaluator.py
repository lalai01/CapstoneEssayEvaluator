import re
from rag import get_similar_essay_context

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

def calculate_grammar_score(essay_text, analysis):
    score = 85
    if analysis['avg_sentence_length'] > 25:
        score -= 5
    elif analysis['avg_sentence_length'] < 8:
        score -= 5
    if analysis['vocabulary_richness'] < 0.4:
        score -= 5
    elif analysis['vocabulary_richness'] > 0.7:
        score += 3
    if re.search(r'\s+[,.!?]', essay_text):
        score -= 2
    if re.search(r'[,.!?]{2,}', essay_text):
        score -= 3
    return max(60, min(98, score))

def calculate_coherence_score(essay_text, analysis):
    score = 80
    paragraphs = essay_text.split('\n\n')
    if len(paragraphs) > 1:
        score += 5
    if analysis['transition_count'] > 3:
        score += 5
    elif analysis['transition_count'] == 0:
        score -= 5
    lower_text = essay_text.lower()
    intro_indicators = ['introduction', 'first', 'begin', 'start', 'purpose']
    conc_indicators = ['conclusion', 'summary', 'finally', 'in conclusion', 'to summarize']
    if any(word in lower_text[:500] for word in intro_indicators):
        score += 3
    if any(word in lower_text[-500:] for word in conc_indicators):
        score += 3
    return max(60, min(98, score))

def calculate_content_score(essay_text, analysis):
    score = 75
    if analysis['word_count'] > 500:
        score += 10
    elif analysis['word_count'] > 300:
        score += 5
    elif analysis['word_count'] < 100:
        score -= 10
    if analysis['vocabulary_richness'] > 0.6:
        score += 5
    evidence_indicators = ['example', 'for instance', 'such as', 'because', 'research', 'study', 'data']
    evidence_count = sum(1 for indicator in evidence_indicators if indicator in essay_text.lower())
    score += min(10, evidence_count * 2)
    return max(60, min(98, score))

def generate_dynamic_feedback(essay_text, scores, analysis, evaluation_type, rag_context=""):
    feedback = []
    if rag_context:
        feedback.append("📚 **Retrieved from similar past evaluations (RAG)**")
        feedback.append(rag_context)
        feedback.append("")

    feedback.append("📊 **ESSAY ANALYSIS**")
    feedback.append(f"Your essay contains {analysis['word_count']} words across {analysis['sentence_count']} sentences.")
    feedback.append(f"Average sentence length: {analysis['avg_sentence_length']:.1f} words.")
    feedback.append("")
    
    # Grammar
    feedback.append(f"📝 **GRAMMAR ANALYSIS (Score: {scores['grammar']})**")
    if scores['grammar'] >= 90:
        feedback.append("- Excellent grammar! Your writing is clear and correct.")
        feedback.append("✅ **Advanced tips:**")
        feedback.append("• Experiment with more complex sentence structures")
        feedback.append("• Consider using stylistic devices for emphasis")
    elif scores['grammar'] >= 70:
        feedback.append("- Good basic grammar with room for refinement.")
        if analysis['transition_count'] < 3:
            feedback.append("- Adding more transition words would improve flow.")
        feedback.append("✅ **Refinement suggestions:**")
        feedback.append("• Vary your sentence structure for better rhythm")
        feedback.append("• Check for consistent tense usage throughout")
    else:
        grammar_issues = []
        if analysis['avg_sentence_length'] > 25:
            grammar_issues.append("- Some sentences are quite long. Consider breaking them into shorter, clearer sentences.")
        if analysis['vocabulary_richness'] < 0.4:
            grammar_issues.append("- Limited vocabulary detected. Try using more varied word choices.")
        if re.search(r'[,.!?][A-Za-z]', essay_text):
            grammar_issues.append("- Missing spaces after punctuation in some places.")
        if grammar_issues:
            feedback.extend(grammar_issues)
        else:
            feedback.append("- Several grammatical patterns need attention. Consider reviewing basic grammar rules.")
        feedback.append("")
        feedback.append("✅ **Suggestions for improvement:**")
        feedback.append("• Read your essay aloud to catch awkward phrasing")
        feedback.append("• Use grammar checking tools to identify specific errors")
        feedback.append("• Review subject-verb agreement in complex sentences")
    
    # Coherence
    feedback.append("")
    feedback.append(f"🔄 **COHERENCE ANALYSIS (Score: {scores['coherence']})**")
    paragraphs = essay_text.split('\n\n')
    if len(paragraphs) > 1:
        feedback.append(f"- Your essay has {len(paragraphs)} paragraphs, which is good for organization.")
    else:
        feedback.append("- Consider breaking your essay into paragraphs for better organization.")
    if analysis['transition_count'] > 3:
        feedback.append(f"- Good use of transition words ({analysis['transition_count']} instances).")
    elif analysis['transition_count'] > 0:
        feedback.append(f"- You used {analysis['transition_count']} transition words – adding a few more would improve flow.")
    else:
        feedback.append("- Adding transition words (however, therefore, moreover) would improve logical flow.")
    
    # Content
    feedback.append("")
    feedback.append(f"📚 **CONTENT ANALYSIS (Score: {scores['content']})**")
    lower_text = essay_text.lower()
    intro_indicators = ['introduction', 'first', 'begin', 'start', 'purpose', 'this essay', 'in this essay']
    conc_indicators = ['conclusion', 'summary', 'finally', 'in conclusion', 'to summarize', 'overall']
    has_intro = any(word in lower_text[:500] for word in intro_indicators)
    has_conclusion = any(word in lower_text[-500:] for word in conc_indicators)
    evidence_words = ['example', 'for instance', 'such as', 'because', 'research', 'study', 'data', 'shows', 'demonstrates']
    evidence_count = sum(1 for word in evidence_words if word in lower_text)
    
    if scores['content'] >= 90:
        feedback.append("- Outstanding content! Your essay is well-developed and insightful.")
        if has_intro:
            feedback.append("- Clear introduction that sets up your argument.")
        else:
            feedback.append("- Consider making your introduction more explicit (state your main thesis).")
        if has_conclusion:
            feedback.append("- Strong conclusion that reinforces your main points.")
        else:
            feedback.append("- Add a brief conclusion to leave a lasting impression.")
        if evidence_count >= 3:
            feedback.append(f"- Excellent use of evidence ({evidence_count} examples).")
        else:
            feedback.append("- To reach perfection, include more specific data or real-world examples.")
    elif scores['content'] >= 75:
        feedback.append("- Good content with room for development.")
        if not has_intro:
            feedback.append("- Add a clear introduction stating your main idea.")
        else:
            feedback.append("- Your introduction is present – try to make it more engaging.")
        if not has_conclusion:
            feedback.append("- Add a conclusion that summarizes your main points.")
        else:
            feedback.append("- Your conclusion is good; consider adding a final thought or call to action.")
        if evidence_count < 2:
            feedback.append("- Include more specific examples or evidence to support your arguments.")
        else:
            feedback.append(f"- Good use of evidence ({evidence_count} instances).")
    else:
        feedback.append("- Content needs significant improvement.")
        if not has_intro:
            feedback.append("- Start with an introduction that tells the reader what your essay is about.")
        if not has_conclusion:
            feedback.append("- End with a conclusion that restates your main idea.")
        if evidence_count == 0:
            feedback.append("- Add examples, facts, or data to support your claims.")
        else:
            feedback.append("- Your evidence is limited – try to elaborate further.")
    
    if len(essay_text) > 100 and scores['content'] < 90:
        sentences = re.split(r'[.!?]+', essay_text)
        sample_sentences = [s.strip() for s in sentences if len(s.strip().split()) > 5]
        if sample_sentences:
            sample = sample_sentences[0][:100] + "..." if len(sample_sentences[0]) > 100 else sample_sentences[0]
            feedback.append("")
            feedback.append("✨ **SAMPLE IMPROVEMENT BASED ON YOUR WRITING:**")
            feedback.append(f"Original: \"{sample}\"")
            feedback.append("Enhanced version would include more specific details and varied vocabulary.")
            feedback.append("Tip: Try to replace general words with more precise terminology.")
    
    return "\n".join(feedback)

def evaluate_essay(essay_text, evaluation_type="analytic", use_rag=True):
    analysis = analyze_essay_content(essay_text)
    scores = {
        "grammar": calculate_grammar_score(essay_text, analysis),
        "coherence": calculate_coherence_score(essay_text, analysis),
        "content": calculate_content_score(essay_text, analysis)
    }
    rag_context = ""
    if use_rag:
        try:
            rag_context = get_similar_essay_context(essay_text)
        except Exception as e:
            print(f"RAG error: {e}")
    feedback = generate_dynamic_feedback(essay_text, scores, analysis, evaluation_type, rag_context)
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
1. **Input Methods**:
   - Type or paste your essay directly
   - Upload an image (JPG, PNG, etc.) of handwritten/printed essay
   - Upload a PDF file (all pages will be processed)
   
2. **For Best OCR Results**:
   - Ensure good lighting when photographing
   - Use clear, legible handwriting
   - Avoid shadows and glare
   - For PDFs, ensure they are text-based or high-quality scans
   
3. **Review Process**:
   - Check extracted text for accuracy
   - Make manual corrections if needed
   - Then click Evaluate for personalized feedback"""
}
import re

def analyze_essay_content(essay_text):
    """Dynamically analyze essay content without static text"""
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
    """Dynamically calculate grammar score based on content"""
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
    """Dynamically calculate coherence score"""
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
    """Dynamically calculate content score"""
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

def generate_dynamic_feedback(essay_text, scores, analysis, evaluation_type):
    """Generate completely dynamic feedback based on actual essay content"""
    
    feedback = []
    
    feedback.append("📊 **ESSAY ANALYSIS**")
    feedback.append(f"Your essay contains {analysis['word_count']} words across {analysis['sentence_count']} sentences.")
    feedback.append(f"Average sentence length: {analysis['avg_sentence_length']:.1f} words.")
    
    feedback.append("")
    feedback.append(f"📝 **GRAMMAR ANALYSIS (Score: {scores['grammar']})**")
    
    if scores['grammar'] < 70:
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
        
    elif scores['grammar'] < 85:
        feedback.append("- Good basic grammar with room for refinement.")
        if analysis['transition_count'] < 3:
            feedback.append("- Adding more transition words would improve flow.")
        feedback.append("")
        feedback.append("✅ **Refinement suggestions:**")
        feedback.append("• Vary your sentence structure for better rhythm")
        feedback.append("• Check for consistent tense usage throughout")
    else:
        feedback.append("- Excellent grammar! Your writing is clear and correct.")
        feedback.append("")
        feedback.append("✅ **Advanced tips:**")
        feedback.append("• Experiment with more complex sentence structures")
        feedback.append("• Consider using stylistic devices for emphasis")
    
    feedback.append("")
    feedback.append(f"🔄 **COHERENCE ANALYSIS (Score: {scores['coherence']})**")
    
    paragraphs = essay_text.split('\n\n')
    if len(paragraphs) > 1:
        feedback.append(f"- Your essay has {len(paragraphs)} paragraphs, which is good for organization.")
    else:
        feedback.append("- Consider breaking your essay into paragraphs for better organization.")
    
    if analysis['transition_count'] > 0:
        feedback.append(f"- You used {analysis['transition_count']} transition words, which helps with flow.")
    else:
        feedback.append("- Adding transition words (however, therefore, moreover) would improve logical flow.")
    
    feedback.append("")
    feedback.append(f"📚 **CONTENT ANALYSIS (Score: {scores['content']})**")
    
    words = essay_text.lower().split()
    if any(word in ['introduction', 'first', 'begin'] for word in words[:50]):
        feedback.append("- Good introduction detected.")
    else:
        feedback.append("- Consider adding a clearer introduction that states your main idea.")
    
    if any(word in ['conclusion', 'summary', 'finally'] for word in words[-50:]):
        feedback.append("- Good conclusion detected that wraps up your ideas.")
    else:
        feedback.append("- Add a conclusion that summarizes your main points.")
    
    evidence_words = ['example', 'because', 'since', 'research', 'data', 'study', 'shows']
    evidence_count = sum(1 for word in evidence_words if word in essay_text.lower())
    if evidence_count > 3:
        feedback.append(f"- Good use of evidence and supporting details ({evidence_count} instances).")
    else:
        feedback.append("- Add more specific examples and evidence to support your arguments.")
    
    if len(essay_text) > 100:
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

def evaluate_essay(essay_text, evaluation_type="analytic"):
    """Main evaluation function - completely dynamic"""
    
    analysis = analyze_essay_content(essay_text)
    
    scores = {
        "grammar": calculate_grammar_score(essay_text, analysis),
        "coherence": calculate_coherence_score(essay_text, analysis),
        "content": calculate_content_score(essay_text, analysis)
    }
    
    feedback = generate_dynamic_feedback(essay_text, scores, analysis, evaluation_type)
    
    return scores, feedback

RUBRIC = {
    "grammar": "Correctness of sentence structure, punctuation, and spelling.",
    "coherence": "Logical flow, transitions, and organization of ideas.",
    "content": "Depth, relevance, evidence, and quality of arguments."
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
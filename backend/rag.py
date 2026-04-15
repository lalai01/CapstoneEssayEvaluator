import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from supabase_client import supabase

# Global cache
_past_essays = []
_past_feedbacks = []
_vectorizer = None

def load_past_data_from_db():
    """
    Fetch essays and feedback from both knowledge_base (accepted evaluations)
    and learning_feedback (teacher overrides).
    """
    essays = []
    feedbacks = []
    
    if not supabase:
        return essays, feedbacks

    # 1. Accepted evaluations from knowledge_base
    kb_result = supabase.table("knowledge_base") \
        .select("essay, teacher_feedback, feedback") \
        .eq("accepted", True) \
        .execute()
    
    for item in kb_result.data:
        fb = item.get("teacher_feedback") or item.get("feedback")
        if fb and item.get("essay"):
            essays.append(item["essay"])
            feedbacks.append(fb)

    # 2. Teacher overrides from learning_feedback (rejected evaluations)
    lf_result = supabase.table("learning_feedback") \
        .select("original_essay, teacher_feedback") \
        .execute()
    
    for item in lf_result.data:
        fb = item.get("teacher_feedback")
        essay = item.get("original_essay")
        if fb and essay:
            essays.append(essay)
            feedbacks.append(f"[Teacher Override] {fb}")

    return essays, feedbacks

def get_similar_essay_context(essay_text, top_k=2):
    """Return a string with feedback from the most similar past essays (including overrides)."""
    global _past_essays, _past_feedbacks, _vectorizer
    
    if not _past_essays:
        _past_essays, _past_feedbacks = load_past_data_from_db()
        if not _past_essays:
            return ""

    # Build corpus: past essays + current essay
    corpus = _past_essays + [essay_text]
    _vectorizer = TfidfVectorizer(stop_words='english').fit(corpus)
    tfidf_matrix = _vectorizer.transform(corpus)
    
    # Similarity between current (last) and all past
    similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    context_parts = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Only include reasonably similar essays
            context_parts.append(f"[Similar essay feedback]:\n{_past_feedbacks[idx]}")
    
    if context_parts:
        return "\n\n--- Similar past evaluation feedback (RAG) ---\n" + "\n\n".join(context_parts)
    return ""
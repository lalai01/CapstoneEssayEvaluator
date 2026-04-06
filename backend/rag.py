import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from supabase_client import supabase

# Global cache
_past_essays = []
_past_feedbacks = []
_vectorizer = None

def load_past_essays_from_db():
    """Fetch accepted evaluations that have teacher_feedback or original feedback."""
    result = supabase.table("knowledge_base").select("essay, teacher_feedback, feedback").eq("accepted", True).execute()
    essays = []
    feedbacks = []
    for item in result.data:
        # Prefer teacher_feedback if exists, otherwise original feedback
        fb = item.get("teacher_feedback") or item.get("feedback")
        if fb and item.get("essay"):
            essays.append(item["essay"])
            feedbacks.append(fb)
    return essays, feedbacks

def get_similar_essay_context(essay_text, top_k=2):
    """Return a string with feedback from the most similar past essays."""
    global _past_essays, _past_feedbacks, _vectorizer
    if not _past_essays:
        _past_essays, _past_feedbacks = load_past_essays_from_db()
        if not _past_essays:
            return ""
    # Build corpus: past essays + current essay
    corpus = _past_essays + [essay_text]
    _vectorizer = TfidfVectorizer(stop_words='english').fit(corpus)
    tfidf_matrix = _vectorizer.transform(corpus)
    # Similarity between current (last) and all past
    similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    context = "\n\n--- Similar past evaluation feedback (RAG) ---\n"
    added = False
    for idx in top_indices:
        if similarities[idx] > 0.1:
            context += f"\n[Similar essay feedback]:\n{_past_feedbacks[idx]}\n"
            added = True
    return context if added else ""
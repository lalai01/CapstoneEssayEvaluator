import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from supabase_client import supabase

_past_essays = []
_past_feedbacks = []
_vectorizer = None


def load_past_data_from_db():
    essays, feedbacks = [], []
    if not supabase:
        return essays, feedbacks

    try:
        kb_result = supabase.table("knowledge_base") \
            .select("essay, teacher_feedback, feedback") \
            .eq("accepted", True) \
            .execute()

        for item in kb_result.data:
            fb = item.get("teacher_feedback") or item.get("feedback")
            if fb and item.get("essay"):
                essays.append(item["essay"])
                feedbacks.append(fb)
    except Exception as e:
        print(f"RAG kb query failed: {e}")

    try:
        lf_result = supabase.table("learning_feedback") \
            .select("original_essay, teacher_feedback") \
            .execute()

        for item in lf_result.data:
            fb = item.get("teacher_feedback")
            essay = item.get("original_essay")
            if fb and essay:
                essays.append(essay)
                feedbacks.append(f"[Teacher Override] {fb}")
    except Exception as e:
        print(f"RAG lf query failed: {e}")

    return essays, feedbacks


def get_similar_essay_context(essay_text, top_k=2):
    global _past_essays, _past_feedbacks, _vectorizer
    if not _past_essays:
        _past_essays, _past_feedbacks = load_past_data_from_db()
        if not _past_essays:
            return ""
    corpus = _past_essays + [essay_text]
    _vectorizer = TfidfVectorizer(stop_words="english").fit(corpus)
    tfidf_matrix = _vectorizer.transform(corpus)
    similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    context_parts = []
    for idx in top_indices:
        if similarities[idx] > 0.1:
            context_parts.append(f"[Similar essay feedback]:\n{_past_feedbacks[idx]}")
    if context_parts:
        return "\n\n--- Similar past evaluation feedback (RAG) ---\n" + "\n\n".join(context_parts)
    return ""
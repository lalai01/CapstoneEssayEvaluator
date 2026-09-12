# backend/repository.py
"""
Repository layer for the AI Essay Evaluator.

All JSON access happens through these classes; main.py never touches
raw files or the json_store module directly.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from json_store import (
    append_to_list,
    delete_from_list,
    get_one,
    load_json,
    query_list,
    update_in_list,
    upsert_in_list,
)


def _now() -> str:
    return datetime.utcnow().isoformat()


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------

class RatingsRepo:
    FILE = "ratings.json"

    def upsert(self, user_id: str, rating: int, comment: Optional[str]) -> Dict[str, Any]:
        return upsert_in_list(
            self.FILE,
            identity_keys=["user_id"],
            identity_values=[user_id],
            updates={
                "user_id": user_id,
                "rating": rating,
                "comment": comment,
                "updated_at": _now(),
            },
        )

    def for_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        return get_one(self.FILE, user_id=user_id)

    def all(self) -> List[Dict[str, Any]]:
        rows = load_json(self.FILE, [])
        rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return rows

    def summary(self) -> Dict[str, Any]:
        rows = load_json(self.FILE, [])
        values = [r["rating"] for r in rows if "rating" in r]
        avg = sum(values) / len(values) if values else 0
        return {
            "average": round(avg, 1),
            "count": len(values),
            "distribution": {str(i): values.count(i) for i in range(1, 6)},
        }


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------

class CommentsRepo:
    FILE = "comments.json"

    def create(self, rating_id: int, parent_id: Optional[int],
               user_id: str, body: str) -> Dict[str, Any]:
        return append_to_list(self.FILE, {
            "rating_id": rating_id,
            "parent_id": parent_id,
            "user_id": user_id,
            "body": body,
        })

    def for_rating(self, rating_id: int) -> List[Dict[str, Any]]:
        rows = query_list(self.FILE, rating_id=rating_id)
        rows.sort(key=lambda r: r.get("created_at", ""))
        return rows


# ---------------------------------------------------------------------------
# Reactions
# ---------------------------------------------------------------------------

class ReactionsRepo:
    FILE = "comment_reactions.json"

    def toggle(self, comment_id: int, user_id: str, reaction_type: str) -> str:
        existing = get_one(
            self.FILE,
            comment_id=comment_id,
            user_id=user_id,
            reaction_type=reaction_type,
        )
        if existing:
            delete_from_list(self.FILE, "id", existing["id"])
            return "removed"
        append_to_list(self.FILE, {
            "comment_id": comment_id,
            "user_id": user_id,
            "reaction_type": reaction_type,
        })
        return "added"

    def for_comments(self, comment_ids: List[int]) -> List[Dict[str, Any]]:
        rows = load_json(self.FILE, [])
        return [r for r in rows if r.get("comment_id") in comment_ids]


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------

class KnowledgeRepo:
    FILE = "knowledge_base.json"

    def save(self, entry: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        payload = dict(entry)
        payload["user_id"] = user_id
        return append_to_list(self.FILE, payload)

    def for_user(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        rows = query_list(self.FILE, user_id=user_id)
        rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return rows[:limit]

    def get(self, entry_id: int, user_id: str) -> Optional[Dict[str, Any]]:
        return get_one(self.FILE, id=entry_id, user_id=user_id)

    def all_accepted(self) -> List[Dict[str, Any]]:
        return [r for r in load_json(self.FILE, []) if r.get("accepted")]


# ---------------------------------------------------------------------------
# Learning Feedback (teacher overrides)
# ---------------------------------------------------------------------------

class OverridesRepo:
    FILE = "learning_feedback.json"

    def save(self, entry: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        payload = dict(entry)
        payload["user_id"] = user_id
        return append_to_list(self.FILE, payload)

    def for_user(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        rows = query_list(self.FILE, user_id=user_id)
        rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return rows[:limit]

    def all(self) -> List[Dict[str, Any]]:
        return load_json(self.FILE, [])


# ---------------------------------------------------------------------------
# Saved Essays
# ---------------------------------------------------------------------------

class SavedEssaysRepo:
    FILE = "saved_essays.json"

    def save(self, title: str, essay: str, user_id: str) -> Dict[str, Any]:
        return append_to_list(self.FILE, {
            "title": title,
            "essay": essay,
            "user_id": user_id,
        })

    def for_user(self, user_id: str) -> List[Dict[str, Any]]:
        rows = query_list(self.FILE, user_id=user_id)
        rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return rows

    def delete(self, entry_id: int, user_id: str) -> bool:
        row = get_one(self.FILE, id=entry_id, user_id=user_id)
        if not row:
            return False
        return delete_from_list(self.FILE, "id", entry_id)


# ---------------------------------------------------------------------------
# Surveys
# ---------------------------------------------------------------------------

class SurveysRepo:
    FILE = "surveys.json"

    def create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return append_to_list(self.FILE, payload)

    def all(self, active_only: bool = False) -> List[Dict[str, Any]]:
        rows = load_json(self.FILE, [])
        if active_only:
            rows = [r for r in rows if r.get("is_active")]
        rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return rows

    def get(self, survey_id: int) -> Optional[Dict[str, Any]]:
        return get_one(self.FILE, id=survey_id)

    def update(self, survey_id: int, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        updates = {k: v for k, v in updates.items() if v is not None}
        updates["updated_at"] = _now()
        return update_in_list(self.FILE, "id", survey_id, updates)

    def delete(self, survey_id: int) -> bool:
        return delete_from_list(self.FILE, "id", survey_id)


# ---------------------------------------------------------------------------
# Survey Questions
# ---------------------------------------------------------------------------

class QuestionsRepo:
    FILE = "survey_questions.json"

    def add(self, survey_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        rows = query_list(self.FILE, survey_id=survey_id)
        next_order = max((r.get("order_number", 0) for r in rows), default=0) + 1
        row = dict(payload)
        row["survey_id"] = survey_id
        row["order_number"] = next_order
        return append_to_list(self.FILE, row)

    def for_survey(self, survey_id: int) -> List[Dict[str, Any]]:
        rows = query_list(self.FILE, survey_id=survey_id)
        rows.sort(key=lambda r: r.get("order_number", 0))
        return rows

    def update(self, question_id: int, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        updates = {k: v for k, v in updates.items() if v is not None}
        return update_in_list(self.FILE, "id", question_id, updates)

    def delete(self, question_id: int) -> bool:
        return delete_from_list(self.FILE, "id", question_id)


# ---------------------------------------------------------------------------
# Survey Responses
# ---------------------------------------------------------------------------

class ResponsesRepo:
    FILE = "survey_responses.json"

    def submit(self, survey_id: int, user_id: str,
               answers: Dict[str, str]) -> int:
        written = 0
        for qid_str, answer in answers.items():
            qid = int(qid_str)
            upsert_in_list(
                self.FILE,
                identity_keys=["survey_id", "question_id", "user_id"],
                identity_values=[survey_id, qid, user_id],
                updates={
                    "survey_id": survey_id,
                    "question_id": qid,
                    "user_id": user_id,
                    "answer": answer,
                },
            )
            written += 1
        return written

    def for_survey(self, survey_id: int) -> List[Dict[str, Any]]:
        return query_list(self.FILE, survey_id=survey_id)

    def for_user(self, survey_id: int, user_id: str) -> List[Dict[str, Any]]:
        return query_list(self.FILE, survey_id=survey_id, user_id=user_id)

    def has_user_submitted(self, survey_id: int, user_id: str) -> bool:
        return bool(self.for_user(survey_id, user_id))


# ---------------------------------------------------------------------------
# User Profiles
# ---------------------------------------------------------------------------

class UserProfilesRepo:
    FILE = "user_profiles.json"

    def upsert(self, user_id: str, full_name: str,
               avatar_url: Optional[str] = None,
               email: Optional[str] = None) -> Dict[str, Any]:
        return upsert_in_list(
            self.FILE,
            identity_keys=["id"],
            identity_values=[user_id],
            updates={
                "id": user_id,
                "full_name": full_name,
                "avatar_url": avatar_url,
                "email": email,
            },
        )

    def many(self, user_ids: List[str]) -> List[Dict[str, Any]]:
        rows = load_json(self.FILE, [])
        wanted = set(user_ids)
        return [r for r in rows if r.get("id") in wanted]

    def get(self, user_id: str) -> Optional[Dict[str, Any]]:
        return get_one(self.FILE, id=user_id)


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

ratings_repo = RatingsRepo()
comments_repo = CommentsRepo()
reactions_repo = ReactionsRepo()
knowledge_repo = KnowledgeRepo()
overrides_repo = OverridesRepo()
saved_essays_repo = SavedEssaysRepo()
surveys_repo = SurveysRepo()
questions_repo = QuestionsRepo()
responses_repo = ResponsesRepo()
user_profiles_repo = UserProfilesRepo()
from typing import TypedDict, Optional, Dict, Any

from langgraph.graph import StateGraph, START, END

from evaluator import (
    is_valid_essay,
    analyze_essay_content,
    calculate_analytic_scores,
    holistic_from_analytic,
    ai_holistic_score,
    holistic_feedback_text,
    generate_rule_based_analytic_feedback,
    generate_rule_based_holistic_feedback,
    enhance_feedback_with_ai,
    HOLISTIC_RUBRIC,
)
from rag import get_similar_essay_context


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------
class EvaluationState(TypedDict, total=False):
    essay_text: str
    evaluation_type: str
    use_rag: bool
    scores: Optional[Dict[str, Any]]
    feedback: Optional[str]
    rag_context: Optional[str]
    error: Optional[str]
    _analysis: Optional[Dict[str, Any]]
    _validated: Optional[bool]
    rubric: Optional[Dict[str, Any]]
    ai_model: Optional[str]   # "auto" | "deepseek" | "local"/"llama" | "gemma"
    strict_model: Optional[bool]   # only the chosen AI (side-by-side comparison)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def validate_node(state: EvaluationState) -> dict:
    valid, err = is_valid_essay(state["essay_text"])
    if not valid:
        return {"error": err, "_validated": False}
    return {"_validated": True}


def score_node(state: EvaluationState) -> dict:
    analysis = analyze_essay_content(state["essay_text"])
    if state.get("evaluation_type") == "holistic":
        # One quick overall judgement; only if every model fails do we fall
        # back to deriving the score from the slower analytic scoring.
        scores = ai_holistic_score(state["essay_text"], state.get("ai_model") or "auto",
                                   strict=bool(state.get("strict_model")))
        if scores is None:
            analytic = calculate_analytic_scores(
                state["essay_text"],
                analysis,
                rubric=state.get("rubric"),
                preference=state.get("ai_model") or "auto",
                strict=bool(state.get("strict_model")),
            )
            score = holistic_from_analytic(analytic, state.get("rubric"))
            by = analytic.get("evaluated_by") or {}
            scores = {
                "holistic_score": score,
                "level_description": HOLISTIC_RUBRIC[score],
                "grammar_issues": analytic.get("grammar_issues") or [],
                "evaluated_by": {**by, "method": "derived", "reason": (
                    (by.get("reason") or "") + " The quick holistic check didn't respond, so the "
                    "score was derived from the analytic criteria instead.").strip()},
            }
    else:
        scores = calculate_analytic_scores(
            state["essay_text"],
            analysis,
            rubric=state.get("rubric"),
            preference=state.get("ai_model") or "auto",
            strict=bool(state.get("strict_model")),
        )
    return {"scores": scores, "_analysis": analysis}


def rag_node(state: EvaluationState) -> dict:
    if not state.get("use_rag"):
        return {"rag_context": ""}
    try:
        context = get_similar_essay_context(state["essay_text"])
    except Exception as e:
        print(f"RAG error: {e}")
        context = ""
    return {"rag_context": context}


def feedback_node(state: EvaluationState) -> dict:
    analysis = state.get("_analysis") or {}
    if state.get("evaluation_type") == "holistic" and state["scores"].get("holistic_feedback"):
        # The holistic model already wrote the feedback: no second AI call.
        return {"feedback": holistic_feedback_text(state["scores"], state.get("rag_context") or "")}
    if state.get("evaluation_type") == "holistic":
        rule_feedback = generate_rule_based_holistic_feedback(
            state["essay_text"],
            state["scores"]["holistic_score"],
            analysis,
            state.get("rag_context") or "",
        )
    else:
        rule_feedback = generate_rule_based_analytic_feedback(
            state["essay_text"],
            state["scores"],
            analysis,
            state.get("rag_context") or "",
            rubric=state.get("rubric"),
        )
    # The AI that scored the essay also writes its feedback.
    by = state["scores"].get("evaluated_by") or {}
    enhanced = enhance_feedback_with_ai(
        state["essay_text"], state["scores"], analysis, rule_feedback,
        rubric=state.get("rubric"), model=by.get("model"), provider=by.get("provider"),
    )
    return {"feedback": enhanced}


def error_node(state: EvaluationState) -> dict:
    return {"feedback": f"⚠️ Invalid Input: {state.get('error', 'Unknown error')}"}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph():
    wf = StateGraph(EvaluationState)

    wf.add_node("validate", validate_node)
    wf.add_node("score", score_node)
    wf.add_node("rag", rag_node)
    wf.add_node("generate_feedback", feedback_node)
    wf.add_node("handle_error", error_node)

    wf.add_edge(START, "validate")
    wf.add_conditional_edges(
        "validate",
        lambda s: "handle_error" if s.get("error") else "score",
        {"handle_error": "handle_error", "score": "score"},
    )
    wf.add_edge("score", "rag")
    wf.add_edge("rag", "generate_feedback")
    wf.add_edge("generate_feedback", END)
    wf.add_edge("handle_error", END)

    return wf.compile()


evaluation_graph = build_graph()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_evaluation(essay_text: str, evaluation_type: str, use_rag: bool, rubric=None,
                   ai_model: str = "auto", strict_model: bool = False) -> dict:
    initial: EvaluationState = {
        "essay_text": essay_text,
        "evaluation_type": evaluation_type,
        "use_rag": use_rag,
        "rubric": rubric,
        "ai_model": ai_model,
        "strict_model": strict_model,
        "scores": None,
        "feedback": None,
        "rag_context": None,
        "error": None,
        "_validated": None,
    }
    return evaluation_graph.invoke(initial)
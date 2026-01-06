"""Pydantic models for LLM responses used by the coach engine."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

class EvalResult(BaseModel):
    """
    Parsed evaluation result returned by the Evaluator LLM.

    Attributes:
        score: Numeric score 0-10.
        short_verdict: Brief verdict text.
        rubric: List of rubric coverage items.
        missing_points: Missing items list.
        incorrect_points: Incorrect/unclear items.
        improvement_tips: Actionable tips.
        ideal_answer: Ideal reference answer.
    """
    score: int = Field(..., ge=0, le=10)
    short_verdict: str
    rubric: List[Dict[str, Any]] = Field(default_factory=list)
    missing_points: List[str] = Field(default_factory=list)
    incorrect_points: List[str] = Field(default_factory=list)
    improvement_tips: List[str] = Field(default_factory=list)
    ideal_answer: str

class AnalysisResult(BaseModel):
    """
    Parsed analysis result returned by the Analyzer LLM.

    Attributes:
        overall_summary: Narrative summary of progress.
        strong_topics: Topic ids performing well.
        weak_topics: Topic ids performing poorly.
        recommendations: Suggested actions.
        topic_scores: Normalized readiness scores per topic.
        topic_confidence: Confidence per topic score.
        next_focus_topics: Topics to focus next.
        why_focus: Short rationale per focus topic.
    """
    overall_summary: str
    strong_topics: List[str] = Field(default_factory=list)
    weak_topics: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    topic_scores: Dict[str, float] = Field(default_factory=dict)
    topic_confidence: Dict[str, float] = Field(default_factory=dict)
    next_focus_topics: List[str] = Field(default_factory=list)
    why_focus: Dict[str, str] = Field(default_factory=dict)

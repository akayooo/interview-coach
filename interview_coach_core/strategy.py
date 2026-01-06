"""Topic/question selection heuristics for interview flow."""

import random
from typing import Any, Dict, List, Optional, Tuple

def topic_weight(topic_id: str, progress: Dict[str, Any], recently: List[str]) -> float:
    """
    Calculate sampling weight for a topic based on recency and performance.

    Args:
        topic_id: Topic identifier.
        progress: Aggregated stats per topic.
        recently: List of recently used topic ids.

    Returns:
        Weight value (>=0.05) used for random choice.
    """
    base = 1.0
    st = progress.get(topic_id)
    if st:
        avg = st.get("avg_score", 0.0)
        attempts = st.get("attempts", 0)
        w = base + (10.0 - avg) * 0.20 + (1.0 / (attempts + 1)) * 0.8
    else:
        w = base + 1.0
    if topic_id in recently[-2:]:
        w *= 0.2
    return max(0.05, w)

def pick_topic(topics: Dict[str, Any], progress: Dict[str, Any], recently: List[str], hint: Optional[Dict[str, Any]] = None) -> str:
    """
    Pick a topic id with weighted randomness and weak-topic bias.

    Args:
        topics: All topics map.
        progress: Aggregated stats per topic.
        recently: Recent topic ids to down-weight.
        hint: Optional analysis hints (weak_topics).

    Returns:
        Selected topic_id.
    """
    ids = list(topics.keys())
    weights = [topic_weight(tid, progress, recently) for tid in ids]
    if hint and hint.get("weak_topics"):
        weak = set(hint["weak_topics"])
        weights = [w * (1.8 if tid in weak else 1.0) for tid, w in zip(ids, weights)]
    return random.choices(ids, weights=weights, k=1)[0]

def pick_question(topic: Dict[str, Any]) -> Tuple[str, Optional[str], str]:
    """
    Pick an unasked question from a topic.

    Args:
        topic: Topic payload with questions and theory.
    Returns:
        Tuple of (question text, expected rubric/checklist, theory excerpt). Empty question triggers generation.
    """
    qs = topic.get("questions", [])
    if not qs:
        return "", None, topic.get("theory", "")[:2000]
    q = random.choice(qs)
    return q["q"], q.get("expected"), topic.get("theory", "")[:2000]

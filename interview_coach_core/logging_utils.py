"""Helpers for creating run directories and persisting JSONL history."""

import os
import json
from datetime import datetime
from typing import Any, Dict, List

def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def new_run_dir(base: str) -> str:
    """
    Create a timestamped subdirectory under base.

    Args:
        base: Root folder for runs.

    Returns:
        Full path to the created run directory.
    """
    ensure_dir(base)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, stamp)
    ensure_dir(run_dir)
    return run_dir

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    """
    Append one JSON object to a .jsonl file.

    Args:
        path: Target file path.
        obj: Object to serialize.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dicts.

    Args:
        path: File path to read.

    Returns:
        Parsed list; empty list if file missing.
    """
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def build_progress_from_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate per-topic stats from history records.

    Args:
        history: List of history entries (question/score/etc).

    Returns:
        Dict of topic_id -> attempts, scores list, last_ts, avg_score.
    """
    stats: Dict[str, Any] = {}
    for h in history:
        tid = h.get("topic_id")
        if not tid:
            continue
        stats.setdefault(tid, {"attempts": 0, "scores": [], "last_ts": None})
        stats[tid]["attempts"] += 1
        stats[tid]["scores"].append(h.get("score", 0))
        stats[tid]["last_ts"] = h.get("ts")
    for tid, st in stats.items():
        st["avg_score"] = sum(st["scores"]) / max(1, len(st["scores"]))
    return stats

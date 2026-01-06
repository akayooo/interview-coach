"""Markdown parser: load conspects, split sections, extract questions, build index."""

import os
import glob
import re
from typing import Any, Dict, List

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
BULLET_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
NUM_RE = re.compile(r"^\s*\d+[\.\)]\s+(.*)$")

def read_markdown_files(notes_dir: str) -> Dict[str, str]:
    """
    Load all markdown files from a directory.

    Args:
        notes_dir: Path to directory with .md files.

    Returns:
        Dict[path, contents].

    Raises:
        FileNotFoundError: If no markdown files are found.
    """
    paths = sorted(glob.glob(os.path.join(notes_dir, "*.md")))
    if not paths:
        raise FileNotFoundError(f"Не нашел .md файлов в {notes_dir}. Положи туда конспекты.")
    out: Dict[str, str] = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            out[p] = f.read()
    return out

def split_by_headings(md_text: str) -> List[Dict[str, Any]]:
    """
    Split markdown text into sections by headings.

    Args:
        md_text: Raw markdown text.

    Returns:
        List of sections with level, title, and text.
    """
    lines = md_text.splitlines()
    sections: List[Dict[str, Any]] = []
    cur = {"level": 0, "title": "ROOT", "text_lines": []}
    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            if cur["text_lines"] or cur["title"] != "ROOT":
                sections.append({"level": cur["level"], "title": cur["title"], "text": "\n".join(cur["text_lines"]).strip()})
            cur = {"level": len(m.group(1)), "title": m.group(2).strip(), "text_lines": []}
        else:
            cur["text_lines"].append(line)
    if cur["text_lines"] or cur["title"] != "ROOT":
        sections.append({"level": cur["level"], "title": cur["title"], "text": "\n".join(cur["text_lines"]).strip()})
    return [s for s in sections if (s["title"].strip() and (s["text"].strip() or s["level"] > 0))]

def extract_questions(text: str) -> List[str]:
    """
    Extract likely interview questions from a block of text.

    Args:
        text: Section text.

    Returns:
        Unique list of question strings.
    """
    qs: List[str] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if raw.lower().startswith(("q:", "q -", "вопрос:", "question:")):
            qs.append(raw.split(":", 1)[-1].strip() or raw)
            continue
        m = BULLET_RE.match(raw) or NUM_RE.match(raw)
        candidate = m.group(1).strip() if m else None
        if candidate and ("?" in candidate or candidate.lower().startswith(("как ", "почему ", "что ", "зачем ", "когда ", "в чем ", "в чём "))):
            qs.append(candidate)
            continue
        if "?" in raw and len(raw) <= 200:
            qs.append(raw)
    uniq, seen = [], set()
    for q in qs:
        qn = re.sub(r"\s+", " ", q).strip()
        if qn.lower() not in seen:
            seen.add(qn.lower())
            uniq.append(qn)
    return uniq

def build_content_index(notes_dir: str) -> Dict[str, Any]:
    """
    Build structured content index from markdown conspects.

    Args:
        notes_dir: Directory with markdown files.

    Returns:
        Dict containing topics keyed by topic_id and source metadata.

    Raises:
        RuntimeError: If no topics could be derived from markdown.
    """
    files = read_markdown_files(notes_dir)
    topics: Dict[str, Any] = {}
    topic_counter = 0
    for path, text in files.items():
        sections = split_by_headings(text)
        for sec in sections:
            if sec["level"] < 3 or sec["level"] > 4:
                continue
            topic_counter += 1
            topic_id = f"t{topic_counter:04d}"
            sec_text = sec["text"].strip()
            questions = extract_questions(sec_text)
            topics[topic_id] = {
                "topic_id": topic_id,
                "title": sec["title"],
                "source_file": os.path.basename(path),
                "theory": sec_text,
                "questions": [{"q": q, "expected": None, "ref": sec["title"]} for q in questions],
            }
    if not topics:
        raise RuntimeError("Не получилось выделить темы из markdown. Проверь, что есть заголовки #/##/###.")
    return {"topics": topics, "notes_dir": notes_dir}

"""FastAPI entrypoint wiring static assets, websocket protocol, and the coach engine."""

import os
import json
import uuid
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from interview_coach_core.config import Config
from interview_coach_core.parsing import build_content_index, build_simple_index
from interview_coach_core.openrouter import build_llms
from interview_coach_core.stt import load_whisper_model, transcribe_bytes
from app.coach_engine import CoachEngine

app = FastAPI(title="InterviewCoachWeb")

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def index() -> FileResponse:
    """Serve the single-page client.

    Returns:
        Static index.html response.
    """
    return FileResponse(os.path.join(static_dir, "index.html"))

cfg = Config.from_env()
engine: Optional[CoachEngine] = None
llm_gen: Optional[Any] = None
content_index: Optional[Dict[str, Any]] = None
simple_index: Dict[str, Any] = {"sources": {}}
simple_sessions: Dict[str, "SimpleSessionState"] = {}
whisper_model: Optional[Any] = None
CHAT_HISTORY_LIMIT = 8
SIMPLE_CHAT_SYSTEM = """Ты — собеседник-наставник. Помогай разбирать ответы на вопросы из конспекта: поясняй, дополняй, приводи примеры, исправляй ошибки. Отвечай по-русски, кратко и по делу. Если пользователь просит уточнение, объясняй проще и структурированно. Не придумывай новые требования, опирайся на вопрос и эталонный ответ. Все формулы оформляй через $...$ (или $$...$$ для блочных)."""

@dataclass
class SimpleSessionState:
    """State for a simple-mode session.

    Attributes:
        session_id: Websocket session id.
        allowed_sources: Selected source files.
        weights: Per-source sampling weights.
        remaining_by_source: Remaining questions by source.
        total_questions: Total questions at session start.
        asked_count: Number of served questions (including повторы).
        last_question: Last shown question.
        last_answer: Last shown answer.
        last_source: Source file for the last shown question.
        last_item: Raw payload of the last question.
        skipped_queue: FIFO queue of skipped questions to repeat later.
        in_repeat: Flag indicating the repeat iteration over skipped questions.
        chat_history: Simple-mode chat history.
    """
    session_id: str
    allowed_sources: List[str]
    weights: Dict[str, int]
    remaining_by_source: Dict[str, List[Dict[str, str]]]
    total_questions: int
    asked_count: int = 0
    last_question: str = ""
    last_answer: str = ""
    last_source: str = ""
    last_item: Optional[Dict[str, str]] = None
    skipped_queue: List[Tuple[str, Dict[str, str]]] = field(default_factory=list)
    in_repeat: bool = False
    chat_history: List[Dict[str, str]] = field(default_factory=list)

def _normalize_simple_weights(sources: List[str], raw_weights: Dict[str, Any]) -> Dict[str, int]:
    """Normalize per-source weights to sum to 100.

    Args:
        sources: Source names to keep.
        raw_weights: Raw weight mapping from client.

    Returns:
        Normalized weights summing to 100.
    """
    if not sources:
        return {}
    parsed = {s: max(0, int(raw_weights.get(s, 0) or 0)) for s in sources}
    total = sum(parsed.values())
    if total <= 0:
        base = 100 // len(sources)
        remainder = 100 - base * len(sources)
        return {s: base + (1 if i < remainder else 0) for i, s in enumerate(sources)}
    scaled = {s: round(parsed[s] / total * 100) for s in sources}
    diff = 100 - sum(scaled.values())
    i = 0
    while diff != 0 and sources:
        src = sources[i % len(sources)]
        if diff > 0:
            scaled[src] += 1
            diff -= 1
        elif scaled[src] > 0:
            scaled[src] -= 1
            diff += 1
        i += 1
    return scaled

def _build_simple_session(session_id: str, sources: List[str], weights: Dict[str, Any]) -> SimpleSessionState:
    """Build simple-mode session state.

    Args:
        session_id: Websocket session id.
        sources: Selected sources or empty for all.
        weights: Raw weight mapping.

    Returns:
        Initialized SimpleSessionState.
    """
    pool = simple_index.get("sources", {})
    allowed = [s for s in (sources or list(pool.keys())) if s in pool]
    remaining_by_source: Dict[str, List[Dict[str, str]]] = {}
    for src in allowed:
        items = list(pool.get(src, []))
        random.shuffle(items)
        remaining_by_source[src] = items
    total_questions = sum(len(v) for v in remaining_by_source.values())
    normalized = _normalize_simple_weights(allowed, weights or {})
    return SimpleSessionState(
        session_id=session_id,
        allowed_sources=allowed,
        weights=normalized,
        remaining_by_source=remaining_by_source,
        total_questions=total_questions,
    )

def _pick_simple_source(st: SimpleSessionState) -> Optional[str]:
    """Pick a source according to remaining questions and weights.

    Args:
        st: Simple session state.

    Returns:
        Selected source or None if empty.
    """
    candidates = [src for src, items in st.remaining_by_source.items() if items]
    if not candidates:
        return None
    weights = [max(0, st.weights.get(src, 0)) for src in candidates]
    if sum(weights) <= 0:
        weights = [1 for _ in candidates]
    return random.choices(candidates, weights=weights, k=1)[0]

def _build_hello_payload(session_id: str) -> Dict[str, Any]:
    """Build hello payload for websocket client.

    Args:
        session_id: Websocket session id.

    Returns:
        JSON payload with configuration and sources.
    """
    sources = sorted({t["source_file"] for t in content_index["topics"].values()}) if content_index else []
    simple_sources = sorted(simple_index.get("sources", {}).keys())
    simple_counts = {src: len(items) for src, items in simple_index.get("sources", {}).items()}
    return {
        "type": "hello",
        "session_id": session_id,
        "has_llm": bool(cfg.openrouter_api_key),
        "has_stt": whisper_model is not None,
        "notes_dir": cfg.notes_dir,
        "runs_dir": cfg.runs_dir,
        "analyze_every_n": cfg.analyze_every_n,
        "model": cfg.openrouter_model,
        "sources": sources,
        "simple_sources": simple_sources,
        "simple_counts": simple_counts,
    }

@app.on_event("startup")
def on_startup() -> None:
    """Load content index and build LLM/STT clients before accepting websocket traffic."""
    global engine, whisper_model, content_index, simple_index, llm_gen
    content_index = build_content_index(cfg.notes_dir)
    simple_index = build_simple_index(cfg.notes_dir)
    if cfg.openrouter_api_key:
        llm_gen_client, llm_eval = build_llms(
            api_key=cfg.openrouter_api_key,
            model=cfg.openrouter_model,
            http_referer=cfg.app_http_referer,
            app_title=cfg.app_title,
            timeout_s=cfg.llm_timeout_s,
        )
        llm_gen = llm_gen_client
        engine = CoachEngine(
            content_index=content_index,
            llm_gen=llm_gen_client,
            llm_eval=llm_eval,
            runs_dir=cfg.runs_dir,
            analyze_every_n=cfg.analyze_every_n,
            max_questions_per_source=cfg.max_questions_per_source,
            model_name=cfg.openrouter_model,
        )
    try:
        whisper_model = load_whisper_model(cfg.whisper_model, device=cfg.whisper_device, compute_type=cfg.whisper_compute_type)
    except Exception as e:
        print(f"[warn] whisper model not loaded: {e}")

@app.post("/stt")
async def stt_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Speech-to-text endpoint using faster-whisper.

    Args:
        file: Uploaded audio file.

    Returns:
        Dict with transcribed text and metadata.
    """
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model не загружен")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Пустой аудиофайл")
    text, info = await run_in_threadpool(
        lambda: transcribe_bytes(
            whisper_model,
            data,
            language=(cfg.whisper_language or None),
            initial_prompt=cfg.whisper_initial_prompt,
            beam_size=cfg.whisper_beam_size,
            condition_on_previous_text=cfg.whisper_condition_on_prev,
        )
    )
    return {"text": text, "language": info.get("language"), "duration": info.get("duration")}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    """
    Primary websocket handler.

    Receives control messages from the browser (start/answer/reset), streams status,
    questions, feedback, and analysis updates back to the client.

    Args:
        ws: Active websocket connection.
    """
    await ws.accept()
    session_id = str(uuid.uuid4())
    session_mode = "smart"

    async def emit(payload: Dict[str, Any]) -> None:
        """
        Send a JSON payload to the current websocket client.

        Args:
            payload: JSON-serializable object to emit.
        """
        await ws.send_text(json.dumps(payload, ensure_ascii=False))

    async def emit_simple_question() -> None:
        """Send the next simple-mode question to the client."""
        st = simple_sessions.get(session_id)
        if not st:
            return
        has_candidates = any(items for items in st.remaining_by_source.values())
        repeat = False
        src: Optional[str] = None
        item: Optional[Dict[str, str]] = None

        if has_candidates:
            src = _pick_simple_source(st)
            if not src:
                # fallback to first available source to avoid dead-end
                src = next((s for s, items in st.remaining_by_source.items() if items), None)
            if src:
                item = st.remaining_by_source[src].pop(0)
                st.in_repeat = False

        if item is None:
            if st.skipped_queue:
                repeat = True
                st.in_repeat = True
                src, item = st.skipped_queue.pop(0)
            else:
                st.in_repeat = False
                await emit(
                    {
                        "type": "simple_done",
                        "reason": "Вопросы закончились для выбранных конспектов.",
                        "total": st.total_questions,
                    }
                )
                await emit({"type": "status", "agent": "SimpleMode", "step": "Готово", "state": "idle", "counter": st.asked_count})
                return

        step_label = "Повторяю пропущенный вопрос…" if repeat else "Выбираю вопрос…"
        await emit({"type": "status", "agent": "SimpleMode", "step": step_label, "state": "running", "counter": st.asked_count})
        st.asked_count += 1
        st.last_source = src or ""
        st.last_item = item
        st.last_question = item.get("q", "")
        st.last_answer = item.get("a", "")
        st.chat_history = []
        remaining_total = sum(len(v) for v in st.remaining_by_source.values()) + len(st.skipped_queue)
        await emit(
            {
                "type": "simple_question",
                "source_file": src,
                "subtopic": item.get("subtopic", ""),
                "question": item.get("q", ""),
                "answer": item.get("a", ""),
                "index": st.asked_count,
                "total": st.total_questions,
                "remaining": remaining_total,
                "repeat": repeat,
                "repeat_remaining": len(st.skipped_queue),
            }
        )
        await emit({"type": "status", "agent": "SimpleMode", "step": "Готово", "state": "idle", "counter": st.asked_count})

    await emit(_build_hello_payload(session_id))

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            t = msg.get("type")

            if t == "start":
                mode = (msg.get("mode") or "smart").lower()
                session_mode = "simple" if mode == "simple" else "smart"
                if session_mode == "simple":
                    simple_sessions.pop(session_id, None)
                    sources = msg.get("sources") or []
                    weights = msg.get("weights") or {}
                    st_simple = _build_simple_session(session_id, sources, weights)
                    simple_sessions[session_id] = st_simple
                    if st_simple.total_questions == 0:
                        await emit(
                            {
                                "type": "simple_done",
                                "reason": "Нет вопросов в выбранных конспектах.",
                                "total": 0,
                            }
                        )
                        continue
                    await emit_simple_question()
                else:
                    simple_sessions.pop(session_id, None)
                    if not engine:
                        await emit(
                            {
                                "type": "message",
                                "role": "system",
                                "title": "Ошибка конфигурации",
                                "body": "OPENROUTER_API_KEY не задан. Установи env и перезапусти сервер.",
                            }
                        )
                        await emit({"type": "status", "agent": "Session", "step": "Ожидание настроек", "state": "idle", "counter": 0})
                        continue
                    sources = msg.get("sources") or []
                    st = engine.get_session(session_id)
                    st.allowed_sources = sources
                    st.asked_topics = set()
                    st.source_question_counts = {}
                    st.recently_topics = []
                    st.progress = {}
                    await emit({"type":"ready","run_dir": st.run_dir, "history_path": st.history_path, "counter": st.counter})
                    await engine.start_or_next_question(session_id, emit)

            elif t == "answer":
                if session_mode != "smart" or not engine:
                    continue
                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                if text.lower() in {"exit","quit","q"}:
                    await emit({"type":"message","role":"system","title":"Сессия завершена","body":"Ок, остановились. Обнови страницу для новой сессии."})
                    await emit({"type":"status","agent":"Session","step":"Остановлено пользователем","state":"idle","counter": engine.get_session(session_id).counter})
                    continue
                await engine.submit_answer(session_id, text, emit)

            elif t == "simple_chat":
                if session_mode != "simple":
                    continue
                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                st_simple = simple_sessions.get(session_id)
                if not st_simple:
                    await emit({"type": "event", "agent": "SimpleChat", "level": "warn", "text": "Сессия simple не инициализирована."})
                    continue
                await emit({"type": "chat", "role": "user", "text": text})
                if not llm_gen:
                    await emit(
                        {
                            "type": "chat",
                            "role": "assistant",
                            "text": "LLM недоступен. Проверь OPENROUTER_API_KEY и перезапусти сервер.",
                        }
                    )
                    continue
                await emit({"type": "status", "agent": "SimpleChat", "step": "Думаю…", "state": "running", "counter": st_simple.asked_count})
                q = st_simple.last_question or "—"
                a = st_simple.last_answer or "—"
                system_prompt = f"{SIMPLE_CHAT_SYSTEM}\n\nВопрос:\n{q}\n\nЭталонный ответ:\n{a}\n"
                history = st_simple.chat_history[-CHAT_HISTORY_LIMIT:]
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(history)
                messages.append({"role": "user", "content": text})
                try:
                    resp = await run_in_threadpool(lambda: llm_gen.invoke(messages).content.strip())
                except Exception as e:
                    await emit({"type": "event", "agent": "SimpleChat", "level": "error", "text": f"LLM error: {e}"})
                    await emit(
                        {
                            "type": "chat",
                            "role": "assistant",
                            "text": "Не удалось получить ответ от LLM. Попробуй позже.",
                        }
                    )
                    await emit({"type": "status", "agent": "SimpleChat", "step": "Ошибка", "state": "error", "counter": st_simple.asked_count})
                    continue
                st_simple.chat_history.append({"role": "user", "content": text})
                st_simple.chat_history.append({"role": "assistant", "content": resp})
                if len(st_simple.chat_history) > CHAT_HISTORY_LIMIT * 2:
                    st_simple.chat_history = st_simple.chat_history[-CHAT_HISTORY_LIMIT * 2:]
                await emit({"type": "chat", "role": "assistant", "text": resp})
                await emit({"type": "status", "agent": "SimpleChat", "step": "Готово", "state": "idle", "counter": st_simple.asked_count})

            elif t == "next":
                if session_mode != "simple":
                    continue
                st_simple = simple_sessions.get(session_id)
                if st_simple and msg.get("weights"):
                    st_simple.weights = _normalize_simple_weights(st_simple.allowed_sources, msg.get("weights") or {})
                await emit_simple_question()

            elif t == "simple_skip":
                if session_mode != "simple":
                    continue
                st_simple = simple_sessions.get(session_id)
                if not st_simple:
                    continue
                if st_simple.last_item is not None and st_simple.last_source:
                    st_simple.skipped_queue.append((st_simple.last_source, st_simple.last_item))
                await emit_simple_question()

            elif t == "reset":
                simple_sessions.pop(session_id, None)
                session_id = str(uuid.uuid4())
                session_mode = "smart"
                await emit(_build_hello_payload(session_id))

            else:
                await emit({"type":"event","agent":"Server","level":"warn","text":f"Неизвестный тип сообщения: {t}"})
    except WebSocketDisconnect:
        return

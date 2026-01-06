"""FastAPI entrypoint wiring static assets, websocket protocol, and the coach engine."""

import os
import json
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from interview_coach_core.config import Config
from interview_coach_core.parsing import build_content_index
from interview_coach_core.openrouter import build_llms
from interview_coach_core.stt import load_whisper_model, transcribe_bytes
from app.coach_engine import CoachEngine

app = FastAPI(title="InterviewCoachWeb")

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def index() -> FileResponse:
    """Serve the single-page client."""
    return FileResponse(os.path.join(static_dir, "index.html"))

cfg = Config.from_env()
engine: Optional[CoachEngine] = None
whisper_model = None

@app.on_event("startup")
def on_startup() -> None:
    """Load content index and build LLM/STT clients before accepting websocket traffic."""
    global engine, whisper_model
    content_index = build_content_index(cfg.notes_dir)
    if cfg.openrouter_api_key:
        llm_gen, llm_eval = build_llms(
            api_key=cfg.openrouter_api_key,
            model=cfg.openrouter_model,
            http_referer=cfg.app_http_referer,
            app_title=cfg.app_title,
            timeout_s=cfg.llm_timeout_s,
        )
        engine = CoachEngine(
            content_index=content_index,
            llm_gen=llm_gen,
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
async def stt_endpoint(file: UploadFile = File(...)) -> dict:
    """Speech-to-text endpoint using faster-whisper."""
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
    """
    await ws.accept()
    session_id = str(uuid.uuid4())

    async def emit(payload: Dict[str, Any]) -> None:
        """
        Send a JSON payload to the current websocket client.

        Args:
            payload: JSON-serializable object to emit.
        """
        await ws.send_text(json.dumps(payload, ensure_ascii=False))

    await emit({
        "type":"hello",
        "session_id": session_id,
        "has_llm": bool(cfg.openrouter_api_key),
        "has_stt": whisper_model is not None,
        "notes_dir": cfg.notes_dir,
        "runs_dir": cfg.runs_dir,
        "analyze_every_n": cfg.analyze_every_n,
        "model": cfg.openrouter_model,
        "sources": sorted({t["source_file"] for t in engine.content_index["topics"].values()}) if engine else [],
    })

    if not cfg.openrouter_api_key:
        await emit({"type":"message","role":"system","title":"Ошибка конфигурации","body":"OPENROUTER_API_KEY не задан. Установи env и перезапусти сервер."})

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            t = msg.get("type")

            if t == "start":
                if not engine:
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
                if not engine:
                    continue
                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                if text.lower() in {"exit","quit","q"}:
                    await emit({"type":"message","role":"system","title":"Сессия завершена","body":"Ок, остановились. Обнови страницу для новой сессии."})
                    await emit({"type":"status","agent":"Session","step":"Остановлено пользователем","state":"idle","counter": engine.get_session(session_id).counter})
                    continue
                await engine.submit_answer(session_id, text, emit)

            elif t == "reset":
                if not engine:
                    continue
                session_id = str(uuid.uuid4())
                await emit({"type":"hello","session_id": session_id, "has_llm": True, "analyze_every_n": cfg.analyze_every_n, "model": cfg.openrouter_model})

            else:
                await emit({"type":"event","agent":"Server","level":"warn","text":f"Неизвестный тип сообщения: {t}"})
    except WebSocketDisconnect:
        return

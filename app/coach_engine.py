"""Session orchestration: selects topics/questions, evaluates answers, and tracks progress."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from fastapi.concurrency import run_in_threadpool

from interview_coach_core.strategy import pick_topic, pick_question
from interview_coach_core.logging_utils import new_run_dir, append_jsonl, load_jsonl, build_progress_from_history
from interview_coach_core.prompts import EVAL_SYSTEM, ANALYSIS_SYSTEM, robust_json_load
from interview_coach_core.models import EvalResult, AnalysisResult

EmitFunc = Callable[[Dict[str, Any]], Awaitable[None]]

@dataclass
class SessionState:
    """
    State container for one websocket interview session.

    Attributes:
        session_id: Unique id for the websocket client.
        run_dir: Folder for session artifacts.
        history_path: Path to JSONL history file.
        counter: Number of answered questions.
        recently_topics: Rolling list of recently asked topic ids.
        asked_topics: Stop-list of already used topic ids.
        source_question_counts: Per-source question counts.
        progress: Aggregated stats by topic.
        analyzer_hint: Hints from last analysis (weak/strong topics).
        last_analysis: Last analysis JSON payload.
        selected_topic_id: Currently active topic id.
        selected_question: Current question text.
        expected_answer: Expected rubric/checklist (text or JSON).
        theory_context: Theory fragment used for context.
    """

    session_id: str
    run_dir: str
    history_path: str
    counter: int = 0
    recently_topics: List[str] = field(default_factory=list)
    asked_topics: Set[str] = field(default_factory=set)
    source_question_counts: Dict[str, int] = field(default_factory=dict)
    progress: Dict[str, Any] = field(default_factory=dict)
    analyzer_hint: Dict[str, Any] = field(default_factory=dict)
    last_analysis: Optional[Dict[str, Any]] = None

    selected_topic_id: Optional[str] = None
    selected_question: Optional[str] = None
    expected_answer: Optional[str] = None
    theory_context: str = ""
    allowed_sources: List[str] = field(default_factory=list)

@dataclass
class CoachEngine:
    """
    Coordinates question selection, evaluation, history, and analysis.

    Attributes:
        content_index: Parsed topics/questions/theory from notes.
        llm_gen: LLM client used for generation (questions/expected).
        llm_eval: LLM client used for evaluation and analysis.
        runs_dir: Base path for per-session run directories.
        analyze_every_n: Frequency for running progress analysis.
        max_questions_per_source: Max questions per source file (0 = unlimited).
        model_name: Name of LLM model used (for logging).
        sessions: Active sessions keyed by session_id.
    """
    content_index: Dict[str, Any]
    llm_gen: Any
    llm_eval: Any
    runs_dir: str
    analyze_every_n: int
    max_questions_per_source: int
    model_name: str

    sessions: Dict[str, SessionState] = field(default_factory=dict)

    def create_session(self, session_id: str) -> SessionState:
        """
        Create and register a new session, snapshotting available topics.

        Args:
            session_id: Unique identifier of the websocket client.

        Returns:
            SessionState initialized with history paths.
        """
        run_dir = new_run_dir(self.runs_dir)
        history_path = f"{run_dir}/history.jsonl"
        with open(f"{run_dir}/topics_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(
                {tid: {"title": t["title"], "source_file": t["source_file"]} for tid, t in self.content_index["topics"].items()},
                f,
                ensure_ascii=False,
                indent=2,
            )
        st = SessionState(session_id=session_id, run_dir=run_dir, history_path=history_path)
        self.sessions[session_id] = st
        return st

    def get_session(self, session_id: str) -> SessionState:
        """
        Return existing session or create a new one.

        Args:
            session_id: Unique identifier of the websocket client.

        Returns:
            SessionState bound to the given id.
        """
        if session_id not in self.sessions:
            return self.create_session(session_id)
        return self.sessions[session_id]

    async def _emit(self, emit: EmitFunc, payload: Dict[str, Any]) -> None:
        """
        Send a payload to the websocket.

        Args:
            emit: Async callable that serializes and sends messages.
            payload: JSON-serializable payload.
        """
        await emit(payload)

    async def set_status(self, emit: EmitFunc, agent: str, step: str, state: str, st: SessionState) -> None:
        """
        Emit a status update to the client.

        Args:
            emit: Async sender.
            agent: Name of logical agent (TopicPicker, Evaluator, etc.).
            step: Human-readable status detail.
            state: One of running/error/idle.
            st: Session state to include counter from.
        """
        await self._emit(emit, {"type": "status", "agent": agent, "step": step, "state": state, "counter": st.counter})

    async def add_event(self, emit: EmitFunc, agent: str, level: str, text: str) -> None:
        """
        Emit a log/event entry.

        Args:
            emit: Async sender.
            agent: Component name.
            level: info/warn/error.
            text: Message to display.
        """
        await self._emit(emit, {"type": "event", "agent": agent, "level": level, "text": text})

    async def add_message(self, emit: EmitFunc, role: str, title: str, body: str) -> None:
        """
        Emit a chat-style message for the feed.

        Args:
            emit: Async sender.
            role: user/system.
            title: Header text.
            body: Markdown/text body.
        """
        await self._emit(emit, {"type": "message", "role": role, "title": title, "body": body})

    async def start_or_next_question(self, session_id: str, emit: EmitFunc) -> SessionState:
        """
        Select the next topic/question, prepare expected rubric, and notify client.

        Args:
            session_id: Current session id.
            emit: Async sender.

        Returns:
            Updated SessionState after scheduling the question.
        """
        st = self.get_session(session_id)
        all_topics = self.content_index["topics"]
        topics = {
            tid: t
            for tid, t in all_topics.items()
            if not st.allowed_sources or t.get("source_file") in st.allowed_sources
        }
        if self.max_questions_per_source > 0:
            topics = {
                tid: t
                for tid, t in topics.items()
                if st.source_question_counts.get(t.get("source_file", ""), 0) < self.max_questions_per_source
            }
        topics = {tid: t for tid, t in topics.items() if tid not in st.asked_topics}
        if not topics:
            await self.add_event(
                emit,
                "Server",
                "warn",
                "Нет доступных тем: исчерпан стоп-лист или лимит вопросов по файлам.",
            )
            return st

        await self.set_status(emit, "TopicPicker", "Выбираю тему…", "running", st)
        tid = pick_topic(topics, st.progress, st.recently_topics, st.analyzer_hint)
        st.selected_topic_id = tid
        st.asked_topics.add(tid)
        await self.add_event(emit, "TopicPicker", "info", f"Тема: {tid} · {topics[tid]['title']}")
        await self.set_status(emit, "TopicPicker", "Готово", "idle", st)

        await self.set_status(emit, "QuestionPicker", "Выбираю/формулирую вопрос…", "running", st)
        topic = topics[tid]
        q, expected, ctx = pick_question(topic)

        if not q:
            await self.add_event(emit, "QuestionPicker", "info", "В конспекте нет явных вопросов — генерирую из теории…")
            prompt = f"""Сгенерируй 1 вопрос для собеседования по теме ниже.

ТРЕБОВАНИЯ:
- Верни ТОЛЬКО сам вопрос, без пояснений, без "коротко", без подсказок, без ответа.
- Вопрос должен быть самодостаточным (кандидат должен думать сам).
- Не включай в вопрос определение/подсказку/полуответ.
- Не добавляй варианты ответа.
- 1–2 предложения максимум.

ТЕМА: {topic['title']}

КОНТЕКСТ (фрагмент конспекта):
{ctx}
"""
            q = await run_in_threadpool(lambda: self.llm_gen.invoke(prompt).content.strip())
            expected = None

        if expected is None:
            await self.add_event(emit, "QuestionPicker", "info", "Генерирую expected (чеклист)…")
            prompt = f"""Составь рубрику идеального ответа для оценки.
Верни ТОЛЬКО JSON:
{{
  "rubric": [
    {{"point":"...", "importance": 1-3, "notes":"кратко что ожидается"}}
  ],
  "ideal_answer": "короткий идеальный ответ 5-10 предложений"
}}

ВОПРОС: {q}

ТЕМА: {topic['title']}
ТЕОРИЯ (фрагмент):
{ctx}
"""
            expected = await run_in_threadpool(lambda: self.llm_gen.invoke(prompt).content.strip())

        st.selected_question = q
        try:
            expected_json = robust_json_load(expected)
            expected = json.dumps(expected_json, ensure_ascii=False, indent=2)
        except Exception:
            pass

        st.expected_answer = expected
        st.theory_context = ctx
        source_key = topic.get("source_file", "")
        st.source_question_counts[source_key] = st.source_question_counts.get(source_key, 0) + 1

        await self.set_status(emit, "QuestionPicker", "Готово", "idle", st)

        await self._emit(
            emit,
            {
                "type": "question",
                "topic_id": tid,
                "topic_title": topic["title"],
                "source_file": topic["source_file"],
                "question": q,
                "theory": (ctx or "")[:2500],
            },
        )

        await self._emit(emit, {"type": "need_input"})
        return st

    async def submit_answer(self, session_id: str, answer: str, emit: EmitFunc) -> SessionState:
        """
        Evaluate user's answer, emit feedback, update history, and advance.

        Args:
            session_id: Current session id.
            answer: Raw user answer text.
            emit: Async sender.

        Returns:
            SessionState after processing and moving to next question.

        Raises:
            ValueError: If JSON parsing fails even after recovery.
        """
        st = self.get_session(session_id)
        if not st.selected_question or not st.selected_topic_id:
            await self.add_event(emit, "Server", "warn", "Ответ пришел без активного вопроса — задаю новый вопрос.")
            return await self.start_or_next_question(session_id, emit)

        await self.add_message(emit, "user", "Ответ", answer)
        await self.set_status(emit, "Evaluator", "Оцениваю ответ…", "running", st)

        topics = self.content_index["topics"]
        topic = topics[st.selected_topic_id]
        q = st.selected_question
        expected = st.expected_answer or ""
        ctx = st.theory_context or ""

        user_prompt = f"""ВОПРОС:
{q}

ОЖИДАНИЕ / ТЕОРИЯ (JSON-рубрика или текст чеклиста):
{expected}

КОНТЕКСТ (фрагмент конспекта):
{ctx}

ОТВЕТ КАНДИДАТА:
{answer}
"""

        resp = await run_in_threadpool(
            lambda: self.llm_eval.invoke(
                [
                    {"role": "system", "content": EVAL_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ]
            ).content
        )

        try:
            data = robust_json_load(resp)
            eval_res = EvalResult(**data)
        except Exception as e:
            await self.add_event(emit, "Evaluator", "warn", f"JSON сломан, восстанавливаю… ({type(e).__name__})")
            fix = await run_in_threadpool(
                lambda: self.llm_eval.invoke(
                    [
                        {"role": "system", "content": "Верни ТОЛЬКО валидный JSON по схеме, без комментариев."},
                        {"role": "user", "content": f"Схема:\n{EVAL_SYSTEM}\n\nТекст:\n{resp}"},
                    ]
                ).content
            )
            data = robust_json_load(fix)
            eval_res = EvalResult(**data)

        score = int(eval_res.score)

        await self._emit(
            emit,
            {
                "type": "feedback",
                "score": score,
                "short_verdict": eval_res.short_verdict,
                "missing_points": eval_res.missing_points,
                "incorrect_points": eval_res.incorrect_points,
                "improvement_tips": eval_res.improvement_tips,
                "ideal_answer": eval_res.ideal_answer,
            },
        )

        await self.set_status(emit, "Evaluator", "Готово", "idle", st)

        st.counter += 1
        await self._emit(emit, {"type": "counter", "counter": st.counter})

        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "n": st.counter,
            "topic_id": st.selected_topic_id,
            "topic_title": topic["title"],
            "question": q,
            "user_answer": answer,
            "expected": expected,
            "score": score,
            "short_verdict": eval_res.short_verdict,
            "missing_points": eval_res.missing_points,
            "incorrect_points": eval_res.incorrect_points,
            "improvement_tips": eval_res.improvement_tips,
            "ideal_answer": eval_res.ideal_answer,
            "model": self.model_name,
        }
        append_jsonl(st.history_path, record)
        st.progress = build_progress_from_history(load_jsonl(st.history_path))
        st.recently_topics = (st.recently_topics + [st.selected_topic_id])[-10:]

        if st.counter > 0 and (st.counter % self.analyze_every_n == 0):
            await self.run_analysis(session_id, emit)

        return await self.start_or_next_question(session_id, emit)

    async def run_analysis(self, session_id: str, emit: EmitFunc) -> AnalysisResult:
        """
        Run periodic progress analysis and emit a summary message.

        Args:
            session_id: Current session id.
            emit: Async sender.

        Returns:
            AnalysisResult emitted to the client.
        """
        st = self.get_session(session_id)
        await self.set_status(emit, "ProgressAnalyzer", f"Анализ прогресса (каждые {self.analyze_every_n})…", "running", st)

        history = load_jsonl(st.history_path)[-120:]
        compact = [
            {
                "n": h.get("n"),
                "topic_id": h.get("topic_id"),
                "score": h.get("score"),
                "question": h.get("question"),
                "short_verdict": h.get("short_verdict", ""),
                "missing_points": (h.get("missing_points") or [])[:5],
            }
            for h in history
        ]

        user_prompt = f"""История (последние {len(compact)}):
{json.dumps(compact, ensure_ascii=False)}

Список тем (topic_id -> title):
{json.dumps({tid: t['title'] for tid,t in self.content_index['topics'].items()}, ensure_ascii=False)}

Сделай анализ готовности и верни JSON по схеме.
"""

        resp = await run_in_threadpool(
            lambda: self.llm_eval.invoke(
                [
                    {"role": "system", "content": ANALYSIS_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ]
            ).content
        )

        try:
            data = robust_json_load(resp)
            analysis = AnalysisResult(**data)
        except Exception as e:
            await self.add_event(emit, "ProgressAnalyzer", "warn", f"JSON сломан, восстанавливаю… ({type(e).__name__})")
            fix = await run_in_threadpool(
                lambda: self.llm_eval.invoke(
                    [
                        {"role": "system", "content": "Верни ТОЛЬКО валидный JSON по схеме, без комментариев."},
                        {"role": "user", "content": f"Схема:\n{ANALYSIS_SYSTEM}\n\nТекст:\n{resp}"},
                    ]
                ).content
            )
            data = robust_json_load(fix)
            analysis = AnalysisResult(**data)

        topics = self.content_index["topics"]
        lines = [analysis.overall_summary.strip()]
        if analysis.strong_topics:
            lines.append(
                "\n**Сильные темы**\n- "
                + "\n- ".join([f"{tid}: {topics[tid]['title']}" for tid in analysis.strong_topics if tid in topics])
            )
        if analysis.weak_topics:
            lines.append(
                "\n**Слабые темы**\n- "
                + "\n- ".join([f"{tid}: {topics[tid]['title']}" for tid in analysis.weak_topics if tid in topics])
            )
        if analysis.recommendations:
            lines.append("\n**Что повторить/как улучшить**\n- " + "\n- ".join(analysis.recommendations))

        await self._emit(emit, {"type": "analysis", "title": f"Progress report (после {st.counter} вопросов)", "body": "\n".join(lines)})
        st.analyzer_hint = {"weak_topics": analysis.weak_topics, "strong_topics": analysis.strong_topics, "topic_scores": analysis.topic_scores}
        st.last_analysis = data

        await self.set_status(emit, "ProgressAnalyzer", "Готово", "idle", st)
        return analysis

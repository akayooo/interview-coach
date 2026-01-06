"""Configuration loader for Interview Coach environment variables."""

import os

from dotenv import load_dotenv
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    """
    Immutable configuration populated from environment variables.

    Attributes:
        notes_dir: Directory with markdown conspects.
        runs_dir: Directory for per-session artifacts.
        analyze_every_n: Frequency for progress analysis.
        max_questions_per_source: Max questions per source file (0 = unlimited).
        openrouter_api_key: API key for OpenRouter.
        openrouter_model: Model name for generation/evaluation.
        app_http_referer: HTTP referer header for OpenRouter.
        app_title: Title header for OpenRouter.
        llm_timeout_s: Timeout for LLM calls in seconds.
        whisper_model: faster-whisper model id.
        whisper_device: Device to run whisper on (cpu/cuda).
        whisper_compute_type: Precision for whisper.
        whisper_initial_prompt: Domain prompt for STT.
        whisper_language: Optional language hint (None/auto if blank).
        whisper_beam_size: Beam size for decoding (lower = faster).
        whisper_condition_on_prev: Whether to condition on previous text.
    """
    notes_dir: str = "conspects"
    runs_dir: str = "runs"
    analyze_every_n: int = 30
    max_questions_per_source: int = 0

    openrouter_api_key: str = ""
    openrouter_model: str = "deepseek/deepseek-chat"

    app_http_referer: str = "http://localhost"
    app_title: str = "InterviewCoachWeb"
    llm_timeout_s: int = 120

    whisper_model: str = "large-v3"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_initial_prompt: str = (
        "Речь в основном на русском, но часто встречаются английские термины и аббревиатуры: "
        "overfitting, regularization, gradient, attention, retrieval, ranking, embedding, CLIP, RAG."
    )
    whisper_language: str = ""
    whisper_beam_size: int = 2
    whisper_condition_on_prev: bool = True

    @staticmethod
    def from_env() -> "Config":
        """
        Build Config from process environment.

        Returns:
            Config instance with environment overrides applied.
        """
        load_dotenv()
        return Config(
            notes_dir=os.environ.get("NOTES_DIR", "conspects"),
            runs_dir=os.environ.get("RUNS_DIR", "runs"),
            analyze_every_n=int(os.environ.get("ANALYZE_EVERY_N", "30")),
            max_questions_per_source=int(os.environ.get("MAX_QUESTIONS_PER_SOURCE", "0")),
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", "").strip(),
            openrouter_model=os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat"),
            app_http_referer=os.environ.get("APP_HTTP_REFERER", "http://localhost"),
            app_title=os.environ.get("APP_TITLE", "InterviewCoachWeb"),
            llm_timeout_s=int(os.environ.get("LLM_TIMEOUT_S", "120")),
            whisper_model=os.environ.get("WHISPER_MODEL", "large-v3"),
            whisper_device=os.environ.get("WHISPER_DEVICE", "cpu"),
            whisper_compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "int8"),
            whisper_initial_prompt=os.environ.get(
                "WHISPER_INITIAL_PROMPT",
                "Речь в основном на русском, но часто встречаются английские термины и аббревиатуры: "
                "overfitting, regularization, gradient, attention, retrieval, ranking, embedding, CLIP, RAG.",
            ),
            whisper_language=os.environ.get("WHISPER_LANGUAGE", ""),
            whisper_beam_size=int(os.environ.get("WHISPER_BEAM_SIZE", "2")),
            whisper_condition_on_prev=os.environ.get("WHISPER_CONDITION_ON_PREV", "true").lower() not in {"0", "false", "no"},
        )

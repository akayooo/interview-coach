"""Speech-to-text helpers using faster-whisper."""

import os
import tempfile
from typing import Optional, Tuple

from faster_whisper import WhisperModel


def load_whisper_model(model_name: str, device: str = "cpu", compute_type: str = "int8") -> WhisperModel:
    """
    Load a faster-whisper model.

    Args:
        model_name: Model id (e.g., "large-v3").
        device: "cpu" or "cuda".
        compute_type: Precision type ("int8", "float16", etc.).

    Returns:
        Initialized WhisperModel.
    """
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe_bytes(
    model: WhisperModel,
    audio_bytes: bytes,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    beam_size: int = 2,
    condition_on_previous_text: bool = True,
) -> Tuple[str, dict]:
    """
    Transcribe audio bytes with VAD and deterministic decoding.

    Args:
        model: Loaded WhisperModel.
        audio_bytes: Raw audio bytes (webm/opus/etc.; ffmpeg handles decode).
        language: Optional language hint (None for auto).
        initial_prompt: Domain prompt for tech terms.
        beam_size: Beam width (lower is faster).
        condition_on_previous_text: Whether to condition on previous text.

    Returns:
        Tuple of (transcribed text, info dict with language/duration).
    """
    fd, path = tempfile.mkstemp(suffix=".webm")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        segments, info = model.transcribe(
            path,
            task="transcribe",
            language=language,
            vad_filter=True,
            beam_size=beam_size,
            temperature=0.0,
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
            word_timestamps=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text, {"language": info.language, "duration": info.duration}
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

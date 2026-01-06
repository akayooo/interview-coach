"""OpenRouter LLM client builders."""

from typing import Tuple

from langchain_openai import ChatOpenAI

def make_llm(api_key: str, model: str, temperature: float, http_referer: str, app_title: str, timeout_s: int = 120) -> ChatOpenAI:
    """
    Construct a ChatOpenAI client configured for OpenRouter.

    Args:
        api_key: OpenRouter API key.
        model: Model id.
        temperature: Sampling temperature.
        http_referer: HTTP referer header for OpenRouter.
        app_title: App title header for OpenRouter.
        timeout_s: Request timeout in seconds.

    Returns:
        Configured ChatOpenAI instance.

    Raises:
        RuntimeError: If api_key is empty.
    """
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY пуст. Установи ключ в env.")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": http_referer, "X-Title": app_title},
        timeout=timeout_s,
    )

def build_llms(api_key: str, model: str, http_referer: str, app_title: str, timeout_s: int = 120) -> Tuple[ChatOpenAI, ChatOpenAI]:
    """
    Build paired LLMs for generation and evaluation.

    Args:
        api_key: OpenRouter API key.
        model: Model id.
        http_referer: HTTP referer header for OpenRouter.
        app_title: App title header.
        timeout_s: Request timeout in seconds.

    Returns:
        Tuple of (llm_gen, llm_eval).
    """
    llm_gen = make_llm(api_key, model, temperature=0.6, http_referer=http_referer, app_title=app_title, timeout_s=timeout_s)
    llm_eval = make_llm(api_key, model, temperature=0.1, http_referer=http_referer, app_title=app_title, timeout_s=timeout_s)
    return llm_gen, llm_eval

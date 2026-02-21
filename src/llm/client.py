"""
LLM client with multi-model support, circuit breaker, and graceful degradation.
Uses GPT-OSS on Groq (primary), with Ollama and HuggingFace fallbacks.

GPT-OSS models are reasoning models — they produce internal reasoning tokens
before the final output. The client automatically adds token overhead for these.

Supports multiple Groq API keys for higher throughput (round-robin rotation).
With Groq Developer tier, a single key provides ~300 RPM (5 req/sec).
"""
from __future__ import annotations


import logging
import os
import threading
from typing import Any, Optional

from openai import OpenAI

from src.config import get_config
from src.utils.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    GracefulDegradation,
    retry_with_jitter,
)

logger = logging.getLogger(__name__)

# Circuit breakers for each backend
_groq_circuit_breaker: Optional[CircuitBreaker] = None
_ollama_circuit_breaker: Optional[CircuitBreaker] = None
_hf_circuit_breaker: Optional[CircuitBreaker] = None

# Groq key rotation state (thread-safe)
_groq_key_index: int = 0
_groq_key_lock = threading.Lock()


def _is_reasoning_model(model_name: str) -> bool:
    """Check if a model is a reasoning model (GPT-OSS) that uses reasoning tokens."""
    return "gpt-oss" in model_name.lower() and "safeguard" not in model_name.lower()


def _get_groq_circuit_breaker() -> CircuitBreaker:
    """Get or create Groq circuit breaker."""
    global _groq_circuit_breaker
    if _groq_circuit_breaker is None:
        config = get_config()
        _groq_circuit_breaker = CircuitBreaker(
            failure_threshold=config.resilience.circuit_breaker_threshold,
            recovery_timeout=config.resilience.circuit_breaker_timeout,
            half_open_max_calls=config.resilience.circuit_breaker_half_open_calls,
            name="groq"
        )
    return _groq_circuit_breaker


def _get_ollama_circuit_breaker() -> CircuitBreaker:
    """Get or create Ollama circuit breaker."""
    global _ollama_circuit_breaker
    if _ollama_circuit_breaker is None:
        config = get_config()
        _ollama_circuit_breaker = CircuitBreaker(
            failure_threshold=config.resilience.circuit_breaker_threshold,
            recovery_timeout=config.resilience.circuit_breaker_timeout,
            half_open_max_calls=config.resilience.circuit_breaker_half_open_calls,
            name="ollama"
        )
    return _ollama_circuit_breaker


def _get_hf_circuit_breaker() -> CircuitBreaker:
    """Get or create HuggingFace circuit breaker."""
    global _hf_circuit_breaker
    if _hf_circuit_breaker is None:
        config = get_config()
        _hf_circuit_breaker = CircuitBreaker(
            failure_threshold=config.resilience.circuit_breaker_threshold,
            recovery_timeout=config.resilience.circuit_breaker_timeout,
            half_open_max_calls=config.resilience.circuit_breaker_half_open_calls,
            name="huggingface"
        )
    return _hf_circuit_breaker


def _get_next_groq_key() -> Optional[str]:
    """
    Get the next Groq API key using round-robin rotation.
    Thread-safe for concurrent requests.

    Returns None if no keys are configured.
    """
    global _groq_key_index
    config = get_config()
    keys = config.llm.groq_api_keys

    if not keys:
        return None

    with _groq_key_lock:
        key = keys[_groq_key_index % len(keys)]
        _groq_key_index = (_groq_key_index + 1) % len(keys)
        return key


def _make_groq_client() -> Optional[OpenAI]:
    """
    Create Groq OpenAI-compatible client with round-robin key rotation.

    With Groq Developer tier (paid), a single key provides:
    - ~300 RPM (5 req/sec) — matches Clay's minimum rate
    - Multiple keys provide redundancy and higher throughput
    """
    api_key = _get_next_groq_key()
    if not api_key:
        return None

    config = get_config()
    return OpenAI(
        base_url=config.llm.groq_base_url,
        api_key=api_key,
        timeout=config.llm.timeout
    )


def _make_ollama_client() -> OpenAI:
    """Create Ollama OpenAI-compatible client."""
    config = get_config()
    return OpenAI(
        base_url=config.llm.ollama_base_url,
        api_key="ollama",  # Ollama doesn't need a real key
        timeout=config.llm.timeout
    )


def _make_hf_client() -> Optional[OpenAI]:
    """Create HuggingFace OpenAI-compatible client."""
    config = get_config()
    token = config.llm.hf_token
    if not token:
        return None
    return OpenAI(
        base_url=config.llm.hf_base_url,
        api_key=token,
        timeout=config.llm.timeout
    )


def _chat_with_groq(
    messages: list[dict[str, Any]],
    max_tokens: int,
    model: Optional[str] = None
) -> str:
    """
    Chat using Groq with automatic model fallback.
    Handles GPT-OSS reasoning models by adding token overhead
    and extracting content from reasoning responses.
    """
    config = get_config()
    client = _make_groq_client()

    if not client:
        raise RuntimeError("Groq API key not configured (set GROQ_API_KEY in .env)")

    models = [model] if model else config.llm.groq_models
    last_error: Optional[Exception] = None

    for model_name in models:
        try:
            logger.debug("Trying Groq model: %s", model_name)

            # GPT-OSS reasoning models use reasoning tokens that count toward max_tokens.
            # Add overhead so there's room for both reasoning + output.
            effective_max_tokens = max_tokens
            if _is_reasoning_model(model_name):
                effective_max_tokens = max_tokens + config.llm.gpt_oss_reasoning_overhead

            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=effective_max_tokens,
            )
            if resp.choices and resp.choices[0].message:
                content = resp.choices[0].message.content
                if content:
                    logger.info("Successfully used Groq model: %s", model_name)
                    return content
            raise RuntimeError(f"Empty response from Groq model {model_name}")
        except Exception as e:
            last_error = e
            logger.warning("Groq model %s failed: %s", model_name, str(e)[:100])
            continue

    raise last_error or RuntimeError("All Groq models failed")


def _chat_with_ollama(
    messages: list[dict[str, Any]],
    max_tokens: int,
    model: Optional[str] = None
) -> str:
    """
    Chat using Ollama with automatic model fallback.
    Tries each configured model in order until one works.
    """
    config = get_config()
    client = _make_ollama_client()
    models = [model] if model else config.llm.ollama_models

    last_error: Optional[Exception] = None

    for model_name in models:
        try:
            logger.debug("Trying Ollama model: %s", model_name)
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
            )
            if resp.choices and resp.choices[0].message:
                content = resp.choices[0].message.content
                if content:
                    logger.info("Successfully used Ollama model: %s", model_name)
                    return content
            raise RuntimeError(f"Empty response from Ollama model {model_name}")
        except Exception as e:
            last_error = e
            logger.warning("Ollama model %s failed: %s", model_name, str(e)[:100])
            continue

    raise last_error or RuntimeError("All Ollama models failed")


def _chat_with_hf(
    messages: list[dict[str, Any]],
    max_tokens: int,
    model: Optional[str] = None
) -> str:
    """
    Chat using HuggingFace Inference API with model fallback.
    Uses free tier - may be rate limited.
    """
    config = get_config()
    client = _make_hf_client()

    if not client:
        raise RuntimeError("HuggingFace token not configured (set HF_TOKEN in .env)")

    models = [model] if model else config.llm.hf_models
    last_error: Optional[Exception] = None

    for model_name in models:
        try:
            logger.debug("Trying HuggingFace model: %s", model_name)
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
            )
            if resp.choices and resp.choices[0].message:
                content = resp.choices[0].message.content
                if content:
                    logger.info("Successfully used HuggingFace model: %s", model_name)
                    return content
            raise RuntimeError(f"Empty response from HuggingFace model {model_name}")
        except Exception as e:
            last_error = e
            logger.warning("HuggingFace model %s failed: %s", model_name, str(e)[:100])
            continue

    raise last_error or RuntimeError("All HuggingFace models failed")


@retry_with_jitter(max_retries=2, base_delay=0.5, max_delay=5.0)
def _groq_with_retry(
    messages: list[dict[str, Any]],
    max_tokens: int,
    model: Optional[str] = None
) -> str:
    """Groq chat with retry and circuit breaker."""
    breaker = _get_groq_circuit_breaker()
    return breaker.execute(_chat_with_groq, messages, max_tokens, model)


@retry_with_jitter(max_retries=2, base_delay=1.0, max_delay=10.0)
def _ollama_with_retry(
    messages: list[dict[str, Any]],
    max_tokens: int,
    model: Optional[str] = None
) -> str:
    """Ollama chat with retry and circuit breaker."""
    breaker = _get_ollama_circuit_breaker()
    return breaker.execute(_chat_with_ollama, messages, max_tokens, model)


@retry_with_jitter(max_retries=2, base_delay=2.0, max_delay=15.0)
def _hf_with_retry(
    messages: list[dict[str, Any]],
    max_tokens: int,
    model: Optional[str] = None
) -> str:
    """HuggingFace chat with retry and circuit breaker."""
    breaker = _get_hf_circuit_breaker()
    return breaker.execute(_chat_with_hf, messages, max_tokens, model)


def chat(
    messages: list[dict[str, Any]],
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
    prefer_local: bool = False,
) -> str:
    """
    Send messages to LLM with automatic fallback.

    Default order: Groq GPT-OSS (fast cloud) -> Ollama (local) -> HuggingFace (backup).
    All backends have circuit breakers to prevent cascading failures.

    Args:
        messages: List of message dicts with 'role' and 'content'
        max_tokens: Maximum tokens in response (default from config).
                    For GPT-OSS models, reasoning overhead is added automatically.
        model: Specific model to use (optional, will use fallback chain if not specified)
        prefer_local: If True, try Ollama first (useful for offline mode)

    Returns:
        Assistant message content as string

    Raises:
        RuntimeError: If all backends fail
    """
    config = get_config()
    max_tokens = max_tokens or config.llm.max_tokens_default

    # Build fallback chain: Groq (fastest) -> Ollama (local) -> HuggingFace (backup)
    if prefer_local:
        handlers = [
            (lambda m=messages, t=max_tokens, mod=model: _ollama_with_retry(m, t, mod), "Ollama"),
            (lambda m=messages, t=max_tokens, mod=model: _groq_with_retry(m, t, mod), "Groq"),
            (lambda m=messages, t=max_tokens, mod=model: _hf_with_retry(m, t, mod), "HuggingFace"),
        ]
    else:
        # Check if Groq is configured, otherwise start with Ollama
        if config.llm.groq_api_key:
            handlers = [
                (lambda m=messages, t=max_tokens, mod=model: _groq_with_retry(m, t, mod), "Groq"),
                (lambda m=messages, t=max_tokens, mod=model: _ollama_with_retry(m, t, mod), "Ollama"),
                (lambda m=messages, t=max_tokens, mod=model: _hf_with_retry(m, t, mod), "HuggingFace"),
            ]
        else:
            handlers = [
                (lambda m=messages, t=max_tokens, mod=model: _ollama_with_retry(m, t, mod), "Ollama"),
                (lambda m=messages, t=max_tokens, mod=model: _hf_with_retry(m, t, mod), "HuggingFace"),
            ]

    degradation = GracefulDegradation(handlers)

    try:
        result, backend, tier = degradation.execute()
        if tier > 0:
            logger.info("Used fallback backend: %s", backend)
        return result
    except CircuitBreakerOpenError as e:
        logger.error("All circuit breakers open: %s", e)
        raise RuntimeError(
            "LLM service temporarily unavailable. "
            "Set GROQ_API_KEY (fastest), ensure Ollama is running, or set HF_TOKEN."
        ) from e
    except Exception as e:
        logger.error("All LLM backends failed: %s", e)
        raise RuntimeError(
            "LLM request failed. "
            "Set GROQ_API_KEY for best speed, or check Ollama status (ollama list)."
        ) from e


def chat_with_json_output(
    messages: list[dict[str, Any]],
    max_tokens: Optional[int] = None,
) -> str:
    """
    Chat expecting JSON output. Adds instruction to return valid JSON.
    Used for structured outputs like query decomposition, claim extraction, etc.
    """
    # Add JSON instruction to system message or create one
    json_instruction = "\nIMPORTANT: Respond with valid JSON only. No markdown, no explanations."

    enhanced_messages = []
    has_system = False

    for msg in messages:
        if msg.get("role") == "system":
            enhanced_messages.append({
                "role": "system",
                "content": msg["content"] + json_instruction
            })
            has_system = True
        else:
            enhanced_messages.append(msg)

    if not has_system:
        enhanced_messages.insert(0, {
            "role": "system",
            "content": "You are a helpful assistant that outputs valid JSON." + json_instruction
        })

    return chat(enhanced_messages, max_tokens)


def health_check() -> dict[str, Any]:
    """
    Check health of LLM backends.

    Returns dict with status of each backend:
    {
        "groq": {"available": bool, "configured": bool, "num_keys": int, "models": list, "error": str or None},
        "ollama": {"available": bool, "models": list, "error": str or None},
        "huggingface": {"available": bool, "configured": bool, "error": str or None}
    }
    """
    config = get_config()
    num_groq_keys = len(config.llm.groq_api_keys)
    status = {
        "groq": {
            "available": False,
            "configured": num_groq_keys > 0,
            "num_keys": num_groq_keys,
            "primary_model": config.llm.groq_models[0] if config.llm.groq_models else None,
            "models": [],
            "error": None
        },
        "ollama": {"available": False, "models": [], "error": None},
        "huggingface": {"available": False, "configured": False, "error": None}
    }

    # Check Groq (fastest option)
    if num_groq_keys > 0:
        try:
            client = _make_groq_client()
            if client:
                # Try a simple completion to verify (use non-reasoning model for health check)
                resp = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5,
                )
                if resp.choices:
                    status["groq"]["available"] = True
                    status["groq"]["models"] = config.llm.groq_models
        except Exception as e:
            status["groq"]["error"] = str(e)[:200]

    # Check Ollama
    try:
        client = _make_ollama_client()
        resp = client.chat.completions.create(
            model=config.llm.ollama_models[0],
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        if resp.choices:
            status["ollama"]["available"] = True
            status["ollama"]["models"] = config.llm.ollama_models
    except Exception as e:
        status["ollama"]["error"] = str(e)[:200]

    # Check HuggingFace
    status["huggingface"]["configured"] = config.llm.hf_token is not None
    if config.llm.hf_token:
        try:
            client = _make_hf_client()
            if client:
                status["huggingface"]["available"] = True
        except Exception as e:
            status["huggingface"]["error"] = str(e)[:200]

    return status


def reset_circuit_breakers() -> None:
    """Reset all circuit breakers (useful for testing or recovery)."""
    global _groq_circuit_breaker, _ollama_circuit_breaker, _hf_circuit_breaker
    if _groq_circuit_breaker:
        _groq_circuit_breaker.reset()
    if _ollama_circuit_breaker:
        _ollama_circuit_breaker.reset()
    if _hf_circuit_breaker:
        _hf_circuit_breaker.reset()
    logger.info("All circuit breakers reset")

"""
LLM client with multi-model support, circuit breaker, and graceful degradation.
Uses GPT-OSS on DeepInfra (primary, cheapest), with Groq fallback, plus Ollama
and HuggingFace as local/free fallbacks.

GPT-OSS models are reasoning models — they produce internal reasoning tokens
before the final output. The client automatically adds token overhead for these.

DeepInfra: 200 concurrent requests, no daily cap, $0.03/$0.14 per 1M tokens.
Groq (fallback): ~300 RPM, $0.075/$0.30 per 1M tokens.
"""
from __future__ import annotations


import logging
import os
import threading
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# ChatResult: structured return type with usage & cost
# ---------------------------------------------------------------------------

@dataclass
class ChatResult:
    """Result from an LLM chat call with token usage and cost."""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    backend: str  # "groq", "ollama", "huggingface"


# Pricing per 1M tokens: (input_price, output_price)
# Backend-specific pricing is resolved via _calculate_cost(model, backend, ...)
DEEPINFRA_PRICING: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-20b": (0.03, 0.14),
}

GROQ_PRICING: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-70b-versatile": (0.59, 0.79),
    "llama3-70b-8192": (0.59, 0.79),
    "llama3-8b-8192": (0.05, 0.08),
}


def _calculate_cost(model: str, prompt_tokens: int, completion_tokens: int, backend: str = "groq") -> float:
    """Calculate cost in USD based on backend, model pricing and token counts."""
    pricing_table = DEEPINFRA_PRICING if backend == "deepinfra" else GROQ_PRICING
    pricing = pricing_table.get(model)
    if not pricing:
        # Partial match fallback
        for key, prices in pricing_table.items():
            if key in model or model in key:
                pricing = prices
                break
    if not pricing:
        return 0.0
    input_price, output_price = pricing
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000


# ---------------------------------------------------------------------------
# Thread-local cost accumulator (for multi-call endpoints like /research)
# ---------------------------------------------------------------------------

_cost_tracker = threading.local()


def start_cost_tracking() -> None:
    """Start accumulating LLM costs for this thread."""
    _cost_tracker.total_cost = 0.0
    _cost_tracker.total_prompt_tokens = 0
    _cost_tracker.total_completion_tokens = 0


def get_accumulated_cost() -> dict[str, Any]:
    """Return accumulated cost since last start_cost_tracking()."""
    return {
        "total_cost_usd": round(getattr(_cost_tracker, "total_cost", 0.0), 8),
        "total_prompt_tokens": getattr(_cost_tracker, "total_prompt_tokens", 0),
        "total_completion_tokens": getattr(_cost_tracker, "total_completion_tokens", 0),
    }


def _accumulate_cost(result: ChatResult) -> None:
    """Add a ChatResult's cost to the thread-local accumulator."""
    if hasattr(_cost_tracker, "total_cost"):
        _cost_tracker.total_cost += result.cost_usd
        _cost_tracker.total_prompt_tokens += result.prompt_tokens
        _cost_tracker.total_completion_tokens += result.completion_tokens

# Circuit breakers for each backend
_deepinfra_circuit_breaker: Optional[CircuitBreaker] = None
_groq_circuit_breaker: Optional[CircuitBreaker] = None
_ollama_circuit_breaker: Optional[CircuitBreaker] = None
_hf_circuit_breaker: Optional[CircuitBreaker] = None

# Groq key rotation state (thread-safe)
_groq_key_index: int = 0
_groq_key_lock = threading.Lock()


def _is_reasoning_model(model_name: str) -> bool:
    """Check if a model is a reasoning model (GPT-OSS) that uses reasoning tokens."""
    return "gpt-oss" in model_name.lower() and "safeguard" not in model_name.lower()


def _get_deepinfra_circuit_breaker() -> CircuitBreaker:
    """Get or create DeepInfra circuit breaker."""
    global _deepinfra_circuit_breaker
    if _deepinfra_circuit_breaker is None:
        config = get_config()
        _deepinfra_circuit_breaker = CircuitBreaker(
            failure_threshold=config.resilience.circuit_breaker_threshold,
            recovery_timeout=config.resilience.circuit_breaker_timeout,
            half_open_max_calls=config.resilience.circuit_breaker_half_open_calls,
            name="deepinfra"
        )
    return _deepinfra_circuit_breaker


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


def _make_deepinfra_client() -> Optional[OpenAI]:
    """
    Create DeepInfra OpenAI-compatible client.

    DeepInfra provides 200 concurrent requests (no daily cap).
    54% cheaper than Groq for GPT-OSS 20B.
    """
    config = get_config()
    api_key = config.llm.deepinfra_api_key
    if not api_key:
        return None

    return OpenAI(
        base_url=config.llm.deepinfra_base_url,
        api_key=api_key,
        timeout=config.llm.timeout
    )


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


def _chat_with_deepinfra(
    messages: list[dict[str, Any]],
    max_tokens: int,
    model: Optional[str] = None
) -> ChatResult:
    """
    Chat using DeepInfra (PRIMARY backend).
    Same GPT-OSS 20B model at 54% lower cost, 200 concurrent requests.
    Returns ChatResult with token usage and cost.
    """
    config = get_config()
    client = _make_deepinfra_client()

    if not client:
        raise RuntimeError("DeepInfra API key not configured (set DEEPINFRA_API_KEY in .env)")

    models = [model] if model else config.llm.deepinfra_models
    last_error: Optional[Exception] = None

    for model_name in models:
        try:
            logger.debug("Trying DeepInfra model: %s", model_name)

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
                    logger.info("Successfully used DeepInfra model: %s", model_name)
                    usage = resp.usage
                    prompt_tokens = usage.prompt_tokens if usage else 0
                    completion_tokens = usage.completion_tokens if usage else 0
                    cost = _calculate_cost(model_name, prompt_tokens, completion_tokens, backend="deepinfra")
                    return ChatResult(
                        content=content,
                        model=model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        cost_usd=cost,
                        backend="deepinfra",
                    )
            raise RuntimeError(f"Empty response from DeepInfra model {model_name}")
        except Exception as e:
            last_error = e
            logger.warning("DeepInfra model %s failed: %s", model_name, str(e)[:100])
            continue

    raise last_error or RuntimeError("All DeepInfra models failed")


def _chat_with_groq(
    messages: list[dict[str, Any]],
    max_tokens: int,
    model: Optional[str] = None
) -> ChatResult:
    """
    Chat using Groq with automatic model fallback (FALLBACK backend).
    Handles GPT-OSS reasoning models by adding token overhead
    and extracting content from reasoning responses.
    Returns ChatResult with token usage and cost.
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
                    # Extract usage from response
                    usage = resp.usage
                    prompt_tokens = usage.prompt_tokens if usage else 0
                    completion_tokens = usage.completion_tokens if usage else 0
                    cost = _calculate_cost(model_name, prompt_tokens, completion_tokens, backend="groq")
                    return ChatResult(
                        content=content,
                        model=model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        cost_usd=cost,
                        backend="groq",
                    )
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
) -> ChatResult:
    """
    Chat using Ollama with automatic model fallback.
    Tries each configured model in order until one works.
    Local inference — cost is always $0.
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
                    usage = resp.usage
                    prompt_tokens = usage.prompt_tokens if usage else 0
                    completion_tokens = usage.completion_tokens if usage else 0
                    return ChatResult(
                        content=content,
                        model=model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        cost_usd=0.0,  # Local inference
                        backend="ollama",
                    )
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
) -> ChatResult:
    """
    Chat using HuggingFace Inference API with model fallback.
    Uses free tier — cost is $0.
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
                    usage = resp.usage
                    prompt_tokens = usage.prompt_tokens if usage else 0
                    completion_tokens = usage.completion_tokens if usage else 0
                    return ChatResult(
                        content=content,
                        model=model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        cost_usd=0.0,  # Free tier
                        backend="huggingface",
                    )
            raise RuntimeError(f"Empty response from HuggingFace model {model_name}")
        except Exception as e:
            last_error = e
            logger.warning("HuggingFace model %s failed: %s", model_name, str(e)[:100])
            continue

    raise last_error or RuntimeError("All HuggingFace models failed")


@retry_with_jitter(max_retries=2, base_delay=0.5, max_delay=5.0)
def _deepinfra_with_retry(
    messages: list[dict[str, Any]],
    max_tokens: int,
    model: Optional[str] = None
) -> ChatResult:
    """DeepInfra chat with retry and circuit breaker."""
    breaker = _get_deepinfra_circuit_breaker()
    return breaker.execute(_chat_with_deepinfra, messages, max_tokens, model)


@retry_with_jitter(max_retries=2, base_delay=0.5, max_delay=5.0)
def _groq_with_retry(
    messages: list[dict[str, Any]],
    max_tokens: int,
    model: Optional[str] = None
) -> ChatResult:
    """Groq chat with retry and circuit breaker."""
    breaker = _get_groq_circuit_breaker()
    return breaker.execute(_chat_with_groq, messages, max_tokens, model)


@retry_with_jitter(max_retries=2, base_delay=1.0, max_delay=10.0)
def _ollama_with_retry(
    messages: list[dict[str, Any]],
    max_tokens: int,
    model: Optional[str] = None
) -> ChatResult:
    """Ollama chat with retry and circuit breaker."""
    breaker = _get_ollama_circuit_breaker()
    return breaker.execute(_chat_with_ollama, messages, max_tokens, model)


@retry_with_jitter(max_retries=2, base_delay=2.0, max_delay=15.0)
def _hf_with_retry(
    messages: list[dict[str, Any]],
    max_tokens: int,
    model: Optional[str] = None
) -> ChatResult:
    """HuggingFace chat with retry and circuit breaker."""
    breaker = _get_hf_circuit_breaker()
    return breaker.execute(_chat_with_hf, messages, max_tokens, model)


def _chat_internal(
    messages: list[dict[str, Any]],
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
    prefer_local: bool = False,
) -> ChatResult:
    """
    Internal: send messages to LLM with automatic fallback, returning full ChatResult.

    Default order: Groq GPT-OSS (fast cloud) -> Ollama (local) -> HuggingFace (backup).
    All backends have circuit breakers to prevent cascading failures.
    """
    config = get_config()
    max_tokens = max_tokens or config.llm.max_tokens_default

    # Build fallback chain: DeepInfra (cheapest) -> Groq (fast) -> Ollama (local) -> HuggingFace (backup)
    if prefer_local:
        handlers = [
            (lambda m=messages, t=max_tokens, mod=model: _ollama_with_retry(m, t, mod), "Ollama"),
            (lambda m=messages, t=max_tokens, mod=model: _deepinfra_with_retry(m, t, mod), "DeepInfra"),
            (lambda m=messages, t=max_tokens, mod=model: _groq_with_retry(m, t, mod), "Groq"),
            (lambda m=messages, t=max_tokens, mod=model: _hf_with_retry(m, t, mod), "HuggingFace"),
        ]
    else:
        handlers = []
        if config.llm.deepinfra_api_key:
            handlers.append(
                (lambda m=messages, t=max_tokens, mod=model: _deepinfra_with_retry(m, t, mod), "DeepInfra")
            )
        if config.llm.groq_api_key:
            handlers.append(
                (lambda m=messages, t=max_tokens, mod=model: _groq_with_retry(m, t, mod), "Groq")
            )
        handlers.append(
            (lambda m=messages, t=max_tokens, mod=model: _ollama_with_retry(m, t, mod), "Ollama")
        )
        handlers.append(
            (lambda m=messages, t=max_tokens, mod=model: _hf_with_retry(m, t, mod), "HuggingFace")
        )

    degradation = GracefulDegradation(handlers)

    try:
        result, backend, tier = degradation.execute()
        if tier > 0:
            logger.info("Used fallback backend: %s", backend)
        # Accumulate cost for multi-call tracking (e.g. /research)
        _accumulate_cost(result)
        return result
    except CircuitBreakerOpenError as e:
        logger.error("All circuit breakers open: %s", e)
        raise RuntimeError(
            "LLM service temporarily unavailable. "
            "Set DEEPINFRA_API_KEY (cheapest) or GROQ_API_KEY, ensure Ollama is running, or set HF_TOKEN."
        ) from e
    except Exception as e:
        logger.error("All LLM backends failed: %s", e)
        raise RuntimeError(
            "LLM request failed. "
            "Set DEEPINFRA_API_KEY (cheapest) or GROQ_API_KEY, or check Ollama status."
        ) from e


def chat(
    messages: list[dict[str, Any]],
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
    prefer_local: bool = False,
) -> str:
    """
    Send messages to LLM with automatic fallback. Returns content string.

    For cost tracking, use chat_with_usage() instead.
    """
    result = _chat_internal(messages, max_tokens, model, prefer_local)
    return result.content


def chat_with_usage(
    messages: list[dict[str, Any]],
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
    prefer_local: bool = False,
) -> ChatResult:
    """
    Send messages to LLM with automatic fallback. Returns ChatResult with cost.

    Same as chat() but returns the full ChatResult including:
    - model used, token counts, cost_usd, backend
    """
    return _chat_internal(messages, max_tokens, model, prefer_local)


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
    has_deepinfra = config.llm.deepinfra_api_key is not None
    status = {
        "deepinfra": {
            "available": False,
            "configured": has_deepinfra,
            "primary_model": config.llm.deepinfra_models[0] if config.llm.deepinfra_models else None,
            "models": [],
            "error": None,
            "role": "primary",
        },
        "groq": {
            "available": False,
            "configured": num_groq_keys > 0,
            "num_keys": num_groq_keys,
            "primary_model": config.llm.groq_models[0] if config.llm.groq_models else None,
            "models": [],
            "error": None,
            "role": "fallback",
        },
        "ollama": {"available": False, "models": [], "error": None},
        "huggingface": {"available": False, "configured": False, "error": None}
    }

    # Check DeepInfra (primary - cheapest)
    if has_deepinfra:
        try:
            client = _make_deepinfra_client()
            if client:
                resp = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5,
                )
                if resp.choices:
                    status["deepinfra"]["available"] = True
                    status["deepinfra"]["models"] = config.llm.deepinfra_models
        except Exception as e:
            status["deepinfra"]["error"] = str(e)[:200]

    # Check Groq (fallback)
    if num_groq_keys > 0:
        try:
            client = _make_groq_client()
            if client:
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
    global _deepinfra_circuit_breaker, _groq_circuit_breaker, _ollama_circuit_breaker, _hf_circuit_breaker
    if _deepinfra_circuit_breaker:
        _deepinfra_circuit_breaker.reset()
    if _groq_circuit_breaker:
        _groq_circuit_breaker.reset()
    if _ollama_circuit_breaker:
        _ollama_circuit_breaker.reset()
    if _hf_circuit_breaker:
        _hf_circuit_breaker.reset()
    logger.info("All circuit breakers reset")

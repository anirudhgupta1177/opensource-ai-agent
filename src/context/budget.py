"""
Enhanced context-window management with semantic chunking and relevance-based allocation.
Handles token estimation, smart truncation, and context building with source citations.
"""
from __future__ import annotations


import logging
from typing import Any, Optional

from src.config import get_config
from src.context.relevance import score_sources

logger = logging.getLogger(__name__)

# Try to import tiktoken for accurate token counting
_TIKTOKEN_AVAILABLE = False
_tiktoken_encoder = None

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    logger.warning("tiktoken not available, using character-based token estimation")


def _get_tiktoken_encoder():
    """Lazily initialize tiktoken encoder."""
    global _tiktoken_encoder
    if _tiktoken_encoder is None and _TIKTOKEN_AVAILABLE:
        import tiktoken
        _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_encoder


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses tiktoken for accurate counting if available,
    otherwise falls back to character-based heuristic.
    """
    if not text:
        return 0

    encoder = _get_tiktoken_encoder()
    if encoder:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass

    # Fallback: ~4 characters per token for English
    config = get_config()
    return max(0, len(text) // config.context.chars_per_token)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to fit within token limit.

    Uses tiktoken for precise truncation if available,
    otherwise uses character-based estimation.
    """
    if not text or max_tokens <= 0:
        return ""

    current_tokens = estimate_tokens(text)
    if current_tokens <= max_tokens:
        return text

    encoder = _get_tiktoken_encoder()
    if encoder:
        try:
            tokens = encoder.encode(text)[:max_tokens]
            return encoder.decode(tokens)
        except Exception:
            pass

    # Fallback: character-based truncation
    config = get_config()
    max_chars = max_tokens * config.context.chars_per_token
    if len(text) <= max_chars:
        return text

    # Try to truncate at word boundary
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.8:  # Keep at least 80% of content
        truncated = truncated[:last_space]

    return truncated


def chunk_semantically(text: str, max_chunk_tokens: int = 1500) -> list[str]:
    """
    Split text into semantic chunks (paragraphs, sections).

    Preserves paragraph boundaries rather than cutting at arbitrary positions.
    This helps maintain context coherence within each chunk.
    """
    if not text:
        return []

    config = get_config()
    min_chunk = config.context.min_chunk_tokens

    # Split by double newlines (paragraphs)
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = estimate_tokens(para)

        # If single paragraph exceeds max, split it further
        if para_tokens > max_chunk_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_tokens = 0

            # Split large paragraph by sentences
            sentences = _split_into_sentences(para)
            for sentence in sentences:
                sentence_tokens = estimate_tokens(sentence)
                if current_tokens + sentence_tokens > max_chunk_tokens:
                    if current_chunk and current_tokens >= min_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                    current_tokens = sentence_tokens
                else:
                    current_chunk += sentence + " "
                    current_tokens += sentence_tokens
        elif current_tokens + para_tokens > max_chunk_tokens:
            # Start new chunk
            if current_chunk and current_tokens >= min_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
            current_tokens = para_tokens
        else:
            current_chunk += para + "\n\n"
            current_tokens += para_tokens

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    import re
    # Simple sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def build_context_with_sources(
    fetched: list[dict[str, Any]],
    max_context_tokens: Optional[int] = None,
    query: Optional[str] = None,
) -> str:
    """
    Build context string from fetched sources with [Source: url] citations.

    If query is provided, sources are ranked by relevance and higher-relevance
    sources get more token budget.

    Args:
        fetched: List of dicts with url, title, snippet, text keys
        max_context_tokens: Maximum tokens for entire context
        query: Optional query for relevance-based prioritization

    Returns:
        Formatted context string with source citations
    """
    config = get_config()
    max_tokens = max_context_tokens or config.context.max_context_tokens

    if not fetched:
        return ""

    # Score and sort by relevance if query provided
    if query:
        fetched = score_sources(query, fetched, text_key="text")
        logger.info("Scored %d sources by relevance to query", len(fetched))

    # Filter out sources with no usable content
    valid_sources = [
        f for f in fetched
        if f.get("url") and f.get("text", "").strip()
        and f.get("text") != "Content unavailable"
    ]

    if not valid_sources:
        return ""

    # Calculate budget allocation based on relevance
    total_relevance = sum(s.get("relevance", 1.0) for s in valid_sources)
    parts: list[str] = []
    used_tokens = 0

    for source in valid_sources:
        url = source.get("url", "")
        text = source.get("text", "").strip()
        relevance = source.get("relevance", 1.0)

        # Allocate budget proportionally to relevance
        if total_relevance > 0:
            budget_ratio = relevance / total_relevance
        else:
            budget_ratio = 1.0 / len(valid_sources)

        source_budget = int(budget_ratio * max_tokens)
        # Ensure minimum budget per source
        source_budget = max(source_budget, config.context.min_chunk_tokens)
        # Don't exceed remaining budget
        source_budget = min(source_budget, max_tokens - used_tokens)

        if source_budget < config.context.min_chunk_tokens:
            break

        # Truncate text to budget
        text = truncate_to_tokens(text, source_budget)
        if not text:
            continue

        # Build source block with citation header
        header = f"[Source: {url}]"
        if relevance and query:
            header += f" (relevance: {relevance:.2f})"
        header += "\n"

        block = header + text
        block_tokens = estimate_tokens(block)

        # Final check against remaining budget
        if used_tokens + block_tokens > max_tokens:
            remaining = max_tokens - used_tokens - estimate_tokens(header)
            if remaining < config.context.min_chunk_tokens:
                break
            text = truncate_to_tokens(text, remaining)
            block = header + text
            block_tokens = estimate_tokens(block)

        parts.append(block)
        used_tokens += block_tokens

        if used_tokens >= max_tokens:
            break

    logger.info(
        "Built context with %d sources, ~%d tokens",
        len(parts),
        used_tokens
    )

    return "\n\n".join(parts)


def build_context_simple(
    fetched: list[dict[str, Any]],
    max_context_tokens: Optional[int] = None,
) -> str:
    """
    Simple context builder without relevance scoring.
    For backward compatibility with existing code.
    """
    return build_context_with_sources(fetched, max_context_tokens, query=None)

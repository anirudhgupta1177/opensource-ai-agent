"""
Enhanced orchestrator with adaptive query handling.
Routes simple queries through direct pipeline, complex queries through ReAct agent.
Includes query decomposition, citation verification, and confidence scoring.
"""
from __future__ import annotations


import logging
from typing import Any, Optional

from src.agent.query_decomposer import decompose_query, aggregate_results, should_decompose
from src.agent.react_agent import run_react_agent
from src.config import get_config
from src.context.budget import build_context_with_sources
from src.crawl.fetcher import fetch_urls
from src.llm.client import chat
from src.llm.prompts import get_research_prompt
from src.search.duckduckgo_search import search as ddg_search
from src.verification.citation_verifier import verify_claims
from src.verification.confidence_scorer import compute_confidence

logger = logging.getLogger(__name__)


def _is_complex_query(query: str) -> bool:
    """
    Determine if a query is complex enough to warrant ReAct agent.

    Uses heuristics to avoid overhead for simple queries.
    """
    config = get_config()

    if not config.agent.enable_react_for_complex:
        return False

    # Check if query decomposition suggests complexity
    if should_decompose(query):
        return True

    # Additional complexity indicators
    word_count = len(query.split())
    if word_count > 25:
        return True

    # Questions requiring analysis or synthesis
    complex_patterns = [
        "how do", "why does", "what causes",
        "explain the relationship",
        "analyze", "evaluate", "assess",
    ]
    query_lower = query.lower()
    if any(p in query_lower for p in complex_patterns):
        return True

    return False


def _run_simple_pipeline(
    prompt: str,
    format_hint: str,
    max_sources: int,
) -> dict[str, Any]:
    """
    Run the simple search-fetch-synthesize pipeline.

    Best for straightforward queries that don't require multi-step reasoning.
    """
    config = get_config()

    # Search
    search_results = []
    try:
        search_results = ddg_search(prompt, max_results=config.agent.search_max_results)
    except Exception as e:
        logger.warning("Search failed: %s", e)
        return {
            "content": "",
            "sources": [],
            "confidence": None,
            "verified_claims": [],
            "error": "Search failed. Please try again.",
        }

    if not search_results:
        return {
            "content": "No search results found for your query.",
            "sources": [],
            "confidence": None,
            "verified_claims": [],
            "error": None,
        }

    # Fetch and extract content
    try:
        fetched = fetch_urls(search_results, max_sources=max_sources)
    except Exception as e:
        logger.warning("Fetch failed: %s", e)
        return {
            "content": "",
            "sources": [
                {"url": r.get("url"), "title": r.get("title"), "snippet": r.get("snippet")}
                for r in search_results[:max_sources]
            ],
            "confidence": None,
            "verified_claims": [],
            "error": "Some sources failed to load.",
        }

    if not fetched:
        return {
            "content": "Could not extract content from any source.",
            "sources": [
                {"url": r.get("url"), "title": r.get("title"), "snippet": r.get("snippet")}
                for r in search_results[:max_sources]
            ],
            "confidence": None,
            "verified_claims": [],
            "error": None,
        }

    # Build context with relevance-based prioritization
    context_str = build_context_with_sources(
        fetched,
        max_context_tokens=config.context.max_context_tokens,
        query=prompt,
    )

    if not context_str.strip():
        return {
            "content": "No usable content could be extracted from the sources.",
            "sources": [
                {"url": f.get("url"), "title": f.get("title"), "snippet": f.get("snippet")}
                for f in fetched
            ],
            "confidence": None,
            "verified_claims": [],
            "error": None,
        }

    # Synthesize response
    system_prompt = get_research_prompt(format_hint)
    user_message = f"{prompt}\n\nContext:\n{context_str}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    try:
        content = chat(messages, max_tokens=config.context.synthesis_max_tokens)
    except RuntimeError as e:
        logger.warning("LLM failed: %s", e)
        return {
            "content": "",
            "sources": [
                {"url": f.get("url"), "title": f.get("title"), "snippet": f.get("snippet")}
                for f in fetched
            ],
            "confidence": None,
            "verified_claims": [],
            "error": "LLM temporarily unavailable. Ensure Ollama is running.",
        }

    # Prepare sources output
    sources_out = [
        {"url": f.get("url"), "title": f.get("title"), "snippet": f.get("snippet")}
        for f in fetched
    ]

    return {
        "content": content.strip(),
        "sources": sources_out,
        "fetched": fetched,  # Keep full data for verification
        "error": None,
    }


def _run_decomposed_pipeline(
    decomposition: dict[str, Any],
    format_hint: str,
    max_sources: int,
) -> dict[str, Any]:
    """
    Run pipeline with query decomposition.

    Executes each sub-query independently and aggregates results.
    """
    sub_queries = decomposition["sub_queries"]
    strategy = decomposition["aggregation_strategy"]
    original_query = decomposition["original_query"]

    logger.info(
        "Running decomposed pipeline: %d sub-queries, strategy=%s",
        len(sub_queries),
        strategy
    )

    sub_results = []
    for sub_query in sub_queries:
        result = _run_simple_pipeline(sub_query, format_hint, max_sources // len(sub_queries) or 3)
        result["sub_query"] = sub_query
        sub_results.append(result)

    # Aggregate results
    aggregated = aggregate_results(sub_results, strategy, original_query)

    return {
        "content": aggregated["content"],
        "sources": aggregated["sources"],
        "sub_query_results": aggregated.get("sub_query_results", []),
        "aggregation_strategy": strategy,
        "error": None,
    }


def run(
    prompt: str,
    *,
    format_hint: Optional[str] = None,
    max_sources: int = 8,
    use_react: Optional[bool] = None,
    verify: bool = True,
) -> dict[str, Any]:
    """
    Main entry point for research queries.

    Adaptively routes queries to appropriate processing pipeline:
    - Simple queries → direct search-fetch-synthesize
    - Complex queries → ReAct agent or decomposed pipeline

    Args:
        prompt: The research query/prompt
        format_hint: Output format ("markdown", "bullet_list", "raw")
        max_sources: Maximum number of sources to fetch
        use_react: Force ReAct agent (None = auto-detect)
        verify: Whether to verify claims and compute confidence

    Returns:
        Dict with:
        - content: The synthesized answer
        - sources: List of source dicts
        - confidence: Confidence score (if verify=True)
        - verified_claims: Claim verifications (if verify=True)
        - error: Error message if any
    """
    config = get_config()

    # Validate input
    if not prompt or len(prompt.strip()) < 3:
        return {
            "content": "",
            "sources": [],
            "confidence": None,
            "verified_claims": [],
            "error": "Prompt is too short.",
        }

    if len(prompt) > 2000:
        return {
            "content": "",
            "sources": [],
            "confidence": None,
            "verified_claims": [],
            "error": "Prompt too long (max 2000 characters).",
        }

    # Normalize parameters
    max_sources = min(max(max_sources, 1), config.agent.max_sources_cap)
    format_hint = format_hint or _infer_format(prompt)

    # Determine processing strategy
    should_use_react = use_react if use_react is not None else _is_complex_query(prompt)

    # Check for query decomposition
    decomposition = decompose_query(prompt)

    if decomposition["needs_decomposition"] and not should_use_react:
        logger.info("Using decomposed pipeline for query")
        result = _run_decomposed_pipeline(decomposition, format_hint, max_sources)
    elif should_use_react:
        logger.info("Using ReAct agent for complex query")
        result = run_react_agent(prompt, max_steps=config.agent.max_react_steps)
    else:
        logger.info("Using simple pipeline for query")
        result = _run_simple_pipeline(prompt, format_hint, max_sources)

    # Handle errors
    if result.get("error") and not result.get("content"):
        return result

    # Verify claims and compute confidence
    if verify and result.get("content"):
        try:
            # Get fetched data for verification (with full text)
            fetched = result.get("fetched") or result.get("sources", [])

            # Verify claims
            verification = verify_claims(result["content"], fetched, use_llm=True)

            # Compute confidence
            confidence = compute_confidence(
                response=result["content"],
                sources=fetched,
                verification_result=verification,
                query=prompt,
            )

            result["confidence"] = {
                "overall": confidence.overall,
                "level": confidence.level,
                "factors": {
                    "source_quality": confidence.factors.source_quality,
                    "citation_density": confidence.factors.citation_density,
                    "verification_rate": confidence.factors.verification_rate,
                    "source_consensus": confidence.factors.source_consensus,
                    "source_diversity": confidence.factors.source_diversity,
                },
                "concerns": confidence.concerns,
                "recommendation": confidence.recommendation,
            }

            result["verified_claims"] = [
                {
                    "claim": c.claim[:200],
                    "supported": c.supported,
                    "confidence": c.confidence,
                    "sources": c.supporting_sources[:3],
                }
                for c in verification.claims[:10]  # Limit to 10 claims
            ]

            logger.info(
                "Verification complete: %.1f%% claims verified, confidence=%.2f",
                verification.verification_rate * 100,
                confidence.overall
            )

        except Exception as e:
            logger.warning("Verification failed: %s", str(e)[:100])
            result["confidence"] = None
            result["verified_claims"] = []

    # Clean up internal data
    result.pop("fetched", None)

    return result


def _infer_format(prompt: str) -> str:
    """Infer output format from prompt content."""
    prompt_lower = prompt.lower()

    if "bullet" in prompt_lower or "list" in prompt_lower:
        return "bullet_list"
    if "raw" in prompt_lower and "format" in prompt_lower:
        return "raw"

    return "markdown"


# Backward compatibility
def run_simple(
    prompt: str,
    format_hint: str = "markdown",
    max_sources: int = 8,
) -> dict[str, Any]:
    """Simple pipeline without verification (for backward compatibility)."""
    return run(prompt, format_hint=format_hint, max_sources=max_sources, verify=False)

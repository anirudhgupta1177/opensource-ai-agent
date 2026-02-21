"""
Query decomposition for complex research queries.
Breaks down multi-part queries into simpler, independent sub-queries.
"""
from __future__ import annotations


import json
import logging
import re
from typing import Any, Optional

from src.config import get_config
from src.llm.client import chat_with_json_output
from src.llm.prompts import get_decomposition_prompt

logger = logging.getLogger(__name__)


# Keywords that suggest query complexity
COMPLEXITY_INDICATORS = [
    # Comparison queries
    "compare", "contrast", "vs", "versus", "difference between",
    "similarities", "pros and cons", "advantages and disadvantages",
    # Multi-topic queries
    "and also", "as well as", "in addition to",
    "both", "multiple", "several", "various",
    # Relationship queries
    "how does .* relate to", "connection between",
    "impact of .* on", "effect of .* on",
    # Comprehensive queries
    "everything about", "complete guide", "comprehensive",
    "all aspects of", "detailed overview",
]


def _estimate_complexity(query: str) -> float:
    """
    Estimate query complexity on a scale of 0-1.

    Considers:
    - Length of query
    - Presence of complexity indicators
    - Number of distinct topics
    """
    query_lower = query.lower()
    score = 0.0

    # Length factor (longer queries tend to be more complex)
    word_count = len(query.split())
    if word_count > 20:
        score += 0.3
    elif word_count > 10:
        score += 0.15

    # Complexity indicators
    for indicator in COMPLEXITY_INDICATORS:
        if re.search(indicator, query_lower):
            score += 0.2
            break

    # Multiple question marks suggest multiple questions
    if query.count('?') > 1:
        score += 0.25

    # Conjunctions suggesting multiple parts
    conjunctions = ['and', 'or', 'but', 'also', 'plus']
    conjunction_count = sum(1 for c in conjunctions if f" {c} " in query_lower)
    score += min(conjunction_count * 0.1, 0.3)

    return min(score, 1.0)


def should_decompose(query: str) -> bool:
    """
    Determine if a query should be decomposed into sub-queries.

    Uses heuristics to avoid unnecessary LLM calls for simple queries.
    """
    config = get_config()

    # Quick heuristic checks
    complexity = _estimate_complexity(query)
    if complexity < 0.3:
        logger.debug("Query complexity %.2f below threshold, skipping decomposition", complexity)
        return False

    # Word count threshold
    word_count = len(query.split())
    if word_count < config.agent.decomposition_word_threshold:
        logger.debug("Query word count %d below threshold, skipping decomposition", word_count)
        return False

    return True


def decompose_query(query: str) -> dict[str, Any]:
    """
    Decompose a complex query into simpler sub-queries using LLM.

    Returns:
        Dict with:
        - needs_decomposition: bool
        - sub_queries: list of sub-query strings
        - aggregation_strategy: "combine" | "compare" | "synthesize"
        - original_query: the input query
    """
    config = get_config()

    # Check if decomposition is likely needed
    if not should_decompose(query):
        return {
            "needs_decomposition": False,
            "sub_queries": [query],
            "aggregation_strategy": None,
            "original_query": query,
        }

    # Use LLM to decompose
    try:
        prompt = get_decomposition_prompt(query)
        response = chat_with_json_output(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )

        # Parse JSON response
        result = _parse_decomposition_response(response)

        if result["needs_decomposition"]:
            logger.info(
                "Decomposed query into %d sub-queries: %s",
                len(result["sub_queries"]),
                result["aggregation_strategy"]
            )
        else:
            logger.debug("LLM determined no decomposition needed")

        result["original_query"] = query
        return result

    except Exception as e:
        logger.warning("Query decomposition failed: %s", str(e)[:100])
        # Fallback to not decomposing
        return {
            "needs_decomposition": False,
            "sub_queries": [query],
            "aggregation_strategy": None,
            "original_query": query,
            "error": str(e),
        }


def _parse_decomposition_response(response: str) -> dict[str, Any]:
    """Parse LLM response for query decomposition."""
    # Try to extract JSON from response
    response = response.strip()

    # Handle markdown code blocks
    if "```json" in response:
        match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            response = match.group(1)
    elif "```" in response:
        match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            response = match.group(1)

    try:
        data = json.loads(response)

        # Validate and normalize
        needs_decomp = data.get("needs_decomposition", False)
        sub_queries = data.get("sub_queries", [])
        strategy = data.get("aggregation_strategy")

        # Ensure sub_queries is a list of strings
        if not isinstance(sub_queries, list):
            sub_queries = []
        sub_queries = [str(q) for q in sub_queries if q]

        # Limit number of sub-queries
        config = get_config()
        if len(sub_queries) > config.agent.max_sub_queries:
            sub_queries = sub_queries[:config.agent.max_sub_queries]

        # Validate aggregation strategy
        valid_strategies = ["combine", "compare", "synthesize"]
        if strategy not in valid_strategies:
            strategy = "combine"

        return {
            "needs_decomposition": bool(needs_decomp and sub_queries),
            "sub_queries": sub_queries if sub_queries else [],
            "aggregation_strategy": strategy if needs_decomp else None,
        }

    except json.JSONDecodeError:
        logger.warning("Failed to parse decomposition JSON: %s", response[:200])
        return {
            "needs_decomposition": False,
            "sub_queries": [],
            "aggregation_strategy": None,
        }


def aggregate_results(
    sub_results: list[dict[str, Any]],
    strategy: str,
    original_query: str,
) -> dict[str, Any]:
    """
    Aggregate results from multiple sub-queries.

    Args:
        sub_results: List of result dicts from each sub-query
        strategy: Aggregation strategy ("combine", "compare", "synthesize")
        original_query: The original query for context

    Returns:
        Combined result dict
    """
    if not sub_results:
        return {
            "content": "No results found for the sub-queries.",
            "sources": [],
            "sub_query_results": [],
        }

    # Collect all sources (deduplicated)
    all_sources = []
    seen_urls = set()
    all_content_parts = []

    for i, result in enumerate(sub_results):
        sub_query = result.get("sub_query", f"Part {i+1}")
        content = result.get("content", "")

        if content:
            all_content_parts.append(f"### {sub_query}\n\n{content}")

        for source in result.get("sources", []):
            url = source.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_sources.append(source)

    # Combine content based on strategy
    if strategy == "compare":
        combined_content = "## Comparison\n\n" + "\n\n---\n\n".join(all_content_parts)
    elif strategy == "synthesize":
        combined_content = "## Synthesis\n\n" + "\n\n".join(all_content_parts)
    else:  # combine
        combined_content = "\n\n".join(all_content_parts)

    return {
        "content": combined_content,
        "sources": all_sources,
        "sub_query_results": sub_results,
        "aggregation_strategy": strategy,
    }

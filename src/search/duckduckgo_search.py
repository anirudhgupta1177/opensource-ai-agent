"""
Search module: duckduckgo-search (free, no API key).
Returns list of {url, title, snippet}.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 15


def search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, Any]]:
    """
    Run text search via DuckDuckGo. Returns list of dicts with url, title, snippet.
    On failure returns empty list and logs.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        try:
            from ddgs import DDGS
        except ImportError:
            logger.error("Neither duckduckgo-search nor ddgs installed")
            return []

    out: list[dict[str, Any]] = []
    try:
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=max_results)):
                if i >= max_results:
                    break
                out.append({
                    "url": r.get("href") or r.get("link") or "",
                    "title": r.get("title") or "",
                    "snippet": r.get("body") or r.get("snippet") or "",
                })
    except Exception as e:
        logger.warning("DuckDuckGo search failed for query=%s: %s", query[:50], e)
        return []
    return out

"""
Web content fetcher with Crawl4AI primary and trafilatura fallback.
Handles URL validation, content extraction, and error recovery.
"""
from __future__ import annotations


import logging
from typing import Any, Optional
from urllib.parse import urlparse

import requests
import trafilatura

from src.config import get_config
from src.crawl.crawl4ai_fetcher import (
    fetch_with_crawl4ai_sync,
    should_use_crawl4ai,
    get_crawl4ai_status,
)

logger = logging.getLogger(__name__)

CONTENT_UNAVAILABLE = "Content unavailable"


def _is_safe_url(url: str) -> bool:
    """Validate URL is safe to fetch (http/https only)."""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or parsed.scheme not in ("http", "https"):
            return False
        if not parsed.netloc:
            return False
        # Block common non-content URLs
        blocked_extensions = ('.pdf', '.zip', '.exe', '.dmg', '.pkg', '.tar', '.gz')
        if parsed.path.lower().endswith(blocked_extensions):
            return False
        return True
    except Exception:
        return False


def _fetch_with_trafilatura(url: str, timeout: int) -> str:
    """
    Fetch and extract content using requests + trafilatura.
    Lightweight, fast, good for static HTML pages.
    """
    config = get_config()

    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": config.crawl.user_agent},
            allow_redirects=True,
        )
        resp.raise_for_status()

        # Check content type
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            logger.warning("Non-HTML content type for %s: %s", url[:60], content_type[:50])
            return CONTENT_UNAVAILABLE

        html = resp.text
        if not html or len(html) < 100:
            return CONTENT_UNAVAILABLE

    except requests.Timeout:
        logger.warning("Request timeout for %s", url[:80])
        return CONTENT_UNAVAILABLE
    except requests.RequestException as e:
        logger.warning("Request failed for %s: %s", url[:80], str(e)[:100])
        return CONTENT_UNAVAILABLE

    # Extract main content with trafilatura
    try:
        text = trafilatura.extract(
            html,
            output_format="txt",
            include_links=False,
            include_comments=False,
            include_tables=True,
        )
        if text and text.strip():
            return text.strip()
    except Exception as e:
        logger.warning("Trafilatura extraction failed for %s: %s", url[:80], str(e)[:100])

    return CONTENT_UNAVAILABLE


def fetch_and_extract(
    url: str,
    timeout: Optional[int] = None,
    prefer_crawl4ai: bool = True,
) -> str:
    """
    Fetch URL and extract main text content.

    Uses Crawl4AI for JavaScript rendering when available and appropriate,
    falls back to trafilatura for simpler extraction.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        prefer_crawl4ai: Whether to prefer Crawl4AI (default True)

    Returns:
        Extracted text content, or CONTENT_UNAVAILABLE on failure
    """
    if not _is_safe_url(url):
        logger.warning("Skipping unsafe or invalid URL: %s", url[:80])
        return CONTENT_UNAVAILABLE

    config = get_config()
    timeout = timeout or config.crawl.fetch_timeout

    # Try Crawl4AI first if appropriate
    if prefer_crawl4ai and should_use_crawl4ai(url):
        result = fetch_with_crawl4ai_sync(url, timeout=config.crawl.crawl4ai_timeout)

        if result["success"]:
            # Prefer markdown for LLM-friendly format
            content = result.get("markdown") or result.get("text") or ""
            if content and len(content) >= config.crawl.min_content_length:
                logger.info("Crawl4AI success for %s (%d chars)", url[:60], len(content))
                # Truncate if too long
                if len(content) > config.crawl.max_content_length:
                    content = content[:config.crawl.max_content_length]
                return content
            logger.debug("Crawl4AI returned insufficient content for %s", url[:60])
        else:
            logger.debug("Crawl4AI failed for %s: %s", url[:60], result.get("error", "unknown"))

    # Fallback to trafilatura
    logger.debug("Using trafilatura for %s", url[:60])
    content = _fetch_with_trafilatura(url, timeout)

    if content != CONTENT_UNAVAILABLE:
        # Truncate if too long
        if len(content) > config.crawl.max_content_length:
            content = content[:config.crawl.max_content_length]

    return content


def fetch_urls(
    url_infos: list[dict[str, Any]],
    max_sources: int = 8,
    per_url_timeout: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Fetch and extract content from multiple URLs.

    Args:
        url_infos: List of dicts with url, title, snippet keys
        max_sources: Maximum number of URLs to fetch
        per_url_timeout: Timeout per URL in seconds

    Returns:
        List of dicts with url, title, snippet, text keys
    """
    config = get_config()
    max_sources = min(max_sources, config.agent.max_sources_cap)
    per_url_timeout = per_url_timeout or config.crawl.fetch_timeout

    results: list[dict[str, Any]] = []
    successful = 0
    failed = 0

    for i, info in enumerate(url_infos):
        if i >= max_sources:
            break

        url = info.get("url", "")
        if not url:
            continue

        title = info.get("title", "")
        snippet = info.get("snippet", "")

        text = fetch_and_extract(url, timeout=per_url_timeout)

        if text == CONTENT_UNAVAILABLE:
            failed += 1
        else:
            successful += 1

        results.append({
            "url": url,
            "title": title,
            "snippet": snippet,
            "text": text,
        })

    logger.info(
        "Fetched %d URLs: %d successful, %d failed",
        len(results),
        successful,
        failed
    )

    return results


def get_fetcher_status() -> dict[str, Any]:
    """Get fetcher capabilities and status."""
    crawl4ai_status = get_crawl4ai_status()
    return {
        "crawl4ai": crawl4ai_status,
        "trafilatura": {"available": True},
        "js_rendering": crawl4ai_status.get("js_rendering", False),
    }

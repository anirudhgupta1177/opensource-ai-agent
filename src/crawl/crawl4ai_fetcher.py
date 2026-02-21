"""
Crawl4AI integration for enhanced web content extraction.
Provides JavaScript rendering and clean markdown output.
Falls back to trafilatura for simpler pages or when Crawl4AI fails.
"""
from __future__ import annotations


import asyncio
import logging
from typing import Any, Optional
from urllib.parse import urlparse

from src.config import get_config

logger = logging.getLogger(__name__)

# Check if Crawl4AI is available
_CRAWL4AI_AVAILABLE = False
try:
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig
    _CRAWL4AI_AVAILABLE = True
except (ImportError, TypeError, Exception) as e:
    # TypeError occurs on Python < 3.10 due to `type | None` syntax in crawl4ai
    logger.warning("crawl4ai not available (%s), will use trafilatura only", type(e).__name__)


async def fetch_with_crawl4ai(
    url: str,
    timeout: Optional[int] = None,
) -> dict[str, Any]:
    """
    Fetch and extract content using Crawl4AI.

    Provides:
    - JavaScript rendering for dynamic pages
    - Clean markdown output
    - Link extraction
    - Metadata extraction

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Dict with 'markdown', 'text', 'links', 'metadata', 'success', 'error'
    """
    if not _CRAWL4AI_AVAILABLE:
        return {
            "success": False,
            "error": "crawl4ai not installed",
            "markdown": "",
            "text": "",
            "links": [],
            "metadata": {}
        }

    config = get_config()
    timeout = timeout or config.crawl.crawl4ai_timeout

    try:
        # Configure browser for headless crawling
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
        )

        # Configure crawler run
        crawler_config = CrawlerRunConfig(
            wait_until="domcontentloaded",
            page_timeout=timeout * 1000,  # Convert to milliseconds
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(
                url=url,
                config=crawler_config,
            )

            if not result.success:
                return {
                    "success": False,
                    "error": result.error_message or "Crawl failed",
                    "markdown": "",
                    "text": "",
                    "links": [],
                    "metadata": {}
                }

            # Extract content
            markdown = result.markdown or ""
            text = result.cleaned_html or result.markdown or ""

            # Extract links
            links = []
            if hasattr(result, 'links') and result.links:
                links = [
                    {"href": link.get("href", ""), "text": link.get("text", "")}
                    for link in result.links[:50]  # Limit to 50 links
                ]

            # Extract metadata
            metadata = {}
            if hasattr(result, 'metadata') and result.metadata:
                metadata = {
                    "title": result.metadata.get("title", ""),
                    "description": result.metadata.get("description", ""),
                    "keywords": result.metadata.get("keywords", ""),
                }

            return {
                "success": True,
                "error": None,
                "markdown": markdown,
                "text": text,
                "links": links,
                "metadata": metadata
            }

    except asyncio.TimeoutError:
        logger.warning("Crawl4AI timeout for %s", url[:80])
        return {
            "success": False,
            "error": "Timeout",
            "markdown": "",
            "text": "",
            "links": [],
            "metadata": {}
        }
    except Exception as e:
        logger.warning("Crawl4AI error for %s: %s", url[:80], str(e)[:100])
        return {
            "success": False,
            "error": str(e),
            "markdown": "",
            "text": "",
            "links": [],
            "metadata": {}
        }


def fetch_with_crawl4ai_sync(
    url: str,
    timeout: Optional[int] = None,
) -> dict[str, Any]:
    """
    Synchronous wrapper for Crawl4AI fetch.
    Creates event loop if needed.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, use run_coroutine_threadsafe
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                fetch_with_crawl4ai(url, timeout),
                loop
            )
            return future.result(timeout=timeout or 15)
        else:
            return loop.run_until_complete(fetch_with_crawl4ai(url, timeout))
    except RuntimeError:
        # No event loop, create a new one
        return asyncio.run(fetch_with_crawl4ai(url, timeout))


def is_js_heavy_site(url: str) -> bool:
    """
    Heuristic to detect if a site likely requires JavaScript rendering.

    Returns True for sites known to be JS-heavy or have dynamic content.
    """
    js_heavy_indicators = [
        # SPA frameworks
        "react", "angular", "vue",
        # JS-heavy platforms
        "twitter.com", "x.com",
        "linkedin.com",
        "facebook.com",
        "instagram.com",
        # Dynamic content sites
        "medium.com",
        "dev.to",
        "hashnode.com",
        # Documentation sites (often use JS frameworks)
        "docs.google.com",
        "notion.so",
    ]

    url_lower = url.lower()
    return any(indicator in url_lower for indicator in js_heavy_indicators)


def should_use_crawl4ai(url: str) -> bool:
    """
    Determine if Crawl4AI should be used for this URL.

    Considers:
    - Crawl4AI availability
    - Site characteristics
    - URL patterns
    """
    if not _CRAWL4AI_AVAILABLE:
        return False

    config = get_config()
    if not config.crawl.enable_js_rendering:
        return False

    # Always try Crawl4AI for known JS-heavy sites
    if is_js_heavy_site(url):
        return True

    # For other sites, use Crawl4AI as primary (better markdown output)
    return True


def get_crawl4ai_status() -> dict[str, Any]:
    """Get Crawl4AI availability status."""
    return {
        "available": _CRAWL4AI_AVAILABLE,
        "js_rendering": _CRAWL4AI_AVAILABLE,
    }

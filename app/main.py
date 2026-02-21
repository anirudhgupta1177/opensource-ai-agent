"""
FastAPI app for the Open Source Web Scraper Agent.
Provides research API with citation verification and confidence scoring.
"""
from __future__ import annotations


import logging
from typing import Any, Optional, Union

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.agent.orchestrator import run as agent_run
from src.config import get_config
from src.crawl.fetcher import get_fetcher_status, fetch_and_extract
from src.llm.client import health_check as llm_health_check, reset_circuit_breakers, chat
from src.email import (
    EmailVerifyRequest,
    EmailVerifyResponse,
    MiscDetails,
    MxRecord,
    ReachabilityStatus,
    SmtpDetails,
    close_reacher_client,
    reacher_health_check,
    reset_reacher_circuit_breaker,
    verify_email_with_retry,
    ReacherError,
    ReacherAuthError,
    ReacherNotConfiguredError,
    ReacherRateLimitError,
    ReacherTimeoutError,
)
from src.utils.resilience import CircuitBreakerOpenError

load_dotenv()

config = get_config()

logging.basicConfig(
    level=getattr(logging, config.app.log_level),
    format=config.app.log_format,
)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Open Source Web Scraper Agent",
    description=(
        "Research & enrichment API powered by GPT-OSS on Groq.\n\n"
        "Features:\n"
        "- GPT-OSS reasoning models (20B primary, 120B fallback)\n"
        "- Multi-step reasoning with ReAct agent\n"
        "- Query decomposition for complex questions\n"
        "- Citation verification and confidence scoring\n"
        "- 10 req/sec throughput for Clay integration"
    ),
    version="3.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.app.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models

class ResearchOptions(BaseModel):
    """Options for research request."""
    format: str = Field(
        default="markdown",
        description="Output format: markdown | bullet_list | raw"
    )
    max_sources: int = Field(
        default=8,
        ge=1,
        le=15,
        description="Maximum URLs to fetch and process"
    )
    use_react: Optional[bool] = Field(
        default=None,
        description="Force ReAct agent (None = auto-detect based on query complexity)"
    )
    verify: bool = Field(
        default=True,
        description="Enable citation verification and confidence scoring"
    )


class ResearchRequest(BaseModel):
    """Research API request body."""
    prompt: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The research query or prompt"
    )
    options: Optional[ResearchOptions] = None


class SourceItem(BaseModel):
    """A single source reference."""
    url: str
    title: str
    snippet: str


class VerifiedClaim(BaseModel):
    """A verified claim from the response."""
    claim: str
    supported: bool
    confidence: float
    sources: list[str]


class ConfidenceFactors(BaseModel):
    """Breakdown of confidence score factors."""
    source_quality: float
    citation_density: float
    verification_rate: float
    source_consensus: float
    source_diversity: float


class ConfidenceScore(BaseModel):
    """Confidence assessment for the response."""
    overall: float = Field(description="Overall confidence (0-1)")
    level: str = Field(description="Confidence level: high | moderate | low")
    factors: ConfidenceFactors
    concerns: list[str] = Field(description="List of concerns about the results")
    recommendation: str


class ResearchResponse(BaseModel):
    """Research API response body."""
    content: str = Field(description="The synthesized research answer")
    sources: list[SourceItem] = Field(description="Sources used")
    confidence: Optional[ConfidenceScore] = Field(
        default=None,
        description="Confidence assessment (when verify=True)"
    )
    verified_claims: list[VerifiedClaim] = Field(
        default=[],
        description="Verified claims from the response"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if something went wrong"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    llm: dict[str, Any]
    crawl: dict[str, Any]


# ============== SCRAPER MODELS ==============

class ScrapeRequest(BaseModel):
    """Website scraper request."""
    url: str = Field(..., description="Website URL to scrape")
    timeout: int = Field(default=8, ge=1, le=30, description="Timeout in seconds")


class ScrapeResponse(BaseModel):
    """Website scraper response - raw content, no AI."""
    url: str
    domain: str
    content: str = Field(description="Raw text content from website")
    content_length: int
    success: bool
    bot_protected: bool = Field(default=False, description="True if site has bot protection (Cloudflare, etc.)")
    error: Optional[str] = None


# ============== AI QUALIFY MODELS ==============

class AIQualifyRequest(BaseModel):
    """AI qualification request - content provided by user."""
    content: str = Field(..., min_length=1, max_length=50000, description="Website content to analyze")
    domain: str = Field(default="", description="Domain name (optional, for context)")
    criteria: str = Field(
        default="B2B SaaS that could integrate with email automation tools",
        description="Qualification criteria"
    )


class AIQualifyResponse(BaseModel):
    """AI qualification response."""
    score: int = Field(description="Score 1-10")
    qualified: bool = Field(description="True if score > 6")
    reasoning: str
    domain: str
    error: Optional[str] = None


# ============== AI PROCESS MODELS ==============

class AIProcessRequest(BaseModel):
    """General AI processing request."""
    prompt: str = Field(..., min_length=1, max_length=10000, description="The prompt/instruction")
    input_data: str = Field(default="", max_length=50000, description="Input data to process")
    max_tokens: int = Field(default=500, ge=50, le=4000, description="Max response tokens")


class AIProcessResponse(BaseModel):
    """General AI processing response."""
    output: str
    success: bool
    error: Optional[str] = None


# ============== LEGACY MODELS (kept for compatibility) ==============

class QualifyRequest(BaseModel):
    """Fast website qualification request."""
    url: str = Field(..., description="Website URL to analyze")
    criteria: str = Field(
        default="B2B SaaS that could integrate with email automation tools",
        description="Qualification criteria"
    )


class QualifyResponse(BaseModel):
    """Fast qualification response."""
    score: int = Field(description="Score 1-10")
    qualified: bool = Field(description="True if score > 6")
    reasoning: str = Field(description="Brief reasoning")
    website: str
    error: Optional[str] = None


# API Endpoints

@app.post("/research", response_model=ResearchResponse)
@limiter.limit(config.app.rate_limit)
async def research(request: Request, body: ResearchRequest):
    """
    Process a research prompt and return cited results.

    The endpoint will:
    1. Search the web using DuckDuckGo (free, no API key)
    2. Fetch and extract content from top results
    3. Use open-source LLMs to synthesize an answer
    4. Verify claims against sources (optional)
    5. Compute confidence score (optional)

    For complex queries, automatically uses:
    - Query decomposition (breaks into sub-queries)
    - ReAct agent (multi-step reasoning)
    """
    opts = body.options or ResearchOptions()

    try:
        result = agent_run(
            body.prompt,
            format_hint=opts.format if opts.format in ("markdown", "bullet_list", "raw") else "markdown",
            max_sources=opts.max_sources,
            use_react=opts.use_react,
            verify=opts.verify,
        )
    except Exception as e:
        logger.exception("Research failed")
        raise HTTPException(
            status_code=503,
            detail="Research failed. Please try again."
        ) from e

    # Handle errors
    if result.get("error") and not result.get("content"):
        raise HTTPException(status_code=503, detail=result["error"])

    # Build response
    sources = [
        SourceItem(
            url=s.get("url", ""),
            title=s.get("title", ""),
            snippet=s.get("snippet", "")
        )
        for s in result.get("sources", [])
    ]

    # Build confidence response if available
    confidence = None
    if result.get("confidence"):
        conf = result["confidence"]
        confidence = ConfidenceScore(
            overall=conf.get("overall", 0),
            level=conf.get("level", "unknown"),
            factors=ConfidenceFactors(
                source_quality=conf.get("factors", {}).get("source_quality", 0),
                citation_density=conf.get("factors", {}).get("citation_density", 0),
                verification_rate=conf.get("factors", {}).get("verification_rate", 0),
                source_consensus=conf.get("factors", {}).get("source_consensus", 0),
                source_diversity=conf.get("factors", {}).get("source_diversity", 0),
            ),
            concerns=conf.get("concerns", []),
            recommendation=conf.get("recommendation", ""),
        )

    # Build verified claims
    verified_claims = [
        VerifiedClaim(
            claim=c.get("claim", ""),
            supported=c.get("supported", False),
            confidence=c.get("confidence", 0),
            sources=c.get("sources", []),
        )
        for c in result.get("verified_claims", [])
    ]

    return ResearchResponse(
        content=result.get("content", ""),
        sources=sources,
        confidence=confidence,
        verified_claims=verified_claims,
        error=result.get("error"),
    )


# ============== ENDPOINT 1: WEBSITE SCRAPER (ULTRA-FAST) ==============

import random
import asyncio

# Rotating User-Agents to avoid bot detection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]

# Browser-like headers to bypass basic bot detection
def get_browser_headers() -> dict:
    """Get realistic browser headers with rotating User-Agent."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }


# Global async HTTP client for connection pooling
_http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    """Get or create async HTTP client with connection pooling."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(15.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            follow_redirects=True,
            http2=True,  # Enable HTTP/2 for better performance
        )
    return _http_client


@app.on_event("shutdown")
async def shutdown_http_client():
    """Close HTTP clients on shutdown."""
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None
    # Close Reacher client
    await close_reacher_client()


async def fetch_with_retry(client: httpx.AsyncClient, url: str, timeout: int, max_retries: int = 3) -> httpx.Response:
    """Fetch URL with retry logic for transient failures."""
    last_error = None

    for attempt in range(max_retries):
        try:
            headers = get_browser_headers()
            resp = await client.get(url, timeout=timeout, headers=headers)
            return resp
        except (httpx.ConnectError, httpx.ReadError, httpx.WriteError) as e:
            last_error = e
            if attempt < max_retries - 1:
                # Exponential backoff: 0.5s, 1s, 2s
                await asyncio.sleep(0.5 * (2 ** attempt))
                continue
            raise
        except httpx.TimeoutException:
            raise  # Don't retry timeouts

    raise last_error


@app.post("/scrape", response_model=ScrapeResponse)
@limiter.limit("200/minute")
async def scrape_website(request: Request, body: ScrapeRequest):
    """
    ULTRA-FAST website scraper with retry logic.

    Features:
    - Rotating User-Agents to avoid bot detection
    - Browser-like headers
    - Retry logic for transient failures (TLS errors, etc.)
    - HTTP/2 support
    - Connection pooling
    """
    import trafilatura

    domain = body.url.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
    timeout = min(body.timeout, 10)  # Max 10 seconds

    try:
        client = await get_http_client()

        # Fetch with retry for transient errors
        resp = await fetch_with_retry(client, body.url, timeout, max_retries=3)

        if resp.status_code != 200:
            # Some sites return 403/503 for bots - try to extract anyway
            if resp.status_code in (403, 503) and len(resp.text) > 500:
                html = resp.text
            else:
                return ScrapeResponse(
                    url=body.url,
                    domain=domain,
                    content="",
                    content_length=0,
                    success=False,
                    bot_protected=resp.status_code in (403, 503, 429),
                    error=f"HTTP {resp.status_code}" + (" (bot protection)" if resp.status_code in (403, 503, 429) else "")
                )
        else:
            html = resp.text

        # Fast text extraction with multiple fallback options
        content = trafilatura.extract(
            html,
            include_links=False,
            include_comments=False,
            include_tables=True,
            no_fallback=False,  # Enable fallback extraction methods
        ) or ""

        # If trafilatura fails, try basic extraction
        if not content and len(html) > 500:
            # Simple fallback: extract text from body
            import re
            # Remove script and style tags
            clean_html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            clean_html = re.sub(r'<style[^>]*>.*?</style>', '', clean_html, flags=re.DOTALL | re.IGNORECASE)
            # Remove all HTML tags
            content = re.sub(r'<[^>]+>', ' ', clean_html)
            # Clean up whitespace
            content = re.sub(r'\s+', ' ', content).strip()

        # Detect bot protection patterns
        bot_protection_patterns = [
            "access denied",
            "access to this page has been denied",
            "checking your browser",
            "please wait while we verify",
            "cloudflare",
            "just a moment",
            "ray id",
            "please enable javascript",
            "captcha",
            "blocked",
            "are you a robot",
            "bot detected",
            "datadome",
            "perimeterx",
        ]
        content_lower = content.lower() if content else ""
        is_bot_protected = (
            (len(content) < 200 and any(p in content_lower for p in bot_protection_patterns))
            or resp.status_code in (403, 503, 429)
        )

        return ScrapeResponse(
            url=body.url,
            domain=domain,
            content=content[:10000] if not is_bot_protected else "",
            content_length=len(content) if not is_bot_protected else 0,
            success=bool(content) and not is_bot_protected,
            bot_protected=is_bot_protected,
            error="Bot protection detected (Cloudflare/similar)" if is_bot_protected else (None if content else "No content extracted")
        )

    except httpx.TimeoutException:
        return ScrapeResponse(
            url=body.url,
            domain=domain,
            content="",
            content_length=0,
            success=False,
            bot_protected=False,
            error="Timeout"
        )
    except Exception as e:
        error_msg = str(e)[:100]
        # Identify common issues
        if "SSL" in error_msg or "TLS" in error_msg or "certificate" in error_msg.lower():
            error_msg = "TLS/SSL error - site may have certificate issues"
        elif "Connect" in error_msg:
            error_msg = "Connection failed - site may be down or blocking"

        return ScrapeResponse(
            url=body.url,
            domain=domain,
            content="",
            content_length=0,
            success=False,
            bot_protected=False,
            error=error_msg
        )


# ============== ENDPOINT 2: AI QUALIFICATION ==============

@app.post("/ai/qualify", response_model=AIQualifyResponse)
@limiter.limit("600/minute")
async def ai_qualify(request: Request, body: AIQualifyRequest):
    """
    AI qualification using GPT-OSS (~1-2 seconds).

    Analyzes pre-provided content - NO web scraping.
    Content should be provided by user (from /scrape or other source).
    Uses GPT-OSS reasoning model for accurate qualification.
    """
    import re

    try:
        # Truncate content if too long
        content = body.content[:4000] if len(body.content) > 4000 else body.content
        domain = body.domain or "unknown"

        prompt = f"""Analyze this content and score based on criteria.

Domain: {domain}
Content: {content}

Criteria: {body.criteria}

Respond in EXACTLY this format (no other text):
SCORE: [1-10]
QUALIFIED: [YES/NO]
REASONING: [One sentence]"""

        response = chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )

        # Parse response
        score = 0
        qualified = False
        reasoning = response.strip()

        score_match = re.search(r'SCORE:\s*(\d+)', response, re.IGNORECASE)
        if score_match:
            score = min(10, max(1, int(score_match.group(1))))

        qual_match = re.search(r'QUALIFIED:\s*(YES|NO)', response, re.IGNORECASE)
        if qual_match:
            qualified = qual_match.group(1).upper() == "YES"
        else:
            qualified = score > 6

        reason_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        return AIQualifyResponse(
            score=score,
            qualified=qualified,
            reasoning=reasoning,
            domain=domain,
        )

    except Exception as e:
        logger.exception("AI qualify failed")
        return AIQualifyResponse(
            score=0,
            qualified=False,
            reasoning="Analysis failed",
            domain=body.domain or "unknown",
            error=str(e)[:200]
        )


# ============== ENDPOINT 3: AI PROCESSING ==============

@app.post("/ai/process", response_model=AIProcessResponse)
@limiter.limit("600/minute")
async def ai_process(request: Request, body: AIProcessRequest):
    """
    AI processing using GPT-OSS (~1-2 seconds).

    General-purpose AI endpoint - NO web scraping.
    Send any prompt + input data, get AI response.
    Uses GPT-OSS reasoning model for high-quality output.
    """
    try:
        # Build message
        if body.input_data:
            full_prompt = f"{body.prompt}\n\nInput:\n{body.input_data[:8000]}"
        else:
            full_prompt = body.prompt

        response = chat(
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=body.max_tokens,
        )

        return AIProcessResponse(
            output=response.strip(),
            success=True,
        )

    except Exception as e:
        logger.exception("AI process failed")
        return AIProcessResponse(
            output="",
            success=False,
            error=str(e)[:200]
        )


# ============== LEGACY ENDPOINTS ==============

def _parse_qualify_response(response: str, url: str) -> QualifyResponse:
    """Parse LLM response into QualifyResponse."""
    import re

    score = 0
    qualified = False
    reasoning = response.strip()

    # Extract score
    score_match = re.search(r'SCORE:\s*(\d+)', response, re.IGNORECASE)
    if score_match:
        score = min(10, max(1, int(score_match.group(1))))

    # Extract qualified
    qual_match = re.search(r'QUALIFIED:\s*(YES|NO)', response, re.IGNORECASE)
    if qual_match:
        qualified = qual_match.group(1).upper() == "YES"
    else:
        qualified = score > 6

    # Extract reasoning
    reason_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if reason_match:
        reasoning = reason_match.group(1).strip()

    return QualifyResponse(
        score=score,
        qualified=qualified,
        reasoning=reasoning,
        website=url,
    )


@app.post("/qualify", response_model=QualifyResponse)
@limiter.limit("600/minute")
async def qualify_website(request: Request, body: QualifyRequest):
    """
    Website qualification using GPT-OSS (~1-2 seconds).

    Uses LLM's knowledge of the domain - NO website fetching.
    """
    try:
        # Extract domain for cleaner prompt
        domain = body.url.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]

        prompt = f"""You are an expert at identifying B2B SaaS companies.

Analyze this company based on the domain name and your knowledge:
Domain: {domain}
URL: {body.url}

Criteria: {body.criteria}

If you don't know this company, make your best guess based on the domain name.

Respond in EXACTLY this format:
SCORE: [1-10]
QUALIFIED: [YES/NO]
REASONING: [One sentence]"""

        response = chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )

        return _parse_qualify_response(response, body.url)

    except Exception as e:
        logger.exception("Qualification failed")
        return QualifyResponse(
            score=0,
            qualified=False,
            reasoning="Analysis failed",
            website=body.url,
            error=str(e)[:200]
        )


@app.post("/qualify-deep", response_model=QualifyResponse)
@limiter.limit("30/minute")
async def qualify_website_deep(request: Request, body: QualifyRequest):
    """
    DEEP website qualification (5-10 seconds).

    Actually fetches and analyzes website content.
    Use this for more accurate results when speed isn't critical.
    """
    try:
        # Fetch website content - returns string directly
        text = fetch_and_extract(body.url, timeout=8)

        domain = body.url.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]

        if not text or text == "Content unavailable":
            # Fallback to LLM knowledge if fetch fails
            prompt = f"""Analyze this company: {domain}
Criteria: {body.criteria}
Respond: SCORE: [1-10] QUALIFIED: [YES/NO] REASONING: [one sentence]"""
        else:
            text = text[:3000]
            prompt = f"""Analyze this website content and score it.

Domain: {domain}
Content: {text[:2500]}

Criteria: {body.criteria}

Respond in EXACTLY this format:
SCORE: [1-10]
QUALIFIED: [YES/NO]
REASONING: [One sentence]"""

        response = chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )

        return _parse_qualify_response(response, body.url)

    except Exception as e:
        logger.exception("Deep qualification failed")
        return QualifyResponse(
            score=0,
            qualified=False,
            reasoning="Analysis failed",
            website=body.url,
            error=str(e)[:200]
        )


# ============== EMAIL VERIFICATION ENDPOINT ==============

@app.post("/email/verify", response_model=EmailVerifyResponse)
@limiter.limit("30/minute")
async def verify_email(request: Request, body: EmailVerifyRequest):
    """
    Verify an email address using Reacher.email.

    Performs comprehensive email verification including:
    - Syntax validation
    - MX record lookup
    - SMTP verification (checks if mailbox exists)
    - Disposable email detection
    - Catch-all domain detection

    Note: SMTP verification can take 5-30 seconds depending on the mail server.

    Requires REACHER_API_SECRET environment variable to be set.
    """
    try:
        # Build options from request
        options = {}
        if body.check_smtp is not None:
            options["check_smtp"] = body.check_smtp
        if body.check_gravatar is not None:
            options["check_gravatar"] = body.check_gravatar

        # Call Reacher with retry and circuit breaker
        result = await verify_email_with_retry(str(body.email), **options)

        # Transform Reacher response to our response model
        return _transform_reacher_response(result, str(body.email))

    except ReacherNotConfiguredError:
        raise HTTPException(
            status_code=503,
            detail="Email verification not configured. Set REACHER_API_SECRET environment variable."
        )
    except ReacherAuthError:
        raise HTTPException(
            status_code=401,
            detail="Email verification service authentication failed"
        )
    except ReacherRateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Email verification rate limit exceeded"
        )
    except ReacherTimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Email verification timed out"
        )
    except CircuitBreakerOpenError:
        raise HTTPException(
            status_code=503,
            detail="Email verification temporarily unavailable (circuit breaker open)"
        )
    except ReacherError as e:
        logger.exception("Email verification failed")
        raise HTTPException(
            status_code=503,
            detail=f"Email verification service unavailable: {str(e)[:100]}"
        )


def _transform_reacher_response(reacher_result: dict, email: str) -> EmailVerifyResponse:
    """Transform Reacher API response to our response model."""
    # Map Reacher's is_reachable to our enum
    is_reachable = reacher_result.get("is_reachable", "unknown")
    reachable_map = {
        "safe": ReachabilityStatus.SAFE,
        "invalid": ReachabilityStatus.INVALID,
        "risky": ReachabilityStatus.RISKY,
        "unknown": ReachabilityStatus.UNKNOWN,
    }
    reachable = reachable_map.get(is_reachable, ReachabilityStatus.UNKNOWN)

    # Extract MX records
    mx_records = []
    mx_data = reacher_result.get("mx", {})
    if mx_data.get("records"):
        for record in mx_data["records"]:
            mx_records.append(MxRecord(
                host=record.get("exchange", ""),
                priority=record.get("priority", 0),
            ))

    # Extract SMTP details
    smtp_data = reacher_result.get("smtp", {})
    smtp = None
    if smtp_data:
        smtp = SmtpDetails(
            can_connect=smtp_data.get("can_connect_smtp", False),
            has_full_inbox=smtp_data.get("has_full_inbox", False),
            is_catch_all=smtp_data.get("is_catch_all", False),
            is_deliverable=smtp_data.get("is_deliverable", False),
            is_disabled=smtp_data.get("is_disabled", False),
        )

    # Extract misc details
    misc_data = reacher_result.get("misc", {})
    misc = None
    if misc_data:
        misc = MiscDetails(
            is_disposable=misc_data.get("is_disposable", False),
            is_role_account=misc_data.get("is_role_account", False),
            gravatar_url=misc_data.get("gravatar_url"),
        )

    # Extract syntax info
    syntax_data = reacher_result.get("syntax", {})

    return EmailVerifyResponse(
        email=email,
        reachable=reachable,
        is_valid_syntax=syntax_data.get("is_valid_syntax", True),
        domain=syntax_data.get("domain", email.split("@")[1] if "@" in email else ""),
        mx_records=mx_records,
        has_mx_records=bool(mx_records),
        smtp=smtp,
        misc=misc,
        verification_time_ms=reacher_result.get("verification_time_ms", 0),
    )


# ============== HEALTH & STATUS ENDPOINTS ==============

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.

    Returns status of:
    - LLM backends (Ollama, HuggingFace)
    - Web crawling capabilities (Crawl4AI, Trafilatura)
    """
    llm_status = llm_health_check()
    crawl_status = get_fetcher_status()

    overall_status = "ok"
    if not llm_status.get("ollama", {}).get("available"):
        if not llm_status.get("huggingface", {}).get("available"):
            overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        llm=llm_status,
        crawl=crawl_status,
    )


@app.post("/reset-circuit-breakers")
async def reset_breakers():
    """
    Reset circuit breakers for LLM backends and email verification.

    Use this if circuit breakers are open after transient failures
    and you want to retry immediately.
    """
    reset_circuit_breakers()
    reset_reacher_circuit_breaker()
    return {"status": "ok", "message": "Circuit breakers reset (LLM + Reacher)"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    primary_model = config.llm.groq_models[0] if config.llm.groq_models else "none"

    return {
        "name": "Open Source Web Scraper Agent",
        "version": "3.0.0",
        "description": "Research & enrichment API powered by GPT-OSS on Groq",
        "model": {
            "primary": primary_model,
            "fallback": config.llm.groq_models[1] if len(config.llm.groq_models) > 1 else None,
            "type": "reasoning",
            "pricing": {
                "gpt-oss-20b": {"input": "$0.075/1M tokens", "output": "$0.30/1M tokens"},
                "gpt-oss-120b": {"input": "$0.15/1M tokens", "output": "$0.60/1M tokens"},
            },
        },
        "endpoints": {
            "POST /scrape": "Ultra-fast website scraper (2-5s)",
            "POST /ai/qualify": "AI qualification via GPT-OSS (~1-2s, 600/min)",
            "POST /ai/process": "General AI processing via GPT-OSS (~1-2s, 600/min)",
            "POST /research": "Full research with citations (10-30s)",
            "POST /email/verify": "Email verification via Reacher (5-30s)",
            "GET /health": "Check system health",
            "GET /status": "Model and capacity status",
        },
        "docs": "/docs",
    }


@app.get("/status")
async def status():
    """
    Get detailed status of GPT-OSS models and throughput capacity.

    Shows:
    - Model configuration (GPT-OSS 20B primary, 120B fallback)
    - Rate limits and Clay compatibility
    - Cost estimates per enrichment
    """
    num_keys = len(config.llm.groq_api_keys)
    primary_model = config.llm.groq_models[0] if config.llm.groq_models else "none"

    return {
        "model": {
            "primary": primary_model,
            "fallback": config.llm.groq_models[1] if len(config.llm.groq_models) > 1 else None,
            "type": "reasoning (GPT-OSS uses internal chain-of-thought)",
            "num_keys": num_keys,
        },
        "rate_limits": {
            "ai_qualify": "600/minute (10 req/sec)",
            "ai_process": "600/minute (10 req/sec)",
            "scrape": "200/minute",
            "research": "10/second",
            "note": "Groq Developer tier provides ~300+ RPM per key",
        },
        "clay_compatibility": {
            "clay_rate": "10 requests/second (600/min)",
            "status": "OK - Groq Developer tier active",
        },
        "cost_per_enrichment": {
            "gpt_oss_20b": "~$0.000275 per record (~$275 per 1M records)",
            "gpt_oss_120b": "~$0.000582 per record (~$582 per 1M records)",
            "vs_gpt4o_mini": "5.3x cheaper (GPT-OSS 20B) / 2.5x cheaper (GPT-OSS 120B)",
        },
    }


@app.get("/api-docs", response_class=HTMLResponse)
async def api_docs(request: Request):
    """Interactive API docs with copyable curl commands."""
    base = str(request.base_url).rstrip("/")
    primary_model = config.llm.groq_models[0] if config.llm.groq_models else "openai/gpt-oss-20b"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>API Docs — Open Source AI Agent</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0a0a0a;color:#e0e0e0;line-height:1.6}}
.container{{max-width:900px;margin:0 auto;padding:24px 20px}}
h1{{font-size:28px;font-weight:700;color:#fff;margin-bottom:6px}}
.subtitle{{color:#888;font-size:14px;margin-bottom:32px}}
.badge{{display:inline-block;background:#1a3a1a;color:#4ade80;font-size:11px;font-weight:600;padding:2px 8px;border-radius:4px;margin-left:8px}}
.endpoint{{background:#141414;border:1px solid #2a2a2a;border-radius:12px;margin-bottom:20px;overflow:hidden}}
.endpoint-header{{display:flex;align-items:center;gap:10px;padding:16px 20px;cursor:pointer;user-select:none}}
.endpoint-header:hover{{background:#1a1a1a}}
.method{{font-size:12px;font-weight:700;padding:4px 10px;border-radius:6px;font-family:monospace}}
.method-post{{background:#1a2a4a;color:#60a5fa}}
.method-get{{background:#1a3a1a;color:#4ade80}}
.path{{font-family:monospace;font-size:15px;font-weight:600;color:#fff}}
.desc{{color:#888;font-size:13px;margin-left:auto;white-space:nowrap}}
.rate{{font-size:11px;color:#666;font-family:monospace}}
.endpoint-body{{display:none;padding:0 20px 20px;border-top:1px solid #2a2a2a}}
.endpoint.open .endpoint-body{{display:block}}
.endpoint.open .endpoint-header{{border-bottom:none}}
.section-title{{font-size:12px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:1px;margin:16px 0 8px}}
.curl-box{{position:relative;background:#0d0d0d;border:1px solid #2a2a2a;border-radius:8px;padding:14px 16px;padding-right:80px;font-family:'SF Mono',Monaco,'Cascadia Code',monospace;font-size:13px;color:#c9d1d9;overflow-x:auto;white-space:pre-wrap;word-break:break-all;line-height:1.5}}
.copy-btn{{position:absolute;top:10px;right:10px;background:#2a2a2a;color:#ccc;border:none;padding:6px 14px;border-radius:6px;font-size:12px;cursor:pointer;font-family:inherit;transition:all 0.15s}}
.copy-btn:hover{{background:#3a3a3a;color:#fff}}
.copy-btn.copied{{background:#1a3a1a;color:#4ade80}}
.response-box{{background:#0d0d0d;border:1px solid #2a2a2a;border-radius:8px;padding:14px 16px;font-family:monospace;font-size:12px;color:#7c8a99;overflow-x:auto;white-space:pre;line-height:1.5}}
.params{{width:100%;border-collapse:collapse;font-size:13px;margin-top:4px}}
.params td{{padding:6px 0;vertical-align:top}}
.params td:first-child{{font-family:monospace;color:#60a5fa;font-weight:600;width:120px}}
.params td:nth-child(2){{color:#888;font-style:italic;width:70px}}
.params td:last-child{{color:#aaa}}
.params tr{{border-bottom:1px solid #1a1a1a}}
.footer{{text-align:center;color:#444;font-size:12px;margin-top:40px;padding:20px}}
.footer a{{color:#60a5fa;text-decoration:none}}
.chevron{{color:#555;margin-left:auto;transition:transform 0.2s;font-size:18px}}
.endpoint.open .chevron{{transform:rotate(90deg)}}
.nav{{display:flex;gap:8px;margin-bottom:24px;flex-wrap:wrap}}
.nav a{{color:#888;font-size:12px;text-decoration:none;padding:4px 12px;border:1px solid #2a2a2a;border-radius:6px;transition:all 0.15s}}
.nav a:hover{{color:#fff;border-color:#444}}
</style>
</head>
<body>
<div class="container">
<h1>Open Source AI Agent API <span class="badge">v3.0.0</span></h1>
<p class="subtitle">Powered by GPT-OSS on Groq &mdash; {primary_model} &mdash; 10 req/sec</p>

<div class="nav">
<a href="#ai-process">AI Process</a>
<a href="#ai-qualify">AI Qualify</a>
<a href="#scrape">Scrape</a>
<a href="#qualify">Qualify</a>
<a href="#qualify-deep">Qualify Deep</a>
<a href="#research">Research</a>
<a href="#email-verify">Email Verify</a>
<a href="#health">Health</a>
<a href="#status">Status</a>
</div>

<!-- AI Process -->
<div class="endpoint open" id="ai-process">
<div class="endpoint-header" onclick="toggle(this)">
<span class="method method-post">POST</span>
<span class="path">/ai/process</span>
<span class="desc">General AI processing</span>
<span class="rate">600/min</span>
<span class="chevron">&#9656;</span>
</div>
<div class="endpoint-body">
<p style="color:#aaa;font-size:13px;margin-bottom:12px">Send any prompt + input data, get AI response via GPT-OSS. No web scraping.</p>
<div class="section-title">Parameters</div>
<table class="params">
<tr><td>prompt</td><td>string</td><td>The prompt/instruction (required)</td></tr>
<tr><td>input_data</td><td>string</td><td>Input data to process (optional)</td></tr>
<tr><td>max_tokens</td><td>int</td><td>Max response tokens, default 500 (50-4000)</td></tr>
</table>
<div class="section-title">Curl</div>
<div class="curl-box">curl -X POST {base}/ai/process \\
  -H "Content-Type: application/json" \\
  -d '{{"prompt": "Summarize this company", "input_data": "Acme Corp builds cloud SaaS tools for enterprise", "max_tokens": 500}}'<button class="copy-btn" onclick="copyCmd(this)">Copy</button></div>
<div class="section-title">Response</div>
<div class="response-box">{{"output": "Acme Corp is a B2B SaaS company...", "success": true, "error": null}}</div>
</div>
</div>

<!-- AI Qualify -->
<div class="endpoint" id="ai-qualify">
<div class="endpoint-header" onclick="toggle(this)">
<span class="method method-post">POST</span>
<span class="path">/ai/qualify</span>
<span class="desc">AI lead qualification</span>
<span class="rate">600/min</span>
<span class="chevron">&#9656;</span>
</div>
<div class="endpoint-body">
<p style="color:#aaa;font-size:13px;margin-bottom:12px">Analyze pre-provided content and score against criteria. No web scraping.</p>
<div class="section-title">Parameters</div>
<table class="params">
<tr><td>content</td><td>string</td><td>Website content to analyze (required, max 50k chars)</td></tr>
<tr><td>domain</td><td>string</td><td>Domain name for context (optional)</td></tr>
<tr><td>criteria</td><td>string</td><td>Qualification criteria (default: B2B SaaS)</td></tr>
</table>
<div class="section-title">Curl</div>
<div class="curl-box">curl -X POST {base}/ai/qualify \\
  -H "Content-Type: application/json" \\
  -d '{{"content": "We build enterprise email automation tools for sales teams.", "domain": "acme.com", "criteria": "B2B SaaS that could integrate with email automation tools"}}'<button class="copy-btn" onclick="copyCmd(this)">Copy</button></div>
<div class="section-title">Response</div>
<div class="response-box">{{"score": 9, "qualified": true, "reasoning": "Enterprise email automation for sales teams is a strong B2B SaaS fit.", "domain": "acme.com", "error": null}}</div>
</div>
</div>

<!-- Scrape -->
<div class="endpoint" id="scrape">
<div class="endpoint-header" onclick="toggle(this)">
<span class="method method-post">POST</span>
<span class="path">/scrape</span>
<span class="desc">Website scraper</span>
<span class="rate">200/min</span>
<span class="chevron">&#9656;</span>
</div>
<div class="endpoint-body">
<p style="color:#aaa;font-size:13px;margin-bottom:12px">Ultra-fast website scraper with rotating User-Agents, retry logic, and bot detection.</p>
<div class="section-title">Parameters</div>
<table class="params">
<tr><td>url</td><td>string</td><td>Website URL to scrape (required)</td></tr>
<tr><td>timeout</td><td>int</td><td>Timeout in seconds, default 8 (1-30)</td></tr>
</table>
<div class="section-title">Curl</div>
<div class="curl-box">curl -X POST {base}/scrape \\
  -H "Content-Type: application/json" \\
  -d '{{"url": "https://example.com", "timeout": 8}}'<button class="copy-btn" onclick="copyCmd(this)">Copy</button></div>
<div class="section-title">Response</div>
<div class="response-box">{{"url": "https://example.com", "domain": "example.com", "content": "Example Domain. This domain is for use in illustrative examples...", "content_length": 1256, "success": true, "bot_protected": false, "error": null}}</div>
</div>
</div>

<!-- Qualify -->
<div class="endpoint" id="qualify">
<div class="endpoint-header" onclick="toggle(this)">
<span class="method method-post">POST</span>
<span class="path">/qualify</span>
<span class="desc">Qualify by domain (no fetch)</span>
<span class="rate">600/min</span>
<span class="chevron">&#9656;</span>
</div>
<div class="endpoint-body">
<p style="color:#aaa;font-size:13px;margin-bottom:12px">Qualify a website using LLM knowledge only. No fetching, instant response.</p>
<div class="section-title">Parameters</div>
<table class="params">
<tr><td>url</td><td>string</td><td>Website URL to analyze (required)</td></tr>
<tr><td>criteria</td><td>string</td><td>Qualification criteria (default: B2B SaaS)</td></tr>
</table>
<div class="section-title">Curl</div>
<div class="curl-box">curl -X POST {base}/qualify \\
  -H "Content-Type: application/json" \\
  -d '{{"url": "https://salesforce.com", "criteria": "B2B SaaS that could integrate with email automation tools"}}'<button class="copy-btn" onclick="copyCmd(this)">Copy</button></div>
<div class="section-title">Response</div>
<div class="response-box">{{"score": 9, "qualified": true, "reasoning": "Salesforce is a leading B2B SaaS CRM platform.", "website": "https://salesforce.com", "error": null}}</div>
</div>
</div>

<!-- Qualify Deep -->
<div class="endpoint" id="qualify-deep">
<div class="endpoint-header" onclick="toggle(this)">
<span class="method method-post">POST</span>
<span class="path">/qualify-deep</span>
<span class="desc">Qualify with content fetch</span>
<span class="rate">30/min</span>
<span class="chevron">&#9656;</span>
</div>
<div class="endpoint-body">
<p style="color:#aaa;font-size:13px;margin-bottom:12px">Fetches website content then qualifies. Slower but more accurate.</p>
<div class="section-title">Parameters</div>
<table class="params">
<tr><td>url</td><td>string</td><td>Website URL to fetch and analyze (required)</td></tr>
<tr><td>criteria</td><td>string</td><td>Qualification criteria (default: B2B SaaS)</td></tr>
</table>
<div class="section-title">Curl</div>
<div class="curl-box">curl -X POST {base}/qualify-deep \\
  -H "Content-Type: application/json" \\
  -d '{{"url": "https://stripe.com", "criteria": "B2B SaaS that could integrate with email automation tools"}}'<button class="copy-btn" onclick="copyCmd(this)">Copy</button></div>
<div class="section-title">Response</div>
<div class="response-box">{{"score": 8, "qualified": true, "reasoning": "Stripe is a B2B payments platform used by businesses.", "website": "https://stripe.com", "error": null}}</div>
</div>
</div>

<!-- Research -->
<div class="endpoint" id="research">
<div class="endpoint-header" onclick="toggle(this)">
<span class="method method-post">POST</span>
<span class="path">/research</span>
<span class="desc">Full research with citations</span>
<span class="rate">10/sec</span>
<span class="chevron">&#9656;</span>
</div>
<div class="endpoint-body">
<p style="color:#aaa;font-size:13px;margin-bottom:12px">Searches the web, fetches content, synthesizes an answer with citations and confidence scoring.</p>
<div class="section-title">Parameters</div>
<table class="params">
<tr><td>prompt</td><td>string</td><td>Research query (required, 3-2000 chars)</td></tr>
<tr><td>options</td><td>object</td><td>Optional: format, max_sources, use_react, verify</td></tr>
</table>
<div class="section-title">Curl</div>
<div class="curl-box">curl -X POST {base}/research \\
  -H "Content-Type: application/json" \\
  -d '{{"prompt": "What are the benefits of renewable energy?", "options": {{"format": "markdown", "max_sources": 5, "verify": true}}}}'<button class="copy-btn" onclick="copyCmd(this)">Copy</button></div>
<div class="section-title">Response</div>
<div class="response-box">{{"content": "## Benefits of Renewable Energy\\n...", "sources": [{{"url": "...", "title": "...", "snippet": "..."}}], "confidence": {{"overall": 0.85, "level": "high", ...}}, "verified_claims": [...], "error": null}}</div>
</div>
</div>

<!-- Email Verify -->
<div class="endpoint" id="email-verify">
<div class="endpoint-header" onclick="toggle(this)">
<span class="method method-post">POST</span>
<span class="path">/email/verify</span>
<span class="desc">Email verification</span>
<span class="rate">30/min</span>
<span class="chevron">&#9656;</span>
</div>
<div class="endpoint-body">
<p style="color:#aaa;font-size:13px;margin-bottom:12px">Verify email via Reacher: syntax, MX, SMTP, disposable detection. Requires REACHER_API_SECRET.</p>
<div class="section-title">Parameters</div>
<table class="params">
<tr><td>email</td><td>string</td><td>Email address to verify (required)</td></tr>
<tr><td>check_smtp</td><td>bool</td><td>Enable SMTP verification (optional)</td></tr>
<tr><td>check_gravatar</td><td>bool</td><td>Check for Gravatar (optional)</td></tr>
</table>
<div class="section-title">Curl</div>
<div class="curl-box">curl -X POST {base}/email/verify \\
  -H "Content-Type: application/json" \\
  -d '{{"email": "test@example.com"}}'<button class="copy-btn" onclick="copyCmd(this)">Copy</button></div>
<div class="section-title">Response</div>
<div class="response-box">{{"email": "test@example.com", "reachable": "safe", "is_valid_syntax": true, "domain": "example.com", "has_mx_records": true, ...}}</div>
</div>
</div>

<!-- Health -->
<div class="endpoint" id="health">
<div class="endpoint-header" onclick="toggle(this)">
<span class="method method-get">GET</span>
<span class="path">/health</span>
<span class="desc">System health check</span>
<span class="rate">&mdash;</span>
<span class="chevron">&#9656;</span>
</div>
<div class="endpoint-body">
<div class="section-title">Curl</div>
<div class="curl-box">curl {base}/health<button class="copy-btn" onclick="copyCmd(this)">Copy</button></div>
<div class="section-title">Response</div>
<div class="response-box">{{"status": "ok", "llm": {{"groq": {{"available": true, "primary_model": "openai/gpt-oss-20b", ...}}}}, "crawl": {{...}}}}</div>
</div>
</div>

<!-- Status -->
<div class="endpoint" id="status">
<div class="endpoint-header" onclick="toggle(this)">
<span class="method method-get">GET</span>
<span class="path">/status</span>
<span class="desc">Model &amp; capacity status</span>
<span class="rate">&mdash;</span>
<span class="chevron">&#9656;</span>
</div>
<div class="endpoint-body">
<div class="section-title">Curl</div>
<div class="curl-box">curl {base}/status<button class="copy-btn" onclick="copyCmd(this)">Copy</button></div>
<div class="section-title">Response</div>
<div class="response-box">{{"model": {{"primary": "openai/gpt-oss-20b", "fallback": "openai/gpt-oss-120b", ...}}, "rate_limits": {{...}}, "cost_per_enrichment": {{...}}}}</div>
</div>
</div>

<div class="footer">
<p>Built with FastAPI + GPT-OSS on Groq &mdash; <a href="/docs">Swagger UI</a> &mdash; <a href="/redoc">ReDoc</a></p>
</div>
</div>
<script>
function toggle(el){{el.parentElement.classList.toggle('open')}}
function copyCmd(btn){{
  const box=btn.parentElement;
  const text=box.innerText.replace('Copy','').replace('Copied!','').trim();
  navigator.clipboard.writeText(text).then(()=>{{
    btn.textContent='Copied!';btn.classList.add('copied');
    setTimeout(()=>{{btn.textContent='Copy';btn.classList.remove('copied')}},2000);
  }});
}}
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

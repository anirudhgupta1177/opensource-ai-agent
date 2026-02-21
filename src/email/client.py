"""
Reacher email verification client with circuit breaker and retry.
Follows patterns from src/llm/client.py and src/utils/resilience.py.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Optional

import httpx

from src.config import get_config
from src.utils.resilience import CircuitBreaker, CircuitBreakerOpenError

logger = logging.getLogger(__name__)


# ============== EXCEPTIONS ==============

class ReacherError(Exception):
    """Base exception for Reacher errors."""
    pass


class ReacherAuthError(ReacherError):
    """Authentication error with Reacher."""
    pass


class ReacherRateLimitError(ReacherError):
    """Rate limit exceeded on Reacher."""
    pass


class ReacherTimeoutError(ReacherError):
    """Timeout waiting for Reacher response."""
    pass


class ReacherNotConfiguredError(ReacherError):
    """Reacher is not configured (missing API secret)."""
    pass


# ============== CIRCUIT BREAKER ==============

_reacher_circuit_breaker: Optional[CircuitBreaker] = None


def _get_reacher_circuit_breaker() -> CircuitBreaker:
    """Get or create Reacher circuit breaker."""
    global _reacher_circuit_breaker
    if _reacher_circuit_breaker is None:
        config = get_config()
        _reacher_circuit_breaker = CircuitBreaker(
            failure_threshold=config.resilience.circuit_breaker_threshold,
            recovery_timeout=config.resilience.circuit_breaker_timeout,
            half_open_max_calls=config.resilience.circuit_breaker_half_open_calls,
            name="reacher"
        )
    return _reacher_circuit_breaker


def reset_reacher_circuit_breaker() -> None:
    """Reset the Reacher circuit breaker."""
    global _reacher_circuit_breaker
    if _reacher_circuit_breaker:
        _reacher_circuit_breaker.reset()
        logger.info("Reacher circuit breaker reset")


# ============== HTTP CLIENT ==============

class ReacherClient:
    """
    Async HTTP client for Reacher email verification.
    Uses httpx.AsyncClient for connection pooling.
    """

    def __init__(self):
        self.config = get_config()
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def base_url(self) -> str:
        """Get Reacher base URL from config/env."""
        return self.config.reacher.base_url_from_env

    @property
    def api_secret(self) -> Optional[str]:
        """Get API secret from config/env."""
        return self.config.reacher.api_secret

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client with connection pooling."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self.config.reacher.timeout,
                    connect=5.0
                ),
                limits=httpx.Limits(
                    max_keepalive_connections=10,
                    max_connections=20
                ),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_headers(self) -> dict[str, str]:
        """Build request headers with authentication."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_secret:
            # Reacher uses x-reacher-secret header for auth
            headers["x-reacher-secret"] = self.api_secret
        return headers

    async def verify_email(self, email: str, **options) -> dict[str, Any]:
        """
        Verify a single email address via Reacher.

        Args:
            email: Email address to verify
            **options: Optional overrides (check_smtp, check_gravatar, etc.)

        Returns:
            Reacher API response as dict

        Raises:
            ReacherError: If verification fails
        """
        if not self.api_secret:
            raise ReacherNotConfiguredError(
                "Reacher API secret not configured. Set REACHER_API_SECRET environment variable."
            )

        client = await self._get_client()

        # Build request payload (Reacher v0 API format)
        payload = {
            "to_email": email,
            "from_email": self.config.reacher.from_email,
            "hello_name": self.config.reacher.hello_name,
        }

        # Apply configuration defaults
        if self.config.reacher.verify_smtp:
            payload["smtp_check"] = True
        if self.config.reacher.check_gravatar:
            payload["check_gravatar"] = True

        # Apply overrides from options
        if "check_smtp" in options and options["check_smtp"] is not None:
            payload["smtp_check"] = options["check_smtp"]
        if "check_gravatar" in options and options["check_gravatar"] is not None:
            payload["check_gravatar"] = options["check_gravatar"]

        start_time = time.time()

        try:
            response = await client.post(
                f"{self.base_url}/v0/check_email",
                json=payload,
                headers=self._get_headers(),
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 401:
                raise ReacherAuthError("Invalid Reacher API secret")

            if response.status_code == 429:
                raise ReacherRateLimitError("Reacher rate limit exceeded")

            response.raise_for_status()

            result = response.json()
            result["verification_time_ms"] = elapsed_ms
            return result

        except httpx.TimeoutException as e:
            raise ReacherTimeoutError(
                f"Reacher timeout after {self.config.reacher.timeout}s"
            ) from e
        except httpx.HTTPStatusError as e:
            raise ReacherError(
                f"Reacher HTTP error: {e.response.status_code}"
            ) from e
        except httpx.RequestError as e:
            raise ReacherError(f"Reacher request failed: {str(e)}") from e


# ============== SINGLETON CLIENT ==============

_reacher_client: Optional[ReacherClient] = None


def get_reacher_client() -> ReacherClient:
    """Get or create the singleton Reacher client."""
    global _reacher_client
    if _reacher_client is None:
        _reacher_client = ReacherClient()
    return _reacher_client


async def close_reacher_client() -> None:
    """Close the Reacher client (call on app shutdown)."""
    global _reacher_client
    if _reacher_client:
        await _reacher_client.close()
        _reacher_client = None


# ============== VERIFY WITH RESILIENCE ==============

async def verify_email_with_retry(
    email: str,
    max_retries: int = 2,
    base_delay: float = 1.0,
    **options
) -> dict[str, Any]:
    """
    Verify email with retry logic and circuit breaker.

    This is the main entry point for email verification.

    Args:
        email: Email address to verify
        max_retries: Number of retry attempts
        base_delay: Base delay between retries (exponential backoff)
        **options: Options to pass to verify_email

    Returns:
        Reacher verification result

    Raises:
        CircuitBreakerOpenError: If circuit breaker is open
        ReacherError: If verification fails after retries
    """
    breaker = _get_reacher_circuit_breaker()
    client = get_reacher_client()

    # Check circuit breaker state
    if breaker.state.value == "open":
        raise CircuitBreakerOpenError(
            f"Circuit breaker 'reacher' is open. Retry after {breaker.recovery_timeout}s."
        )

    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            result = await client.verify_email(email, **options)
            breaker._on_success()
            return result

        except (ReacherTimeoutError, httpx.RequestError) as e:
            # Retryable errors
            last_error = e
            breaker._on_failure(e)

            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt)
                jitter = delay * 0.5 * random.random()
                actual_delay = delay + jitter

                logger.warning(
                    "Retry %d/%d for email verification after %.2fs (error: %s)",
                    attempt + 1,
                    max_retries,
                    actual_delay,
                    str(e)[:100]
                )
                await asyncio.sleep(actual_delay)
            else:
                logger.error("All retries exhausted for email verification")

        except (ReacherAuthError, ReacherNotConfiguredError, ReacherRateLimitError) as e:
            # Non-retryable errors
            breaker._on_failure(e)
            raise

        except Exception as e:
            # Unexpected errors
            last_error = e
            breaker._on_failure(e)
            raise

    if last_error:
        raise last_error
    raise RuntimeError("Unexpected retry state")


# ============== HEALTH CHECK ==============

async def reacher_health_check() -> dict[str, Any]:
    """
    Check Reacher service health.

    Returns:
        Dict with availability status and configuration info
    """
    config = get_config()
    status = {
        "available": False,
        "configured": bool(config.reacher.api_secret),
        "base_url": config.reacher.base_url_from_env,
        "circuit_breaker": "closed",
        "error": None,
    }

    # Check circuit breaker state
    breaker = _get_reacher_circuit_breaker()
    status["circuit_breaker"] = breaker.state.value

    if not status["configured"]:
        status["error"] = "REACHER_API_SECRET not set"
        return status

    if breaker.state.value == "open":
        status["error"] = "Circuit breaker is open"
        return status

    # Try a quick connectivity check (syntax-only, no SMTP)
    try:
        client = get_reacher_client()
        # Test with a well-known email, SMTP disabled for speed
        await client.verify_email("test@gmail.com", check_smtp=False)
        status["available"] = True
    except ReacherAuthError:
        status["error"] = "Invalid API secret"
    except ReacherTimeoutError:
        status["error"] = "Connection timeout"
    except Exception as e:
        status["error"] = str(e)[:100]

    return status

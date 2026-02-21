"""
Resilience patterns: circuit breaker, retry with jitter, graceful degradation.
These patterns prevent cascading failures and improve system reliability.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failing, reject requests immediately
    HALF_OPEN = "half_open"  # Testing recovery, allow limited requests


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and rejecting requests."""
    pass


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by:
    1. Tracking failures and opening circuit after threshold
    2. Rejecting requests while open (fail fast)
    3. Allowing test requests after recovery timeout
    4. Closing circuit after successful test requests

    Usage:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)

        @breaker
        def risky_operation():
            ...
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
        name: str = "default",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_successes = 0
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to wrap function with circuit breaker."""
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.execute(func, *args, **kwargs)
        return wrapper

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection."""
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.OPEN:
                logger.warning(
                    "Circuit breaker '%s' is OPEN, rejecting request",
                    self.name
                )
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open. "
                    f"Retry after {self.recovery_timeout}s."
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise

    def _check_state_transition(self) -> None:
        """Check if circuit should transition states based on time."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                logger.info(
                    "Circuit breaker '%s' transitioning to HALF_OPEN after %0.1fs",
                    self.name,
                    elapsed
                )
                self._state = CircuitState.HALF_OPEN
                self._half_open_successes = 0

    def _on_success(self) -> None:
        """Handle successful execution."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_max_calls:
                    logger.info(
                        "Circuit breaker '%s' transitioning to CLOSED after %d successes",
                        self.name,
                        self._half_open_successes
                    )
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def _on_failure(self, error: Exception) -> None:
        """Handle failed execution."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            logger.warning(
                "Circuit breaker '%s' recorded failure %d/%d: %s",
                self.name,
                self._failure_count,
                self.failure_threshold,
                str(error)[:100]
            )

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens circuit
                logger.info(
                    "Circuit breaker '%s' transitioning to OPEN (failed in half-open)",
                    self.name
                )
                self._state = CircuitState.OPEN
                self._half_open_successes = 0
            elif self._failure_count >= self.failure_threshold:
                logger.info(
                    "Circuit breaker '%s' transitioning to OPEN (threshold reached)",
                    self.name
                )
                self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_successes = 0
            logger.info("Circuit breaker '%s' manually reset to CLOSED", self.name)


def retry_with_jitter(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter_factor: float = 0.5,
    retryable_exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry decorator with exponential backoff and jitter.

    Jitter helps prevent thundering herd problem when multiple clients
    retry at the same time after a failure.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff calculation
        jitter_factor: Random jitter as fraction of delay (0-1)
        retryable_exceptions: Tuple of exception types that trigger retry

    Usage:
        @retry_with_jitter(max_retries=3, base_delay=1.0)
        def unreliable_operation():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt < max_retries - 1:
                        # Calculate delay with exponential backoff
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        # Add jitter: random value between 0 and jitter_factor * delay
                        jitter = delay * jitter_factor * random.random()
                        actual_delay = delay + jitter

                        logger.warning(
                            "Retry %d/%d for %s after %.2fs (error: %s)",
                            attempt + 1,
                            max_retries,
                            func.__name__,
                            actual_delay,
                            str(e)[:100]
                        )
                        time.sleep(actual_delay)
                    else:
                        logger.error(
                            "All %d retries exhausted for %s",
                            max_retries,
                            func.__name__
                        )

            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry state")

        return wrapper
    return decorator


class GracefulDegradation:
    """
    Graceful degradation handler for multi-tier fallbacks.

    Tries each handler in order until one succeeds.
    Returns result along with which tier succeeded.

    Usage:
        degradation = GracefulDegradation([
            (primary_handler, "Primary Service"),
            (backup_handler, "Backup Service"),
            (cache_handler, "Cached Response"),
        ])

        result, tier_name, tier_level = degradation.execute(args)
    """

    def __init__(self, handlers: list[tuple[Callable, str]]):
        """
        Initialize with list of (handler_function, description) tuples.
        Handlers are tried in order until one succeeds.
        """
        if not handlers:
            raise ValueError("At least one handler is required")
        self.handlers = handlers

    def execute(self, *args: Any, **kwargs: Any) -> tuple[Any, str, int]:
        """
        Execute handlers in order until one succeeds.

        Returns:
            Tuple of (result, handler_description, tier_level)
            tier_level is 0-indexed position of successful handler

        Raises:
            RuntimeError if all handlers fail
        """
        errors: list[tuple[str, Exception]] = []

        for tier, (handler, description) in enumerate(self.handlers):
            try:
                result = handler(*args, **kwargs)
                if tier > 0:
                    logger.info(
                        "Graceful degradation: using tier %d (%s)",
                        tier,
                        description
                    )
                return result, description, tier
            except Exception as e:
                errors.append((description, e))
                logger.warning(
                    "Handler '%s' (tier %d) failed: %s",
                    description,
                    tier,
                    str(e)[:100]
                )
                continue

        # All handlers failed
        error_summary = "; ".join(
            f"{name}: {str(err)[:50]}" for name, err in errors
        )
        raise RuntimeError(f"All fallback handlers failed: {error_summary}")

"""
Email verification module using Reacher.email backend.
"""
from src.email.models import (
    EmailVerifyRequest,
    EmailVerifyResponse,
    MiscDetails,
    MxRecord,
    ReachabilityStatus,
    SmtpDetails,
)
from src.email.client import (
    ReacherClient,
    ReacherError,
    ReacherAuthError,
    ReacherNotConfiguredError,
    ReacherRateLimitError,
    ReacherTimeoutError,
    close_reacher_client,
    get_reacher_client,
    reacher_health_check,
    reset_reacher_circuit_breaker,
    verify_email_with_retry,
)

__all__ = [
    # Models
    "EmailVerifyRequest",
    "EmailVerifyResponse",
    "MiscDetails",
    "MxRecord",
    "ReachabilityStatus",
    "SmtpDetails",
    # Client
    "ReacherClient",
    "get_reacher_client",
    "close_reacher_client",
    "verify_email_with_retry",
    "reacher_health_check",
    "reset_reacher_circuit_breaker",
    # Exceptions
    "ReacherError",
    "ReacherAuthError",
    "ReacherNotConfiguredError",
    "ReacherRateLimitError",
    "ReacherTimeoutError",
]

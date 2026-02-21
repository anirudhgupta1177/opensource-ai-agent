"""
Pydantic models for email verification API.
Based on Reacher.email API response format.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class ReachabilityStatus(str, Enum):
    """Email reachability status from Reacher."""
    SAFE = "safe"           # Email exists and is deliverable
    INVALID = "invalid"     # Email does not exist
    RISKY = "risky"         # Email may bounce (catch-all, etc.)
    UNKNOWN = "unknown"     # Could not determine


class EmailVerifyRequest(BaseModel):
    """Email verification request body."""
    email: EmailStr = Field(
        ...,
        description="Email address to verify"
    )
    check_smtp: Optional[bool] = Field(
        default=None,
        description="Override SMTP verification setting"
    )
    check_gravatar: Optional[bool] = Field(
        default=None,
        description="Check if email has Gravatar"
    )


class MxRecord(BaseModel):
    """MX record information."""
    host: str = Field(description="MX server hostname")
    priority: int = Field(description="MX priority (lower = higher priority)")


class SmtpDetails(BaseModel):
    """SMTP verification details."""
    can_connect: bool = Field(description="Could connect to SMTP server")
    has_full_inbox: bool = Field(description="Mailbox is full")
    is_catch_all: bool = Field(description="Domain accepts all emails")
    is_deliverable: bool = Field(description="Email is deliverable")
    is_disabled: bool = Field(description="Account is disabled")


class MiscDetails(BaseModel):
    """Miscellaneous verification details."""
    is_disposable: bool = Field(description="Email is from disposable domain")
    is_role_account: bool = Field(description="Email is role-based (info@, support@)")
    gravatar_url: Optional[str] = Field(default=None, description="Gravatar URL if exists")


class EmailVerifyResponse(BaseModel):
    """Email verification response."""
    email: str = Field(description="Verified email address")
    reachable: ReachabilityStatus = Field(description="Reachability status")
    is_valid_syntax: bool = Field(description="Email has valid syntax")

    # Domain information
    domain: str = Field(description="Email domain")
    mx_records: list[MxRecord] = Field(default=[], description="MX records")
    has_mx_records: bool = Field(description="Domain has MX records")

    # SMTP verification results
    smtp: Optional[SmtpDetails] = Field(default=None, description="SMTP verification details")

    # Additional checks
    misc: Optional[MiscDetails] = Field(default=None, description="Miscellaneous checks")

    # Metadata
    verification_time_ms: int = Field(description="Verification time in milliseconds")
    error: Optional[str] = Field(default=None, description="Error message if verification failed")

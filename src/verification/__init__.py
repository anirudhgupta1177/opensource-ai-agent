"""Verification modules for hallucination prevention and citation checking."""
from src.verification.citation_verifier import (
    CitationVerifier,
    verify_claims,
    extract_claims,
)
from src.verification.confidence_scorer import (
    ConfidenceScorer,
    compute_confidence,
)

__all__ = [
    "CitationVerifier",
    "verify_claims",
    "extract_claims",
    "ConfidenceScorer",
    "compute_confidence",
]

"""
Citation verification for hallucination prevention.
Verifies that claims in the response are supported by source documents.
"""
from __future__ import annotations


import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from src.config import get_config
from src.context.relevance import score_relevance
from src.llm.client import chat_with_json_output
from src.llm.prompts import get_citation_verification_prompt, get_claim_extraction_prompt

logger = logging.getLogger(__name__)


@dataclass
class ClaimVerification:
    """Result of verifying a single claim."""
    claim: str
    supported: bool
    confidence: float
    supporting_sources: list[str]
    supporting_quote: Optional[str] = None
    explanation: Optional[str] = None


@dataclass
class VerificationResult:
    """Overall verification result for a response."""
    total_claims: int
    verified_claims: int
    unverified_claims: int
    verification_rate: float
    claims: list[ClaimVerification]
    high_confidence_claims: int
    low_confidence_claims: int


class CitationVerifier:
    """
    Verifies claims in LLM responses against source documents.

    Uses a combination of:
    1. Semantic similarity matching
    2. Keyword overlap detection
    3. LLM-based verification for ambiguous cases
    """

    def __init__(
        self,
        min_similarity_threshold: float = 0.3,
        use_llm_verification: bool = True,
    ):
        self.min_similarity_threshold = min_similarity_threshold
        self.use_llm_verification = use_llm_verification

    def extract_claims(self, text: str) -> list[str]:
        """
        Extract individual factual claims from text.

        Uses LLM to identify distinct verifiable statements.
        """
        if not text or len(text) < 20:
            return []

        try:
            prompt = get_claim_extraction_prompt(text)
            response = chat_with_json_output(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )

            # Parse JSON array
            claims = self._parse_claims_response(response)
            logger.debug("Extracted %d claims from text", len(claims))
            return claims

        except Exception as e:
            logger.warning("Claim extraction failed: %s", str(e)[:100])
            # Fallback: split by sentences and filter
            return self._simple_claim_extraction(text)

    def _parse_claims_response(self, response: str) -> list[str]:
        """Parse LLM response for extracted claims."""
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
            claims = json.loads(response)
            if isinstance(claims, list):
                return [str(c) for c in claims if c and len(str(c)) > 10]
        except json.JSONDecodeError:
            pass

        return []

    def _simple_claim_extraction(self, text: str) -> list[str]:
        """Simple sentence-based claim extraction fallback."""
        # Remove citations for cleaner extraction
        text = re.sub(r'\[Source:[^\]]+\]', '', text)

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter out short sentences, questions, and meta-statements
            if len(sentence) < 30:
                continue
            if sentence.endswith('?'):
                continue
            if any(skip in sentence.lower() for skip in ['i think', 'in my opinion', 'might be', 'could be']):
                continue
            claims.append(sentence)

        return claims[:10]  # Limit to 10 claims

    def verify_claim(
        self,
        claim: str,
        sources: list[dict[str, Any]],
    ) -> ClaimVerification:
        """
        Verify a single claim against source documents.

        Args:
            claim: The claim to verify
            sources: List of source dicts with 'url' and 'text' keys

        Returns:
            ClaimVerification with support status and confidence
        """
        if not claim or not sources:
            return ClaimVerification(
                claim=claim,
                supported=False,
                confidence=0.0,
                supporting_sources=[],
            )

        supporting_sources = []
        max_similarity = 0.0
        best_quote = None

        for source in sources:
            source_text = source.get("text", "")
            if not source_text:
                continue

            # Calculate semantic similarity
            similarity = score_relevance(claim, source_text)

            if similarity > self.min_similarity_threshold:
                supporting_sources.append(source.get("url", "unknown"))
                if similarity > max_similarity:
                    max_similarity = similarity
                    # Try to find supporting quote
                    best_quote = self._find_supporting_quote(claim, source_text)

        # Calculate confidence based on similarity and source count
        confidence = self._calculate_confidence(max_similarity, len(supporting_sources))

        # Use LLM verification for borderline cases
        if self.use_llm_verification and 0.3 <= max_similarity <= 0.6 and sources:
            llm_result = self._llm_verify_claim(claim, sources[0].get("text", "")[:2000])
            if llm_result:
                # Blend LLM result with similarity-based result
                confidence = (confidence + llm_result.get("confidence", 0.5)) / 2
                if llm_result.get("supporting_quote"):
                    best_quote = llm_result["supporting_quote"]

        supported = confidence >= 0.5 and len(supporting_sources) > 0

        return ClaimVerification(
            claim=claim,
            supported=supported,
            confidence=confidence,
            supporting_sources=supporting_sources[:5],  # Limit sources
            supporting_quote=best_quote,
        )

    def _find_supporting_quote(self, claim: str, text: str, max_length: int = 200) -> Optional[str]:
        """Find a quote from the text that supports the claim."""
        # Simple keyword-based quote finding
        claim_words = set(claim.lower().split())
        claim_words -= {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'to', 'of', 'and', 'or', 'in', 'on', 'at'}

        sentences = re.split(r'(?<=[.!?])\s+', text)
        best_sentence = None
        best_overlap = 0

        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(claim_words & sentence_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence

        if best_sentence and best_overlap >= 2:
            return best_sentence[:max_length]
        return None

    def _calculate_confidence(self, similarity: float, source_count: int) -> float:
        """Calculate confidence score from similarity and source count."""
        # Base confidence from similarity
        confidence = similarity

        # Boost for multiple supporting sources
        if source_count > 1:
            confidence = min(1.0, confidence * (1 + 0.1 * (source_count - 1)))

        return confidence

    def _llm_verify_claim(self, claim: str, source_text: str) -> Optional[dict]:
        """Use LLM to verify a claim against source text."""
        try:
            prompt = get_citation_verification_prompt(claim, source_text)
            response = chat_with_json_output(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
            )

            # Parse JSON response
            response = response.strip()
            if "```" in response:
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
                if match:
                    response = match.group(1)

            return json.loads(response)

        except Exception as e:
            logger.debug("LLM verification failed: %s", str(e)[:50])
            return None

    def verify_response(
        self,
        response: str,
        sources: list[dict[str, Any]],
    ) -> VerificationResult:
        """
        Verify all claims in a response against sources.

        Args:
            response: The LLM-generated response text
            sources: List of source documents

        Returns:
            VerificationResult with overall and per-claim results
        """
        # Extract claims
        claims = self.extract_claims(response)

        if not claims:
            return VerificationResult(
                total_claims=0,
                verified_claims=0,
                unverified_claims=0,
                verification_rate=1.0,  # No claims means nothing to verify
                claims=[],
                high_confidence_claims=0,
                low_confidence_claims=0,
            )

        # Verify each claim
        verifications = []
        verified_count = 0
        high_conf_count = 0
        low_conf_count = 0

        for claim in claims:
            verification = self.verify_claim(claim, sources)
            verifications.append(verification)

            if verification.supported:
                verified_count += 1

            if verification.confidence >= 0.7:
                high_conf_count += 1
            elif verification.confidence < 0.4:
                low_conf_count += 1

        verification_rate = verified_count / len(claims) if claims else 0

        logger.info(
            "Verified %d/%d claims (%.1f%% rate)",
            verified_count,
            len(claims),
            verification_rate * 100
        )

        return VerificationResult(
            total_claims=len(claims),
            verified_claims=verified_count,
            unverified_claims=len(claims) - verified_count,
            verification_rate=verification_rate,
            claims=verifications,
            high_confidence_claims=high_conf_count,
            low_confidence_claims=low_conf_count,
        )


# Convenience functions
def verify_claims(
    response: str,
    sources: list[dict[str, Any]],
    use_llm: bool = True,
) -> VerificationResult:
    """Verify claims in a response against sources."""
    verifier = CitationVerifier(use_llm_verification=use_llm)
    return verifier.verify_response(response, sources)


def extract_claims(text: str) -> list[str]:
    """Extract claims from text."""
    verifier = CitationVerifier()
    return verifier.extract_claims(text)

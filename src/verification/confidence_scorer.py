"""
Confidence scoring for research responses.
Computes overall confidence based on multiple factors.
"""
from __future__ import annotations


import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from src.config import get_config
from src.context.relevance import compute_source_diversity
from src.verification.citation_verifier import VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceFactors:
    """Individual factors contributing to confidence score."""
    source_quality: float  # Quality of sources (domain reputation, content length)
    citation_density: float  # How well the response cites sources
    verification_rate: float  # Percentage of claims verified
    source_consensus: float  # Agreement between multiple sources
    source_diversity: float  # Diversity of source domains
    relevance_score: float  # How relevant sources are to query


@dataclass
class ConfidenceScore:
    """Overall confidence assessment."""
    overall: float  # 0.0 to 1.0
    factors: ConfidenceFactors
    level: str  # "high", "moderate", "low"
    concerns: list[str]
    recommendation: str


class ConfidenceScorer:
    """
    Computes confidence scores for research responses.

    Considers multiple factors:
    - Source quality and diversity
    - Citation density in response
    - Claim verification rate
    - Source consensus/agreement
    """

    def __init__(self):
        config = get_config()
        self.high_threshold = config.agent.high_confidence_threshold
        self.min_threshold = config.agent.min_confidence_threshold

    def score_source_quality(self, sources: list[dict[str, Any]]) -> float:
        """
        Score the quality of sources.

        Considers:
        - Content length (more content generally means more authoritative)
        - Domain reputation (simple heuristics)
        - Number of sources
        """
        if not sources:
            return 0.0

        scores = []
        for source in sources:
            text = source.get("text", "")
            url = source.get("url", "")

            # Base score from content length
            length_score = min(1.0, len(text) / 5000)

            # Domain reputation bonus
            domain_score = 0.5  # Default
            reputable_domains = [
                'wikipedia.org', 'github.com', 'stackoverflow.com',
                '.edu', '.gov', 'nature.com', 'science.org',
                'arxiv.org', 'acm.org', 'ieee.org',
            ]
            for domain in reputable_domains:
                if domain in url.lower():
                    domain_score = 0.8
                    break

            # Combine scores
            source_score = (length_score + domain_score) / 2
            scores.append(source_score)

        # Average with bonus for multiple sources
        avg_score = sum(scores) / len(scores)
        source_count_bonus = min(0.2, len(sources) * 0.05)

        return min(1.0, avg_score + source_count_bonus)

    def score_citation_density(self, response: str, sources: list[dict[str, Any]]) -> float:
        """
        Score how well the response cites its sources.

        A well-cited response should have citations distributed throughout.
        """
        if not response or not sources:
            return 0.0

        # Count citations in response
        citation_pattern = r'\[Source:[^\]]+\]'
        citations = re.findall(citation_pattern, response)
        citation_count = len(citations)

        # Calculate expected citations based on response length
        word_count = len(response.split())
        expected_citations = max(1, word_count // 100)  # ~1 citation per 100 words

        # Score based on actual vs expected
        if expected_citations > 0:
            ratio = citation_count / expected_citations
            density_score = min(1.0, ratio)
        else:
            density_score = 1.0 if citation_count > 0 else 0.0

        # Check citation distribution (not all bunched at the end)
        if citation_count > 2:
            first_half = response[:len(response)//2]
            first_half_citations = len(re.findall(citation_pattern, first_half))
            distribution_score = min(1.0, first_half_citations / (citation_count / 2))
            density_score = (density_score + distribution_score) / 2

        return density_score

    def score_source_consensus(
        self,
        verification_result: Optional[VerificationResult],
        sources: list[dict[str, Any]],
    ) -> float:
        """
        Score agreement between sources on verified claims.

        Higher score when multiple sources agree on claims.
        """
        if not verification_result or not verification_result.claims:
            return 0.5  # Neutral when no verification data

        # Check how many claims have multiple supporting sources
        multi_source_claims = 0
        for claim in verification_result.claims:
            if len(claim.supporting_sources) > 1:
                multi_source_claims += 1

        if verification_result.total_claims > 0:
            consensus_ratio = multi_source_claims / verification_result.total_claims
        else:
            consensus_ratio = 0.5

        return consensus_ratio

    def compute(
        self,
        response: str,
        sources: list[dict[str, Any]],
        verification_result: Optional[VerificationResult] = None,
        query: Optional[str] = None,
    ) -> ConfidenceScore:
        """
        Compute overall confidence score for a response.

        Args:
            response: The generated response text
            sources: List of source documents used
            verification_result: Optional claim verification results
            query: Optional original query for relevance scoring

        Returns:
            ConfidenceScore with overall score and breakdown
        """
        concerns = []

        # Calculate individual factors
        source_quality = self.score_source_quality(sources)
        if source_quality < 0.4:
            concerns.append("Low source quality")

        citation_density = self.score_citation_density(response, sources)
        if citation_density < 0.3:
            concerns.append("Insufficient citations in response")

        if verification_result:
            verification_rate = verification_result.verification_rate
            if verification_rate < 0.5:
                concerns.append(f"Only {verification_rate*100:.0f}% of claims verified")
        else:
            verification_rate = 0.5  # Neutral when not verified

        source_consensus = self.score_source_consensus(verification_result, sources)
        if source_consensus < 0.3:
            concerns.append("Limited agreement between sources")

        source_diversity = compute_source_diversity(sources)
        if source_diversity < 0.3 and len(sources) > 2:
            concerns.append("Sources lack diversity (mostly same domain)")

        # Calculate relevance score if query provided
        if query and sources:
            relevance_scores = []
            for source in sources:
                rel = source.get("relevance", 0.5)
                relevance_scores.append(rel)
            relevance_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
        else:
            relevance_score = 0.5

        if relevance_score < 0.3:
            concerns.append("Sources may not be highly relevant to query")

        # Create factors object
        factors = ConfidenceFactors(
            source_quality=source_quality,
            citation_density=citation_density,
            verification_rate=verification_rate,
            source_consensus=source_consensus,
            source_diversity=source_diversity,
            relevance_score=relevance_score,
        )

        # Calculate weighted overall score
        weights = {
            "source_quality": 0.15,
            "citation_density": 0.15,
            "verification_rate": 0.30,  # Verification is most important
            "source_consensus": 0.15,
            "source_diversity": 0.10,
            "relevance_score": 0.15,
        }

        overall = (
            factors.source_quality * weights["source_quality"] +
            factors.citation_density * weights["citation_density"] +
            factors.verification_rate * weights["verification_rate"] +
            factors.source_consensus * weights["source_consensus"] +
            factors.source_diversity * weights["source_diversity"] +
            factors.relevance_score * weights["relevance_score"]
        )

        # Determine confidence level
        if overall >= self.high_threshold:
            level = "high"
            recommendation = "Results are well-supported by sources"
        elif overall >= self.min_threshold:
            level = "moderate"
            recommendation = "Results are partially supported; verify critical claims"
        else:
            level = "low"
            recommendation = "Results should be independently verified"

        logger.info(
            "Confidence score: %.2f (%s) - %d concerns",
            overall,
            level,
            len(concerns)
        )

        return ConfidenceScore(
            overall=overall,
            factors=factors,
            level=level,
            concerns=concerns,
            recommendation=recommendation,
        )


def compute_confidence(
    response: str,
    sources: list[dict[str, Any]],
    verification_result: Optional[VerificationResult] = None,
    query: Optional[str] = None,
) -> ConfidenceScore:
    """Convenience function to compute confidence score."""
    scorer = ConfidenceScorer()
    return scorer.compute(response, sources, verification_result, query)

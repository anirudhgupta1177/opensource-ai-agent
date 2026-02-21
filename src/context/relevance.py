"""
Relevance scoring for context prioritization.
Uses TF-IDF similarity to rank sources by relevance to the query.
"""
from __future__ import annotations


import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import sklearn for TF-IDF, fall back to simple keyword matching
_SKLEARN_AVAILABLE = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available, using simple keyword matching for relevance")


def _preprocess_text(text: str) -> str:
    """Preprocess text for similarity comparison."""
    # Lowercase and remove extra whitespace
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.strip()


def _simple_keyword_score(query: str, text: str) -> float:
    """
    Simple keyword-based relevance scoring.
    Used as fallback when sklearn is not available.
    """
    query_words = set(_preprocess_text(query).split())
    text_words = set(_preprocess_text(text).split())

    if not query_words or not text_words:
        return 0.0

    # Jaccard-like similarity
    intersection = len(query_words & text_words)
    union = len(query_words | text_words)

    if union == 0:
        return 0.0

    base_score = intersection / union

    # Boost if query words appear multiple times
    text_lower = text.lower()
    word_count_boost = 0
    for word in query_words:
        count = text_lower.count(word)
        if count > 1:
            word_count_boost += min(count * 0.05, 0.2)  # Cap boost per word

    return min(1.0, base_score + word_count_boost)


def score_relevance(query: str, text: str) -> float:
    """
    Calculate relevance score between query and text.

    Uses TF-IDF cosine similarity if sklearn is available,
    otherwise falls back to simple keyword matching.

    Args:
        query: The search query or research prompt
        text: The text content to score

    Returns:
        Float between 0.0 and 1.0, where higher is more relevant
    """
    if not query or not text:
        return 0.0

    query = _preprocess_text(query)
    text = _preprocess_text(text)

    if not query or not text:
        return 0.0

    if _SKLEARN_AVAILABLE:
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2)  # Include bigrams for better matching
            )
            tfidf_matrix = vectorizer.fit_transform([query, text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning("TF-IDF scoring failed, using fallback: %s", e)
            return _simple_keyword_score(query, text)
    else:
        return _simple_keyword_score(query, text)


def score_sources(
    query: str,
    sources: list[dict],
    text_key: str = "text"
) -> list[dict]:
    """
    Score and sort sources by relevance to query.

    Args:
        query: The search query
        sources: List of source dicts containing text content
        text_key: Key in source dict containing the text to score

    Returns:
        Sources sorted by relevance (highest first) with 'relevance' key added
    """
    scored = []
    for source in sources:
        text = source.get(text_key, "") or ""
        # Also consider title and snippet if available
        title = source.get("title", "") or ""
        snippet = source.get("snippet", "") or ""

        # Combine text sources for scoring, weighted towards main text
        combined_text = f"{title} {title} {snippet} {text}"  # Title weighted 2x

        relevance = score_relevance(query, combined_text)
        scored.append({**source, "relevance": relevance})

    # Sort by relevance descending
    scored.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    return scored


def filter_by_relevance(
    sources: list[dict],
    min_relevance: float = 0.1,
    max_sources: Optional[int] = None
) -> list[dict]:
    """
    Filter sources by minimum relevance threshold.

    Args:
        sources: List of source dicts with 'relevance' key
        min_relevance: Minimum relevance score to include
        max_sources: Maximum number of sources to return

    Returns:
        Filtered list of sources meeting the threshold
    """
    filtered = [s for s in sources if s.get("relevance", 0) >= min_relevance]

    if max_sources:
        filtered = filtered[:max_sources]

    return filtered


def compute_source_diversity(sources: list[dict]) -> float:
    """
    Compute diversity score of sources based on their URLs.
    Higher diversity means sources from different domains.

    Returns:
        Float between 0.0 and 1.0
    """
    if not sources:
        return 0.0

    domains = set()
    for source in sources:
        url = source.get("url", "")
        # Extract domain from URL
        match = re.search(r'://([^/]+)', url)
        if match:
            domain = match.group(1)
            # Remove www prefix
            domain = re.sub(r'^www\.', '', domain)
            domains.add(domain)

    if len(sources) <= 1:
        return 1.0

    # Diversity = unique domains / total sources
    return len(domains) / len(sources)

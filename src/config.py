"""
Configuration management for the Open Source Web Scraper Agent.
All settings are centralized here with sensible defaults.
"""
from __future__ import annotations


import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """LLM backend configuration."""
    # Groq settings (PRIMARY - fast cloud inference)
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_models: list[str] = field(default_factory=lambda: [
        "openai/gpt-oss-20b",       # Primary - fastest, cheapest ($0.075/$0.30 per 1M tokens)
        "openai/gpt-oss-120b",      # Fallback - higher quality reasoning ($0.15/$0.60 per 1M tokens)
        "llama-3.1-8b-instant",     # Fallback - non-reasoning, fast
    ])

    # GPT-OSS models use reasoning tokens that count toward max_tokens.
    # We add this overhead to ensure enough room for both reasoning + output.
    gpt_oss_reasoning_overhead: int = 1500

    # Ollama settings (LOCAL fallback)
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_models: list[str] = field(default_factory=lambda: [
        "qwen2.5:7b",       # Primary - fast, good quality (use 7B for speed)
        "mistral:7b-instruct",  # Fallback 1 - fast, efficient
        "qwen2.5:14b",      # Fallback 2 - higher quality when needed
    ])

    # HuggingFace settings (free tier fallback)
    hf_base_url: str = "https://router.huggingface.co/v1"
    hf_models: list[str] = field(default_factory=lambda: [
        "Qwen/Qwen2.5-14B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ])

    # Request settings
    timeout: float = 60.0
    max_retries: int = 3
    max_tokens_default: int = 4096

    @property
    def groq_api_key(self) -> Optional[str]:
        """Get first Groq API key from environment (for backward compatibility)."""
        keys = self.groq_api_keys
        return keys[0] if keys else None

    @property
    def groq_api_keys(self) -> list[str]:
        """
        Get all Groq API keys from environment.
        Supports multiple keys via GROQ_API_KEYS (comma-separated) or single GROQ_API_KEY.
        Multiple keys allow higher throughput by rotating between them.

        Example: GROQ_API_KEYS=key1,key2,key3
        """
        # First try comma-separated keys
        keys_str = os.getenv("GROQ_API_KEYS", "").strip()
        if keys_str:
            keys = [k.strip() for k in keys_str.split(",") if k.strip()]
            if keys:
                return keys

        # Fallback to single key
        single_key = os.getenv("GROQ_API_KEY", "").strip()
        return [single_key] if single_key else []

    @property
    def hf_token(self) -> Optional[str]:
        """Get HuggingFace token from environment."""
        token = os.getenv("HF_TOKEN", "").strip()
        return token if token else None


@dataclass
class AgentConfig:
    """Agent behavior configuration."""
    # ReAct agent settings
    max_react_steps: int = 5
    enable_react_for_complex: bool = True

    # Query decomposition
    decomposition_word_threshold: int = 15  # Decompose queries longer than this
    max_sub_queries: int = 4

    # Confidence thresholds
    min_confidence_threshold: float = 0.5
    high_confidence_threshold: float = 0.8

    # Source settings (reduced for speed - increase for quality)
    max_sources: int = 5
    max_sources_cap: int = 10
    search_max_results: int = 8


@dataclass
class ContextConfig:
    """Context window and token management."""
    max_context_tokens: int = 6000
    reserved_for_response: int = 1024
    synthesis_max_tokens: int = 2048

    # Token estimation
    chars_per_token: int = 4  # Heuristic fallback

    # Chunking
    min_chunk_tokens: int = 100
    max_chunk_tokens: int = 1500


@dataclass
class CrawlConfig:
    """Web crawling configuration."""
    fetch_timeout: int = 10
    max_fetches_total_time: int = 30
    user_agent: str = "Mozilla/5.0 (compatible; WebScraperAgent/1.0; +research)"

    # Crawl4AI settings
    enable_js_rendering: bool = True
    crawl4ai_timeout: int = 15

    # Content extraction
    min_content_length: int = 100
    max_content_length: int = 50000


@dataclass
class ResilienceConfig:
    """Resilience patterns configuration."""
    # Circuit breaker
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    circuit_breaker_half_open_calls: int = 3

    # Retry settings
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0
    retry_jitter_factor: float = 0.5


@dataclass
class AppConfig:
    """Application-level configuration."""
    # API settings
    rate_limit: str = "10/second"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


@dataclass
class ReacherConfig:
    """Reacher email verification configuration."""
    # Reacher instance settings
    base_url: str = "http://localhost:8080"
    timeout: float = 30.0  # SMTP verification can be slow

    # Verification options
    verify_smtp: bool = True  # Enable SMTP verification for accuracy
    check_gravatar: bool = False  # Optional gravatar check
    check_catch_all: bool = True  # Check if domain is catch-all

    # HELLO/FROM settings for SMTP (Reacher uses these for verification)
    hello_name: str = "localhost"
    from_email: str = "noreply@localhost"

    @property
    def api_secret(self) -> Optional[str]:
        """Get Reacher API secret from environment."""
        secret = os.getenv("REACHER_API_SECRET", "").strip()
        return secret if secret else None

    @property
    def base_url_from_env(self) -> str:
        """Get Reacher base URL from environment (or default)."""
        return os.getenv("REACHER_BASE_URL", self.base_url).strip()


class Config:
    """Main configuration container."""

    def __init__(self):
        self.llm = LLMConfig()
        self.agent = AgentConfig()
        self.context = ContextConfig()
        self.crawl = CrawlConfig()
        self.resilience = ResilienceConfig()
        self.app = AppConfig()
        self.reacher = ReacherConfig()

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration with environment variable overrides."""
        config = cls()

        # Override from environment if set
        if os.getenv("OLLAMA_BASE_URL"):
            config.llm.ollama_base_url = os.getenv("OLLAMA_BASE_URL")

        if os.getenv("MAX_CONTEXT_TOKENS"):
            config.context.max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS"))

        if os.getenv("MAX_REACT_STEPS"):
            config.agent.max_react_steps = int(os.getenv("MAX_REACT_STEPS"))

        if os.getenv("LOG_LEVEL"):
            config.app.log_level = os.getenv("LOG_LEVEL")

        # Reacher overrides
        if os.getenv("REACHER_BASE_URL"):
            config.reacher.base_url = os.getenv("REACHER_BASE_URL")
        if os.getenv("REACHER_TIMEOUT"):
            config.reacher.timeout = float(os.getenv("REACHER_TIMEOUT"))
        if os.getenv("REACHER_HELLO_NAME"):
            config.reacher.hello_name = os.getenv("REACHER_HELLO_NAME")
        if os.getenv("REACHER_FROM_EMAIL"):
            config.reacher.from_email = os.getenv("REACHER_FROM_EMAIL")

        return config


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reset_config() -> None:
    """Reset configuration (useful for testing)."""
    global _config
    _config = None

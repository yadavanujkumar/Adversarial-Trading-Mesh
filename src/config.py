"""
src/config.py — Centralised, validated configuration for Aura-Swarm-Quant.

Uses pydantic-settings so every environment variable is type-checked and
validated at startup.  A single Settings instance is created at import time
and shared across the entire application.

Benefits over raw os.getenv()
------------------------------
* All config in one place — no scattered os.getenv("KEY", "default") calls.
* Startup fails fast with a descriptive error if a required variable is missing
  or a value is out of range (e.g. STOP_LOSS_PCT=2.0 instead of 0.02).
* Easy to unit-test: swap Settings() for a custom instance.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application-wide configuration loaded from the environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key (required for LLM agents)")
    llm_model: str = Field(default="gpt-4o-mini", description="OpenAI model identifier")
    llm_max_retries: int = Field(default=3, ge=1, le=10, description="LLM retry attempts on transient errors")
    llm_retry_delay_secs: float = Field(default=1.0, ge=0.0, description="Initial retry back-off in seconds")

    # ── Exa.ai sentiment ─────────────────────────────────────────────────────
    exa_api_key: str = Field(default="", description="Exa.ai API key (optional, enables live sentiment)")
    exa_num_results: int = Field(default=8, ge=1, le=50)

    # ── Binance WebSocket ─────────────────────────────────────────────────────
    ws_symbol: str = Field(default="btcusdt", description="Binance stream symbol (lowercase)")
    ws_throttle_secs: float = Field(default=1.0, ge=0.1, le=60.0,
                                    description="Minimum seconds between yielded snapshots")

    # ── Risk parameters ───────────────────────────────────────────────────────
    bull_threshold: float = Field(default=0.8, ge=0.0, le=1.0,
                                  description="Min bull confidence required to issue BUY")
    bear_threshold: float = Field(default=0.3, ge=0.0, le=1.0,
                                  description="Max bear confidence allowed for a BUY")
    stop_loss_pct: float = Field(default=0.02, gt=0.0, le=0.5,
                                 description="Stop-loss distance below entry price (e.g. 0.02 = 2 %)")

    # ── Dashboard ─────────────────────────────────────────────────────────────
    dashboard_host: str = Field(default="0.0.0.0")
    dashboard_port: int = Field(default=8000, ge=1, le=65535)
    # Optional: require this header value for mutating API calls
    api_auth_token: str = Field(default="", description="If set, /api/trigger requires X-API-Token header")
    # Requests per minute per IP for /api/trigger
    api_rate_limit_rpm: int = Field(default=30, ge=1, le=600)

    # ── Memory / persistence ──────────────────────────────────────────────────
    memory_path: str = Field(default="agent_memory.json")
    memory_max_lessons: int = Field(default=10, ge=1, le=100,
                                    description="Maximum lessons kept per agent")

    # ── Alpaca order execution ────────────────────────────────────────────────
    alpaca_api_key: str = Field(default="")
    alpaca_secret_key: str = Field(default="")
    alpaca_base_url: str = Field(default="https://paper-api.alpaca.markets")
    alpaca_enabled: bool = Field(default=False,
                                 description="Set True to submit real orders through Alpaca")

    # ── Resilience ────────────────────────────────────────────────────────────
    circuit_breaker_threshold: int = Field(
        default=5, ge=1,
        description="Consecutive failures before a circuit opens",
    )
    circuit_breaker_timeout_secs: float = Field(
        default=60.0, ge=5.0,
        description="Seconds a circuit stays OPEN before attempting recovery",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
    )
    log_json: bool = Field(default=False, description="Emit machine-readable JSON log lines")

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("ws_symbol")
    @classmethod
    def normalise_symbol(cls, v: str) -> str:
        return v.lower().strip()

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> "Settings":
        if self.bull_threshold <= self.bear_threshold:
            raise ValueError(
                f"bull_threshold ({self.bull_threshold}) must be greater than "
                f"bear_threshold ({self.bear_threshold}) for BUY signals to ever trigger."
            )
        return self

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def memory_path_obj(self) -> Path:
        return Path(self.memory_path)

    def warn_if_demo(self) -> None:
        """Log a warning when critical keys are not configured."""
        if not self.openai_api_key:
            logger.warning(
                "OPENAI_API_KEY is not set — LLM agents will fail on every call."
            )
        if not self.exa_api_key:
            logger.info(
                "EXA_API_KEY not set — Bull agent will run without live sentiment."
            )
        if self.alpaca_enabled and (not self.alpaca_api_key or not self.alpaca_secret_key):
            logger.warning(
                "ALPACA_ENABLED=true but ALPACA_API_KEY / ALPACA_SECRET_KEY are missing."
            )


# Module-level singleton — imported everywhere in the application.
settings = Settings()

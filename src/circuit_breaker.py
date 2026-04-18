"""
src/circuit_breaker.py — Async circuit breaker for external API calls.

Implements the classic three-state circuit breaker pattern
(CLOSED → OPEN → HALF-OPEN) to prevent cascade failures when the OpenAI
or Exa.ai APIs are unavailable or rate-limiting the application.

State machine
-------------
  CLOSED   → Normal operation.  Every call passes through.
  OPEN     → Failure threshold exceeded.  Calls are rejected immediately
              with CircuitBreakerError (fast-fail).
  HALF_OPEN → Recovery probe.  One call is allowed through.  Success → CLOSED.
               Failure → back to OPEN with a refreshed timeout.

Usage
-----
    result = await openai_circuit.call(llm.ainvoke, messages)

    # Or use as async context manager for manual control:
    async with openai_circuit:
        result = await llm.ainvoke(messages)
"""
from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(RuntimeError):
    """Raised when a call is rejected because the circuit is OPEN."""


class CircuitBreaker:
    """
    Async circuit breaker with configurable failure threshold and recovery timeout.

    Parameters
    ----------
    name:               Identifier used in log messages and /api/circuit-breakers.
    failure_threshold:  Consecutive failures before transitioning to OPEN.
    recovery_timeout:   Seconds the circuit stays OPEN before a recovery probe.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        self.name = name
        self._threshold = failure_threshold
        self._timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._total_failures = 0
        self._opened_at: float = 0.0
        self._last_failure_at: float = 0.0
        self._lock = asyncio.Lock()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    def status(self) -> dict[str, Any]:
        """Return a JSON-serialisable status dict for the health endpoint."""
        return {
            "name": self.name,
            "state": self._state.value,
            "consecutive_failures": self._failures,
            "total_failures": self._total_failures,
            "failure_threshold": self._threshold,
            "recovery_timeout_secs": self._timeout,
            "opened_at": self._opened_at or None,
            "seconds_until_probe": max(
                0.0,
                self._timeout - (time.monotonic() - self._opened_at)
            ) if self._state == CircuitState.OPEN else 0.0,
        }

    # ── Core call method ──────────────────────────────────────────────────────

    async def call(
        self,
        coro_fn: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute *coro_fn(*args, **kwargs)* through the circuit breaker.

        Raises
        ------
        CircuitBreakerError
            If the circuit is OPEN and the recovery timeout has not elapsed.
        """
        async with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self._timeout:
                    logger.info("Circuit [%s] → HALF-OPEN (probing recovery)", self.name)
                    self._state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerError(
                        f"Circuit '{self.name}' is OPEN — call rejected "
                        f"({self._timeout - elapsed:.0f}s until recovery probe)."
                    )

        try:
            result = await coro_fn(*args, **kwargs)
        except Exception as exc:
            await self._on_failure(exc)
            raise

        await self._on_success()
        return result

    # ── Private state transitions ─────────────────────────────────────────────

    async def _on_failure(self, exc: Exception) -> None:
        async with self._lock:
            self._failures += 1
            self._total_failures += 1
            self._last_failure_at = time.monotonic()

            if self._state == CircuitState.HALF_OPEN or self._failures >= self._threshold:
                prev = self._state
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                logger.error(
                    "Circuit [%s] %s → OPEN after %d consecutive failures. Error: %s",
                    self.name,
                    prev.value,
                    self._failures,
                    exc,
                )

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state != CircuitState.CLOSED:
                logger.info(
                    "Circuit [%s] %s → CLOSED (recovered after %d failures)",
                    self.name,
                    self._state.value,
                    self._failures,
                )
            self._state = CircuitState.CLOSED
            self._failures = 0


# ── Module-level circuit breakers ─────────────────────────────────────────────
# Imported by agents and the Exa.ai sentiment fetcher.

openai_circuit = CircuitBreaker("openai", failure_threshold=5, recovery_timeout=60.0)
exa_circuit = CircuitBreaker("exa_ai", failure_threshold=3, recovery_timeout=30.0)
alpaca_circuit = CircuitBreaker("alpaca", failure_threshold=3, recovery_timeout=30.0)

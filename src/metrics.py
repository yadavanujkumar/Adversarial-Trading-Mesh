"""
src/metrics.py — In-process metrics for Aura-Swarm-Quant.

Provides lightweight, thread-safe counters and latency histograms that can be
exposed via the /api/metrics endpoint without requiring an external Prometheus
push-gateway or a sidecar agent.

Usage
-----
    from src.metrics import metrics

    metrics.increment("reasoning_cycles_total")
    with metrics.timer("reasoning_cycle_ms"):
        await trading_graph.ainvoke(state)
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Generator


@dataclass
class _LatencyBucket:
    """Stores the last N duration samples and computes percentile summaries."""

    _window: deque = field(default_factory=lambda: deque(maxlen=1000))
    _lock: Lock = field(default_factory=Lock)

    def record(self, duration_ms: float) -> None:
        with self._lock:
            self._window.append(duration_ms)

    def summary(self) -> dict[str, float]:
        with self._lock:
            data = sorted(self._window)
        if not data:
            return {"count": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
        n = len(data)
        return {
            "count": n,
            "avg_ms": round(sum(data) / n, 2),
            "p50_ms": round(data[int(n * 0.50)], 2),
            "p95_ms": round(data[min(int(n * 0.95), n - 1)], 2),
            "p99_ms": round(data[min(int(n * 0.99), n - 1)], 2),
        }


class MetricsRegistry:
    """
    Simple thread-safe counter + latency registry suitable for a single-process
    asyncio application.

    For multi-process deployments, swap this for a Prometheus client or
    OpenTelemetry SDK — the same interface is preserved.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._latencies: dict[str, _LatencyBucket] = {}
        self._started_at: float = time.time()

    # ── Counters ──────────────────────────────────────────────────────────────

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a named counter by *amount*."""
        with self._lock:
            self._counters[name] += amount

    def get_counter(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, 0)

    # ── Latencies ─────────────────────────────────────────────────────────────

    def record_latency(self, name: str, duration_ms: float) -> None:
        """Record a single latency sample in milliseconds."""
        if name not in self._latencies:
            with self._lock:
                if name not in self._latencies:
                    self._latencies[name] = _LatencyBucket()
        self._latencies[name].record(duration_ms)

    @contextmanager
    def timer(self, name: str) -> Generator[None, None, None]:
        """Context manager that records wall-clock time as a latency sample."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.record_latency(name, elapsed_ms)

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time snapshot of all metrics for the /api/metrics endpoint."""
        with self._lock:
            counters = dict(self._counters)
        latencies = {k: v.summary() for k, v in self._latencies.items()}
        return {
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "counters": counters,
            "latencies": latencies,
        }


# Module-level singleton imported by all components.
metrics = MetricsRegistry()

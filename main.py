"""
main.py — Entry point for Aura-Swarm-Quant.

This module wires together:
1. The Binance WebSocket feed  (src/websocket_feed.py)
2. The LangGraph reasoning loop (src/graph.py)
3. The FastAPI dashboard        (dashboard/app.py)

All three components run concurrently inside a single asyncio event loop.

Usage
-----
    python main.py

Environment variables (see .env.example):
    OPENAI_API_KEY   — required for LLM agents
    EXA_API_KEY      — optional; enables live social sentiment
    WS_SYMBOL        — Binance symbol to stream (default: btcusdt)
    BULL_THRESHOLD   — consensus gate for the Bull  (default: 0.8)
    BEAR_THRESHOLD   — consensus gate for the Bear  (default: 0.3)
    STOP_LOSS_PCT    — fraction below entry to place stop (default: 0.02)
    LOG_JSON         — set to "true" for structured JSON log output
    ALPACA_ENABLED   — set to "true" to submit real orders via Alpaca

Architecture
------------
                WebSocket Feed
                     │
                     │ MarketSnapshot (every ~1 s)
                     ▼
              ┌──────────────┐
              │ LangGraph    │  ← async reasoning cycle
              │  War Room    │     (Bull ‖ Bear → Judge → [PostMortem])
              └──────┬───────┘
                     │ SwarmState
                     ▼
              ┌──────────────┐
              │   FastAPI    │  ← /api/health, /api/metrics, /ws/live, HTML dashboard
              │  Dashboard   │
              └──────────────┘
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Optional

import uvicorn
from dotenv import load_dotenv

load_dotenv()

# Config must be imported before any other src modules so that the singleton
# is constructed once with the fully loaded environment.
from src.config import settings
from src.graph import trading_graph
from src.metrics import metrics
from src.state import MarketSnapshot, SwarmState, empty_risk_metadata
from src.websocket_feed import binance_market_stream
from dashboard.app import app as fastapi_app, broadcast_state

# ── Logging setup ─────────────────────────────────────────────────────────────

def _configure_logging() -> None:
    """
    Configure the root logger.

    When LOG_JSON=true, each log record is emitted as a single-line JSON
    object — ideal for log aggregation systems (Datadog, Loki, CloudWatch).
    Otherwise, a human-readable format is used for local development.
    """
    if settings.log_json:
        import json as _json

        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                payload = {
                    "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                if record.exc_info:
                    payload["exc"] = self.formatException(record.exc_info)
                return _json.dumps(payload)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logging.root.handlers = [handler]
    else:
        logging.basicConfig(
            level=settings.log_level,
            format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )

    logging.root.setLevel(settings.log_level)


_configure_logging()
logger = logging.getLogger("main")

# ── Graceful shutdown ─────────────────────────────────────────────────────────

_shutdown_event = asyncio.Event()


def _handle_signal(sig: int) -> None:
    logger.info("Received signal %s — initiating graceful shutdown…", signal.Signals(sig).name)
    _shutdown_event.set()


# ── Core logic ────────────────────────────────────────────────────────────────

def _build_initial_state(snap: MarketSnapshot) -> SwarmState:
    """Construct a fresh SwarmState from a MarketSnapshot."""
    return SwarmState(
        market_data=snap,
        bull_signal=0.0,
        bear_signal=1.0,
        bull_rationale="",
        risk_metadata=empty_risk_metadata(),
        final_decision=None,
        post_mortem=None,
        stop_loss_hit=False,
        agent_memory={},
    )


async def _monitor_stop_loss(
    state: SwarmState,
    live_price: float,
    pending_state_ref: list[Optional[SwarmState]],
) -> None:
    """
    Coroutine that checks whether the current live price has breached the
    stop-loss level set by the Judge.  If so, it injects a stop_loss_hit
    flag and triggers a post-mortem cycle.

    In production this would be replaced by an exchange order event
    (filled / stopped) delivered via a separate WebSocket channel.
    """
    decision = state.get("final_decision")
    if not decision or decision.get("action") != "BUY":
        return

    stop_price = decision.get("stop_loss_price", 0.0)
    if stop_price and live_price <= stop_price:
        logger.warning(
            "🚨 Stop-loss triggered!  Entry=%.4f  Stop=%.4f  Current=%.4f",
            decision["entry_price"], stop_price, live_price,
        )
        metrics.increment("stop_losses_triggered_total")
        # Build a post-mortem state with current price
        pm_snap: MarketSnapshot = {**state["market_data"], "price": live_price}
        pm_state: SwarmState = {**state, "market_data": pm_snap, "stop_loss_hit": True}
        result: SwarmState = await trading_graph.ainvoke(pm_state)
        await broadcast_state(result)
        pending_state_ref[0] = result  # propagate updated memory forward


async def reasoning_loop(symbol: str = settings.ws_symbol) -> None:
    """
    Main async loop: receives MarketSnapshots from the Binance WebSocket
    and drives the LangGraph reasoning cycle for each update.

    The loop maintains the last known SwarmState so that agent_memory
    persists across cycles without an external database.

    Exits cleanly when *_shutdown_event* is set.
    """
    logger.info("Starting reasoning loop for symbol=%s", symbol.upper())
    last_state: Optional[SwarmState] = None

    async for snap in binance_market_stream(symbol=symbol):
        if _shutdown_event.is_set():
            logger.info("Shutdown requested — exiting reasoning loop.")
            break

        logger.info(
            "⚡ New snapshot: %s @ %.4f  RSI=%.1f",
            snap["symbol"], snap["price"], snap["rsi_14"],
        )

        # Check stop-loss on existing open position before starting a new cycle
        if last_state is not None:
            pending: list[Optional[SwarmState]] = [last_state]
            await _monitor_stop_loss(last_state, snap["price"], pending)
            last_state = pending[0]

        # Build initial state for this cycle, carrying forward agent_memory
        initial = _build_initial_state(snap)
        if last_state is not None:
            initial["agent_memory"] = last_state.get("agent_memory", {})

        try:
            with metrics.timer("reasoning_cycle_ms"):
                result: SwarmState = await trading_graph.ainvoke(initial)
            last_state = result
            metrics.increment("reasoning_cycles_total")
            await broadcast_state(result)

            decision = result.get("final_decision")
            if decision:
                logger.info(
                    "📊 Decision: %s  bull=%.3f  bear=%.3f  pos=%.3f  score=%.3f",
                    decision["action"],
                    result.get("bull_signal", 0.0),
                    result.get("bear_signal", 0.0),
                    decision.get("position_size_pct", 0.0),
                    decision.get("consensus_score", 0.0),
                )
        except Exception as exc:
            logger.error("Reasoning cycle failed: %s", exc, exc_info=True)
            metrics.increment("reasoning_cycle_errors_total")


async def _run_dashboard() -> None:
    """Start the FastAPI / uvicorn server in the background."""
    config = uvicorn.Config(
        app=fastapi_app,
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    await server.serve()


async def main() -> None:
    """Co-schedule the dashboard server and the trading reasoning loop."""
    # Validate and log configuration
    settings.warn_if_demo()

    logger.info("🚀 Aura-Swarm-Quant v2 starting…")
    logger.info("   Dashboard    → http://%s:%d", settings.dashboard_host, settings.dashboard_port)
    logger.info("   Symbol       → %s", settings.ws_symbol.upper())
    logger.info("   Bull / Bear  → %.2f / %.2f", settings.bull_threshold, settings.bear_threshold)
    logger.info("   Stop-loss    → %.1f%%", settings.stop_loss_pct * 100)
    logger.info("   Alpaca exec  → %s", "ENABLED" if settings.alpaca_enabled else "disabled")
    logger.info("   JSON logs    → %s", settings.log_json)

    # Register OS signal handlers for graceful shutdown (SIGTERM for containers)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal, sig)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler for all signals
            pass

    await asyncio.gather(
        _run_dashboard(),
        reasoning_loop(settings.ws_symbol),
    )

    logger.info("✅ Aura-Swarm-Quant shut down cleanly.")


if __name__ == "__main__":
    asyncio.run(main())

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
              │   FastAPI    │  ← /api/status, /ws/live, HTML dashboard
              │  Dashboard   │
              └──────────────┘
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from src.graph import trading_graph
from src.state import MarketSnapshot, SwarmState, empty_risk_metadata
from src.websocket_feed import binance_market_stream
from dashboard.app import app as fastapi_app, broadcast_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

WS_SYMBOL: str = os.getenv("WS_SYMBOL", "btcusdt")
DASHBOARD_HOST: str = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "8000"))


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
        # Build a post-mortem state with current price
        pm_snap: MarketSnapshot = {**state["market_data"], "price": live_price}
        pm_state: SwarmState = {**state, "market_data": pm_snap, "stop_loss_hit": True}
        result: SwarmState = await trading_graph.ainvoke(pm_state)
        await broadcast_state(result)
        pending_state_ref[0] = result  # propagate updated memory forward


async def reasoning_loop(symbol: str = WS_SYMBOL) -> None:
    """
    Main async loop: receives MarketSnapshots from the Binance WebSocket
    and drives the LangGraph reasoning cycle for each update.

    The loop maintains the last known SwarmState so that agent_memory
    persists across cycles without an external database.
    """
    logger.info("Starting reasoning loop for symbol=%s", symbol.upper())
    last_state: Optional[SwarmState] = None

    async for snap in binance_market_stream(symbol=symbol):
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
            result: SwarmState = await trading_graph.ainvoke(initial)
            last_state = result
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


async def _run_dashboard() -> None:
    """Start the FastAPI / uvicorn server in the background."""
    config = uvicorn.Config(
        app=fastapi_app,
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    await server.serve()


async def main() -> None:
    """Co-schedule the dashboard server and the trading reasoning loop."""
    logger.info("🚀 Aura-Swarm-Quant starting…")
    logger.info("   Dashboard → http://%s:%d", DASHBOARD_HOST, DASHBOARD_PORT)

    await asyncio.gather(
        _run_dashboard(),
        reasoning_loop(WS_SYMBOL),
    )


if __name__ == "__main__":
    asyncio.run(main())

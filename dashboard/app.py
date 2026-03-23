"""
dashboard/app.py — FastAPI monitoring dashboard for Aura-Swarm-Quant.

Endpoints
---------
GET  /                  HTML status page (human-readable)
GET  /api/status        Latest swarm state as JSON
GET  /api/decisions     Decision history (last 100)
GET  /api/memory/{agent} Agent lesson memory
POST /api/trigger       Manually trigger a reasoning cycle with custom data
WS   /ws/live           Real-time swarm state updates via WebSocket
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.memory import get_lessons
from src.state import MarketSnapshot, SwarmState, empty_risk_metadata

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Aura-Swarm-Quant Monitor",
    description="Real-time observability dashboard for the adversarial trading swarm.",
    version="1.0.0",
)

# ── In-memory state store ─────────────────────────────────────────────────────

_latest_state: dict[str, Any] = {}
_decision_history: deque[dict[str, Any]] = deque(maxlen=100)
_live_connections: list[WebSocket] = []


def update_dashboard_state(state: SwarmState) -> None:
    """Called by main.py after each reasoning cycle to update the dashboard."""
    global _latest_state
    _latest_state = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": state["market_data"]["symbol"],
        "price": state["market_data"]["price"],
        "rsi_14": state["market_data"]["rsi_14"],
        "bull_signal": state.get("bull_signal", 0.0),
        "bear_signal": state.get("bear_signal", 0.0),
        "bull_rationale": state.get("bull_rationale", ""),
        "bear_rationale": state.get("risk_metadata", {}).get("bear_rationale", ""),
        "volatility_regime": state.get("risk_metadata", {}).get("volatility_regime", "unknown"),
        "decision": state.get("final_decision"),
        "post_mortem": state.get("post_mortem"),
    }

    decision = state.get("final_decision")
    if decision:
        record = {**_latest_state, "decision": decision}
        _decision_history.appendleft(record)


async def broadcast_state(state: SwarmState) -> None:
    """Broadcast state to all live WebSocket subscribers."""
    update_dashboard_state(state)
    payload = json.dumps(_latest_state, default=str)
    dead: list[WebSocket] = []
    for ws in _live_connections:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _live_connections.remove(ws)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    decision = _latest_state.get("decision") or {}
    action = decision.get("action", "—") if decision else "—"
    bull = _latest_state.get("bull_signal", 0.0)
    bear = _latest_state.get("bear_signal", 0.0)
    price = _latest_state.get("price", 0.0)
    symbol = _latest_state.get("symbol", "—")
    ts = _latest_state.get("timestamp", "—")

    action_color = {"BUY": "#00e676", "REJECT": "#ff5252", "HOLD": "#ffab40"}.get(action, "#ccc")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="3">
  <title>Aura-Swarm-Quant</title>
  <style>
    body {{ background: #0d1117; color: #c9d1d9; font-family: monospace; padding: 2rem; }}
    h1 {{ color: #58a6ff; }}
    .card {{ background: #161b22; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }}
    .signal {{ font-size: 2rem; font-weight: bold; }}
    .action {{ color: {action_color}; font-size: 2.5rem; font-weight: bold; }}
    .row {{ display: flex; gap: 2rem; }}
    .metric {{ flex: 1; }}
    label {{ color: #8b949e; font-size: 0.8rem; }}
    .bar-bg {{ background:#21262d; border-radius:4px; height:8px; margin-top:4px; }}
    .bar-fill {{ height:8px; border-radius:4px; }}
  </style>
</head>
<body>
  <h1>⚡ Aura-Swarm-Quant — War Room</h1>
  <div class="card">
    <div class="row">
      <div class="metric"><label>Symbol</label><div class="signal">{symbol}</div></div>
      <div class="metric"><label>Price</label><div class="signal">${price:,.4f}</div></div>
      <div class="metric"><label>Last Decision</label><div class="action">{action}</div></div>
    </div>
    <small>Updated: {ts}</small>
  </div>
  <div class="card">
    <div class="row">
      <div class="metric">
        <label>Bull Signal</label>
        <div style="font-size:1.5rem; color:#00e676">{bull:.3f}</div>
        <div class="bar-bg"><div class="bar-fill" style="width:{bull*100:.0f}%;background:#00e676"></div></div>
      </div>
      <div class="metric">
        <label>Bear Signal</label>
        <div style="font-size:1.5rem; color:#ff5252">{bear:.3f}</div>
        <div class="bar-bg"><div class="bar-fill" style="width:{bear*100:.0f}%;background:#ff5252"></div></div>
      </div>
    </div>
  </div>
  <div class="card">
    <label>Bull Rationale</label>
    <p>{_latest_state.get('bull_rationale', '—')}</p>
    <label>Bear Rationale</label>
    <p>{_latest_state.get('bear_rationale', '—')}</p>
  </div>
  <p style="color:#8b949e">
    <a href="/api/status" style="color:#58a6ff">/api/status</a> &nbsp;·&nbsp;
    <a href="/api/decisions" style="color:#58a6ff">/api/decisions</a> &nbsp;·&nbsp;
    <a href="/docs" style="color:#58a6ff">/docs</a>
  </p>
</body>
</html>"""


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    """Latest swarm reasoning cycle snapshot."""
    return _latest_state or {"status": "waiting for first cycle"}


@app.get("/api/decisions")
async def api_decisions() -> list[dict[str, Any]]:
    """Last 100 trade decisions (most recent first)."""
    return list(_decision_history)


@app.get("/api/memory/{agent}")
async def api_memory(agent: str) -> list[dict[str, Any]]:
    """Retrieve stored lessons for a named agent (bull / bear / judge)."""
    if agent not in ("bull", "bear", "judge"):
        raise HTTPException(status_code=400, detail="agent must be 'bull', 'bear', or 'judge'")
    return get_lessons(agent)


class ManualTriggerPayload(BaseModel):
    symbol: str = "BTCUSDT"
    price: float
    volume_24h: float = 0.0
    bid_ask_spread: float = 0.0
    order_book_imbalance: float = 0.0
    rsi_14: float = 50.0
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    stop_loss_hit: bool = False


@app.post("/api/trigger")
async def api_trigger(payload: ManualTriggerPayload) -> dict[str, Any]:
    """
    Manually inject market data and trigger a full reasoning cycle.
    Useful for backtesting or demo purposes without a live WebSocket feed.
    """
    from src.graph import trading_graph

    snap: MarketSnapshot = {
        "symbol": payload.symbol,
        "price": payload.price,
        "volume_24h": payload.volume_24h,
        "bid_ask_spread": payload.bid_ask_spread,
        "order_book_imbalance": payload.order_book_imbalance,
        "rsi_14": payload.rsi_14,
        "macd_line": payload.macd_line,
        "macd_signal": payload.macd_signal,
        "macd_hist": payload.macd_hist,
        "sentiment_score": 0.0,
        "sentiment_sources": [],
    }

    initial_state: SwarmState = {
        "market_data": snap,
        "bull_signal": 0.0,
        "bear_signal": 1.0,
        "bull_rationale": "",
        "risk_metadata": empty_risk_metadata(),
        "final_decision": None,
        "post_mortem": None,
        "stop_loss_hit": payload.stop_loss_hit,
        "agent_memory": {},
    }

    result: SwarmState = await trading_graph.ainvoke(initial_state)
    await broadcast_state(result)
    return {
        "bull_signal": result.get("bull_signal"),
        "bear_signal": result.get("bear_signal"),
        "decision": result.get("final_decision"),
        "post_mortem": result.get("post_mortem"),
    }


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    """Real-time push of each reasoning cycle result."""
    await websocket.accept()
    _live_connections.append(websocket)
    try:
        while True:
            await asyncio.sleep(60)  # keep connection alive; data is pushed from broadcast_state
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _live_connections:
            _live_connections.remove(websocket)

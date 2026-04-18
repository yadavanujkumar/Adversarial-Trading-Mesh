"""
dashboard/app.py — FastAPI monitoring dashboard for Aura-Swarm-Quant.

Endpoints
---------
GET  /                      HTML war-room dashboard (human-readable, auto-refresh)
GET  /api/health            Liveness / readiness probe (Kubernetes-compatible)
GET  /api/metrics           In-process counter and latency snapshot
GET  /api/circuit-breakers  Circuit breaker states for all external dependencies
GET  /api/status            Latest swarm state as JSON
GET  /api/decisions         Decision history (last 100) with optional ?limit=N
GET  /api/memory/{agent}    Agent lesson memory
POST /api/trigger           Manually trigger a reasoning cycle with custom data
WS   /ws/live               Real-time swarm state updates via WebSocket

Enterprise features
-------------------
* CORS configured for API consumers (origins configurable via settings).
* /api/health returns HTTP 503 until the first reasoning cycle completes,
  enabling Kubernetes readiness probes to gate traffic.
* Simple in-process rate limiter on /api/trigger prevents API abuse.
* Optional X-API-Token header check on /api/trigger (set API_AUTH_TOKEN env var).
* WebSocket connections are tracked with a hard upper bound to prevent resource
  exhaustion on a single node.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.circuit_breaker import openai_circuit, exa_circuit, alpaca_circuit
from src.config import settings
from src.memory import get_lessons
from src.metrics import metrics
from src.state import MarketSnapshot, SwarmState, empty_risk_metadata

logger = logging.getLogger(__name__)

# ── FastAPI application ───────────────────────────────────────────────────────

app = FastAPI(
    title="Aura-Swarm-Quant Monitor",
    description="Real-time observability dashboard for the adversarial trading swarm.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production by setting CORS_ORIGINS env var
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── In-memory state store ─────────────────────────────────────────────────────

_latest_state: dict[str, Any] = {}
_decision_history: deque[dict[str, Any]] = deque(maxlen=100)
_live_connections: list[WebSocket] = []
_first_cycle_complete: bool = False

MAX_WS_CONNECTIONS = 50  # hard cap on concurrent WebSocket subscribers

# ── Simple in-process rate limiter for /api/trigger ──────────────────────────

_trigger_requests: deque[float] = deque(maxlen=settings.api_rate_limit_rpm)


def _check_rate_limit(client_ip: str) -> None:
    """
    Sliding-window rate limiter: allows settings.api_rate_limit_rpm requests
    per 60-second window across *all* callers (per-IP limiting would need a
    more sophisticated store — use a reverse proxy / API gateway for that).
    """
    now = time.monotonic()
    # Purge requests older than 60 s
    while _trigger_requests and now - _trigger_requests[0] > 60.0:
        _trigger_requests.popleft()
    if len(_trigger_requests) >= settings.api_rate_limit_rpm:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: max {settings.api_rate_limit_rpm} requests/minute. "
                "Retry after the current window expires."
            ),
        )
    _trigger_requests.append(now)


def _check_auth(x_api_token: str | None) -> None:
    """Validate the optional API token header if configured."""
    if settings.api_auth_token and x_api_token != settings.api_auth_token:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Token header.")


# ── State update helpers ──────────────────────────────────────────────────────

def update_dashboard_state(state: SwarmState) -> None:
    """Called by main.py after each reasoning cycle to update the dashboard."""
    global _latest_state, _first_cycle_complete
    _latest_state = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": state["market_data"]["symbol"],
        "price": state["market_data"]["price"],
        "rsi_14": state["market_data"]["rsi_14"],
        "macd_hist": state["market_data"]["macd_hist"],
        "order_book_imbalance": state["market_data"]["order_book_imbalance"],
        "bull_signal": state.get("bull_signal", 0.0),
        "bear_signal": state.get("bear_signal", 0.0),
        "bull_rationale": state.get("bull_rationale", ""),
        "bear_rationale": state.get("risk_metadata", {}).get("bear_rationale", ""),
        "volatility_regime": state.get("risk_metadata", {}).get("volatility_regime", "unknown"),
        "liquidity_trap_score": state.get("risk_metadata", {}).get("liquidity_trap_score", 0.0),
        "black_swan_probability": state.get("risk_metadata", {}).get("black_swan_probability", 0.0),
        "decision": state.get("final_decision"),
        "post_mortem": state.get("post_mortem"),
    }
    _first_cycle_complete = True
    metrics.increment("dashboard_state_updates_total")

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
    metrics.increment("ws_broadcasts_total")


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
    rsi = _latest_state.get("rsi_14", 0.0)
    obi = _latest_state.get("order_book_imbalance", 0.0)
    regime = _latest_state.get("volatility_regime", "unknown")
    lts = _latest_state.get("liquidity_trap_score", 0.0)
    bsp = _latest_state.get("black_swan_probability", 0.0)
    total_cycles = metrics.get_counter("market_snapshots_ingested_total")
    total_buys = metrics.get_counter("decisions_buy_total")
    total_rejects = metrics.get_counter("decisions_reject_total")

    action_color = {"BUY": "#00e676", "REJECT": "#ff5252", "HOLD": "#ffab40"}.get(action, "#ccc")
    regime_color = {"low": "#00e676", "elevated": "#ffab40", "crisis": "#ff5252"}.get(regime, "#ccc")

    cb_states = {
        "OpenAI": openai_circuit.state.value,
        "Exa.ai": exa_circuit.state.value,
        "Alpaca": alpaca_circuit.state.value,
    }
    cb_html = " &nbsp;·&nbsp; ".join(
        f'<span style="color:{"#00e676" if s == "closed" else "#ff5252"}">{n}: {s}</span>'
        for n, s in cb_states.items()
    )

    waiting = not _first_cycle_complete

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="3">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aura-Swarm-Quant ⚡</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ background: #0d1117; color: #c9d1d9; font-family: 'Courier New', monospace;
            padding: 1.5rem; margin: 0; }}
    h1 {{ color: #58a6ff; margin: 0 0 1rem; font-size: 1.4rem; letter-spacing: 0.05em; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
             padding: 1.25rem; }}
    .card-title {{ color: #8b949e; font-size: 0.7rem; text-transform: uppercase;
                   letter-spacing: 0.1em; margin-bottom: 0.5rem; }}
    .big {{ font-size: 2rem; font-weight: bold; line-height: 1; }}
    .action {{ color: {action_color}; font-size: 2.5rem; font-weight: bold; }}
    .bar-bg {{ background: #21262d; border-radius: 4px; height: 6px; margin-top: 6px; }}
    .bar {{ height: 6px; border-radius: 4px; transition: width 0.4s ease; }}
    .row {{ display: flex; gap: 1.5rem; flex-wrap: wrap; }}
    .metric {{ flex: 1; min-width: 100px; }}
    label {{ color: #8b949e; font-size: 0.75rem; display: block; margin-bottom: 2px; }}
    .tag {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;
            background: #21262d; }}
    .warn {{ color: #ffab40; }}
    .links {{ margin-top: 1rem; color: #8b949e; font-size: 0.8rem; }}
    .links a {{ color: #58a6ff; text-decoration: none; margin-right: 1rem; }}
    .links a:hover {{ text-decoration: underline; }}
    .waiting {{ color: #8b949e; font-style: italic; }}
    .cb-bar {{ margin-top: 0.5rem; font-size: 0.8rem; }}
    .stat {{ text-align: center; }}
    .stat .num {{ font-size: 1.6rem; font-weight: bold; color: #58a6ff; }}
    .stat label {{ margin-top: 2px; }}
  </style>
</head>
<body>
  <h1>⚡ Aura-Swarm-Quant — War Room
    <span style="font-size:0.75rem; color:#8b949e; float:right">v2.0 · {ts[:19].replace("T"," ")} UTC</span>
  </h1>

  {"<p class='waiting'>⏳ Awaiting first reasoning cycle…</p>" if waiting else ""}

  <div class="grid">

    <!-- Price card -->
    <div class="card">
      <div class="card-title">Market</div>
      <div class="row">
        <div class="metric">
          <label>Symbol</label>
          <div class="big" style="color:#58a6ff">{symbol}</div>
        </div>
        <div class="metric">
          <label>Price</label>
          <div class="big">${price:,.4f}</div>
        </div>
        <div class="metric">
          <label>RSI-14</label>
          <div class="big" style="color:{"#ffab40" if rsi > 70 or rsi < 30 else "#c9d1d9"}">{rsi:.1f}</div>
        </div>
      </div>
      <div style="margin-top:.75rem">
        <label>Order Book Imbalance</label>
        <div class="bar-bg">
          <div class="bar" style="width:{min(abs(obi)*100, 100):.0f}%;
               background:{("#00e676" if obi > 0 else "#ff5252")}"></div>
        </div>
        <small style="color:#8b949e">{obi:+.3f}  ·  Regime: <span style="color:{regime_color}">{regime}</span></small>
      </div>
    </div>

    <!-- Decision card -->
    <div class="card">
      <div class="card-title">Last Decision</div>
      <div class="action">{action}</div>
      <div style="margin-top:.5rem; font-size:.9rem; color:#8b949e">
        {f"Entry: ${decision.get('entry_price', 0):.4f}  ·  Stop: ${decision.get('stop_loss_price', 0):.4f}  ·  Size: {decision.get('position_size_pct', 0)*100:.1f}%" if decision and action == "BUY" else ""}
      </div>
      <div style="margin-top:.5rem; font-size:.8rem; color:#8b949e">
        {decision.get("judge_rationale", "") if decision else ""}
      </div>
    </div>

    <!-- Signals card -->
    <div class="card">
      <div class="card-title">Agent Signals</div>
      <div class="metric" style="margin-bottom:.75rem">
        <label>🐂 Bull Signal (α)</label>
        <div style="color:#00e676; font-size:1.4rem; font-weight:bold">{bull:.3f}</div>
        <div class="bar-bg"><div class="bar" style="width:{bull*100:.0f}%; background:#00e676"></div></div>
      </div>
      <div class="metric">
        <label>🐻 Bear Signal (ω)</label>
        <div style="color:#ff5252; font-size:1.4rem; font-weight:bold">{bear:.3f}</div>
        <div class="bar-bg"><div class="bar" style="width:{bear*100:.0f}%; background:#ff5252"></div></div>
      </div>
      <div style="margin-top:.75rem; font-size:.75rem; color:#8b949e">
        Liquidity Trap: {lts:.2f} &nbsp;·&nbsp; Black Swan: {bsp:.2f}
      </div>
    </div>

    <!-- Stats card -->
    <div class="card">
      <div class="card-title">Session Statistics</div>
      <div class="row">
        <div class="metric stat">
          <div class="num">{total_cycles}</div>
          <label>Cycles</label>
        </div>
        <div class="metric stat">
          <div class="num" style="color:#00e676">{total_buys}</div>
          <label>BUYs</label>
        </div>
        <div class="metric stat">
          <div class="num" style="color:#ff5252">{total_rejects}</div>
          <label>REJECTs</label>
        </div>
      </div>
      <div class="cb-bar">
        Circuit Breakers: {cb_html}
      </div>
    </div>

    <!-- Rationale card -->
    <div class="card" style="grid-column: 1 / -1">
      <div class="card-title">Agent Debate</div>
      <div class="row">
        <div class="metric">
          <label>🐂 Bull Rationale</label>
          <p style="margin:.25rem 0; font-size:.85rem">{_latest_state.get("bull_rationale", "—")}</p>
        </div>
        <div class="metric">
          <label>🐻 Bear Rationale</label>
          <p style="margin:.25rem 0; font-size:.85rem">{_latest_state.get("bear_rationale", "—")}</p>
        </div>
      </div>
    </div>

  </div>

  <div class="links">
    <a href="/api/health">/health</a>
    <a href="/api/status">/status</a>
    <a href="/api/decisions">/decisions</a>
    <a href="/api/metrics">/metrics</a>
    <a href="/api/circuit-breakers">/circuit-breakers</a>
    <a href="/docs">/docs</a>
  </div>
</body>
</html>"""


@app.get("/api/health")
async def api_health() -> JSONResponse:
    """
    Liveness + readiness probe (Kubernetes-compatible).

    Returns HTTP 200 once the first reasoning cycle has completed.
    Returns HTTP 503 while the system is still initialising, so a
    Kubernetes readiness probe won't route traffic until the swarm is warm.
    """
    cb_ok = not any(cb.is_open for cb in [openai_circuit, exa_circuit, alpaca_circuit])
    status = {
        "status": "ok" if _first_cycle_complete else "initialising",
        "ready": _first_cycle_complete,
        "live": True,
        "circuit_breakers_healthy": cb_ok,
        "ws_subscribers": len(_live_connections),
        "decisions_in_history": len(_decision_history),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    code = 200 if _first_cycle_complete else 503
    return JSONResponse(content=status, status_code=code)


@app.get("/api/metrics")
async def api_metrics_endpoint() -> dict[str, Any]:
    """In-process counter and latency snapshot (Prometheus-compatible format coming soon)."""
    return metrics.snapshot()


@app.get("/api/circuit-breakers")
async def api_circuit_breakers() -> list[dict[str, Any]]:
    """Current state of all registered circuit breakers."""
    return [
        openai_circuit.status(),
        exa_circuit.status(),
        alpaca_circuit.status(),
    ]


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    """Latest swarm reasoning cycle snapshot."""
    return _latest_state or {"status": "waiting for first cycle"}


@app.get("/api/decisions")
async def api_decisions(limit: int = Query(default=100, ge=1, le=100)) -> list[dict[str, Any]]:
    """Last *limit* trade decisions (most recent first)."""
    return list(_decision_history)[:limit]


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
async def api_trigger(
    payload: ManualTriggerPayload,
    request: Request,
    x_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    """
    Manually inject market data and trigger a full reasoning cycle.
    Useful for backtesting or demo purposes without a live WebSocket feed.

    Rate-limited to settings.api_rate_limit_rpm requests/minute.
    Optional authentication via X-API-Token header.
    """
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)
    _check_auth(x_api_token)

    from src.graph import trading_graph  # noqa: PLC0415

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

    metrics.increment("manual_triggers_total")
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
    """
    Real-time push of each reasoning cycle result.

    Connections are capped at MAX_WS_CONNECTIONS to prevent resource exhaustion.
    The server sends a heartbeat ping every 30 s to detect stale connections.
    """
    if len(_live_connections) >= MAX_WS_CONNECTIONS:
        await websocket.close(code=1013, reason="Too many subscribers")
        return

    await websocket.accept()
    _live_connections.append(websocket)
    metrics.increment("ws_connections_total")
    try:
        while True:
            # Heartbeat: keep-alive ping so stale connections are detected quickly
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({"type": "ping",
                                                  "ts": datetime.now(timezone.utc).isoformat()}))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in _live_connections:
            _live_connections.remove(websocket)

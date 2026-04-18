# ⚡ Aura-Swarm-Quant — Adversarial Trading Mesh

A production-ready, real-time, multi-agent autonomous trading system built on **LangGraph** and **FastAPI**. Three specialist AI agents engage in a structured adversarial debate before any trade is executed — only mathematical consensus unlocks capital.

---

## Architecture: The War Room

```
                 WebSocket Feed (Binance)
                        │
                        │ MarketSnapshot (1 s throttle)
                        ▼
              ┌──────────────────────┐
              │   ingest_market_data  │  validate + normalise
              └───┬──────────────────┘
                  │ fan-out (parallel)
        ┌─────────┴─────────┐
        ▼                   ▼
 ┌─────────────┐   ┌─────────────────┐
 │  Agent A    │   │   Agent B        │
 │  The Bull   │   │   The Bear       │
 │  (Alpha)    │   │   (Omega)        │
 └──────┬──────┘   └────────┬─────────┘
        │   fan-in           │
        └────────┬───────────┘
                 ▼
        ┌────────────────┐
        │   Agent C       │
        │   The Judge     │   consensus gate: Bull > 0.8 AND Bear < 0.3
        │   (Sigma)       │
        └────────┬────────┘
                 │ conditional
        ┌────────┴──────────┐
        │                   │
   stop-loss hit?        normal exit
        │
        ▼
 ┌──────────────────┐
 │  Post-Mortem      │  analyse failure → update agent memory
 └──────────────────┘
```

---

## Adversarial Consensus Logic

The core innovation is the **Adversarial Consensus Gate**.  The system deliberately creates cognitive conflict between an optimistic momentum agent and a pessimistic risk agent.  A trade only executes when both thresholds are simultaneously satisfied:

```
EXECUTE  iff  bull_signal > 0.8  AND  bear_signal < 0.3
HOLD     iff  ambiguous zone (0.6 < bull < 0.8 or 0.3 < bear < 0.5)
REJECT   iff  bear_signal ≥ 0.3  OR  bull_signal ≤ 0.8
```

| Agent | Role | Signal | Personality |
|-------|------|--------|-------------|
| **Alpha (Bull)** | Entry strength analyst | `bull_signal ∈ [0,1]` | Forensically optimistic; backs momentum with RSI, MACD, and Exa.ai social sentiment |
| **Omega (Bear)** | Risk auditor | `bear_signal ∈ [0,1]` | Perpetually skeptical; hunts OBI manipulation, liquidity traps, and Black Swan events |
| **Sigma (Judge)** | Executive arbiter | `TradeDecision` | Emotionally detached; enforces the consensus gate with no exceptions |

The adversarial structure prevents both "groupthink" (all agents agree, no one challenges) and "paralysis" (agents always disagree, no trades ever execute).  The mathematical gate ensures only the highest-conviction, lowest-risk setups pass.

### Why This Works Better Than a Single Agent

A single LLM asked to make trading decisions tends to either hallucinate bullish narratives or become overly cautious.  By separating the *optimist* and *pessimist* roles and forcing them to argue against each other's position in every cycle, the system surfaces genuine tension in the data — which is exactly what a human trading desk does.

---

## File Structure

```
Adversarial-Trading-Mesh/
├── main.py                     # Entry point: asyncio event loop + graceful shutdown
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template (fully documented)
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Pydantic Settings — centralised, validated configuration
│   ├── metrics.py              # In-process counters + latency histograms
│   ├── circuit_breaker.py      # Async circuit breaker (OpenAI / Exa.ai / Alpaca)
│   ├── state.py                # TypedDict: SwarmState, MarketSnapshot, RiskMetadata …
│   ├── memory.py               # Async-safe JSON-backed persistent agent lesson store
│   ├── graph.py                # LangGraph workflow (parallel Bull/Bear + Post-Mortem)
│   ├── websocket_feed.py       # Binance WebSocket → MarketSnapshot (RSI/MACD incremental)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── bull_agent.py       # Agent A — momentum + Exa.ai sentiment
│   │   ├── bear_agent.py       # Agent B — OBI + liquidity trap + Black Swan
│   │   └── judge_agent.py      # Agent C — Kelly sizing + consensus gate + Alpaca execution
│   └── execution/
│       ├── __init__.py
│       └── alpaca_broker.py    # Alpaca paper/live order execution with safety guardrails
│
└── dashboard/
    ├── __init__.py
    └── app.py                  # FastAPI: HTML dashboard + /api/* + /ws/live
```

---

## Enterprise-Grade Features

### 🔒 Centralised, Validated Configuration (`src/config.py`)
All environment variables are parsed and type-validated at startup via **Pydantic Settings**.  The application fails fast with a descriptive error rather than silently using wrong defaults.  Attempting to set `bull_threshold ≤ bear_threshold` raises a `ValidationError` immediately.

### ⚡ Async Circuit Breakers (`src/circuit_breaker.py`)
Three-state circuit breakers (CLOSED → OPEN → HALF-OPEN) guard every external API call:
- **`openai_circuit`** — wraps all LLM calls across Bull, Bear, Judge, and Post-Mortem nodes.
- **`exa_circuit`** — wraps Exa.ai neural search in the Bull agent.
- **`alpaca_circuit`** — wraps Alpaca order submission.

When a dependency fails 5 consecutive times the circuit opens and all subsequent calls are rejected immediately (fast-fail) for 60 s before a single recovery probe is attempted.  This prevents cascade failures and LLM API quota exhaustion under partial outages.

### 📊 In-Process Metrics (`src/metrics.py`)
Thread-safe counters and p50/p95/p99 latency histograms are maintained in-process with zero external dependencies.  Accessible at `/api/metrics` for scraping by Prometheus exporters or log aggregators.

### 🏥 Health / Readiness Probes (`/api/health`)
Returns HTTP **503** during initialisation and HTTP **200** once the first reasoning cycle completes — compatible with Kubernetes `readinessProbe` configuration so no traffic is routed until the swarm is warm.

### 🛡️ Rate Limiting & Auth (`/api/trigger`)
A sliding-window rate limiter caps `/api/trigger` at `API_RATE_LIMIT_RPM` requests per minute (default 30).  Set `API_AUTH_TOKEN` to require an `X-API-Token` header on all trigger calls.

### 🔄 Graceful Shutdown (`main.py`)
Registers `SIGINT` / `SIGTERM` handlers so the reasoning loop and dashboard drain cleanly when containerised (Docker, Kubernetes).

### 📝 Structured JSON Logging
Set `LOG_JSON=true` to emit machine-readable JSON log lines compatible with Datadog, Loki, and CloudWatch — no log-parsing configuration needed on the aggregation side.

### 💼 Alpaca Order Execution (`src/execution/alpaca_broker.py`)
When `ALPACA_ENABLED=true`, BUY decisions are automatically translated into real (or paper) market orders via the Alpaca API:
- Position size computed from Kelly fraction × portfolio equity.
- Hard notional cap (`MAX_NOTIONAL_USD = $10,000`) prevents runaway sizing bugs.
- Stop-loss bracket order submitted alongside the buy.
- Protected by `alpaca_circuit` — broker outages don't stall the reasoning pipeline.

### 🧵 Async-Safe Persistent Memory (`src/memory.py`)
The in-process `asyncio.Lock` prevents concurrent post-mortem lessons from corrupting the JSON store.  Atomic writes via `os.replace()` on a temp file ensure the file is never half-written.  An in-memory cache avoids repeated disk reads during every reasoning cycle.

---

## Agent System Prompts

### Agent Alpha — The Bull
> *"You are relentlessly optimistic but intellectually honest. You believe every market dip is a buying opportunity and every consolidation is a coiled spring. You back your conviction with hard data. You are NOT reckless — your optimism is forensic."*

**Signals it uses:** RSI-14, MACD histogram, Exa.ai social sentiment (neural search over recent headlines), pre-computed technical momentum score.

### Agent Omega — The Bear
> *"You are forensically skeptical. Every data point is suspect until independently verified. You have personally watched traders lose everything because they ignored order-book manipulation, thin liquidity, and macro tail risks. Your DEFAULT stance is: prove to me this isn't a trap."*

**Signals it uses:** Order Book Imbalance (OBI) — a measure of bid vs. ask pressure; liquidity trap score; Black Swan probability derived from RSI extremes, MACD magnitude, and bid-ask spread stress.

### Agent Sigma — The Judge
> *"You are the most emotionally detached entity in this room. You do not root for the Bull. You do not fear with the Bear. Adjectives are for analysts — you deal in decisions."*

**Logic:** Hard-rules gate (cannot be overridden by LLM output) + Kelly half-fraction position sizing (capped at 25 % of portfolio) + optional Alpaca order submission.

---

## Self-Correction: The Post-Mortem Loop

When a stop-loss is triggered, the **Post-Mortem node** is automatically activated:

1. Calculates the percentage loss and identifies which risk factor the Bear correctly flagged (or missed).
2. Categorises the failure: `momentum_fade`, `liquidity_trap`, `black_swan`, or `other`.
3. Extracts actionable lessons and persists them to `agent_memory.json`.
4. On the next cycle, **all three agents receive their respective lessons injected into their system prompt**, making the swarm genuinely adaptive.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yadavanujkumar/Adversarial-Trading-Mesh.git
cd Adversarial-Trading-Mesh
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY

# 3. Run
python main.py
# Dashboard → http://localhost:8000
```

### Manual trigger (no WebSocket needed)
```bash
curl -X POST http://localhost:8000/api/trigger \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "price": 67500.0,
    "rsi_14": 62.5,
    "macd_hist": 120.0,
    "order_book_imbalance": 0.15,
    "bid_ask_spread": 0.5
  }'
```

### Docker / Kubernetes
```bash
docker build -t aura-swarm-quant .
docker run -e OPENAI_API_KEY=sk-... -p 8000:8000 aura-swarm-quant

# Kubernetes readiness probe (waits for first cycle before routing traffic)
readinessProbe:
  httpGet:
    path: /api/health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
```

---

## Real-Time Integration

The `src/websocket_feed.py` module connects to Binance's combined stream:

```
wss://stream.binance.com:9443/stream?streams=btcusdt@trade/btcusdt@bookTicker
```

Technical indicators are computed **incrementally** on a rolling 50-price window — no historical data download required.  The feed throttles to one snapshot per second to avoid overwhelming the LLM API quota.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** Powers all three LLM agents. |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model identifier. |
| `LLM_MAX_RETRIES` | `3` | Retry attempts on transient LLM errors. |
| `EXA_API_KEY` | — | Optional. Enables live social sentiment in the Bull agent. |
| `EXA_NUM_RESULTS` | `8` | Number of Exa.ai search results to fetch. |
| `WS_SYMBOL` | `btcusdt` | Binance stream symbol. |
| `WS_THROTTLE_SECS` | `1.0` | Minimum seconds between yielded snapshots. |
| `BULL_THRESHOLD` | `0.8` | Minimum bull confidence for a BUY decision. |
| `BEAR_THRESHOLD` | `0.3` | Maximum bear confidence allowed for a BUY decision. |
| `STOP_LOSS_PCT` | `0.02` | Stop-loss distance below entry price (e.g. `0.02` = 2 %). |
| `ALPACA_ENABLED` | `false` | Set `true` to submit real orders through Alpaca. |
| `ALPACA_API_KEY` | — | Alpaca API key (required when enabled). |
| `ALPACA_SECRET_KEY` | — | Alpaca secret key (required when enabled). |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` | Alpaca endpoint (change to live URL for real capital). |
| `DASHBOARD_PORT` | `8000` | FastAPI dashboard port. |
| `API_AUTH_TOKEN` | — | If set, `/api/trigger` requires `X-API-Token` header. |
| `API_RATE_LIMIT_RPM` | `30` | Max `/api/trigger` requests per minute. |
| `CIRCUIT_BREAKER_THRESHOLD` | `5` | Consecutive failures before a circuit opens. |
| `CIRCUIT_BREAKER_TIMEOUT_SECS` | `60.0` | Seconds a circuit stays OPEN before recovery probe. |
| `MEMORY_PATH` | `agent_memory.json` | Path for persistent lesson storage. |
| `MEMORY_MAX_LESSONS` | `10` | Maximum lessons retained per agent. |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `LOG_JSON` | `false` | Set `true` for structured JSON log output. |

---

## Dashboard Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Auto-refreshing HTML war room with signal bars, circuit breaker status, and session stats |
| `/api/health` | GET | Liveness + readiness probe (HTTP 503 until first cycle, 200 after) |
| `/api/metrics` | GET | In-process counters and latency percentiles |
| `/api/circuit-breakers` | GET | Current state of all circuit breakers |
| `/api/status` | GET | Latest reasoning cycle as JSON |
| `/api/decisions` | GET | Last 100 trade decisions (`?limit=N` to paginate) |
| `/api/memory/{agent}` | GET | Stored lessons for `bull`, `bear`, or `judge` |
| `/api/trigger` | POST | Manually inject market data + trigger full cycle (rate-limited) |
| `/ws/live` | WS | Real-time push of each cycle result (heartbeat every 30 s) |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc UI |

---

## Technical Stack

| Component | Library |
|-----------|---------|
| Agent orchestration | `langgraph` |
| LLM calls | `langchain-openai` (GPT-4o-mini) |
| Real-time feed | `websockets` (Binance) |
| Social sentiment | `httpx` + Exa.ai Neural Search |
| Technical indicators | incremental RSI/MACD (numpy) |
| Monitoring API | `fastapi` + `uvicorn` |
| Position sizing | Half-Kelly criterion |
| Configuration | `pydantic-settings` |
| Order execution | `alpaca-py` |
| Resilience | Async circuit breakers (built-in) |

---

## Licence

MIT — see [LICENSE](./LICENSE).

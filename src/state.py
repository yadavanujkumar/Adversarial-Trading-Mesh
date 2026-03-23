"""
Aura-Swarm-Quant – shared type definitions.

The SwarmState dict travels through every node in the LangGraph graph.
All fields are intentionally typed so that mypy / pyright can catch
state mutation errors at development time.
"""
from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class MarketSnapshot(TypedDict):
    """Raw price / order-book data delivered by the WebSocket feed."""

    symbol: str
    price: float
    volume_24h: float
    bid_ask_spread: float          # proxy for liquidity depth
    order_book_imbalance: float    # (bid_volume – ask_volume) / total_volume ∈ [–1, 1]
    rsi_14: float                  # 0–100
    macd_line: float
    macd_signal: float
    macd_hist: float
    sentiment_score: float         # –1 (fear) … +1 (greed); populated by Exa.ai
    sentiment_sources: list[str]   # headlines / links used for sentiment


class RiskMetadata(TypedDict):
    """Enriched risk payload written by the Bear agent."""

    obi_flag: bool                 # True when order-book imbalance indicates a trap
    liquidity_trap_score: float    # 0–1; higher = more dangerous
    black_swan_probability: float  # 0–1; drawn from VIX-analogue and news scan
    volatility_regime: str         # "low" | "elevated" | "crisis"
    bear_rationale: str            # free-text explanation from Agent B


class TradeDecision(TypedDict):
    """Populated by the Judge agent after adversarial consensus."""

    action: str                    # "BUY" | "HOLD" | "REJECT"
    entry_price: float
    stop_loss_price: float
    position_size_pct: float       # fraction of portfolio (0–1)
    consensus_score: float         # composite Bull – Bear signal
    judge_rationale: str


class PostMortemReport(TypedDict):
    """Produced by the Post-Mortem node when a stop-loss is triggered."""

    triggered_at_price: float
    entry_price: float
    loss_pct: float
    failure_category: str          # "momentum_fade" | "liquidity_trap" | "black_swan" | "other"
    lessons_learned: str           # LLM-generated analysis
    updated_risk_weight: float     # suggested adjustment to future bear threshold


class SwarmState(TypedDict):
    """
    The single shared state object that flows through the entire LangGraph.

    Design note
    -----------
    LangGraph merges node return dicts into this state using a reduce
    strategy (last-write-wins per key).  Nodes should only return the
    keys they actually modify so that parallel nodes don't silently
    overwrite each other's work.
    """

    # ── Input ────────────────────────────────────────────────────────────────
    market_data: MarketSnapshot

    # ── Agent outputs (0–1 confidence scores) ────────────────────────────────
    bull_signal: float          # Agent A output
    bear_signal: float          # Agent B output
    bull_rationale: str         # narrative from Agent A
    risk_metadata: RiskMetadata # enriched risk info from Agent B

    # ── Execution output ─────────────────────────────────────────────────────
    final_decision: Optional[TradeDecision]

    # ── Self-correction ──────────────────────────────────────────────────────
    post_mortem: Optional[PostMortemReport]
    stop_loss_hit: bool         # signal flag set by the order-monitor coroutine

    # ── Persistent cross-cycle memory (passed through every cycle) ───────────
    agent_memory: dict[str, Any]  # keyed by agent name; stores lessons


def empty_risk_metadata() -> RiskMetadata:
    """Return a zeroed-out RiskMetadata for initialising a fresh SwarmState."""
    return RiskMetadata(
        obi_flag=False,
        liquidity_trap_score=0.0,
        black_swan_probability=0.0,
        volatility_regime="unknown",
        bear_rationale="",
    )

"""
src/agents/bear_agent.py — Agent B: "The Bear"

Personality: forensic, perpetually skeptical risk auditor.  The Bear
treats every trade as guilty until proven innocent.  It hunts for
hidden liquidity traps, order-book manipulation, and tail-risk events
that the optimists ignore.  It speaks in the voice of a veteran risk
manager who has lived through multiple market collapses.

Responsibilities
----------------
* Evaluate Order Book Imbalance (OBI) for manipulation signals.
* Compute a liquidity-trap score and Black Swan probability.
* Output a bear_signal ∈ [0, 1] representing the danger of entering a trade.
  (High score = high danger → Judge should REJECT)
"""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.circuit_breaker import openai_circuit
from src.config import settings
from src.memory import format_lessons_for_prompt
from src.state import MarketSnapshot, RiskMetadata, SwarmState

logger = logging.getLogger(__name__)


# ── Order Book Imbalance analysis ─────────────────────────────────────────────

def _analyse_order_book(snap: MarketSnapshot) -> tuple[bool, float]:
    """
    Analyse the order-book imbalance (OBI) for signs of a liquidity trap.

    OBI ∈ [–1, 1]:
      • Strongly negative (< –0.4): asks dominate → selling pressure, bear flag
      • Near zero                  : balanced book → neutral
      • Strongly positive (> +0.6): bids dominate → *potential* bid wall (trap)
        A bid wall that is disproportionately large can be a spoofing indicator.

    Returns (obi_flag: bool, liquidity_trap_score ∈ [0, 1]).
    """
    obi = snap["order_book_imbalance"]
    spread = snap["bid_ask_spread"]

    # Aggressive ask-side pressure
    if obi < -0.4:
        return True, min(1.0, 0.6 + abs(obi + 0.4) * 1.5)

    # Suspicious over-concentration of bids (classic pump-and-dump setup)
    if obi > 0.6:
        # Wide spread with dominated bids → suspicious
        spread_factor = min(1.0, spread / 0.01)  # normalise to a 1 % spread
        trap_score = 0.5 + spread_factor * 0.4
        return True, min(1.0, trap_score)

    # Balanced but wide spread → moderate concern
    if spread > 0.005:
        return False, min(0.4, spread / 0.01)

    return False, 0.1


def _estimate_black_swan_probability(snap: MarketSnapshot) -> tuple[float, str]:
    """
    Heuristic Black Swan probability based on volatility proxies.

    Uses:
    • RSI extreme values (> 85 or < 15) as instability indicators
    • MACD histogram magnitude as momentum-collapse proxy
    • Bid-ask spread as liquidity stress indicator

    Returns (probability ∈ [0, 1], volatility_regime label).
    """
    rsi = snap["rsi_14"]
    hist = abs(snap["macd_hist"])
    spread = snap["bid_ask_spread"]

    # RSI extremes → heightened fragility
    rsi_factor = 0.0
    if rsi > 85 or rsi < 15:
        rsi_factor = 0.4
    elif rsi > 75 or rsi < 25:
        rsi_factor = 0.2

    # Very large MACD histogram magnitude = rapid momentum → fragile
    macd_factor = min(0.3, math.log1p(hist) / math.log1p(200) * 0.3)

    # Spread stress
    spread_factor = min(0.3, spread / 0.01 * 0.3)

    probability = min(1.0, rsi_factor + macd_factor + spread_factor)

    if probability < 0.15:
        regime = "low"
    elif probability < 0.40:
        regime = "elevated"
    else:
        regime = "crisis"

    return probability, regime


# ── System prompt ─────────────────────────────────────────────────────────────

_BEAR_SYSTEM = """You are **Agent Omega — The Bear**, the chief risk auditor inside the
Aura-Swarm-Quant war room.

YOUR PERSONALITY: You are forensically skeptical.  Every data point is
suspect until independently verified.  You have personally watched traders
lose everything because they ignored order-book manipulation, thin liquidity,
and macro tail risks.  You are not a permabear — you will admit when the
setup looks clean — but your DEFAULT stance is: prove to me this isn't a trap.

YOUR TASK: Given the current market snapshot and pre-computed risk metrics,
produce a **bear_signal** score between 0.0 and 1.0, where:
  • 0.0 = virtually no risk — the setup is clean, I have no objection
  • 1.0 = severe, multi-layered risk — this trade is a near-certain disaster

OUTPUT FORMAT (JSON only — no prose outside the JSON block):
{
  "bear_signal": <float 0-1>,
  "obi_flag": <bool>,
  "liquidity_trap_score": <float 0-1>,
  "black_swan_probability": <float 0-1>,
  "volatility_regime": "<low|elevated|crisis>",
  "rationale": "<2-3 sentences itemising the most dangerous signals you found>"
}

RULES:
- You must quantify every claim.  Do not use vague language like "might be risky".
- If the OBI flag is True, your bear_signal must be >= 0.5.
- Acknowledge the Bull's argument but specifically identify its blind spots.
- Never output a score < 0.05 — markets are never completely risk-free.

{lessons}
"""


# ── LangGraph node ─────────────────────────────────────────────────────────────

async def run_bear_agent(state: SwarmState) -> dict[str, Any]:
    """
    LangGraph node: Bear Agent.

    Reads market_data from state, computes OBI and Black Swan metrics,
    asks the LLM to deliver a rigorous risk verdict, and returns the
    partial state update including risk_metadata.
    """
    snap: MarketSnapshot = state["market_data"]

    obi_flag, liquidity_trap_score = _analyse_order_book(snap)
    black_swan_prob, volatility_regime = _estimate_black_swan_probability(snap)

    lessons = format_lessons_for_prompt("bear")
    system_prompt = _BEAR_SYSTEM.replace("{lessons}", lessons)

    human_content = (
        f"## Market Snapshot\n"
        f"Symbol                  : {snap['symbol']}\n"
        f"Price                   : {snap['price']:.4f}\n"
        f"Bid-Ask Spread          : {snap['bid_ask_spread']:.6f}\n"
        f"Order Book Imbalance    : {snap['order_book_imbalance']:+.3f}\n"
        f"RSI-14                  : {snap['rsi_14']:.2f}\n"
        f"MACD Histogram          : {snap['macd_hist']:.4f}\n"
        f"Sentiment Score         : {snap.get('sentiment_score', 0.0):+.3f}\n\n"
        f"## Pre-computed Risk Metrics\n"
        f"OBI Flag                : {obi_flag}\n"
        f"Liquidity Trap Score    : {liquidity_trap_score:.3f}\n"
        f"Black Swan Probability  : {black_swan_prob:.3f}\n"
        f"Volatility Regime       : {volatility_regime}\n\n"
        f"Bull's conviction (from previous context): "
        f"{state.get('bull_rationale', 'not yet available')}"
    )

    llm = ChatOpenAI(model=settings.llm_model, temperature=0.1)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]

    try:
        response = await openai_circuit.call(llm.ainvoke, messages)
        raw = response.content.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(json_match.group()) if json_match else {}

        bear_signal = float(parsed.get("bear_signal",
                                       max(0.05, liquidity_trap_score * 0.5 + black_swan_prob * 0.5)))
        bear_rationale = parsed.get("rationale", "Quantitative risk metrics used as fallback.")

        risk_meta: RiskMetadata = {
            "obi_flag": bool(parsed.get("obi_flag", obi_flag)),
            "liquidity_trap_score": float(parsed.get("liquidity_trap_score", liquidity_trap_score)),
            "black_swan_probability": float(parsed.get("black_swan_probability", black_swan_prob)),
            "volatility_regime": str(parsed.get("volatility_regime", volatility_regime)),
            "bear_rationale": bear_rationale,
        }
    except Exception as exc:
        logger.error("Bear agent LLM call failed: %s", exc)
        bear_signal = max(0.05, liquidity_trap_score * 0.5 + black_swan_prob * 0.5)
        bear_rationale = f"LLM unavailable; composite risk score: {bear_signal:.3f}"
        risk_meta = RiskMetadata(
            obi_flag=obi_flag,
            liquidity_trap_score=liquidity_trap_score,
            black_swan_probability=black_swan_prob,
            volatility_regime=volatility_regime,
            bear_rationale=bear_rationale,
        )

    bear_signal = max(0.05, min(1.0, bear_signal))
    logger.info("Bear signal=%.3f  regime=%s  obi=%s",
                bear_signal, risk_meta["volatility_regime"], risk_meta["obi_flag"])

    return {
        "bear_signal": bear_signal,
        "risk_metadata": risk_meta,
    }

"""
src/agents/bull_agent.py — Agent A: "The Bull"

Personality: relentlessly optimistic momentum hunter.  The Bull scours
price action, technical indicators, and social-media sentiment to
quantify entry strength.  It speaks in the voice of a seasoned growth
trader who has seen every bull run and genuinely believes in the
opportunity in front of it.

Responsibilities
----------------
* Evaluate RSI / MACD momentum signals.
* Query the configured provider for news headline sentiment.
* Output a bull_signal ∈ [0, 1] representing conviction in a long entry.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.circuit_breaker import openai_circuit
from src.config import settings
from src.memory import format_lessons_for_prompt
from src.state import MarketSnapshot, SwarmState
from src.sentiment import fetch_sentiment

logger = logging.getLogger(__name__)

# ── Signal calculation ────────────────────────────────────────────────────────

def _compute_technical_score(snap: MarketSnapshot) -> float:
    """
    Combine RSI and MACD into a momentum score ∈ [0, 1].

    RSI contribution
    ----------------
    • RSI < 30  → oversold, strong buy signal   → 1.0
    • RSI 30–50 → recovering                     → linear 0.5 – 0.8
    • RSI 50–70 → healthy uptrend                → 0.8 – 1.0
    • RSI > 70  → overbought, caution            → 0.4 (risk of reversal)
    • RSI > 80  → very overbought                → 0.2

    MACD histogram contribution (normalised to ±0.2 additive bonus).
    """
    rsi = snap["rsi_14"]
    if rsi < 30:
        rsi_score = 1.0
    elif rsi < 50:
        rsi_score = 0.5 + (rsi - 30) / 20 * 0.3
    elif rsi <= 70:
        rsi_score = 0.8 + (rsi - 50) / 20 * 0.2
    elif rsi <= 80:
        rsi_score = 0.4
    else:
        rsi_score = 0.2

    # MACD histogram: positive = bullish momentum
    hist = snap["macd_hist"]
    macd_bonus = max(-0.2, min(0.2, hist / 50.0))  # scale: ±50 → ±0.2

    return max(0.0, min(1.0, rsi_score + macd_bonus))


# ── System prompt ─────────────────────────────────────────────────────────────

_BULL_SYSTEM = """You are **Agent Alpha — The Bull**, the senior momentum analyst inside the
Aura-Swarm-Quant war room.

YOUR PERSONALITY: You are relentlessly optimistic but intellectually honest.
You believe every market dip is a buying opportunity and every consolidation
is a coiled spring.  You back your conviction with hard data.  You are NOT
reckless — your optimism is forensic.

YOUR TASK: Given the current market snapshot and a pre-computed technical
score, synthesise all available signals into a single **bull_signal** score
between 0.0 and 1.0, where:
  • 0.0 = absolutely no basis for a long entry
  • 1.0 = rare, near-perfect setup — all systems are green

OUTPUT FORMAT (JSON only — no prose outside the JSON block):
{
  "bull_signal": <float 0-1>,
  "rationale": "<2-3 sentences explaining your conviction>"
}

RULES:
- Never output a score > 0.95 unless RSI, MACD, AND sentiment are all
  simultaneously bullish.
- Acknowledge the Bear's primary concern (provided in the prompt) but argue
  against it if your data refutes it.
- Do NOT fabricate data.  If a signal is ambiguous, say so and lower your score.

{lessons}
"""


# ── LangGraph node ─────────────────────────────────────────────────────────────

async def run_bull_agent(state: SwarmState) -> dict[str, Any]:
    """
    LangGraph node: Bull Agent.

    Reads market_data from state, fetches live sentiment, computes a
    momentum score, asks the LLM to synthesise a final bull_signal, and
    returns the partial state update.
    """
    snap: MarketSnapshot = state["market_data"]

    # Parallel: fetch sentiment while computing technical score
    sentiment_score, sentiment_sources = await fetch_sentiment(snap["symbol"])

    # Enrich snapshot with live sentiment (mutate a copy — state is immutable)
    enriched_snap: MarketSnapshot = {
        **snap,
        "sentiment_score": sentiment_score,
        "sentiment_sources": sentiment_sources,
    }

    tech_score = _compute_technical_score(enriched_snap)

    lessons = format_lessons_for_prompt("bull")
    system_prompt = _BULL_SYSTEM.replace("{lessons}", lessons)

    human_content = (
        f"## Market Snapshot\n"
        f"Symbol       : {snap['symbol']}\n"
        f"Price        : {snap['price']:.4f}\n"
        f"Volume 24h   : {snap['volume_24h']:.2f}\n"
        f"RSI-14       : {snap['rsi_14']:.2f}\n"
        f"MACD line    : {snap['macd_line']:.4f}  |  Signal: {snap['macd_signal']:.4f}  "
        f"|  Hist: {snap['macd_hist']:.4f}\n"
        f"Sentiment    : {sentiment_score:+.3f} "
        f"(sources: {len(sentiment_sources)})\n"
        f"Pre-computed tech score: {tech_score:.3f}\n\n"
        f"Bear's primary concern (from previous context): "
        f"{state.get('risk_metadata', {}).get('bear_rationale', 'not yet available')}"
    )

    llm = ChatOpenAI(model=settings.llm_model, temperature=0.2)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]

    try:
        response = await openai_circuit.call(llm.ainvoke, messages)
        raw = response.content.strip()
        # Extract JSON even if wrapped in markdown fences
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(json_match.group()) if json_match else {}
        bull_signal = float(parsed.get("bull_signal", tech_score))
        bull_rationale = parsed.get("rationale", "Technical score used as fallback.")
    except Exception as exc:
        logger.error("Bull agent LLM call failed: %s", exc)
        bull_signal = tech_score
        bull_rationale = f"LLM unavailable; using pure technical score: {tech_score:.3f}"

    bull_signal = max(0.0, min(1.0, bull_signal))
    logger.info("Bull signal=%.3f  rationale=%s", bull_signal, bull_rationale[:80])

    return {
        "bull_signal": bull_signal,
        "bull_rationale": bull_rationale,
        "market_data": enriched_snap,  # propagate enriched data to downstream nodes
    }

"""
src/agents/judge_agent.py — Agent C: "The Judge"

Personality: cold, methodical executive who respects the adversarial
process.  The Judge does not have opinions — it has rules and numbers.
It acts only when both sides of the war room have exhausted their
arguments and the mathematics unambiguously favour entry.  If there is
any doubt, it withholds.  Capital preservation is its first law.

Responsibilities
----------------
* Apply the consensus gate: Bull > BULL_THRESHOLD AND Bear < BEAR_THRESHOLD.
* Compute position size using a Kelly-fraction heuristic.
* Emit a TradeDecision (BUY / HOLD / REJECT).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.memory import format_lessons_for_prompt
from src.state import SwarmState, TradeDecision

logger = logging.getLogger(__name__)

BULL_THRESHOLD = float(os.getenv("BULL_THRESHOLD", "0.8"))
BEAR_THRESHOLD = float(os.getenv("BEAR_THRESHOLD", "0.3"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))


# ── Kelly-fraction position sizing ───────────────────────────────────────────

def _kelly_position_size(bull: float, bear: float) -> float:
    """
    Simplified Kelly criterion:
      f* = (p * b – q) / b
    where:
      p  = probability of win  (bull_signal)
      q  = probability of loss (bear_signal)
      b  = win/loss ratio (approximated as 1/STOP_LOSS_PCT)

    Capped at 25 % of portfolio to avoid ruin through over-sizing.
    """
    p = bull
    q = bear
    b = 1.0 / STOP_LOSS_PCT  # expected win multiple relative to stop distance
    raw_kelly = (p * b - q) / b
    # Half-Kelly as a conservative measure
    half_kelly = raw_kelly / 2.0
    return max(0.0, min(0.25, half_kelly))


# ── System prompt ─────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """You are **Agent Sigma — The Judge**, the silent executive of the
Aura-Swarm-Quant war room.

YOUR PERSONALITY: You are the most emotionally detached entity in this room.
You do not root for the Bull.  You do not fear with the Bear.  You have one
purpose: protect capital and grow it, in that order.  You speak in precise,
declarative sentences.  Adjectives are for analysts — you deal in decisions.

YOUR TASK: Given the Bull's conviction score, the Bear's risk score, the
consensus gate thresholds, and a pre-computed position size, deliver a final
**TradeDecision**.

Consensus Gate (hard rules — applied in this exact order):
  1. BUY    : bull_signal > {bull_threshold} AND bear_signal < {bear_threshold}
  2. REJECT : bear_signal >= {bear_threshold}
  3. HOLD   : neither BUY nor REJECT conditions are met (bull is below threshold
              but bear has not yet flagged danger)

OUTPUT FORMAT (JSON only — no prose outside the JSON block):
{{
  "action": "<BUY|HOLD|REJECT>",
  "position_size_pct": <float 0-0.25>,
  "consensus_score": <float>,
  "rationale": "<1-2 declarative sentences with no hedging>"
}}

CRITICAL: If the hard gate is not met, you MUST output REJECT or HOLD regardless
of how compelling the Bull's argument sounds.  The rules exist for a reason.

{lessons}
"""


# ── LangGraph node ─────────────────────────────────────────────────────────────

async def run_judge_agent(state: SwarmState) -> dict[str, Any]:
    """
    LangGraph node: Judge Agent.

    Synthesises the Bull and Bear outputs into a final TradeDecision.
    Position sizing uses a Kelly fraction.  The LLM adds interpretive
    rationale but cannot override the hard consensus gate.
    """
    bull = state.get("bull_signal", 0.0)
    bear = state.get("bear_signal", 1.0)
    snap = state["market_data"]

    # Hard-rule gate (enforced before LLM to prevent hallucination overrides)
    if bull > BULL_THRESHOLD and bear < BEAR_THRESHOLD:
        hard_action = "BUY"
    elif bear >= BEAR_THRESHOLD:
        hard_action = "REJECT"
    else:
        hard_action = "HOLD"

    position_size = _kelly_position_size(bull, bear) if hard_action == "BUY" else 0.0
    consensus_score = bull - bear  # simple signed margin

    lessons = format_lessons_for_prompt("judge")
    system_prompt = (
        _JUDGE_SYSTEM
        .replace("{bull_threshold}", str(BULL_THRESHOLD))
        .replace("{bear_threshold}", str(BEAR_THRESHOLD))
        .replace("{lessons}", lessons)
    )

    human_content = (
        f"## War Room Summary\n"
        f"Symbol            : {snap['symbol']}\n"
        f"Current Price     : {snap['price']:.4f}\n\n"
        f"Bull Signal       : {bull:.3f}  (threshold > {BULL_THRESHOLD})\n"
        f"Bear Signal       : {bear:.3f}  (threshold < {BEAR_THRESHOLD})\n"
        f"Consensus Score   : {consensus_score:+.3f}\n"
        f"Hard-rule action  : {hard_action}\n"
        f"Proposed pos size : {position_size:.3f} of portfolio\n\n"
        f"Bull's Rationale  : {state.get('bull_rationale', '—')}\n"
        f"Bear's Rationale  : {state.get('risk_metadata', {}).get('bear_rationale', '—')}\n"
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]

    try:
        response = await llm.ainvoke(messages)
        raw = response.content.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(json_match.group()) if json_match else {}
        judge_rationale = parsed.get("rationale", "Decision based on hard consensus gate.")
        # LLM CAN suggest a more conservative position size, but cannot flip action
        llm_size = float(parsed.get("position_size_pct", position_size))
        position_size = min(position_size, llm_size) if hard_action == "BUY" else 0.0
    except Exception as exc:
        logger.error("Judge agent LLM call failed: %s", exc)
        judge_rationale = f"Hard-rule gate applied. LLM unavailable ({exc})."

    stop_loss_price = snap["price"] * (1.0 - STOP_LOSS_PCT) if hard_action == "BUY" else 0.0

    decision: TradeDecision = {
        "action": hard_action,
        "entry_price": snap["price"] if hard_action == "BUY" else 0.0,
        "stop_loss_price": stop_loss_price,
        "position_size_pct": round(position_size, 4),
        "consensus_score": round(consensus_score, 4),
        "judge_rationale": judge_rationale,
    }

    logger.info(
        "Judge decision=%s  bull=%.3f  bear=%.3f  pos=%.3f",
        hard_action, bull, bear, position_size,
    )

    return {"final_decision": decision}

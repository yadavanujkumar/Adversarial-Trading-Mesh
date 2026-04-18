"""
src/graph.py — LangGraph stateful workflow for Aura-Swarm-Quant.

Architecture
============

                 ┌─────────────────────────────────┐
                 │           START                  │
                 └────────────┬────────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │   ingest_market_data │  (normalise + validate)
                   └──────────┬──────────┘
                              │
              ┌───────────────┴───────────────┐
              │  PARALLEL WAR ROOM DEBATE      │
              │                               │
     ┌────────▼────────┐          ┌───────────▼────────┐
     │   bull_agent    │          │    bear_agent       │
     │  (Agent Alpha)  │          │   (Agent Omega)     │
     └────────┬────────┘          └───────────┬─────────┘
              │                               │
              └───────────────┬───────────────┘
                              │  (fan-in after both complete)
                   ┌──────────▼──────────┐
                   │    judge_agent       │  (consensus gate)
                   └──────────┬──────────┘
                              │
               ┌──────────────▼──────────────┐
               │     should_do_postmortem?    │  (conditional)
               └──┬──────────────────────┬───┘
            yes (stop-loss hit)       no (normal)
               │                          │
     ┌─────────▼──────────┐         ┌────▼────┐
     │  post_mortem_node  │         │   END   │
     └─────────┬──────────┘         └─────────┘
               │
         ┌─────▼──────┐
         │    END      │
         └─────────────┘

Parallel execution of bull_agent and bear_agent is achieved via
LangGraph's Send API / fan-out pattern so both agents run concurrently
in the same asyncio event loop.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from src.circuit_breaker import openai_circuit
from src.config import settings
from src.memory import async_append_lesson, format_lessons_for_prompt
from src.metrics import metrics
from src.state import PostMortemReport, SwarmState
from src.agents.bull_agent import run_bull_agent
from src.agents.bear_agent import run_bear_agent
from src.agents.judge_agent import run_judge_agent

logger = logging.getLogger(__name__)


# ── Ingest node ───────────────────────────────────────────────────────────────

async def ingest_market_data(state: SwarmState) -> dict[str, Any]:
    """
    Validate and normalise incoming market data.

    In production this node could enrich the snapshot with order-book
    depth from the exchange REST API.  Here it performs defensive clamping
    so downstream agents never receive NaN / out-of-range values.
    """
    snap = state["market_data"]

    # Defensive clamp
    clean = {
        **snap,
        "rsi_14": max(0.0, min(100.0, snap.get("rsi_14", 50.0))),
        "order_book_imbalance": max(-1.0, min(1.0, snap.get("order_book_imbalance", 0.0))),
        "bid_ask_spread": max(0.0, snap.get("bid_ask_spread", 0.0)),
        "sentiment_score": max(-1.0, min(1.0, snap.get("sentiment_score", 0.0))),
        "sentiment_sources": snap.get("sentiment_sources", []),
    }

    logger.info(
        "Ingested %s @ %.4f  RSI=%.1f  OBI=%+.3f",
        clean["symbol"], clean["price"], clean["rsi_14"], clean["order_book_imbalance"],
    )

    metrics.increment("market_snapshots_ingested_total")

    return {
        "market_data": clean,
        "bull_signal": 0.0,
        "bear_signal": 1.0,
        "bull_rationale": "",
        "stop_loss_hit": state.get("stop_loss_hit", False),
        "post_mortem": None,
        "final_decision": None,
        "agent_memory": state.get("agent_memory", {}),
    }


# ── Post-Mortem node ──────────────────────────────────────────────────────────

_POSTMORTEM_SYSTEM = """You are the Aura-Swarm-Quant post-mortem analyst.
A live trade just hit its stop-loss.  Your task is to dissect the failure
with clinical precision and extract actionable lessons for the trading agents.

Failure categories:
  "momentum_fade"   — the momentum signal reversed faster than anticipated
  "liquidity_trap"  — a bid wall evaporated and price gapped through the stop
  "black_swan"      — an unforeseeable macro event caused a sudden spike
  "other"           — does not fit the above

OUTPUT FORMAT (JSON only):
{
  "failure_category": "<category>",
  "lessons_learned": "<2-3 specific, actionable lessons>",
  "updated_risk_weight": <float 0-1, suggested new bear threshold>
}
"""


async def post_mortem_node(state: SwarmState) -> dict[str, Any]:
    """
    LangGraph node: Post-Mortem analysis.

    Triggered when a stop-loss is hit.  Calls the LLM to diagnose the
    failure, then persists lessons into agent memory so future cycles
    are wiser.
    """
    decision = state.get("final_decision")
    snap = state["market_data"]

    if not decision or decision.get("action") != "BUY":
        # Nothing to analyse — no trade was taken
        return {"post_mortem": None}

    entry_price = decision["entry_price"]
    current_price = snap["price"]
    loss_pct = (entry_price - current_price) / entry_price if entry_price > 0 else 0.0

    human_content = (
        f"## Failed Trade Summary\n"
        f"Symbol       : {snap['symbol']}\n"
        f"Entry Price  : {entry_price:.4f}\n"
        f"Stop-Loss at : {decision.get('stop_loss_price', 0.0):.4f}\n"
        f"Current Price: {current_price:.4f}\n"
        f"Loss %       : {loss_pct * 100:.2f}%\n\n"
        f"Bull Rationale that led to entry:\n{state.get('bull_rationale', '—')}\n\n"
        f"Bear Rationale that was OVERRULED:\n"
        f"{state.get('risk_metadata', {}).get('bear_rationale', '—')}\n\n"
        f"Judge Rationale:\n{decision.get('judge_rationale', '—')}\n"
    )

    llm = ChatOpenAI(model=settings.llm_model, temperature=0.3)
    messages = [SystemMessage(content=_POSTMORTEM_SYSTEM), HumanMessage(content=human_content)]

    try:
        response = await openai_circuit.call(llm.ainvoke, messages)
        raw = response.content.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(json_match.group()) if json_match else {}

        report: PostMortemReport = {
            "triggered_at_price": current_price,
            "entry_price": entry_price,
            "loss_pct": round(loss_pct, 6),
            "failure_category": parsed.get("failure_category", "other"),
            "lessons_learned": parsed.get("lessons_learned", "No lessons extracted."),
            "updated_risk_weight": float(parsed.get("updated_risk_weight",
                                                     settings.bear_threshold)),
        }
    except Exception as exc:
        logger.error("Post-mortem LLM call failed: %s", exc)
        report = PostMortemReport(
            triggered_at_price=current_price,
            entry_price=entry_price,
            loss_pct=round(loss_pct, 6),
            failure_category="other",
            lessons_learned=f"LLM unavailable: {exc}",
            updated_risk_weight=settings.bear_threshold,
        )

    # Persist lessons to agent memory (async-safe)
    lesson = {
        "category": report["failure_category"],
        "summary": report["lessons_learned"][:200],
        "loss_pct": report["loss_pct"],
        "risk_weight": report["updated_risk_weight"],
    }
    await async_append_lesson("bull", lesson)
    await async_append_lesson("bear", lesson)
    await async_append_lesson("judge", lesson)

    metrics.increment("post_mortems_total")
    metrics.increment(f"post_mortem_{report['failure_category']}_total")

    logger.warning(
        "Post-mortem complete: category=%s  loss=%.2f%%  new_risk_weight=%.3f",
        report["failure_category"], report["loss_pct"] * 100, report["updated_risk_weight"],
    )

    return {"post_mortem": report, "stop_loss_hit": False}


# ── Routing logic ─────────────────────────────────────────────────────────────

def _route_after_judge(state: SwarmState) -> str:
    """
    Conditional edge: direct to post_mortem if the stop-loss flag is set,
    otherwise terminate the cycle.
    """
    if state.get("stop_loss_hit", False):
        return "post_mortem"
    return END


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and compile the Aura-Swarm-Quant LangGraph.

    The Bull and Bear nodes are wired as parallel branches from the
    ingest node so that the LLM calls run concurrently via asyncio.
    """
    graph = StateGraph(SwarmState)

    # Register nodes
    graph.add_node("ingest", ingest_market_data)
    graph.add_node("bull_agent", run_bull_agent)
    graph.add_node("bear_agent", run_bear_agent)
    graph.add_node("judge", run_judge_agent)
    graph.add_node("post_mortem", post_mortem_node)

    # Entry point
    graph.set_entry_point("ingest")

    # Fan-out: ingest → bull AND bear (parallel)
    graph.add_edge("ingest", "bull_agent")
    graph.add_edge("ingest", "bear_agent")

    # Fan-in: both agents → judge
    graph.add_edge("bull_agent", "judge")
    graph.add_edge("bear_agent", "judge")

    # Conditional routing after judge
    graph.add_conditional_edges(
        "judge",
        _route_after_judge,
        {"post_mortem": "post_mortem", END: END},
    )

    # Post-mortem always terminates
    graph.add_edge("post_mortem", END)

    return graph.compile()


# Module-level compiled graph (imported by main.py and dashboard)
trading_graph = build_graph()

"""
src/execution/alpaca_broker.py — Alpaca paper/live order execution.

Translates a TradeDecision emitted by the Judge agent into a real (or
paper-trading) market order submitted to the Alpaca REST API.

Architecture notes
------------------
* Alpaca's Python SDK (alpaca-py) is async-capable via httpx under the hood.
* All order submission is guarded by the `alpaca_circuit` circuit breaker so
  a single broker outage cannot stall the entire reasoning pipeline.
* The broker is only activated when ``settings.alpaca_enabled is True`` and
  valid API credentials are present.  The codebase degrades gracefully when
  the broker is disabled.

Order lifecycle
---------------
1. Judge emits TradeDecision(action="BUY", position_size_pct=0.12, …)
2. submit_order() is called with the current portfolio value (from Alpaca).
3. A fractional market buy order is placed for the correct USD notional.
4. Stop-loss bracket is attached as a separate stop order.
5. Order IDs are returned for audit logging.

Safety guardrails
-----------------
* Hard-cap: never submit a notional greater than MAX_NOTIONAL_USD or
  position_size_pct * portfolio_value — whichever is smaller.
* Paper trading mode is default; ``ALPACA_BASE_URL`` must be changed to
  the live endpoint to switch to real capital.
* All order errors are caught and logged; the reasoning loop continues.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.circuit_breaker import alpaca_circuit
from src.config import settings
from src.state import TradeDecision

logger = logging.getLogger(__name__)

# Absolute maximum single-trade notional in USD — a hard safety guardrail.
MAX_NOTIONAL_USD = 10_000.0

# Alpaca SDK classes — imported lazily so the heavy SDK is only loaded when needed.
try:
    from alpaca.trading.client import TradingClient  # type: ignore[import]
    from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest  # type: ignore[import]
    from alpaca.trading.enums import OrderSide, TimeInForce  # type: ignore[import]
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False


def _get_trading_client() -> Any:
    """
    Return an Alpaca TradingClient using the configured credentials.
    Raises ImportError if alpaca-py is not installed.
    """
    if not _ALPACA_AVAILABLE:
        raise ImportError("alpaca-py is not installed. Run: pip install alpaca-py")
    return TradingClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
        paper=("paper" in settings.alpaca_base_url),
    )


async def _fetch_portfolio_value(client: Any) -> float:
    """Return current portfolio equity in USD from Alpaca (sync SDK wrapped in thread)."""
    try:
        account = await asyncio.to_thread(client.get_account)
        return float(account.equity)
    except Exception as exc:
        logger.warning("Could not fetch portfolio value from Alpaca: %s", exc)
        return 0.0


async def submit_order(decision: TradeDecision, symbol: str) -> dict[str, Any]:
    """
    Submit a market BUY order and an accompanying stop-loss order to Alpaca.

    Parameters
    ----------
    decision : TradeDecision
        The Judge's final trade decision.  Must have action == "BUY".
    symbol : str
        Ticker symbol in Alpaca format (e.g. "BTCUSD", "AAPL").

    Returns
    -------
    dict
        Order metadata (order_id, notional, stop_order_id) for audit logging.
        Returns {"status": "skipped", "reason": "..."} when execution is
        disabled or the guard rails prevent submission.
    """
    if not settings.alpaca_enabled:
        logger.debug("Alpaca execution disabled — skipping order submission.")
        return {"status": "skipped", "reason": "alpaca_enabled=False"}

    if decision.get("action") != "BUY":
        return {"status": "skipped", "reason": f"action={decision.get('action')}"}

    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        logger.error("Alpaca credentials missing — cannot submit order.")
        return {"status": "error", "reason": "missing_credentials"}

    async def _execute() -> dict[str, Any]:
        client = _get_trading_client()
        portfolio_value = await _fetch_portfolio_value(client)

        if portfolio_value <= 0:
            return {"status": "skipped", "reason": "portfolio_value_unavailable"}

        # Calculate notional, applying both Kelly fraction and the hard cap
        raw_notional = portfolio_value * decision["position_size_pct"]
        notional = min(raw_notional, MAX_NOTIONAL_USD)

        if notional < 1.0:
            return {"status": "skipped", "reason": f"notional_too_small: ${notional:.2f}"}

        # Submit market buy order (sync SDK call wrapped in thread)
        buy_req = MarketOrderRequest(
            symbol=symbol.upper(),
            notional=round(notional, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        buy_order = await asyncio.to_thread(client.submit_order, order_data=buy_req)
        logger.info(
            "Alpaca BUY submitted: symbol=%s  notional=$%.2f  order_id=%s",
            symbol.upper(), notional, buy_order.id,
        )

        # Submit stop-loss order
        stop_price = decision.get("stop_loss_price", 0.0)
        stop_order_id = None
        if stop_price and stop_price > 0:
            stop_req = StopOrderRequest(
                symbol=symbol.upper(),
                notional=round(notional, 2),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                stop_price=round(stop_price, 4),
            )
            stop_order = await asyncio.to_thread(client.submit_order, order_data=stop_req)
            stop_order_id = str(stop_order.id)
            logger.info(
                "Alpaca STOP submitted: stop_price=%.4f  order_id=%s",
                stop_price, stop_order_id,
            )

        return {
            "status": "submitted",
            "order_id": str(buy_order.id),
            "symbol": symbol.upper(),
            "notional_usd": round(notional, 2),
            "stop_order_id": stop_order_id,
            "entry_price": decision.get("entry_price", 0.0),
            "stop_loss_price": stop_price,
        }

    try:
        return await alpaca_circuit.call(_execute)
    except Exception as exc:
        logger.error("Alpaca order submission failed: %s", exc, exc_info=True)
        return {"status": "error", "reason": str(exc)}

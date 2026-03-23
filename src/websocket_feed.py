"""
src/websocket_feed.py — Real-time Binance WebSocket price feed.

Connects to the Binance public trade-stream for a given symbol and
converts each raw message into a MarketSnapshot that can be fed
directly into the LangGraph reasoning loop.

Order-book imbalance is approximated from the stream's best-bid /
best-ask quantities.  For production use, subscribe to the full
depth-update stream or poll the REST order-book endpoint.

Technical indicators (RSI-14, MACD) are computed incrementally from
a rolling window of closing prices so the system stays responsive
without needing a full historical download on startup.
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import time
from collections import deque
from typing import AsyncIterator, Optional

import numpy as np
import websockets

from src.state import MarketSnapshot

logger = logging.getLogger(__name__)

WS_SYMBOL: str = os.getenv("WS_SYMBOL", "btcusdt").lower()
BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream?streams="

# Rolling window for incremental indicator computation
_WINDOW_SIZE = 50  # keep last 50 candles worth of prices


# ── Incremental technical indicators ─────────────────────────────────────────

class _IndicatorEngine:
    """
    Maintains a rolling deque of prices and computes RSI-14 and MACD
    (12/26/9) incrementally without storing the full price history.
    """

    def __init__(self, window: int = _WINDOW_SIZE) -> None:
        self._prices: deque[float] = deque(maxlen=window)
        self._gains: deque[float] = deque(maxlen=15)
        self._losses: deque[float] = deque(maxlen=15)
        self._macd_history: deque[float] = deque(maxlen=9)

    def update(self, price: float) -> None:
        if self._prices:
            delta = price - self._prices[-1]
            self._gains.append(max(delta, 0.0))
            self._losses.append(max(-delta, 0.0))
        self._prices.append(price)
        # Keep MACD line history updated so the signal EMA is always current
        if len(self._prices) >= 26:
            self._macd_history.append(self._ema(12) - self._ema(26))

    @property
    def rsi_14(self) -> float:
        if len(self._gains) < 14:
            return 50.0  # neutral until we have enough data
        avg_gain = sum(list(self._gains)[-14:]) / 14.0
        avg_loss = sum(list(self._losses)[-14:]) / 14.0
        if avg_loss == 0.0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _ema(self, period: int) -> float:
        prices = list(self._prices)
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        k = 2.0 / (period + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = p * k + ema * (1.0 - k)
        return ema

    @property
    def macd(self) -> tuple[float, float, float]:
        """Returns (macd_line, signal_line, histogram)."""
        if len(self._prices) < 26:
            return 0.0, 0.0, 0.0
        line = self._ema(12) - self._ema(26)

        if len(self._macd_history) < 2:
            return line, line, 0.0

        # Signal line: 9-period EMA of the stored MACD line history
        k = 2.0 / (9 + 1)
        signal = list(self._macd_history)[0]
        for v in list(self._macd_history)[1:]:
            signal = v * k + signal * (1.0 - k)

        hist = line - signal
        return line, signal, hist


# ── Binance stream parser ─────────────────────────────────────────────────────

def _parse_binance_message(msg: dict, engine: _IndicatorEngine) -> Optional[MarketSnapshot]:
    """
    Convert a raw Binance bookTicker / trade message into a MarketSnapshot.

    Supports:
    • Individual symbol trade stream:  wss://.../ws/{symbol}@trade
    • Best book ticker:                wss://.../ws/{symbol}@bookTicker
    """
    data = msg.get("data", msg)
    event = data.get("e", "")

    if event == "trade":
        price = float(data["p"])
        qty = float(data["q"])
        engine.update(price)
        rsi = engine.rsi_14
        macd_line, macd_signal, macd_hist = engine.macd

        return MarketSnapshot(
            symbol=data.get("s", WS_SYMBOL.upper()),
            price=price,
            volume_24h=qty,            # trade qty, not 24h; updated by ticker
            bid_ask_spread=0.0,        # not available in trade stream
            order_book_imbalance=0.0,  # not available in trade stream
            rsi_14=rsi,
            macd_line=macd_line,
            macd_signal=macd_signal,
            macd_hist=macd_hist,
            sentiment_score=0.0,       # populated by Bull agent via Exa.ai
            sentiment_sources=[],
        )

    if event == "bookTicker":
        best_bid_qty = float(data.get("B", 1.0))
        best_ask_qty = float(data.get("A", 1.0))
        total = best_bid_qty + best_ask_qty
        obi = (best_bid_qty - best_ask_qty) / total if total > 0 else 0.0
        best_bid = float(data.get("b", 0.0))
        best_ask = float(data.get("a", 0.0))
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2.0 if best_bid and best_ask else 0.0

        if mid:
            engine.update(mid)
        rsi = engine.rsi_14
        macd_line, macd_signal, macd_hist = engine.macd

        return MarketSnapshot(
            symbol=data.get("s", WS_SYMBOL.upper()),
            price=mid,
            volume_24h=0.0,
            bid_ask_spread=spread,
            order_book_imbalance=obi,
            rsi_14=rsi,
            macd_line=macd_line,
            macd_signal=macd_signal,
            macd_hist=macd_hist,
            sentiment_score=0.0,
            sentiment_sources=[],
        )

    return None


# ── Public async generator ───────────────────────────────────────────────────

async def binance_market_stream(
    symbol: str = WS_SYMBOL,
    throttle_secs: float = 1.0,
) -> AsyncIterator[MarketSnapshot]:
    """
    Async generator that yields a MarketSnapshot for every price update
    from Binance, throttled to at most one snapshot per *throttle_secs*.

    Usage::

        async for snap in binance_market_stream("btcusdt"):
            await trigger_reasoning_loop(snap)

    Reconnects automatically on disconnect with exponential back-off.
    """
    engine = _IndicatorEngine()
    symbol_lower = symbol.lower()
    # Subscribe to both trade and book-ticker streams
    streams = f"{symbol_lower}@trade/{symbol_lower}@bookTicker"
    url = BINANCE_WS_BASE + streams

    backoff = 1.0
    last_yield = 0.0

    while True:
        try:
            logger.info("Connecting to Binance WebSocket: %s", url)
            async with websockets.connect(url, ping_interval=20) as ws:
                backoff = 1.0  # reset on successful connect
                async for raw in ws:
                    msg = _json.loads(raw)
                    snap = _parse_binance_message(msg, engine)
                    if snap and snap["price"] > 0:
                        now = time.monotonic()
                        if now - last_yield >= throttle_secs:
                            last_yield = now
                            yield snap
        except Exception as exc:
            logger.warning("WebSocket disconnected: %s. Reconnecting in %.1fs…", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

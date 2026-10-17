"""News sentiment providers used by the Bull node.

Parallel Search MCP is keyless. Existing Exa credentials retain precedence in
auto mode; explicit provider selections never fall back to another service.
"""
from __future__ import annotations

import asyncio
import json
import logging
from time import monotonic
import uuid
from datetime import timedelta
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Implementation

from src.circuit_breaker import exa_circuit, parallel_circuit
from src.config import settings

logger = logging.getLogger(__name__)

EXA_ENDPOINT = "https://api.exa.ai/search"
PARALLEL_ENDPOINT = "https://search.parallel.ai/mcp"
USER_AGENT = (
    "Adversarial-Trading-Mesh/1.0 "
    "(+https://github.com/yadavanujkumar/Adversarial-Trading-Mesh)"
)
# Stable for this application's lifetime, including repeated reasoning cycles.
_session_id = uuid.uuid4().hex
_parallel_cache: dict[str, tuple[float, tuple[float, list[str]]]] = {}
_parallel_lock = asyncio.Lock()


def _score_headlines(results: list[dict]) -> tuple[float, list[str]]:
    """Keep the existing title-only keyword score and source URL contract."""
    bull_kw = {"bullish", "surge", "rally", "breakout", "buy", "uptrend", "momentum", "growth"}
    bear_kw = {"bearish", "crash", "dump", "sell", "downtrend", "collapse", "risk", "fear"}
    score = 0.0
    sources: list[str] = []
    for result in results:
        title = (result.get("title") or "").lower()
        sources.append(result.get("url", ""))
        score += sum(1 for word in bull_kw if word in title)
        score -= sum(1 for word in bear_kw if word in title)
    max_possible = len(results) * max(len(bull_kw), len(bear_kw))
    normalised = max(-1.0, min(1.0, score / max(max_possible, 1)))
    return normalised, [source for source in sources if source]


async def _fetch_exa_sentiment(symbol: str) -> tuple[float, list[str]]:
    if not settings.exa_api_key:
        logger.debug("EXA_API_KEY not set; skipping Exa sentiment.")
        return 0.0, []

    async def request() -> list[dict]:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(
                EXA_ENDPOINT,
                json={
                    "query": f"{symbol} price prediction bullish bearish analysis",
                    "numResults": settings.exa_num_results,
                    "useAutoprompt": True,
                    "type": "neural",
                },
                headers={"x-api-key": settings.exa_api_key, "Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json().get("results", [])

    try:
        return _score_headlines(await exa_circuit.call(request))
    except Exception as exc:
        logger.warning("Exa.ai sentiment unavailable: %s", type(exc).__name__)
        return 0.0, []


async def _search_parallel(symbol: str) -> tuple[float, list[str]]:
    # The SDK owns initialization, HTTP/SSE negotiation and session cleanup.
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, timeout=settings.sentiment_timeout_secs,
    ) as client:
        async with streamable_http_client(PARALLEL_ENDPOINT, http_client=client) as (read, write, _):
            async with ClientSession(
                read, write,
                read_timeout_seconds=timedelta(seconds=settings.sentiment_timeout_secs),
                client_info=Implementation(name="Adversarial-Trading-Mesh", version="1.0"),
            ) as session:
                await session.initialize()
                tools = await session.list_tools()
                if not any(tool.name == "web_search" for tool in tools.tools):
                    raise ValueError("Parallel MCP does not expose web_search")
                result = await session.call_tool("web_search", {
                    "objective": f"Find recent news headlines about {symbol} for market sentiment analysis.",
                    "search_queries": [f"{symbol} latest market news", f"{symbol} bullish bearish analysis"],
                    "session_id": _session_id,
                    "model_name": settings.llm_model,
                })

    if result.isError:
        raise ValueError("Parallel MCP search returned an error")
    data = result.structuredContent
    if data is None:
        # Older MCP servers encode their JSON response as a text content block.
        data = json.loads(next(block.text for block in result.content if block.type == "text"))
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("Parallel MCP search returned invalid results")
    headlines = []
    for item in data["results"]:
        if not isinstance(item, dict):
            continue
        url, title = item.get("url"), item.get("title")
        if not isinstance(url, str) or not isinstance(title, (str, type(None))):
            continue
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
                continue
        except ValueError:
            continue
        headlines.append({"url": url, "title": title})
    return _score_headlines(headlines)


async def _fetch_parallel_sentiment(symbol: str) -> tuple[float, list[str]]:
    # Serialize cache misses so concurrent triggers do not duplicate searches.
    # Bound both the total request time and the cache size. Failures are cached
    # as neutral too, preventing a one-second feed from hammering an outage.
    async with _parallel_lock:
        cached = _parallel_cache.get(symbol)
        if cached and cached[0] > monotonic():
            return cached[1][0], list(cached[1][1])

        async def request() -> tuple[float, list[str]]:
            return await asyncio.wait_for(_search_parallel(symbol), settings.sentiment_timeout_secs)

        try:
            value = await parallel_circuit.call(request)
        except Exception as exc:
            logger.warning("Parallel sentiment unavailable: %s", type(exc).__name__)
            value = (0.0, [])
        if len(_parallel_cache) >= 32:
            _parallel_cache.pop(next(iter(_parallel_cache)))
        _parallel_cache[symbol] = (monotonic() + settings.sentiment_cache_secs, value)
        return value[0], list(value[1])


async def fetch_sentiment(symbol: str) -> tuple[float, list[str]]:
    """Resolve the user's provider choice before making any network request."""
    provider = settings.resolved_sentiment_provider
    if provider == "none":
        return 0.0, []
    if provider == "exa":
        return await _fetch_exa_sentiment(symbol)
    return await _fetch_parallel_sentiment(symbol)

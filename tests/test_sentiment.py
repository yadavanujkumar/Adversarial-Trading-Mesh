"""Exercise provider resolution through the Bull node and real MCP SDK transport.

The LLM is scripted; these tests do not make model or external network calls.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from src import sentiment
from src.agents import bull_agent
from src.circuit_breaker import CircuitBreaker
from src.config import Settings

REAL_CLIENT = httpx.AsyncClient
HEADLINES = [
    {"title": "Bitcoin bullish rally and growth", "url": "https://news.example/btc"},
    {"title": "Bitcoin risk", "url": "https://news.example/risk"},
]
SNAPSHOT = {
    "symbol": "BTCUSDT", "price": 67000.0, "volume_24h": 1000.0,
    "rsi_14": 62.0, "macd_line": 1.0, "macd_signal": 0.5, "macd_hist": 0.5,
    "order_book_imbalance": 0.1, "bid_ask_spread": 0.1,
}


class MCPFixture:
    def __init__(self, *, sse=False, error=None, data=None, text_only=False):
        self.requests = []
        self.sse, self.error, self.text_only = sse, error, text_only
        self.data = {"results": HEADLINES} if data is None else data
        self.blocked = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def __call__(self, request):
        payload = json.loads(request.content) if request.content else {}
        self.requests.append((request, payload))
        method = payload.get("method")
        if method == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "notifications/initialized":
            return httpx.Response(202)
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        elif method == "tools/call":
            if self.error == "http":
                return httpx.Response(503, json={"error": "service unavailable"})
            if self.error == "rpc":
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"],
                                                "error": {"code": -32603, "message": "failed"}})
            if self.error == "blocked":
                self.blocked.set()
                try:
                    await asyncio.sleep(60)
                finally:
                    self.cancelled.set()
            result = {"content": [{"type": "text", "text": json.dumps(self.data)}],
                      "isError": self.error == "tool"}
            if not self.text_only:
                result["structuredContent"] = self.data
        else:
            return httpx.Response(405)
        response = {"jsonrpc": "2.0", "id": payload["id"], "result": result}
        if self.sse:
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text="event: message\ndata: " + json.dumps(response) + "\n\n")
        return httpx.Response(200, json=response)


class SentimentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.config = Settings(_env_file=None)
        self.messages = []
        self.fixture = MCPFixture()
        self.exa_requests = []
        sentiment._parallel_cache.clear()
        sentiment._parallel_lock = asyncio.Lock()
        self.circuit = CircuitBreaker("parallel-test", failure_threshold=1)

        async def llm_invoke(messages):
            self.messages.append(messages)
            return SimpleNamespace(content='{"bull_signal": 0.7, "rationale": "scripted"}')

        async def handler(request):
            if str(request.url) == sentiment.EXA_ENDPOINT:
                self.exa_requests.append(request)
                return httpx.Response(200, json={"results": HEADLINES})
            return await self.fixture(request)

        def client(**kwargs):
            return REAL_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

        for patcher in (
            patch.object(sentiment, "settings", self.config),
            patch.object(bull_agent, "settings", self.config),
            patch.object(sentiment, "parallel_circuit", self.circuit),
            patch.object(sentiment, "exa_circuit", CircuitBreaker("exa-test")),
            patch.object(sentiment.httpx, "AsyncClient", side_effect=client),
            patch.object(bull_agent, "ChatOpenAI", return_value=SimpleNamespace(ainvoke=llm_invoke)),
            patch.object(bull_agent, "format_lessons_for_prompt", return_value=""),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    async def run_bull(self):
        return await bull_agent.run_bull_agent({"market_data": dict(SNAPSHOT)})

    def assert_mcp_requests(self):
        self.assertEqual([p["method"] for _, p in self.fixture.requests],
                         ["initialize", "notifications/initialized", "tools/list", "tools/call"])
        for request, _ in self.fixture.requests:
            self.assertEqual(str(request.url), "https://search.parallel.ai/mcp")
            self.assertEqual(request.headers["User-Agent"], sentiment.USER_AGENT)
            self.assertNotIn("Authorization", request.headers)
            self.assertNotIn("x-api-key", request.headers)
        call = self.fixture.requests[-1][1]["params"]
        self.assertEqual(call["name"], "web_search")
        self.assertEqual(call["arguments"]["session_id"], sentiment._session_id)
        self.assertEqual(call["arguments"]["model_name"], self.config.llm_model)
        self.assertIn("BTCUSDT", call["arguments"]["objective"])
        self.assertEqual(len(call["arguments"]["search_queries"]), 2)

    async def test_fresh_no_key_reaches_bull_prompt_and_state(self):
        self.assertEqual(self.config.resolved_sentiment_provider, "parallel")
        result = await self.run_bull()
        self.assert_mcp_requests()
        self.assertEqual(result["market_data"]["sentiment_score"], 0.125)
        self.assertEqual(result["market_data"]["sentiment_sources"], [h["url"] for h in HEADLINES])
        self.assertIn("Sentiment    : +0.125 (sources: 2)", self.messages[0][1].content)
        self.assertEqual(result["bull_signal"], 0.7)
        self.assertFalse(self.exa_requests)

    async def test_sse_and_legacy_text_result_reach_bull(self):
        self.fixture.sse = True
        self.fixture.text_only = True
        result = await self.run_bull()
        self.assertEqual(result["market_data"]["sentiment_score"], 0.125)
        self.assert_mcp_requests()

    async def test_existing_exa_key_keeps_request_and_scoring(self):
        self.config.exa_api_key = "saved-exa-key"
        self.config.exa_num_results = 4
        result = await self.run_bull()
        self.assertFalse(self.fixture.requests)
        request = self.exa_requests[0]
        self.assertEqual(request.headers["x-api-key"], "saved-exa-key")
        self.assertEqual(json.loads(request.content), {
            "query": "BTCUSDT price prediction bullish bearish analysis", "numResults": 4,
            "useAutoprompt": True, "type": "neural",
        })
        self.assertEqual(result["market_data"]["sentiment_score"], 0.125)

    async def test_explicit_parallel_overrides_exa_key(self):
        self.config.exa_api_key = "saved-exa-key"
        self.config.sentiment_provider = "parallel"
        await self.run_bull()
        self.assert_mcp_requests()
        self.assertFalse(self.exa_requests)

    async def test_disabled_even_with_cached_result_makes_no_request(self):
        await self.run_bull()
        self.config.sentiment_provider = "none"
        self.fixture.requests.clear()
        self.config.exa_api_key = "saved-exa-key"
        result = await self.run_bull()
        self.assertEqual(result["market_data"]["sentiment_sources"], [])
        self.assertEqual(result["market_data"]["sentiment_score"], 0.0)
        self.assertFalse(self.fixture.requests or self.exa_requests)

    async def test_explicit_exa_without_key_makes_no_request(self):
        self.config.sentiment_provider = "exa"
        result = await self.run_bull()
        self.assertEqual(result["market_data"]["sentiment_sources"], [])
        self.assertFalse(self.fixture.requests or self.exa_requests)

    async def test_exa_failure_does_not_switch_to_parallel(self):
        self.config.exa_api_key = "saved-exa-key"
        for selection in ("auto", "exa"):
            self.config.sentiment_provider = selection
            with patch.object(sentiment.exa_circuit, "call", side_effect=RuntimeError("failed")):
                result = await self.run_bull()
            self.assertEqual(result["market_data"]["sentiment_sources"], [])
        self.assertFalse(self.fixture.requests)

    async def test_cache_and_stable_session_id(self):
        with patch.object(sentiment, "monotonic", return_value=1000.0):
            await self.run_bull()
            await self.run_bull()
        self.assertEqual(len(self.fixture.requests), 4)
        with patch.object(sentiment, "monotonic", return_value=1061.0):
            await self.run_bull()
        calls = [p["params"]["arguments"] for _, p in self.fixture.requests if p["method"] == "tools/call"]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["session_id"], calls[1]["session_id"])

    async def test_concurrent_triggers_share_one_search(self):
        results = await asyncio.gather(self.run_bull(), self.run_bull())
        self.assertEqual(len(self.fixture.requests), 4)
        self.assertEqual(results[0]["market_data"], results[1]["market_data"])

    async def test_cached_sources_are_not_mutated_by_caller(self):
        first = await self.run_bull()
        first["market_data"]["sentiment_sources"].clear()
        second = await self.run_bull()
        self.assertEqual(len(second["market_data"]["sentiment_sources"]), 2)

    async def test_invalid_urls_and_malformed_results(self):
        self.fixture.data = {"results": [*HEADLINES, {"url": "javascript:alert(1)", "title": "bullish"},
                             {"url": "https://user:pass@news.example", "title": "bullish"},
                             {"url": "https://[invalid", "title": "bullish"}, {"url": "", "title": 123}, None]}
        result = await self.run_bull()
        self.assertEqual(result["market_data"]["sentiment_sources"], [h["url"] for h in HEADLINES])
        self.assertEqual(result["market_data"]["sentiment_score"], 0.125)
        sentiment._parallel_cache.clear()
        self.fixture.data = {"unexpected": []}
        result = await self.run_bull()
        self.assertEqual(result["market_data"]["sentiment_sources"], [])

    async def test_http_rpc_and_tool_errors_are_neutral_without_provider_switch(self):
        for error in ("http", "rpc", "tool"):
            with self.subTest(error=error):
                self.fixture.error = error
                self.fixture.requests.clear()
                sentiment._parallel_cache.clear()
                self.circuit = CircuitBreaker("test")
                with patch.object(sentiment, "parallel_circuit", self.circuit):
                    result = await self.run_bull()
                self.assertEqual(result["market_data"]["sentiment_score"], 0.0)
                self.assertEqual(result["market_data"]["sentiment_sources"], [])
                self.assertFalse(self.exa_requests)
                self.assertEqual(self.circuit.status()["total_failures"], 1)

    async def test_request_deadline_cancels_transport_and_failure_is_cached(self):
        self.config.sentiment_timeout_secs = 0.1
        self.fixture.error = "blocked"
        result = await asyncio.wait_for(self.run_bull(), 2.0)
        self.assertEqual(result["market_data"]["sentiment_sources"], [])
        self.assertTrue(self.fixture.cancelled.is_set())
        self.assertEqual(self.circuit.status()["total_failures"], 1)
        count = len(self.fixture.requests)
        await self.run_bull()
        self.assertEqual(len(self.fixture.requests), count)

    async def test_caller_cancellation_propagates_and_releases_lock(self):
        self.fixture.error = "blocked"
        task = asyncio.create_task(self.run_bull())
        await asyncio.wait_for(self.fixture.blocked.wait(), 2.0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.fixture.cancelled.is_set())
        self.assertFalse(sentiment._parallel_lock.locked())
        self.assertFalse(sentiment._parallel_cache)
        self.assertEqual(self.circuit.status()["total_failures"], 0)
        self.assertFalse(self.messages)

    async def test_registered_graph_consumes_parallel_results(self):
        from src import graph
        from src.agents import bear_agent, judge_agent

        async def other_llm(messages):
            return SimpleNamespace(content='{"bear_signal": 0.8, "rationale": "scripted"}')

        with patch.object(bear_agent, "ChatOpenAI", return_value=SimpleNamespace(ainvoke=other_llm)), \
             patch.object(judge_agent, "ChatOpenAI", return_value=SimpleNamespace(ainvoke=other_llm)):
            result = await graph.trading_graph.ainvoke({"market_data": dict(SNAPSHOT)})
        self.assertEqual(result["market_data"]["sentiment_score"], 0.125)
        self.assertEqual(len(result["market_data"]["sentiment_sources"]), 2)
        self.assertEqual(result["final_decision"]["action"], "REJECT")
        self.assert_mcp_requests()

    async def test_api_permissions_gate_news_requests(self):
        from dashboard import app as dashboard
        self.config.api_auth_token = "test-token"
        with patch.object(dashboard, "settings", self.config):
            async with REAL_CLIENT(transport=httpx.ASGITransport(app=dashboard.app), base_url="http://test") as client:
                response = await client.post("/api/trigger", json={"price": 67000})
        self.assertEqual(response.status_code, 401)
        self.assertFalse(self.fixture.requests or self.exa_requests or self.messages)



class SettingsTests(unittest.TestCase):
    def test_saved_settings_and_environment_override(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / ".env"
            path.write_text('SENTIMENT_PROVIDER=none\nEXA_API_KEY=saved\n')
            self.assertEqual(Settings(_env_file=path).resolved_sentiment_provider, "none")
            with patch.dict(os.environ, {"SENTIMENT_PROVIDER": "parallel"}):
                self.assertEqual(Settings(_env_file=path).resolved_sentiment_provider, "parallel")
            path.write_text('EXA_API_KEY=saved\n')
            self.assertEqual(Settings(_env_file=path).resolved_sentiment_provider, "exa")

    def test_invalid_provider_fails_at_startup(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValidationError):
            Settings(_env_file=None, sentiment_provider="typo")


if __name__ == "__main__":
    unittest.main()

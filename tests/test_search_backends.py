import os
import sys
from pathlib import Path

import pytest

os.environ["LOG_REQUESTS"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import claude_proxy
import gluey_proxy.claude_websearch as claude_websearch
import gluey_proxy.codex_responses as codex_responses
import gluey_proxy.search as search
from claude_proxy import _execute_codex_search_and_followup, run_search, synthesize_search_sse


class FollowupResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"status": "completed", "output": []}


class FollowupClient:
    async def post(self, *args, **kwargs):
        return FollowupResponse()


class SearxngResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "results": [
                {
                    "title": "SearXNG result",
                    "url": "https://example.test/result",
                    "content": "Mapped from SearXNG content.",
                }
            ]
        }


class SearxngClient:
    def __init__(self):
        self.calls = []

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return SearxngResponse()


@pytest.mark.asyncio
async def test_run_search_dispatches_to_named_backend(monkeypatch):
    async def fake_search(query):
        return [{"title": "Fake", "url": "https://example.test", "content": query}]

    monkeypatch.setitem(claude_proxy.SEARCH_BACKENDS, "fake", fake_search)

    hits = await run_search("dispatch me", backend="fake")

    assert hits == [
        {
            "title": "Fake",
            "url": "https://example.test",
            "content": "dispatch me",
        }
    ]


@pytest.mark.asyncio
async def test_auto_search_falls_back_to_next_backend(monkeypatch):
    calls = []

    async def failing(query):
        calls.append("first")
        return [{"title": "Search failed: boom", "url": "", "content": ""}]

    async def working(query):
        calls.append("second")
        return [{"title": "Good", "url": "https://ok.test", "content": query}]

    monkeypatch.setitem(search.SEARCH_BACKENDS, "first", failing)
    monkeypatch.setitem(search.SEARCH_BACKENDS, "second", working)
    monkeypatch.setattr(search, "SEARCH_AUTO_ORDER", ["first", "second"])

    hits = await run_search("auto me", backend="auto")

    # First backend tried and failed, fell through to the second.
    assert calls == ["first", "second"]
    assert hits == [{"title": "Good", "url": "https://ok.test", "content": "auto me"}]


@pytest.mark.asyncio
async def test_auto_search_returns_first_successful_backend(monkeypatch):
    calls = []

    async def working(query):
        calls.append("first")
        return [{"title": "Good", "url": "https://ok.test", "content": query}]

    async def never(query):
        calls.append("second")
        return [{"title": "Should not run", "url": "https://no.test", "content": ""}]

    monkeypatch.setitem(search.SEARCH_BACKENDS, "first", working)
    monkeypatch.setitem(search.SEARCH_BACKENDS, "second", never)
    monkeypatch.setattr(search, "SEARCH_AUTO_ORDER", ["first", "second"])

    hits = await run_search("auto me", backend="auto")

    # First backend succeeded; second is never tried.
    assert calls == ["first"]
    assert hits[0]["url"] == "https://ok.test"


@pytest.mark.asyncio
async def test_auto_search_returns_last_failure_when_all_fail(monkeypatch):
    async def fail_a(query):
        return [{"title": "Search failed: a", "url": "", "content": ""}]

    async def fail_b(query):
        return [{"title": "Search failed: b", "url": "", "content": ""}]

    monkeypatch.setitem(search.SEARCH_BACKENDS, "a", fail_a)
    monkeypatch.setitem(search.SEARCH_BACKENDS, "b", fail_b)
    monkeypatch.setattr(search, "SEARCH_AUTO_ORDER", ["a", "b"])

    hits = await run_search("auto me", backend="auto")

    # All failed; the last backend's failure sentinel is surfaced.
    assert hits == [{"title": "Search failed: b", "url": "", "content": ""}]



@pytest.mark.asyncio
async def test_searxng_backend_requires_base_url(monkeypatch):
    monkeypatch.setattr(search, "SEARXNG_BASE_URL", "")

    hits = await run_search("query", backend="searxng")

    assert hits == [
        {
            "title": "Search failed: SEARXNG_BASE_URL is required for searxng backend",
            "url": "",
            "content": "",
        }
    ]


@pytest.mark.asyncio
async def test_searxng_backend_maps_json_results(monkeypatch):
    fake_client = SearxngClient()
    monkeypatch.setattr(search, "SEARXNG_BASE_URL", "https://searxng.example.test")
    monkeypatch.setattr(search, "client", fake_client)

    hits = await run_search("OpenAI latest model", backend="searxng")

    assert fake_client.calls == [
        (
            "https://searxng.example.test/search",
            {
                "params": {
                    "q": "OpenAI latest model",
                    "format": "json",
                    "language": "auto",
                    "safesearch": 0,
                },
                "timeout": 30.0,
            },
        )
    ]
    assert hits == [
        {
            "title": "SearXNG result",
            "url": "https://example.test/result",
            "content": "Mapped from SearXNG content.",
        }
    ]


@pytest.mark.asyncio
async def test_claude_web_search_uses_claude_backend(monkeypatch):
    calls = []

    async def fake_run_search(query, backend=None):
        calls.append((query, backend))
        return [
            {
                "title": "Claude backend result",
                "url": "https://example.test/claude",
                "content": "Claude selected search backend.",
            }
        ]

    monkeypatch.setattr(claude_websearch, "CLAUDE_SEARCH_BACKEND", "tavily")
    monkeypatch.setattr(claude_websearch, "run_search", fake_run_search)

    body = b"".join(
        [chunk async for chunk in synthesize_search_sse("test-model", "query", "rid")]
    )

    assert calls == [("query", "tavily")]
    assert b"Claude backend result" in body


@pytest.mark.asyncio
async def test_codex_web_search_uses_codex_backend(monkeypatch):
    calls = []

    async def fake_run_search(query, backend=None):
        calls.append((query, backend))
        return [
            {
                "title": "Codex backend result",
                "url": "https://example.test/codex",
                "content": "Codex selected search backend.",
            }
        ]

    monkeypatch.setattr(codex_responses, "CODEX_SEARCH_BACKEND", "tavily")
    monkeypatch.setattr(codex_responses, "run_search", fake_run_search)
    monkeypatch.setattr(codex_responses, "client", FollowupClient())

    first_resp = {
        "output": [
            {
                "type": "function_call",
                "call_id": "call_search",
                "name": "web_search",
                "arguments": '{"query":"query"}',
            }
        ]
    }
    payload = {
        "model": "test-model",
        "input": [{"role": "user", "content": "Search the web"}],
        "tools": [{"type": "function", "name": "web_search"}],
    }

    merged = await _execute_codex_search_and_followup(
        first_resp, payload, "rid", 0.0, {}, user_auth="Bearer user-key"
    )

    assert calls == [("query", "tavily")]
    assert any(
        item.get("type") == "function_call_output"
        and "Codex selected search backend" in item.get("output", "")
        for item in merged["output"]
    )

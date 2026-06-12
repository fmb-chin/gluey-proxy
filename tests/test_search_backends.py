import os
import sys
from pathlib import Path

import pytest

os.environ["LOG_REQUESTS"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import claude_proxy
import gluey_proxy.claude_websearch as claude_websearch
import gluey_proxy.codex_responses as codex_responses
from claude_proxy import _execute_codex_search_and_followup, run_search, synthesize_search_sse


class FollowupResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"status": "completed", "output": []}


class FollowupClient:
    async def post(self, *args, **kwargs):
        return FollowupResponse()


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

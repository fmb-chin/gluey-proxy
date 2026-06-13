import os
import sys
from pathlib import Path

import pytest

os.environ["LOG_REQUESTS"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import claude_proxy
import gluey_proxy.codex_responses as codex_responses
from claude_proxy import _execute_codex_search_and_followup, _sanitize_responses_input_for_upstream


class FollowupResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "id": "resp_final",
            "status": "completed",
            "output": [
                {
                    "id": "msg_final",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The current answer used the search result.",
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {"input_tokens": 7, "output_tokens": 11},
        }


class FollowupClient:
    async def post(self, *args, **kwargs):
        return FollowupResponse()


@pytest.mark.asyncio
async def test_codex_web_search_merge_keeps_result_context_for_later_turns(monkeypatch):
    async def fake_search(query, backend=None):
        assert query == "OpenAI latest model"
        assert backend == codex_responses.CODEX_SEARCH_BACKEND
        return [
            {
                "title": "OpenAI model release",
                "url": "https://example.test/model",
                "content": "GPT-5.5 was released in this synthetic result.",
            }
        ]

    monkeypatch.setattr(codex_responses, "run_search", fake_search)
    monkeypatch.setattr(codex_responses, "client", FollowupClient())

    first_resp = {
        "output": [
            {
                "type": "function_call",
                "call_id": "call_search",
                "name": "web_search",
                "arguments": '{"query":"OpenAI latest model"}',
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 5},
    }
    payload = {
        "model": "test-model",
        "input": [{"role": "user", "content": "Search the web"}],
        "tools": [{"type": "function", "name": "web_search"}],
    }

    merged = await _execute_codex_search_and_followup(
        first_resp, payload, "rid", 0.0, {}, user_auth="Bearer user-key"
    )

    output = merged["output"]
    assert any(item.get("type") == "web_search_call" for item in output)
    search_result_items = [
        item
        for item in output
        if item.get("type") == "function_call_output"
        and item.get("call_id") == "call_search"
    ]
    assert len(search_result_items) == 1
    assert "GPT-5.5 was released" in search_result_items[0]["output"]

    next_turn_input = _sanitize_responses_input_for_upstream(output, {})
    assert not any(item.get("type") == "web_search_call" for item in next_turn_input)
    assert any(
        item.get("type") == "function_call_output"
        and "GPT-5.5 was released" in item.get("output", "")
        for item in next_turn_input
    )


def test_sanitize_synthesizes_missing_function_call_for_orphan_output():
    # A history carrying a function_call_output without a paired function_call
    # (as produced by the web_search intercept) must not be sent to upstream
    # bare: upstream rejects it with "tool id not found". The sanitizer should
    # synthesize the missing function_call so the pair is self-consistent.
    history = [
        {"role": "user", "content": "search please"},
        {"type": "function_call_output", "call_id": "call_orphan_1", "output": "result text"},
    ]

    sanitized = _sanitize_responses_input_for_upstream(history, {})

    calls = [i for i in sanitized if i.get("type") == "function_call" and i.get("call_id") == "call_orphan_1"]
    outputs = [i for i in sanitized if i.get("type") == "function_call_output" and i.get("call_id") == "call_orphan_1"]
    assert len(calls) == 1
    assert len(outputs) == 1
    # The synthesized call must come before its output.
    assert sanitized.index(calls[0]) < sanitized.index(outputs[0])


def test_sanitize_leaves_paired_function_call_output_untouched():
    history = [
        {"type": "function_call", "call_id": "c1", "name": "web_search", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "result"},
    ]

    sanitized = _sanitize_responses_input_for_upstream(history, {})

    # No extra synthesized call; the pair passes through unchanged.
    calls = [i for i in sanitized if i.get("type") == "function_call" and i.get("call_id") == "c1"]
    assert len(calls) == 1


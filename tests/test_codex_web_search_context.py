import os
import sys
from pathlib import Path

import pytest

os.environ["LOG_REQUESTS"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import claude_proxy
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
    async def fake_search(query):
        assert query == "OpenAI latest model"
        return [
            {
                "title": "OpenAI model release",
                "url": "https://example.test/model",
                "content": "GPT-5.5 was released in this synthetic result.",
            }
        ]

    monkeypatch.setattr(claude_proxy, "run_search", fake_search)
    monkeypatch.setattr(claude_proxy, "client", FollowupClient())

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

import json
import time
import uuid
from typing import Any, Optional

from .config import (
    CLAUDE_SEARCH_BACKEND,
    LOG_DIR,
    LOG_REQUESTS,
    WEB_SEARCH_QUERY_PREFIX,
    WEB_SEARCH_SYSTEM_MARKER,
)
from .search import run_search


def _system_has_marker(system_field: Any) -> bool:
    if not system_field:
        return False
    if isinstance(system_field, str):
        return WEB_SEARCH_SYSTEM_MARKER in system_field
    if isinstance(system_field, list):
        for block in system_field:
            if isinstance(block, dict):
                txt = block.get("text", "")
                if isinstance(txt, str) and WEB_SEARCH_SYSTEM_MARKER in txt:
                    return True
    return False


def _tools_has_web_search(tools: Any) -> bool:
    if not isinstance(tools, list):
        return False
    for t in tools:
        if isinstance(t, dict) and isinstance(t.get("type"), str):
            if t["type"].startswith("web_search_"):
                return True
    return False


def _extract_query(messages: Any) -> Optional[str]:
    if not isinstance(messages, list) or not messages:
        return None
    first = messages[0]
    if not isinstance(first, dict):
        return None
    content = first.get("content")
    text = None
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                break
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith(WEB_SEARCH_QUERY_PREFIX):
        return text[len(WEB_SEARCH_QUERY_PREFIX):].strip()
    return None


def is_websearch_subrequest(payload: dict) -> Optional[str]:
    """Return the query string if payload is a Claude Code web_search sub-request, else None."""
    if not isinstance(payload, dict):
        return None
    if not _system_has_marker(payload.get("system")):
        return None
    if not _tools_has_web_search(payload.get("tools")):
        return None
    return _extract_query(payload.get("messages"))


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


async def synthesize_search_sse(model: str, query: str, rid: str):
    hits = await run_search(query, CLAUDE_SEARCH_BACKEND)

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    tool_use_id = f"srvtoolu_{uuid.uuid4().hex[:24]}"
    in_tokens = max(1, len(query) // 4)
    out_tokens = max(1, sum(len((h.get("title") or "") + (h.get("url") or "")) for h in hits) // 4)

    if LOG_REQUESTS:
        (LOG_DIR / f"{rid}.synthetic.json").write_text(
            json.dumps(
                {
                    "query": query,
                    "backend": CLAUDE_SEARCH_BACKEND,
                    "tool_use_id": tool_use_id,
                    "hits": hits,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "model": model or "web-search", "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": in_tokens, "output_tokens": 0},
        },
    })

    yield _sse("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {
            "type": "server_tool_use", "id": tool_use_id,
            "name": "web_search", "input": {},
        },
    })
    yield _sse("content_block_delta", {
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": json.dumps({"query": query}, ensure_ascii=False)},
    })
    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})

    result_content = [
        {
            "type": "web_search_result",
            "title": h.get("title") or "",
            "url": h.get("url") or "",
            "encrypted_content": h.get("content") or "",
            "page_age": None,
        }
        for h in hits
        if h.get("url")
    ]

    yield _sse("content_block_start", {
        "type": "content_block_start", "index": 1,
        "content_block": {
            "type": "web_search_tool_result", "tool_use_id": tool_use_id,
            "content": result_content,
        },
    })
    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 1})

    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
    })
    yield _sse("message_stop", {"type": "message_stop"})
    yield b"data: [DONE]\n\n"

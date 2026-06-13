import json
import time
import uuid
from typing import Any, Optional

from .config import CODEX_SEARCH_BACKEND, LOG_REQUESTS, UPSTREAM, UPSTREAM_API_KEY
from .http_client import client
from .request_logging import _log_json
from .search import _format_search_results_text, run_search


def _is_responses_path(full_path: str) -> bool:
    path = "/" + full_path.strip("/")
    return path in {"/v1/responses", "/responses"}


def _find_web_search_call(output: list) -> Optional[dict]:
    """Find a function_call for web_search in the response output."""
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call" and item.get("name") == "web_search":
            return item
    return None


def _without_web_search_tools(tools: Any) -> list:
    """Return upstream-ready tools that can remain available after web_search completes."""
    if not isinstance(tools, list):
        return []
    kept = []
    for tool in tools:
        if not isinstance(tool, dict):
            kept.append(tool)
            continue
        if tool.get("name") == "web_search" or tool.get("type") == "web_search":
            continue
        kept.append(tool)
    return kept


def _sanitize_responses_input_for_upstream(input_items: Any, request_annotations: dict) -> Any:
    """Convert or remove Codex Responses output-only items that Ollama/LiteLLM cannot accept as input.

    Codex clients may send these item types back in subsequent turns as part of
    conversation history. Upstream backends only understand a subset of item
    types (message, function_call, function_call_output, reasoning). We must:

    - custom_tool_call       -> convert to function_call (preserves name/call_id/arguments)
    - custom_tool_call_output -> convert to function_call_output (preserves call_id/output)
    - tool_search_call       -> strip (metadata only, no useful content for the model)
    - tool_search_output     -> strip (metadata only, no useful content for the model)
    - web_search_call         -> strip (search metadata only; actual results are in
                                 function_call_output from our proxy's follow-up)
    """
    if not isinstance(input_items, list):
        return input_items
    # Collect call_ids that already have a paired function_call. A function_call_output
    # whose call_id has no matching call makes upstream reject the request with
    # "tool id not found". This happens with web_search histories: the intercept path
    # surfaces a function_call_output (the search result) without the originating
    # function_call. We synthesize the missing call so the pair is self-consistent
    # for upstream while preserving the search context for the model.
    function_call_ids = {
        item.get("call_id")
        for item in input_items
        if isinstance(item, dict) and item.get("type") == "function_call" and item.get("call_id")
    }
    sanitized = []
    stripped = {}
    for item in input_items:
        if not isinstance(item, dict):
            sanitized.append(item)
            continue
        item_type = item.get("type")
        if item_type == "web_search_call":
            stripped[item_type] = stripped.get(item_type, 0) + 1
            continue
        if item_type == "custom_tool_call":
            raw_input = item.get("input", "{}")
            if isinstance(raw_input, str):
                try:
                    json.loads(raw_input)
                    arguments = raw_input
                except (json.JSONDecodeError, ValueError):
                    arguments = json.dumps({"_raw_input": raw_input}, ensure_ascii=False)
            else:
                arguments = json.dumps(raw_input, ensure_ascii=False)
            converted = {
                "type": "function_call",
                "call_id": item.get("call_id", ""),
                "name": item.get("name", ""),
                "arguments": arguments,
            }
            sanitized.append(converted)
            request_annotations.setdefault("codex_input_converted", {})
            request_annotations["codex_input_converted"]["custom_tool_call"] = request_annotations["codex_input_converted"].get("custom_tool_call", 0) + 1
            continue
        if item_type == "custom_tool_call_output":
            converted = {
                "type": "function_call_output",
                "call_id": item.get("call_id", ""),
                "output": item.get("output", ""),
            }
            sanitized.append(converted)
            request_annotations.setdefault("codex_input_converted", {})
            request_annotations["codex_input_converted"]["custom_tool_call_output"] = request_annotations["codex_input_converted"].get("custom_tool_call_output", 0) + 1
            continue
        if item_type in ("tool_search_call", "tool_search_output"):
            stripped[item_type] = stripped.get(item_type, 0) + 1
            continue
        if item_type == "function_call_output":
            call_id = item.get("call_id")
            if call_id and call_id not in function_call_ids:
                # Synthesize the missing function_call so upstream sees a complete
                # tool round-trip instead of an orphaned result.
                sanitized.append({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": "web_search",
                    "arguments": "{}",
                })
                function_call_ids.add(call_id)
                request_annotations["synthesized_function_call"] = (
                    request_annotations.get("synthesized_function_call", 0) + 1
                )
        sanitized.append(item)
    if stripped:
        request_annotations["codex_input_items_stripped"] = stripped
    return sanitized


async def _execute_codex_search_and_followup(
    first_resp: dict, payload: dict, rid: str, started: float, request_annotations: dict,
    user_auth: str = "",
) -> dict:
    """Handle the Codex web_search tool loop."""
    ws_call = _find_web_search_call(first_resp.get("output", []))
    if not ws_call:
        return first_resp

    try:
        args = json.loads(ws_call.get("arguments", "{}"))
    except Exception:
        args = {}
    query = args.get("query", "")
    if not query:
        request_annotations["codex_web_search_error"] = "empty query"
        return first_resp

    search_started = time.time()
    hits = await run_search(query, CODEX_SEARCH_BACKEND)
    search_text = _format_search_results_text(hits)

    request_annotations["codex_web_search"] = {
        "query": query,
        "backend": CODEX_SEARCH_BACKEND,
        "hits": len(hits),
        "search_ms": int((time.time() - search_started) * 1000),
    }

    if LOG_REQUESTS:
        _log_json(
            rid,
            "codex.search",
            {"query": query, "backend": CODEX_SEARCH_BACKEND, "hits": hits},
        )

    call_id = ws_call.get("call_id", "functions.web_search:0")
    followup_input = list(_sanitize_responses_input_for_upstream(payload.get("input", []), request_annotations) or [])

    followup_input.append({
        "type": "function_call",
        "call_id": call_id,
        "name": "web_search",
        "arguments": ws_call.get("arguments", "{}"),
    })
    followup_input.append({
        "type": "function_call_output",
        "call_id": call_id,
        "output": search_text,
    })

    followup_payload = {
        "model": payload.get("model"),
        "input": followup_input,
        "stream": False,
    }
    followup_tools = _without_web_search_tools(payload.get("tools"))
    if followup_tools:
        followup_payload["tools"] = followup_tools
        request_annotations["codex_followup_tools"] = len(followup_tools)
    if payload.get("instructions"):
        followup_payload["instructions"] = payload["instructions"]
    if payload.get("temperature") is not None:
        followup_payload["temperature"] = payload["temperature"]

    followup_body = json.dumps(followup_payload, ensure_ascii=False).encode("utf-8")
    followup_url = f"{UPSTREAM.rstrip('/')}/v1/responses"

    followup_headers = {"Content-Type": "application/json"}
    auth_key = user_auth or UPSTREAM_API_KEY
    if auth_key:
        followup_headers["Authorization"] = f"Bearer {auth_key}"

    followup_started = time.time()
    try:
        r = await client.post(followup_url, headers=followup_headers, content=followup_body, timeout=300.0)
        r.raise_for_status()
        second_resp = r.json()
    except Exception as e:
        request_annotations["codex_followup_error"] = repr(e)
        fallback = dict(first_resp)
        fallback["status"] = "completed"
        fallback["output"] = [
            item for item in first_resp.get("output", [])
            if isinstance(item, dict) and item.get("type") == "reasoning"
        ]
        fallback["output"].append({
            "id": f"ws_{uuid.uuid4().hex[:24]}",
            "type": "web_search_call",
            "status": "completed",
            "action": {"type": "search", "query": query},
        })
        fallback["output"].append({
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": search_text,
                "annotations": [],
            }],
        })
        return fallback

    request_annotations["codex_followup_ms"] = int((time.time() - followup_started) * 1000)

    ws_output_item = {
        "id": f"ws_{uuid.uuid4().hex[:24]}",
        "type": "web_search_call",
        "status": "completed",
        "action": {
            "type": "search",
            "query": query,
        },
    }
    ws_result_item = {
        "type": "function_call_output",
        "call_id": call_id,
        "output": search_text,
    }

    merged_output = []
    for item in first_resp.get("output", []):
        if item.get("type") == "reasoning":
            merged_output.append(item)

    # For the injected path (clients like Kilo that never registered web_search),
    # do NOT surface the web_search_call / function_call_output round-trip. The
    # client doesn't know this tool and would (a) flag it as an invalid tool call
    # mid-stream and (b) echo the dangling function_call_output back next turn,
    # which upstream rejects ("tool id not found"). The search results are already
    # folded into the follow-up message, so a clean message is all the client needs.
    if not request_annotations.get("web_search_injected"):
        merged_output.append(ws_output_item)
        merged_output.append(ws_result_item)

    for item in second_resp.get("output", []):
        merged_output.append(item)

    merged = dict(second_resp)
    merged["output"] = merged_output
    u1 = first_resp.get("usage") or {}
    u2 = second_resp.get("usage") or {}

    def _merge_usage(a, b):
        result = {}
        for k in set(list(a.keys()) + list(b.keys())):
            va, vb = a.get(k), b.get(k)
            if isinstance(va, dict) or isinstance(vb, dict):
                result[k] = vb if vb is not None else va
            else:
                result[k] = (va or 0) + (vb or 0)
        return result

    merged["usage"] = _merge_usage(u1, u2)

    return merged

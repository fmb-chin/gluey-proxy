"""
Claude Code / Codex aware proxy.

Default behaviour: transparent forwarding to UPSTREAM (currently litellm).

Special cases:

1. Claude Code emits a sub-request whenever it wants to run a WebSearch tool.
   The sub-request has a recognisable shape — intercepted here and handled
   via search API, returning synthetic Anthropic SSE.

2. Codex includes a `type: web_search` tool in `/v1/responses` requests.
   When the model calls this tool, we intercept the response, execute the
   search via Ollama or Tavily (env-controlled), then send a follow-up
   request so the model can incorporate the results.
"""
import json
import time
import uuid
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from gluey_proxy.claude_websearch import is_websearch_subrequest, synthesize_search_sse
from gluey_proxy.codex_responses import (
    _execute_codex_search_and_followup,
    _find_web_search_call,
    _is_responses_path,
    _sanitize_responses_input_for_upstream,
)
from gluey_proxy.config import LOG_REQUESTS, UPSTREAM
from gluey_proxy.http_client import client
from gluey_proxy.mcp_tools import _convert_function_call_to_namespaced, _convert_mcp_calls_in_response
from gluey_proxy.request_logging import _log_bytes, _log_json, _log_meta, _log_req
from gluey_proxy.responses_sse import (
    _buffer_responses_stream,
    _build_sse_from_response,
    _parse_sse_output,
    _responses_sse_event,
)
from gluey_proxy.search import (
    SEARCH_BACKENDS,
    run_ollama_search,
    run_search,
    run_tavily_search,
)


app = FastAPI()


# ---------------------- proxy core ----------------------

@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy(full_path: str, request: Request):
    rid = f"{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
    started = time.time()
    request_annotations = {}

    body = await request.body()
    req_headers = {k: v for k, v in request.headers.items()}
    _log_req(rid, req_headers, body)

    payload: Optional[dict] = None
    if body:
        try:
            payload = json.loads(body)
        except Exception:
            payload = None

    # --- Intercept Claude Code WebSearch sub-requests ---
    if (
        request.method == "POST"
        and full_path.endswith("v1/messages")
        and isinstance(payload, dict)
    ):
        query = is_websearch_subrequest(payload)
        if query:
            model = payload.get("model") or "web-search"
            _log_meta(rid, {
                "rid": rid,
                "intercepted": "claude_code_web_search",
                "query": query,
                "model": model,
                "elapsed_ms_to_decision": int((time.time() - started) * 1000),
            })
            return StreamingResponse(
                synthesize_search_sse(model, query, rid),
                status_code=200,
                headers={"cache-control": "no-cache"},
                media_type="text/event-stream",
            )

    # --- Codex request sanitization ---
    is_codex_responses = (
        request.method == "POST"
        and isinstance(payload, dict)
        and _is_responses_path(full_path)
    )
    has_web_search = False

    # Track flattened namespace → sub-tool mapping so we can rewrite
    # function_call responses with the correct namespace for the Codex client.
    # mcp_namespace_map: {"mcp__ns__tool_name": {"namespace": "mcp__ns", "name": "tool_name"}}
    mcp_namespace_map: dict = {}

    if request.method == "POST" and isinstance(payload, dict):
        # Process tools: flatten namespace into function tools, convert web_search
        tools = payload.get("tools")
        if isinstance(tools, list):
            new_tools = []
            for t in tools:
                if not isinstance(t, dict):
                    new_tools.append(t)
                    continue
                ttype = t.get("type")
                if ttype == "namespace":
                    # Flatten namespace tools into top-level function tools.
                    # Ollama strips the inner "tools" array from namespace objects,
                    # so the model can't see sub-tools inside a namespace.
                    # By flattening, each sub-tool becomes a standalone function the
                    # model can see and call.  We record the mapping so that when
                    # the model calls "mcp__ns__tool", we can rewrite the response
                    # function_call with the correct namespace (which the Codex client
                    # needs to route the call to the right MCP server).
                    ns_name = t.get("name", "")
                    ns_desc = t.get("description", "")
                    sub_tools = t.get("tools") or []
                    if isinstance(sub_tools, list) and sub_tools:
                        for sub in sub_tools:
                            if not isinstance(sub, dict) or sub.get("type") != "function":
                                continue
                            flat = dict(sub)
                            inner_name = sub.get("name", "")
                            flat_name = f"{ns_name}__{inner_name}"
                            flat["name"] = flat_name
                            if ns_desc:
                                flat["description"] = (
                                    f"[{ns_name}] {ns_desc}\n\n{sub.get('description', '')}"
                                )
                            new_tools.append(flat)
                            # Record mapping for response conversion
                            mcp_namespace_map[flat_name] = {
                                "namespace": ns_name,
                                "name": inner_name,
                            }
                        request_annotations.setdefault("codex_tools_flattened", 0)
                        request_annotations["codex_tools_flattened"] += 1
                    else:
                        # No sub-tools — keep the namespace as-is so the model
                        # at least sees the name and description.
                        new_tools.append(t)
                elif ttype == "web_search":
                    # Convert to a standard function tool so the model knows how to call it
                    has_web_search = True
                    new_tools.append({
                        "type": "function",
                        "name": "web_search",
                        "description": "Search the web for current information. Use this tool whenever you need up-to-date information, news, or facts beyond your training data.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query string"
                                }
                            },
                            "required": ["query"]
                        }
                    })
                    request_annotations["codex_web_search_converted"] = True
                else:
                    new_tools.append(t)
            payload["tools"] = new_tools

        # Strip Responses output-only items that are not valid input for Ollama/LiteLLM.
        if is_codex_responses and isinstance(payload.get("input"), list):
            payload["input"] = _sanitize_responses_input_for_upstream(payload.get("input"), request_annotations)

        # Downgrade unsupported reasoning effort values (e.g. "xhigh" -> "high")
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            if isinstance(effort, str) and effort not in {"high", "medium", "low", "max", "none"}:
                reasoning["effort"] = "high"
                request_annotations["reasoning_downgraded"] = f"{effort}->high"
        elif isinstance(reasoning, str) and reasoning not in {"high", "medium", "low", "max", "none"}:
            payload["reasoning"] = "high"
            request_annotations["reasoning_downgraded"] = f"{reasoning}->high"

        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if LOG_REQUESTS:
            _log_bytes(rid, "upstream.body", body)
            tool_names = [t.get("name") for t in payload.get("tools", []) if isinstance(t, dict)]
            _log_json(rid, "upstream.tools", tool_names)

    # --- Forward to upstream ---
    upstream_url = f"{UPSTREAM.rstrip('/')}/{full_path}"
    if request.url.query:
        upstream_url += f"?{request.url.query}"

    fwd_headers = {}
    for k, v in request.headers.items():
        kl = k.lower()
        if kl in ("host", "content-length", "connection", "transfer-encoding"):
            continue
        fwd_headers[k] = v

    upstream_started = time.time()
    # If the request body was modified (e.g. Codex namespace tools flattened,
    # input items sanitized), we need to re-serialize the modified payload and
    # send THAT to upstream, not the original body bytes.
    modified = request_annotations.get("codex_tools_flattened") or request_annotations.get("codex_input_items_stripped") or request_annotations.get("codex_input_converted")
    if modified and payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        upstream_resp = await client.send(
            client.build_request(
                request.method,
                upstream_url,
                headers=fwd_headers,
                content=body,
            ),
            stream=True,
        )
    except Exception as e:
        _log_meta(rid, {
            "rid": rid,
            "upstream_url": upstream_url,
            "method": request.method,
            "error": repr(e),
            **request_annotations,
            "elapsed_ms": int((time.time() - started) * 1000),
        })
        return Response(content=f"upstream error: {e}", status_code=502)

    resp_headers = dict(upstream_resp.headers)
    if LOG_REQUESTS:
        _log_json(rid, "resp.headers", resp_headers)

    ctype = upstream_resp.headers.get("content-type", "")
    is_stream = "text/event-stream" in ctype or "stream" in ctype

    # --- Codex web_search tool loop ---
    # Buffering strategy:
    #   - If request has NO web_search tool → stream through directly (no buffering)
    #   - If request HAS web_search tool → MUST buffer because we need to check if the
    #     model called web_search and, if so, execute the search and send a follow-up.
    #
    # In theory we could stream-while-scanning SSE events and detect function_call:web_search
    # mid-stream, then stop and execute the search. However, this requires either:
    #   (a) Not sending any events to the client until we know if web_search was called
    #       (which is effectively buffering), or
    #   (b) Sending early events and then "switching" to a merged response, which breaks
    #       the SSE protocol since we can't undo already-sent events.
    #
    # The practical optimization already applied: requests WITHOUT web_search tool bypass
    # buffering entirely and stream through with low latency. Codex always includes web_search
    # in its tool list, but if the model doesn't call it, we still buffer. This is unavoidable
    # because we must see the complete response to confirm no web_search call was made.
    needs_web_search_intercept = is_codex_responses and upstream_resp.status_code == 200 and has_web_search

    # For Codex /v1/responses requests with flattened MCP tools, we must buffer
    # the response to rewrite function_call items with the correct namespace
    # so the Codex client can route MCP tool calls to the right MCP server.
    # The Codex router (build_tool_call) uses ResponseItem::FunctionCall →
    # ToolName::new(namespace, name) → ToolName::namespaced(ns, name) to look up
    # the MCP handler. We must keep type="function_call" (NOT "custom_tool_call")
    # because CustomToolCall uses ToolName::plain(name) which ignores namespace.
    needs_mcp_call_conversion = (
        is_codex_responses
        and upstream_resp.status_code == 200
        and mcp_namespace_map
        and not needs_web_search_intercept  # web_search path handles this separately
    )

    if needs_mcp_call_conversion:
        # Buffer the entire upstream response to rewrite function_call namespace
        raw_body = await _buffer_responses_stream(upstream_resp)
        if LOG_REQUESTS:
            _log_bytes(rid, "resp.body", raw_body)

        if is_stream:
            output_items = _parse_sse_output(raw_body)
            # Reconstruct a full response from SSE events to do the conversion
            # We need to find the response.completed event to get the base response dict
            resp_dict = {}
            text_data = raw_body.decode("utf-8", errors="replace")
            for line in text_data.split("\n"):
                ev = _responses_sse_event(line)
                if ev and ev.get("type") == "response.completed":
                    resp_dict = ev.get("response", {})
                    break
            if resp_dict:
                resp_dict = _convert_mcp_calls_in_response(
                    resp_dict, mcp_namespace_map, request_annotations
                )
                sse_bytes = _build_sse_from_response(resp_dict)
                if LOG_REQUESTS:
                    _log_bytes(rid, "resp.client.sse", sse_bytes)
                _log_meta(rid, {
                    "rid": rid,
                    "upstream_url": upstream_url,
                    "method": request.method,
                    "status": 200,
                    "is_stream": True,
                    "content_type": ctype,
                    **request_annotations,
                    "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                    "elapsed_ms": int((time.time() - started) * 1000),
                })
                return Response(
                    content=sse_bytes,
                    status_code=200,
                    headers={"content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache"},
                    media_type="text/event-stream",
                )
            # Fallback: return raw if we couldn't parse
            _log_meta(rid, {
                "rid": rid,
                "upstream_url": upstream_url,
                "method": request.method,
                "status": upstream_resp.status_code,
                "is_stream": True,
                "content_type": ctype,
                **request_annotations,
                "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                "elapsed_ms": int((time.time() - started) * 1000),
            })
            return Response(
                content=raw_body,
                status_code=upstream_resp.status_code,
                headers={"content-type": ctype or "text/event-stream; charset=utf-8"},
                media_type=ctype or "text/event-stream",
            )
        else:
            # Non-streaming: parse JSON, convert, return
            try:
                resp_dict = json.loads(raw_body)
            except Exception:
                resp_dict = {}
            if resp_dict:
                resp_dict = _convert_mcp_calls_in_response(
                    resp_dict, mcp_namespace_map, request_annotations
                )
                if LOG_REQUESTS:
                    _log_json(rid, "resp.client.json", resp_dict)
                _log_meta(rid, {
                    "rid": rid,
                    "upstream_url": upstream_url,
                    "method": request.method,
                    "status": upstream_resp.status_code,
                    "is_stream": False,
                    "content_type": ctype,
                    **request_annotations,
                    "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                    "elapsed_ms": int((time.time() - started) * 1000),
                })
                return Response(
                    content=json.dumps(resp_dict, ensure_ascii=False).encode("utf-8"),
                    status_code=upstream_resp.status_code,
                    headers={"content-type": "application/json"},
                    media_type="application/json",
                )
            # Fallback
            _log_meta(rid, {
                "rid": rid,
                "upstream_url": upstream_url,
                "method": request.method,
                "status": upstream_resp.status_code,
                "is_stream": False,
                "content_type": ctype,
                **request_annotations,
                "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                "elapsed_ms": int((time.time() - started) * 1000),
            })
            out_headers = {}
            for k, v in resp_headers.items():
                kl = k.lower()
                if kl in ("content-length", "transfer-encoding", "connection"):
                    continue
                out_headers[k] = v
            return Response(
                content=raw_body,
                status_code=upstream_resp.status_code,
                headers=out_headers,
                media_type=ctype or None,
            )

    if needs_web_search_intercept:
        # Buffer the entire upstream response
        raw_body = await _buffer_responses_stream(upstream_resp)
        if LOG_REQUESTS:
            _log_bytes(rid, "resp.body", raw_body)

        # Parse output items from the response
        first_resp = None
        if is_stream:
            output_items = _parse_sse_output(raw_body)
        else:
            try:
                first_resp = json.loads(raw_body)
                output_items = first_resp.get("output", [])
            except Exception:
                output_items = []

        # Check if model called web_search
        ws_call = _find_web_search_call(output_items)

        if ws_call:
            # Need to rebuild first_resp for SSE case
            if first_resp is None:
                # Reconstruct from SSE events
                first_resp = {"output": output_items, "status": "completed"}
                # Try to get usage from the completed event
                text_data = raw_body.decode("utf-8", errors="replace")
                for line in text_data.split("\n"):
                    ev = _responses_sse_event(line)
                    if ev and ev.get("type") == "response.completed":
                        first_resp.update(ev.get("response", {}))
                        break

            # Execute search + follow-up
            user_auth = req_headers.get("authorization", "") or req_headers.get("Authorization", "")
            merged = await _execute_codex_search_and_followup(
                first_resp, payload, rid, started, request_annotations, user_auth=user_auth
            )

            # Rewrite MCP function_call namespace in merged response so Codex
            # client can route to the right MCP server.
            if mcp_namespace_map:
                merged = _convert_mcp_calls_in_response(
                    merged, mcp_namespace_map, request_annotations
                )

            # Return merged response
            if is_stream:
                # Re-encode as SSE
                sse_bytes = _build_sse_from_response(merged)
                _log_meta(rid, {
                    "rid": rid,
                    "upstream_url": upstream_url,
                    "method": request.method,
                    "status": 200,
                    "is_stream": True,
                    "content_type": ctype,
                    "codex_web_search_handled": True,
                    **request_annotations,
                    "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                    "elapsed_ms": int((time.time() - started) * 1000),
                })
                return Response(
                    content=sse_bytes,
                    status_code=200,
                    headers={"content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache"},
                    media_type="text/event-stream",
                )
            else:
                _log_meta(rid, {
                    "rid": rid,
                    "upstream_url": upstream_url,
                    "method": request.method,
                    "status": 200,
                    "is_stream": False,
                    "content_type": ctype,
                    "codex_web_search_handled": True,
                    **request_annotations,
                    "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                    "elapsed_ms": int((time.time() - started) * 1000),
                })
                return Response(
                    content=json.dumps(merged, ensure_ascii=False).encode("utf-8"),
                    status_code=200,
                    headers={"content-type": "application/json"},
                    media_type="application/json",
                )

        # No web_search call found — still need to convert MCP calls if any
        if mcp_namespace_map:
            if is_stream:
                # Parse SSE, convert, re-encode
                output_items = _parse_sse_output(raw_body)
                resp_dict = {}
                text_data = raw_body.decode("utf-8", errors="replace")
                for line in text_data.split("\n"):
                    ev = _responses_sse_event(line)
                    if ev and ev.get("type") == "response.completed":
                        resp_dict = ev.get("response", {})
                        break
                if resp_dict:
                    resp_dict = _convert_mcp_calls_in_response(
                        resp_dict, mcp_namespace_map, request_annotations
                    )
                    sse_bytes = _build_sse_from_response(resp_dict)
                    _log_meta(rid, {
                        "rid": rid,
                        "upstream_url": upstream_url,
                        "method": request.method,
                        "status": upstream_resp.status_code,
                        "is_stream": True,
                        "content_type": ctype,
                        "codex_buffered": True,
                        **request_annotations,
                        "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                        "elapsed_ms": int((time.time() - started) * 1000),
                    })
                    return Response(
                        content=sse_bytes,
                        status_code=upstream_resp.status_code,
                        headers={"content-type": "text/event-stream; charset=utf-8"},
                        media_type="text/event-stream",
                    )
            else:
                # Non-streaming: parse JSON, convert, return
                try:
                    resp_dict = json.loads(raw_body)
                except Exception:
                    resp_dict = {}
                if resp_dict:
                    resp_dict = _convert_mcp_calls_in_response(
                        resp_dict, mcp_namespace_map, request_annotations
                    )
                    _log_meta(rid, {
                        "rid": rid,
                        "upstream_url": upstream_url,
                        "method": request.method,
                        "status": upstream_resp.status_code,
                        "is_stream": False,
                        "content_type": ctype,
                        "codex_buffered": True,
                        **request_annotations,
                        "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                        "elapsed_ms": int((time.time() - started) * 1000),
                    })
                    out_headers = {}
                    for k, v in resp_headers.items():
                        kl = k.lower()
                        if kl in ("content-length", "transfer-encoding", "connection"):
                            continue
                        out_headers[k] = v
                    return Response(
                        content=json.dumps(resp_dict, ensure_ascii=False).encode("utf-8"),
                        status_code=upstream_resp.status_code,
                        headers=out_headers,
                        media_type=ctype or None,
                    )

        # No web_search call found and no MCP conversion needed — return as-is
        if is_stream:
            _log_meta(rid, {
                "rid": rid,
                "upstream_url": upstream_url,
                "method": request.method,
                "status": upstream_resp.status_code,
                "is_stream": True,
                "content_type": ctype,
                "codex_buffered": True,
                **request_annotations,
                "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                "elapsed_ms": int((time.time() - started) * 1000),
            })
            return Response(
                content=raw_body,
                status_code=upstream_resp.status_code,
                headers={"content-type": ctype or "text/event-stream; charset=utf-8"},
                media_type=ctype or "text/event-stream",
            )
        else:
            _log_meta(rid, {
                "rid": rid,
                "upstream_url": upstream_url,
                "method": request.method,
                "status": upstream_resp.status_code,
                "is_stream": False,
                "content_type": ctype,
                "codex_buffered": True,
                **request_annotations,
                "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                "elapsed_ms": int((time.time() - started) * 1000),
            })
            out_headers = {}
            for k, v in resp_headers.items():
                kl = k.lower()
                if kl in ("content-length", "transfer-encoding", "connection"):
                    continue
                out_headers[k] = v
            return Response(
                content=raw_body,
                status_code=upstream_resp.status_code,
                headers=out_headers,
                media_type=ctype or None,
            )

    # --- Default: transparent stream forward (non-Codex or error responses) ---
    async def streamer():
        streamed_response_bytes = 0
        try:
            async for chunk in upstream_resp.aiter_raw():
                streamed_response_bytes += len(chunk)
                yield chunk
        finally:
            await upstream_resp.aclose()
            _log_meta(rid, {
                "rid": rid,
                "upstream_url": upstream_url,
                "method": request.method,
                "status": upstream_resp.status_code,
                "is_stream": is_stream,
                "content_type": ctype,
                "streamed_response_bytes": streamed_response_bytes,
                **request_annotations,
                "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                "elapsed_ms": int((time.time() - started) * 1000),
            })

    out_headers = {}
    for k, v in resp_headers.items():
        kl = k.lower()
        if kl in ("content-length", "transfer-encoding", "connection"):
            continue
        out_headers[k] = v

    return StreamingResponse(
        streamer(),
        status_code=upstream_resp.status_code,
        headers=out_headers,
        media_type=ctype or None,
    )


@app.on_event("shutdown")
async def _shutdown():
    await client.aclose()

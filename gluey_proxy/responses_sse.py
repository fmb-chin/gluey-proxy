import json
from typing import Optional


def _responses_sse_event(data_line: str) -> Optional[dict]:
    """Parse a single SSE data line into a dict. Returns None if not parseable."""
    line = data_line.strip()
    if line.startswith("data: "):
        payload = line[6:]
        if payload == "[DONE]":
            return {"type": "done"}
        try:
            return json.loads(payload)
        except Exception:
            return None
    return None


async def _buffer_responses_stream(upstream_resp) -> bytes:
    """Buffer the entire upstream SSE response into decoded bytes."""
    chunks = []
    async for chunk in upstream_resp.aiter_bytes():
        chunks.append(chunk)
    await upstream_resp.aclose()
    return b"".join(chunks)


def _parse_sse_output(raw: bytes) -> list:
    """Extract output items from a buffered SSE stream."""
    output_items = []
    text_data = raw.decode("utf-8", errors="replace")
    for line in text_data.split("\n"):
        ev = _responses_sse_event(line)
        if not ev:
            continue
        if ev.get("type") == "response.output_item.done":
            item = ev.get("item")
            if item:
                output_items.append(item)
        elif ev.get("type") == "response.completed":
            resp = ev.get("response", {})
            if resp.get("output"):
                output_items = resp["output"]
    return output_items


def _build_sse_from_response(resp: dict) -> bytes:
    """Convert a complete response dict into SSE stream bytes."""
    parts = []

    # response.created
    parts.append(f"data: {json.dumps({'type': 'response.created', 'response': resp}, ensure_ascii=False)}\n\n")

    # output items
    for i, item in enumerate(resp.get("output", [])):
        parts.append(f"data: {json.dumps({'type': 'response.output_item.added', 'output_index': i, 'item': item}, ensure_ascii=False)}\n\n")
        parts.append(f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': i, 'item': item}, ensure_ascii=False)}\n\n")

    # in_progress
    parts.append(f"data: {json.dumps({'type': 'response.in_progress', 'response': resp}, ensure_ascii=False)}\n\n")

    # completed
    parts.append(f"data: {json.dumps({'type': 'response.completed', 'response': resp}, ensure_ascii=False)}\n\n")

    parts.append("data: [DONE]\n\n")

    return "".join(parts).encode("utf-8")

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
    """Convert a complete response dict into a faithful Responses SSE stream.

    Replicates the upstream event ordering and field shapes so that strict
    clients (e.g. the Vercel AI SDK's Responses parser used by Kilo) can
    reconstruct assistant text and tool calls. Message items emit the full
    output_text.delta sequence; function_call items emit their arguments delta.
    """
    parts = []
    seq = 0

    def emit(event: dict) -> None:
        nonlocal seq
        event["sequence_number"] = seq
        seq += 1
        parts.append(f"data: {json.dumps(event, ensure_ascii=False)}\n\n")

    emit({"type": "response.created", "response": resp})
    emit({"type": "response.in_progress", "response": resp})

    for index, item in enumerate(resp.get("output", [])):
        item_type = item.get("type")
        item_id = item.get("id", "")

        # output_item.added carries the item in its in-progress shape.
        added_item = dict(item)
        if item_type == "message":
            added_item["status"] = "in_progress"
            added_item["content"] = []
        emit({
            "type": "response.output_item.added",
            "output_index": index,
            "item": added_item,
        })

        if item_type == "message":
            for content_index, block in enumerate(item.get("content", [])):
                if block.get("type") != "output_text":
                    continue
                text = block.get("text", "")
                annotations = block.get("annotations", [])
                emit({
                    "type": "response.content_part.added",
                    "item_id": item_id,
                    "output_index": index,
                    "content_index": content_index,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                })
                if text:
                    emit({
                        "type": "response.output_text.delta",
                        "item_id": item_id,
                        "output_index": index,
                        "content_index": content_index,
                        "delta": text,
                    })
                emit({
                    "type": "response.output_text.done",
                    "item_id": item_id,
                    "output_index": index,
                    "content_index": content_index,
                    "text": text,
                })
                emit({
                    "type": "response.content_part.done",
                    "item_id": item_id,
                    "output_index": index,
                    "content_index": content_index,
                    "part": {"type": "output_text", "text": text, "annotations": annotations},
                })
        elif item_type == "function_call":
            arguments = item.get("arguments", "")
            if arguments:
                emit({
                    "type": "response.function_call_arguments.delta",
                    "item_id": item_id,
                    "output_index": index,
                    "delta": arguments,
                })
            emit({
                "type": "response.function_call_arguments.done",
                "item_id": item_id,
                "output_index": index,
                "arguments": arguments,
            })

        emit({
            "type": "response.output_item.done",
            "output_index": index,
            "item": item,
        })

    emit({"type": "response.completed", "response": resp})
    parts.append("data: [DONE]\n\n")

    return "".join(parts).encode("utf-8")

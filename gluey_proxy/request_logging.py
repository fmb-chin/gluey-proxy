import json

from .config import LOG_BODY_MAX_CHARS, LOG_REQUESTS


def _emit(event: str, rid: str, payload: dict) -> None:
    if not LOG_REQUESTS:
        return
    record = {"event": f"gluey_proxy.{event}", "rid": rid}
    record.update(payload)
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


def _preview_bytes(body: bytes) -> dict:
    text = body.decode("utf-8", errors="replace")
    truncated = len(text) > LOG_BODY_MAX_CHARS
    if truncated:
        text = text[:LOG_BODY_MAX_CHARS] + "...<truncated>"
    return {
        "byte_length": len(body),
        "truncated": truncated,
        "text": text,
    }


def _log_json(rid: str, name: str, data) -> None:
    _emit(name, rid, {"data": data})


def _log_text(rid: str, name: str, text: str) -> None:
    char_length = len(text)
    truncated = len(text) > LOG_BODY_MAX_CHARS
    if truncated:
        text = text[:LOG_BODY_MAX_CHARS] + "...<truncated>"
    _emit(name, rid, {"char_length": char_length, "truncated": truncated, "text": text})


def _log_bytes(rid: str, name: str, body: bytes) -> None:
    _emit(name, rid, _preview_bytes(body))


def _log_req(rid: str, headers: dict, body: bytes) -> None:
    if not LOG_REQUESTS:
        return
    _log_json(rid, "req.headers", headers)
    _log_bytes(rid, "req.body", body)
    try:
        parsed = json.loads(body) if body else None
        if parsed is not None:
            _log_json(rid, "req.json", parsed)
    except Exception:
        pass


def _log_meta(rid: str, meta: dict) -> None:
    if not LOG_REQUESTS:
        return
    _emit("meta", rid, meta)

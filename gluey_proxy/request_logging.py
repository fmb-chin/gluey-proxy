import json

from .config import LOG_DIR, LOG_REQUESTS


def _log_req(rid: str, headers: dict, body: bytes) -> None:
    if not LOG_REQUESTS:
        return
    (LOG_DIR / f"{rid}.req.headers.json").write_text(
        json.dumps(headers, ensure_ascii=False, indent=2)
    )
    (LOG_DIR / f"{rid}.req.body").write_bytes(body)
    try:
        parsed = json.loads(body) if body else None
        if parsed is not None:
            (LOG_DIR / f"{rid}.req.json").write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2)
            )
    except Exception:
        pass


def _log_meta(rid: str, meta: dict) -> None:
    if not LOG_REQUESTS:
        return
    (LOG_DIR / f"{rid}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )

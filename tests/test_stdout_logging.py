import json
import os
import sys
from pathlib import Path

os.environ["LOG_REQUESTS"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gluey_proxy.request_logging as request_logging


def test_request_debug_logs_are_emitted_to_stdout(monkeypatch, capsys):
    monkeypatch.setattr(request_logging, "LOG_REQUESTS", True)
    monkeypatch.setattr(request_logging, "LOG_BODY_MAX_CHARS", 10)

    body = json.dumps(
        {
            "model": "test-model",
            "tools": [{"type": "web_search"}],
        }
    ).encode("utf-8")

    request_logging._log_req("rid-1", {"authorization": "Bearer test"}, body)
    request_logging._log_meta("rid-1", {"codex_web_search_converted": True})

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert [line["event"] for line in lines] == [
        "gluey_proxy.req.headers",
        "gluey_proxy.req.body",
        "gluey_proxy.req.json",
        "gluey_proxy.meta",
    ]
    assert all(line["rid"] == "rid-1" for line in lines)
    assert lines[1]["byte_length"] == len(body)
    assert lines[1]["truncated"] is True
    assert lines[2]["data"]["tools"] == [{"type": "web_search"}]
    assert lines[3]["codex_web_search_converted"] is True

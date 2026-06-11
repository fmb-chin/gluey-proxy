import gzip
import os
import sys
from pathlib import Path

import pytest

os.environ["LOG_REQUESTS"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claude_proxy import _buffer_responses_stream


class EncodedStreamResponse:
    def __init__(self, decoded: bytes):
        self.decoded = decoded
        self.encoded = gzip.compress(decoded)
        self.closed = False

    async def aiter_raw(self):
        yield self.encoded

    async def aiter_bytes(self):
        yield self.decoded

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_buffer_responses_stream_returns_decoded_utf8_sse_bytes():
    decoded = b'data: {"type":"response.completed","response":{"output":[]}}\n\n'
    response = EncodedStreamResponse(decoded)

    buffered = await _buffer_responses_stream(response)

    assert buffered == decoded
    assert buffered.decode("utf-8").startswith("data: ")
    assert response.closed is True

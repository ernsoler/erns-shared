import asyncio
import pytest
from starlette.responses import StreamingResponse

from erns_shared.http import SSEEvent, sse_stream


class TestSSEEventEncode:
    def test_plain_string(self):
        assert SSEEvent(data="hello").encode() == "data: hello\n\n"

    def test_dict_auto_serializes_to_json(self):
        event = SSEEvent(data={"text": "hi"})
        assert event.encode() == 'data: {"text": "hi"}\n\n'

    def test_list_auto_serializes_to_json(self):
        event = SSEEvent(data=[1, 2, 3])
        assert event.encode() == "data: [1, 2, 3]\n\n"

    def test_event_field(self):
        event = SSEEvent(data="ping", event="heartbeat")
        encoded = event.encode()
        assert "event: heartbeat\n" in encoded
        assert "data: ping\n" in encoded

    def test_id_field(self):
        event = SSEEvent(data="x", id="42")
        assert event.encode().startswith("id: 42\n")

    def test_retry_field(self):
        event = SSEEvent(data="x", retry=3000)
        assert "retry: 3000\n" in event.encode()

    def test_all_fields_ordering(self):
        event = SSEEvent(data="msg", event="update", id="1", retry=1000)
        lines = event.encode().split("\n")
        assert lines[0] == "id: 1"
        assert lines[1] == "event: update"
        assert lines[2] == "retry: 1000"
        assert lines[3] == "data: msg"

    def test_multiline_data(self):
        event = SSEEvent(data="line1\nline2")
        assert event.encode() == "data: line1\ndata: line2\n\n"

    def test_ends_with_double_newline(self):
        assert SSEEvent(data="x").encode().endswith("\n\n")


class TestSseStream:
    def test_returns_streaming_response(self):
        async def gen():
            yield SSEEvent(data="x")

        response = sse_stream(gen())
        assert isinstance(response, StreamingResponse)

    def test_media_type(self):
        async def gen():
            yield SSEEvent(data="x")

        assert sse_stream(gen()).media_type == "text/event-stream"

    def test_cache_control_header(self):
        async def gen():
            yield SSEEvent(data="x")

        assert sse_stream(gen()).headers["cache-control"] == "no-cache"

    def test_streams_encoded_events(self):
        async def gen():
            yield SSEEvent(data="hello", event="msg")
            yield SSEEvent(data={"k": "v"})

        async def collect():
            chunks = []
            async for chunk in sse_stream(gen()).body_iterator:
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(collect())
        assert chunks[0] == "event: msg\ndata: hello\n\n"
        assert chunks[1] == 'data: {"k": "v"}\n\n'

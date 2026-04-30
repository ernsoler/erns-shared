import json
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional

from starlette.responses import StreamingResponse


@dataclass
class SSEEvent:
    """A single Server-Sent Event.

    Fields follow the SSE spec (https://html.spec.whatwg.org/multipage/server-sent-events.html):

    - data: the payload. Dicts and lists are automatically JSON-serialized; everything
            else is converted to string. Multi-line strings produce one `data:` line each.
    - event: optional event type. Clients can listen for specific types via
             `addEventListener("delta", handler)`. Defaults to "message" if omitted.
    - id: optional event ID. The browser uses this to resume from the last seen event
          when reconnecting (sent back as `Last-Event-ID` header).
    - retry: optional reconnection delay in milliseconds. Tells the browser how long
             to wait before reconnecting after a dropped connection.

    Example wire format for SSEEvent(data={"text": "hi"}, event="delta", id="1"):

        id: 1
        event: delta
        data: {"text": "hi"}

        (blank line signals end of event)
    """

    data: Any
    event: Optional[str] = None
    id: Optional[str] = None
    retry: Optional[int] = None

    def encode(self) -> str:
        """Serialize this event to the SSE wire format."""
        lines = []
        if self.id is not None:
            lines.append(f"id: {self.id}")
        if self.event is not None:
            lines.append(f"event: {self.event}")
        if self.retry is not None:
            lines.append(f"retry: {self.retry}")
        data_str = (
            json.dumps(self.data)
            if isinstance(self.data, (dict, list))
            else str(self.data)
        )
        for line in data_str.splitlines():
            lines.append(f"data: {line}")
        return "\n".join(lines) + "\n\n"


def sse_stream(source: AsyncGenerator[SSEEvent, None]) -> StreamingResponse:
    """Wrap an async generator of SSEEvents into a FastAPI/Starlette StreamingResponse.

    The response sets the correct headers so the browser (or any SSE client) keeps
    the connection open and receives events as they are yielded.

    Args:
        source: an async generator that yields SSEEvent instances.

    Returns:
        A StreamingResponse with media_type "text/event-stream".

    Example::

        @app.post("/chat")
        async def chat(request: ChatRequest):
            async def generate():
                async for chunk in claude_client.stream(...):
                    yield SSEEvent(data={"text": chunk.text}, event="delta")
                yield SSEEvent(data="[DONE]", event="done")

            return sse_stream(generate())
    """

    async def _generate() -> AsyncGenerator[str, None]:
        async for event in source:
            yield event.encode()

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",       # prevent proxies from buffering the stream
            "X-Accel-Buffering": "no",          # disable nginx response buffering
        },
    )

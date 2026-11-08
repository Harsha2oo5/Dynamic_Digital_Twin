"""Fan-out of live telemetry events to WebSocket clients.

`publish` may be called from any thread (MQTT network thread, simulator thread, request worker threads); events
are handed to the event loop with call_soon_threadsafe. Each client has a bounded queue: a slow client loses its
oldest events instead of blocking ingestion, and is told so (`dropped` counter).
"""

import asyncio
import logging
import threading

logger = logging.getLogger(__name__)


class Subscriber:
    def __init__(self, maxsize: int = 1000):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def put(self, event: dict) -> None:
        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:
                pass
        self.queue.put_nowait(event)


class TelemetryHub:
    def __init__(self):
        self._subs: set[Subscriber] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> Subscriber:
        sub = Subscriber()
        with self._lock:
            self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        with self._lock:
            self._subs.discard(sub)

    @property
    def clients(self) -> int:
        return len(self._subs)

    def publish(self, event: dict) -> None:
        loop = self._loop
        if loop is None or loop.is_closed() or not self._subs:
            return
        with self._lock:
            subs = list(self._subs)

        def deliver():
            for s in subs:
                s.put(event)

        try:
            loop.call_soon_threadsafe(deliver)
        except RuntimeError:  # loop shutting down
            pass

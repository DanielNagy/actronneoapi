"""Shared realtime transport primitives.

The realtime clients for Neo and Que will live in separate modules, but they
share a small set of common event types and a protocol for the consumer-facing
surface.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

_LOGGER = logging.getLogger(__name__)

DEFAULT_EVENT_QUEUE_MAXSIZE = 256

# A valid JSON escape, or an apostrophe escape that JSON does not allow. The
# valid alternative is listed first so it consumes both characters of a "\\"
# pair, which keeps a legitimately escaped backslash before an apostrophe from
# being misread as an invalid escape.
_ESCAPE_SEQUENCE = re.compile(r"""\\(?:u[0-9a-fA-F]{4}|["\\/bfnrtu])|\\'""")


class RealtimeTransportType(str, Enum):
    """Supported realtime transport families."""

    MQTT = "mqtt"
    SIGNALR = "signalr"


class RealtimeConnectionState(str, Enum):
    """Connection state shared by all realtime transports."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class RealtimeEventKind(str, Enum):
    """Kinds of realtime events emitted by a transport."""

    CONNECTION = "connection"
    MESSAGE = "message"


@dataclass(frozen=True, slots=True)
class RealtimeEvent:
    """Base realtime event payload.

    Attributes:
        transport: Transport family that emitted the event.
        kind: Event category.
    """

    transport: RealtimeTransportType
    kind: RealtimeEventKind


@dataclass(frozen=True, slots=True)
class RealtimeConnectionEvent(RealtimeEvent):
    """Connection state transition emitted by a realtime transport."""

    state: RealtimeConnectionState
    previous_state: RealtimeConnectionState | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RealtimeMessage(RealtimeEvent):
    """Message received from a realtime transport."""

    topic: str
    payload: dict[str, Any]
    raw_payload: bytes | None = None
    domain_model: Any | None = None


@dataclass(frozen=True, slots=True)
class RealtimeConnectionDetails:
    """Connection details required to open a realtime transport."""

    endpoint: str
    port: int
    protocol: str
    user_id: str

    def __post_init__(self) -> None:
        """Validate the connection settings."""
        if not self.endpoint.strip():
            raise ValueError("endpoint cannot be empty")
        if self.port <= 0:
            raise ValueError("port must be greater than zero")
        if not self.protocol.strip():
            raise ValueError("protocol cannot be empty")
        if not self.user_id.strip():
            raise ValueError("user_id cannot be empty")

    @property
    def uses_tls(self) -> bool:
        """Return whether the transport should use TLS."""
        return self.protocol.strip().lower() in {"ssl", "tls", "mqtts"}

    @property
    def scheme(self) -> str:
        """Return the transport URI scheme."""
        return "ssl" if self.uses_tls else "tcp"


def repair_apostrophe_escapes(text: str) -> str:
    r"""Rewrite JavaScript-style ``\'`` escapes as plain apostrophes.

    The Actron cloud escapes apostrophes inside string values, which JSON does
    not permit: RFC 8259 allows only ``" \ / b f n r t`` and ``uXXXX`` after a
    backslash, and an apostrophe needs no escaping at all. A zone named
    "Kurt's Office" therefore makes the whole payload unparseable. The escaping
    follows ``addslashes`` semantics, where ``\'`` denotes a literal
    apostrophe, so that is what it is restored to.
    """

    def _replace(match: re.Match[str]) -> str:
        # Valid escapes are matched only so they are stepped over intact.
        return "'" if match.group(0) == "\\'" else match.group(0)

    return _ESCAPE_SEQUENCE.sub(_replace, text)


def loads_repairing_escapes(text: str) -> Any:
    """Parse JSON, retrying once with the vendor's invalid escapes corrected.

    A well-formed payload is parsed strictly and never inspected further; the
    repair runs only after a failure. If the payload is still unparseable
    afterwards, that second failure is what propagates: it points at the defect
    that remains, whereas the original error points at an apostrophe escape
    this function has already dealt with.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = repair_apostrophe_escapes(text)
        if repaired == text:
            raise
    payload = json.loads(repaired)
    _LOGGER.debug("Parsed a payload containing invalid apostrophe escapes")
    return payload


def new_event_queue(
    maxsize: int = DEFAULT_EVENT_QUEUE_MAXSIZE,
) -> asyncio.Queue[RealtimeEvent]:
    """Create the bounded queue a transport uses to buffer realtime events."""
    return asyncio.Queue(maxsize=maxsize)


def put_event_dropping_oldest(
    queue: asyncio.Queue[RealtimeEvent],
    event: RealtimeEvent,
) -> None:
    """Queue an event without blocking, discarding the oldest entry when full.

    Transports emit events whether or not anything consumes ``iter_events``,
    and the Neo heartbeat topic produces them continuously. Bounding the queue
    keeps recent events available to a late consumer without retaining every
    event for the life of the process, and dropping rather than blocking keeps
    a stalled consumer from stalling the transport.
    """
    while True:
        try:
            queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            try:
                dropped = queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - drained concurrently
                continue
            _LOGGER.debug(
                "Realtime event queue is full; dropped the oldest %s",
                type(dropped).__name__,
            )


@runtime_checkable
class RealtimeClient(Protocol):
    """Protocol shared by platform-specific realtime clients."""

    transport_type: RealtimeTransportType

    async def connect(self) -> None:
        """Connect the realtime transport."""

    async def disconnect(self) -> None:
        """Disconnect the realtime transport."""

    async def subscribe(self, topic: str) -> None:
        """Subscribe to a transport-specific topic."""

    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a transport-specific topic."""

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        """Publish a transport-specific payload."""

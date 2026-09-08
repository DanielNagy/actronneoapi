"""Tests for the realtime package scaffold."""

from __future__ import annotations

import json

import pytest

from actron_neo_api.rt import (
    MQTTRTClient,
    RealtimeClient,
    RealtimeConnectionDetails,
    RealtimeConnectionEvent,
    RealtimeConnectionState,
    RealtimeEventKind,
    RealtimeMessage,
    RealtimeTransportType,
    SignalRRTClient,
)
from actron_neo_api.rt.base import loads_repairing_escapes, repair_apostrophe_escapes


def test_realtime_package_exports() -> None:
    """The rt package should expose the shared transport primitives."""
    assert RealtimeTransportType.MQTT.value == "mqtt"
    assert RealtimeTransportType.SIGNALR.value == "signalr"
    assert RealtimeConnectionState.CONNECTED.value == "connected"
    assert RealtimeEventKind.MESSAGE.value == "message"
    assert RealtimeConnectionEvent.__name__ == "RealtimeConnectionEvent"
    assert RealtimeMessage.__name__ == "RealtimeMessage"
    assert RealtimeClient.__name__ == "RealtimeClient"
    assert RealtimeConnectionDetails.__name__ == "RealtimeConnectionDetails"
    assert MQTTRTClient.__name__ == "MQTTRTClient"
    assert SignalRRTClient.__name__ == "SignalRRTClient"


class TestApostropheEscapeRepair:
    r"""The Actron cloud escapes apostrophes as ``\'``, which JSON rejects."""

    def test_valid_documents_are_parsed_unchanged(self) -> None:
        """A well-formed payload must never be inspected or rewritten."""
        for document in (
            '{"a":"say \\"hi\\""}',
            '{"a":"tab\\there"}',
            '{"a":"\\u00e9"}',
            '{"a":"back\\\\slash"}',
            '{"a":"no escapes at all"}',
        ):
            assert loads_repairing_escapes(document) == json.loads(document)

    def test_escaped_apostrophe_is_restored(self) -> None:
        """A zone title with an apostrophe must survive with the apostrophe intact."""
        payload = '{"NV_Title":"Kurt\\\'s Office"}'

        with pytest.raises(json.JSONDecodeError):
            json.loads(payload)

        assert loads_repairing_escapes(payload) == {"NV_Title": "Kurt's Office"}

    def test_escaped_backslash_before_apostrophe_is_preserved(self) -> None:
        r"""A legitimate ``\\`` must not be misread as the start of an escape."""
        payload = '{"a":"path\\\\","b":"it\\\'s"}'

        assert loads_repairing_escapes(payload) == {"a": "path\\", "b": "it's"}

    def test_unrelated_syntax_error_raises_untouched(self) -> None:
        """A payload with nothing to repair fails on its original error."""
        with pytest.raises(json.JSONDecodeError) as error:
            loads_repairing_escapes('{"a": }')

        assert error.value.doc == '{"a": }'

    def test_payload_still_invalid_after_repair_reports_the_remaining_defect(self) -> None:
        """The reported failure must be the one still unresolved.

        The original error points at the apostrophe escape that was repaired,
        so surfacing it would send a reader after a byte that is already
        handled.
        """
        payload = '{"a":"it\\\'s","b":"bad\\q"}'

        with pytest.raises(json.JSONDecodeError) as error:
            loads_repairing_escapes(payload)

        reported = error.value.doc[error.value.pos : error.value.pos + 2]
        assert reported == "\\q"

    def test_repair_is_a_no_op_for_text_without_invalid_escapes(self) -> None:
        """The repair itself leaves valid escapes alone."""
        text = '{"a":"say \\"hi\\"","b":"back\\\\slash","c":"\\u00e9"}'

        assert repair_apostrophe_escapes(text) == text

"""Regression tests for enabled-zone state after command delivery."""

import asyncio
from collections.abc import Iterator
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from actron_neo_api import ActronAirAPI, ActronAirStatus
from actron_neo_api.exceptions import ActronAirAPIError
from actron_neo_api.rt.base import RealtimeEventKind, RealtimeMessage, RealtimeTransportType


@pytest.fixture
def debounce_seconds() -> float:
    """Use a short window for concurrent zone commands."""
    return 0.01


@pytest.fixture
def zone_request() -> MagicMock:
    """Mock the HTTP boundary without bypassing command delivery."""
    response = MagicMock()
    response.status = 200
    response.json = AsyncMock(return_value={"success": True})
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=response)
    context.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=context)


@pytest.fixture
def zone_api(
    debounce_seconds: float, zone_request: MagicMock, mock_oauth: MagicMock
) -> Iterator[ActronAirAPI]:
    """Create a system with the live reproduction's initial zone state."""
    api = ActronAirAPI(debounce_seconds=debounce_seconds)
    api._initialized = True
    api.oauth2_auth = mock_oauth
    session = MagicMock()
    session.closed = False
    session.request = zone_request
    api._session = session
    api.state_manager.process_status_update(
        "abc123",
        ActronAirStatus(
            lastKnownState={
                "UserAirconSettings": {
                    "EnabledZones": [True, False, True, False, True, False, False, False],
                    "Mode": "COOL",
                },
                "RemoteZoneInfo": [{} for _ in range(8)],
            }
        ),
    )
    with patch.object(api, "_get_system_link", return_value="commands"):
        yield api


@pytest.mark.asyncio
async def test_grouped_enable_updates_merged_state(
    zone_api: ActronAirAPI, zone_request: MagicMock
) -> None:
    """Both callers see the transmitted state before any realtime update."""
    status = zone_api.state_manager.get_status("abc123")
    assert status is not None
    expected = [True, False, True, False, True, True, True, False]

    await asyncio.gather(status.zones[5].enable(), status.zones[6].enable())

    zone_request.assert_called_once()
    payload = zone_request.call_args.kwargs["json"]
    assert payload == {
        "command": {"type": "set-settings", "UserAirconSettings.EnabledZones": expected}
    }
    assert status.user_aircon_settings.enabled_zones == expected
    assert status.last_known_state["UserAirconSettings"]["EnabledZones"] == expected
    assert status.last_known_state["UserAirconSettings"]["Mode"] == "COOL"
    assert (
        status.user_aircon_settings.enabled_zones
        is not payload["command"]["UserAirconSettings.EnabledZones"]
    )

    updated = await zone_api._merge_mqtt_status_change(
        "abc123", {"lastKnownState": {"UserAirconSettings": {"FanMode": "LOW"}}}
    )
    assert updated is not None
    assert updated.user_aircon_settings.enabled_zones == expected
    assert updated.last_known_state["UserAirconSettings"]["EnabledZones"] == expected


@pytest.mark.parametrize(
    ("debounce_seconds", "zone_ids"),
    [
        pytest.param(0.01, (5, 6), id="coalesced"),
        pytest.param(0, (5,), id="direct"),
    ],
)
@pytest.mark.asyncio
async def test_enable_failure_preserves_state(
    zone_api: ActronAirAPI, zone_request: MagicMock, zone_ids: tuple[int, ...]
) -> None:
    """Failed delivery leaves parsed and raw state untouched."""
    status = zone_api.state_manager.get_status("abc123")
    assert status is not None
    original = status.model_dump()
    response = zone_request.return_value.__aenter__.return_value
    response.status = 500
    response.text = AsyncMock(return_value="Delivery failed")

    results = await asyncio.gather(
        *(status.zones[zone_id].enable() for zone_id in zone_ids), return_exceptions=True
    )

    zone_request.assert_called_once()
    assert len(results) == len(zone_ids)
    assert all(isinstance(result, ActronAirAPIError) for result in results)
    assert status.model_dump() == original


@pytest.mark.parametrize(
    "debounce_seconds", [pytest.param(0, id="direct"), pytest.param(0.01, id="coalesced")]
)
@pytest.mark.asyncio
async def test_single_zone_disable_updates_state(
    zone_api: ActronAirAPI, zone_request: MagicMock
) -> None:
    """Single-zone commands update both representations with or without batching."""
    status = zone_api.state_manager.get_status("abc123")
    assert status is not None
    expected = [False, False, True, False, True, False, False, False]

    await status.zones[0].enable(False)

    zone_request.assert_called_once()
    assert (
        zone_request.call_args.kwargs["json"]["command"]["UserAirconSettings.EnabledZones"]
        == expected
    )
    assert status.user_aircon_settings.enabled_zones == expected
    assert status.last_known_state["UserAirconSettings"]["EnabledZones"] == expected


@pytest.mark.asyncio
async def test_delivery_updates_replaced_status(
    zone_api: ActronAirAPI, zone_request: MagicMock
) -> None:
    """A status replacement during delivery receives the successful zone update."""
    original = zone_api.state_manager.get_status("abc123")
    assert original is not None
    replacement = ActronAirStatus.model_validate(original.model_dump(by_alias=True))
    response = zone_request.return_value.__aenter__.return_value

    async def receive_response() -> MagicMock:
        zone_api.state_manager.process_status_update("abc123", replacement)
        return response

    zone_request.return_value.__aenter__.side_effect = receive_response

    await original.zones[5].enable()

    expected = [True, False, True, False, True, True, False, False]
    assert zone_api.state_manager.get_status("abc123") is replacement
    assert replacement.user_aircon_settings.enabled_zones == expected
    assert replacement.last_known_state["UserAirconSettings"]["EnabledZones"] == expected


async def _queue_zone_call(
    status: ActronAirStatus, zone_id: int, enabled: bool
) -> asyncio.Task[None]:
    """Let a caller enqueue before the test explicitly flushes its batch."""
    task = asyncio.create_task(status.zones[zone_id].enable(enabled))
    await asyncio.sleep(0)
    return task


@pytest.mark.parametrize("debounce_seconds", [3600])
@pytest.mark.parametrize(
    ("requests", "expected"),
    [
        pytest.param(
            ((0, False), (0, True)),
            [True, False, True, False, True, False, False, False],
            id="off-then-on-at-baseline",
        ),
        pytest.param(
            ((5, True), (5, False)),
            [True, False, True, False, True, False, False, False],
            id="on-then-off-at-baseline",
        ),
        pytest.param(
            ((0, False), (5, True), (0, True), (5, False), (6, True), (2, False)),
            [True, False, False, False, True, False, True, False],
            id="mixed-zones-last-request-wins",
        ),
    ],
)
@pytest.mark.asyncio
async def test_explicit_zone_intent_last_request_wins(
    zone_api: ActronAirAPI,
    zone_request: MagicMock,
    requests: tuple[tuple[int, bool], ...],
    expected: list[bool],
) -> None:
    """A request matching the original baseline still overrides earlier intent."""
    status = zone_api.state_manager.get_status("abc123")
    assert status is not None
    tasks = []
    for zone_id, enabled in requests:
        tasks.append(await _queue_zone_call(status, zone_id, enabled))

    await zone_api._coalescer.flush_all()
    await asyncio.gather(*tasks)

    zone_request.assert_called_once()
    assert zone_request.call_args.kwargs["json"]["command"] == {
        "type": "set-settings",
        "UserAirconSettings.EnabledZones": expected,
    }
    assert status.user_aircon_settings.enabled_zones == expected
    assert status.last_known_state["UserAirconSettings"]["EnabledZones"] == expected


@pytest.mark.parametrize(
    "debounce_seconds", [pytest.param(3600, id="coalesced"), pytest.param(0, id="direct")]
)
@pytest.mark.parametrize(
    ("first_http_status", "first_result_type", "expected"),
    [
        pytest.param(
            200,
            type(None),
            [True, False, True, False, True, True, True, False],
            id="first-succeeds",
        ),
        pytest.param(
            500,
            ActronAirAPIError,
            [True, False, True, False, True, False, True, False],
            id="first-fails",
        ),
    ],
)
@pytest.mark.asyncio
async def test_same_system_batches_wait_for_delivery(
    zone_api: ActronAirAPI,
    zone_request: MagicMock,
    first_http_status: int,
    first_result_type: type[None] | type[ActronAirAPIError],
    expected: list[bool],
) -> None:
    """The next batch builds only after the previous delivery and state update."""
    status = zone_api.state_manager.get_status("abc123")
    assert status is not None
    initial = status.model_dump()
    entered = asyncio.Event()
    release = asyncio.Event()
    normal_context = zone_request.return_value
    first_response = MagicMock()
    first_response.status = first_http_status
    first_response.json = AsyncMock(return_value={"success": True})
    first_response.text = AsyncMock(return_value="Delivery failed")

    async def first_delivery() -> MagicMock:
        entered.set()
        await release.wait()
        return first_response

    first_context = MagicMock()
    first_context.__aenter__ = AsyncMock(side_effect=first_delivery)
    first_context.__aexit__ = AsyncMock(return_value=False)
    zone_request.side_effect = [first_context, normal_context]

    first = await _queue_zone_call(status, 5, True)
    first_flush = asyncio.create_task(zone_api._coalescer._flush("abc123"))
    await asyncio.wait_for(entered.wait(), timeout=1)
    second = await _queue_zone_call(status, 6, True)
    second_flush = asyncio.create_task(zone_api._coalescer._flush("abc123"))
    await asyncio.sleep(0)

    try:
        assert zone_request.call_count == 1
        assert not second.done()
        assert status.model_dump() == initial
    finally:
        release.set()
        results = await asyncio.gather(first, second, return_exceptions=True)
        await asyncio.gather(first_flush, second_flush)

    assert isinstance(results[0], first_result_type)
    assert results[1] is None
    assert zone_request.call_count == 2
    assert zone_request.call_args_list[0].kwargs["json"]["command"][
        "UserAirconSettings.EnabledZones"
    ] == [True, False, True, False, True, True, False, False]
    assert (
        zone_request.call_args_list[1].kwargs["json"]["command"]["UserAirconSettings.EnabledZones"]
        == expected
    )
    assert status.user_aircon_settings.enabled_zones == expected
    assert status.last_known_state["UserAirconSettings"]["EnabledZones"] == expected


@pytest.mark.parametrize(
    "debounce_seconds", [pytest.param(3600, id="coalesced"), pytest.param(0, id="direct")]
)
@pytest.mark.asyncio
async def test_other_system_progresses_while_zone_delivery_waits(
    zone_api: ActronAirAPI, zone_request: MagicMock
) -> None:
    """Holding one system's zone lock does not block another system."""
    first_status = zone_api.state_manager.get_status("abc123")
    assert first_status is not None
    other = ActronAirStatus.model_validate(first_status.model_dump(by_alias=True))
    zone_api.state_manager.process_status_update("other", other)
    entered = asyncio.Event()
    release = asyncio.Event()
    normal_context = zone_request.return_value
    response = normal_context.__aenter__.return_value

    async def first_delivery() -> MagicMock:
        entered.set()
        await release.wait()
        return response

    first_context = MagicMock()
    first_context.__aenter__ = AsyncMock(side_effect=first_delivery)
    first_context.__aexit__ = AsyncMock(return_value=False)
    zone_request.side_effect = [first_context, normal_context]
    first = await _queue_zone_call(first_status, 5, True)
    first_flush = asyncio.create_task(zone_api._coalescer._flush("abc123"))
    await asyncio.wait_for(entered.wait(), timeout=1)

    try:
        second = await _queue_zone_call(other, 6, True)
        await asyncio.wait_for(zone_api._coalescer._flush("other"), timeout=1)
        await asyncio.wait_for(second, timeout=1)
        assert not first.done()
        assert first_status.user_aircon_settings.enabled_zones[5] is False
        assert other.user_aircon_settings.enabled_zones[6] is True
        assert other.last_known_state["UserAirconSettings"]["EnabledZones"][6] is True
    finally:
        release.set()
        await asyncio.gather(first, first_flush)

    assert zone_request.call_count == 2
    assert first_status.user_aircon_settings.enabled_zones[5] is True
    assert first_status.user_aircon_settings.enabled_zones[6] is False
    assert other.user_aircon_settings.enabled_zones[5] is False


@pytest.mark.asyncio
async def test_zone_from_uncached_status_remains_usable(
    zone_api: ActronAirAPI, zone_request: MagicMock
) -> None:
    """Manually attached status models can initialize the command state cache."""
    status = zone_api.state_manager.status.pop("abc123")

    await status.zones[5].enable()

    zone_request.assert_called_once()
    assert zone_api.state_manager.get_status("abc123") is status
    assert status.user_aircon_settings.enabled_zones[5] is True
    assert status.last_known_state["UserAirconSettings"]["EnabledZones"][5] is True


@pytest.mark.parametrize(
    ("debounce_seconds", "expected"),
    [
        pytest.param(
            3600,
            [True, False, True, False, True, False, True, False],
            id="coalesced-full-replacement",
        ),
        pytest.param(
            0,
            [True, False, True, False, True, False, True, False],
            id="direct-full-replacement",
        ),
    ],
)
@pytest.mark.asyncio
async def test_raw_zone_batch_waits_without_blocking_scalar_commands(
    zone_api: ActronAirAPI, zone_request: MagicMock, expected: list[bool]
) -> None:
    """Legacy array callers share zone ordering while scalar-only sends proceed."""
    status = zone_api.state_manager.get_status("abc123")
    assert status is not None
    entered = asyncio.Event()
    release = asyncio.Event()
    normal_context = zone_request.return_value
    response = normal_context.__aenter__.return_value

    async def first_delivery() -> MagicMock:
        entered.set()
        await release.wait()
        return response

    first_context = MagicMock()
    first_context.__aenter__ = AsyncMock(side_effect=first_delivery)
    first_context.__aexit__ = AsyncMock(return_value=False)
    zone_request.side_effect = [first_context, normal_context, normal_context]
    first = await _queue_zone_call(status, 5, True)
    first_flush = asyncio.create_task(zone_api._coalescer._flush("abc123"))
    await asyncio.wait_for(entered.wait(), timeout=1)

    raw_zones = [True, False, True, False, True, False, True, False]
    second = asyncio.create_task(
        zone_api.send_command(
            "ABC123",
            {"command": {"type": "set-settings", "UserAirconSettings.EnabledZones": raw_zones}},
        )
    )
    await asyncio.sleep(0)
    second_flush = asyncio.create_task(zone_api._coalescer._flush("abc123"))
    await asyncio.sleep(0)

    try:
        assert zone_request.call_count == 1
        scalar = asyncio.create_task(
            zone_api.send_command(
                "abc123", {"command": {"type": "set-settings", "UserAirconSettings.Mode": "HEAT"}}
            )
        )
        await asyncio.sleep(0)
        await asyncio.wait_for(zone_api._coalescer._flush("abc123"), timeout=1)
        await asyncio.wait_for(scalar, timeout=1)
        assert not first.done()
        assert not second.done()
        assert zone_request.call_count == 2
        assert zone_request.call_args.kwargs["json"]["command"] == {
            "type": "set-settings",
            "UserAirconSettings.Mode": "HEAT",
        }
    finally:
        release.set()
        await asyncio.gather(first, second, first_flush, second_flush)

    assert zone_request.call_count == 3
    assert (
        zone_request.call_args.kwargs["json"]["command"]["UserAirconSettings.EnabledZones"]
        == expected
    )
    assert status.user_aircon_settings.enabled_zones == expected
    assert status.last_known_state["UserAirconSettings"]["EnabledZones"] == expected


@pytest.mark.parametrize(
    "zone_id", [pytest.param(-1, id="negative"), pytest.param(8, id="out-of-range")]
)
@pytest.mark.asyncio
async def test_invalid_zone_intent_does_not_send(
    zone_api: ActronAirAPI, zone_request: MagicMock, zone_id: int
) -> None:
    """Reject invalid zone indices before queueing any changes."""
    status = zone_api.state_manager.get_status("abc123")
    assert status is not None
    zone = status.zones[0]
    zone.zone_id = zone_id

    with pytest.raises(ValueError, match="out of range"):
        await zone.enable()

    zone_request.assert_not_called()


@pytest.mark.asyncio
async def test_fetched_status_is_authoritative_for_zone_commands(
    zone_api: ActronAirAPI, zone_request: MagicMock
) -> None:
    """A command from even an old zone model preserves newer fetched zone state."""
    original = zone_api.state_manager.get_status("abc123")
    assert original is not None
    payload = {"lastKnownState": deepcopy(original.last_known_state)}
    payload["lastKnownState"]["UserAirconSettings"]["EnabledZones"][5] = True
    response = zone_request.return_value.__aenter__.return_value
    response.json.return_value = payload
    observer = MagicMock()
    zone_api.state_manager.add_observer(observer)

    fetched = await zone_api.get_ac_status("ABC123")

    assert zone_api.state_manager.get_status("abc123") is fetched
    assert fetched is not original
    observer.assert_called_once_with("abc123", fetched.last_known_state)
    assert fetched.user_aircon_settings.enabled_zones[5] is True
    assert original.user_aircon_settings.enabled_zones[5] is False

    await original.zones[6].enable()

    expected = [True, False, True, False, True, True, True, False]
    assert (
        zone_request.call_args.kwargs["json"]["command"]["UserAirconSettings.EnabledZones"]
        == expected
    )
    assert fetched.user_aircon_settings.enabled_zones == expected
    assert fetched.last_known_state["UserAirconSettings"]["EnabledZones"] == expected
    assert observer.call_count == 1


@pytest.mark.asyncio
async def test_polled_status_notifies_once(zone_api: ActronAirAPI, zone_request: MagicMock) -> None:
    """Polling stores the fetched model without a second observer notification."""
    original = zone_api.state_manager.get_status("abc123")
    assert original is not None
    zone_request.return_value.__aenter__.return_value.json.return_value = {
        "lastKnownState": deepcopy(original.last_known_state)
    }
    observer = MagicMock()
    zone_api.state_manager.add_observer(observer)

    result = await zone_api.update_status("abc123")

    current = zone_api.state_manager.get_status("abc123")
    assert current is not None
    assert current is not original
    assert result["abc123"] is current
    observer.assert_called_once_with("abc123", current.last_known_state)
    zone_request.assert_called_once()


@pytest.mark.asyncio
async def test_realtime_fetch_notifies_each_subscriber_once(
    zone_api: ActronAirAPI, zone_request: MagicMock
) -> None:
    """Realtime invalidation fetches once, then dispatches one push callback."""
    original = zone_api.state_manager.get_status("abc123")
    assert original is not None
    zone_request.return_value.__aenter__.return_value.json.return_value = {
        "lastKnownState": deepcopy(original.last_known_state)
    }
    observer = MagicMock()
    push_callback = MagicMock()
    zone_api.state_manager.add_observer(observer)
    zone_api.subscribe_system_updates("abc123", push_callback)
    event = RealtimeMessage(
        transport=RealtimeTransportType.MQTT,
        kind=RealtimeEventKind.MESSAGE,
        topic="actron-cloud/u/neo/abc123/mwc/status-change",
        payload={},
    )

    await zone_api._handle_realtime_event(event)

    current = zone_api.state_manager.get_status("abc123")
    assert current is not None
    assert current is not original
    observer.assert_called_once_with("abc123", current.last_known_state)
    push_callback.assert_called_once_with(current)
    zone_request.assert_called_once()


@pytest.mark.asyncio
async def test_failed_fetch_preserves_cache_and_does_not_notify(
    zone_api: ActronAirAPI, zone_request: MagicMock
) -> None:
    """Unsuccessful fetches do not replace the authoritative status."""
    original = zone_api.state_manager.get_status("abc123")
    observer = MagicMock()
    zone_api.state_manager.add_observer(observer)
    response = zone_request.return_value.__aenter__.return_value
    response.status = 500
    response.text = AsyncMock(return_value="Fetch failed")

    with pytest.raises(ActronAirAPIError):
        await zone_api.get_ac_status("abc123")

    assert zone_api.state_manager.get_status("abc123") is original
    observer.assert_not_called()


@pytest.mark.parametrize("debounce_seconds", [3600])
@pytest.mark.asyncio
async def test_replacement_survives_explicit_change_that_initializes_cache(
    zone_api: ActronAirAPI, zone_request: MagicMock
) -> None:
    """Seeding the cache must not discard an already queued replacement."""
    status = zone_api.state_manager.status.pop("abc123")
    zones = [True, False, True, False, True, True, False, False]
    replacement = asyncio.create_task(
        zone_api.send_command(
            "abc123",
            {"command": {"type": "set-settings", "UserAirconSettings.EnabledZones": zones}},
        )
    )
    await asyncio.sleep(0)
    explicit = await _queue_zone_call(status, 6, True)

    await zone_api._coalescer.flush_all()
    await asyncio.gather(replacement, explicit)

    expected = [True, False, True, False, True, True, True, False]
    zone_request.assert_called_once()
    assert (
        zone_request.call_args.kwargs["json"]["command"]["UserAirconSettings.EnabledZones"]
        == expected
    )
    assert status.user_aircon_settings.enabled_zones == expected
    assert status.last_known_state["UserAirconSettings"]["EnabledZones"] == expected


@pytest.mark.parametrize("debounce_seconds", [0.01])
@pytest.mark.asyncio
async def test_debounce_timer_queues_behind_blocked_delivery(
    zone_api: ActronAirAPI, zone_request: MagicMock
) -> None:
    """Real debounce callbacks seal a second batch while the first send waits."""
    status = zone_api.state_manager.get_status("abc123")
    assert status is not None
    entered = asyncio.Event()
    release = asyncio.Event()
    second_timer_fired = asyncio.Event()
    normal_context = zone_request.return_value
    response = normal_context.__aenter__.return_value

    async def first_delivery() -> MagicMock:
        entered.set()
        await release.wait()
        return response

    first_context = MagicMock()
    first_context.__aenter__ = AsyncMock(side_effect=first_delivery)
    first_context.__aexit__ = AsyncMock(return_value=False)
    zone_request.side_effect = [first_context, normal_context]
    flush_events = iter((asyncio.Event(), second_timer_fired))
    original_flush = zone_api._coalescer._flush

    async def observe_flush(serial: str) -> None:
        next(flush_events).set()
        await original_flush(serial)

    with patch.object(zone_api._coalescer, "_flush", side_effect=observe_flush):
        first = await _queue_zone_call(status, 5, True)
        await asyncio.wait_for(entered.wait(), timeout=2)
        second = await _queue_zone_call(status, 6, True)
        try:
            await asyncio.wait_for(second_timer_fired.wait(), timeout=2)
            assert zone_request.call_count == 1
            assert not second.done()
        finally:
            release.set()
            await asyncio.gather(first, second)

    expected = [True, False, True, False, True, True, True, False]
    assert zone_request.call_count == 2
    assert (
        zone_request.call_args.kwargs["json"]["command"]["UserAirconSettings.EnabledZones"]
        == expected
    )
    assert status.user_aircon_settings.enabled_zones == expected
    assert status.last_known_state["UserAirconSettings"]["EnabledZones"] == expected


@pytest.mark.parametrize(
    "debounce_seconds", [pytest.param(0.01, id="coalesced"), pytest.param(0, id="direct")]
)
@pytest.mark.parametrize(
    "cancel_index",
    [pytest.param(0, id="cancel-in-flight-caller"), pytest.param(1, id="cancel-waiting-caller")],
)
@pytest.mark.asyncio
async def test_cancelled_caller_does_not_interrupt_delivery_or_shutdown(
    zone_api: ActronAirAPI, zone_request: MagicMock, cancel_index: int
) -> None:
    """Cancellation ends the wait; close still drains accepted zone commands."""
    status = zone_api.state_manager.get_status("abc123")
    assert status is not None
    session = zone_api._session
    assert session is not None
    entered = asyncio.Event()
    release = asyncio.Event()
    normal_context = zone_request.return_value
    response = normal_context.__aenter__.return_value

    async def first_delivery() -> MagicMock:
        entered.set()
        await release.wait()
        return response

    first_context = MagicMock()
    first_context.__aenter__ = AsyncMock(side_effect=first_delivery)
    first_context.__aexit__ = AsyncMock(return_value=False)
    zone_request.side_effect = [first_context, normal_context]
    first = await _queue_zone_call(status, 5, True)
    await asyncio.wait_for(entered.wait(), timeout=2)
    second = await _queue_zone_call(status, 6, True)
    callers = (first, second)
    callers[cancel_index].cancel()
    with pytest.raises(asyncio.CancelledError):
        await callers[cancel_index]

    with patch.object(session, "close", new_callable=AsyncMock) as close_session:
        closing = asyncio.create_task(zone_api.close())
        await asyncio.sleep(0)
        try:
            assert not closing.done()
            assert zone_request.call_count == 1
            close_session.assert_not_awaited()
        finally:
            release.set()
            results = await asyncio.gather(*callers, return_exceptions=True)
            await asyncio.wait_for(closing, timeout=2)
        close_session.assert_awaited_once()

    assert isinstance(results[cancel_index], asyncio.CancelledError)
    assert results[1 - cancel_index] is None
    assert zone_request.call_count == 2
    expected = [True, False, True, False, True, True, True, False]
    assert status.user_aircon_settings.enabled_zones == expected
    assert status.last_known_state["UserAirconSettings"]["EnabledZones"] == expected
    assert not zone_api._coalescer._pending_tasks
    assert not zone_api._coalescer._batches

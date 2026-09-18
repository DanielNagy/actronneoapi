"""Regression tests for enabled-zone state after command delivery."""

import asyncio
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from actron_neo_api import ActronAirAPI, ActronAirStatus
from actron_neo_api.exceptions import ActronAirAPIError


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

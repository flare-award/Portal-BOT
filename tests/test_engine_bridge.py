"""Unit tests for Source Engine log parser and State Fusion."""

import os
import tempfile
import pytest

from portal_bot.core.types import GameState, PortalGunState
from portal_bot.engine_bridge.console_reader import ConsoleLogReader
from portal_bot.engine_bridge.game_fusion import StateFusion


def test_console_log_reader_parsing():
    reader = ConsoleLogReader()
    
    # 1. Map Load Event
    reader._parse_line("Map: testchmb_a_02")
    assert reader.state.current_map == "testchmb_a_02"
    assert reader.state.current_chamber == 2
    assert reader.state.gun_state == PortalGunState.SINGLE_PORTAL_BLUE

    # 2. Portal Gun Pickup
    reader._parse_line("Picked up weapon_portalgun dual")
    assert reader.state.gun_state == PortalGunState.DUAL_PORTAL

    # 3. Portal Placement
    reader._parse_line("CWeaponPortalgun::FirePortal: Blue")
    assert reader.state.blue_portal_placed is True

    # 4. Button & Door
    reader._parse_line("func_button pressed down")
    assert reader.state.button_pressed is True

    reader._parse_line("door_exit open")
    assert reader.state.door_open is True


def test_state_fusion_override():
    reader = ConsoleLogReader()
    reader.state.connected = True
    reader.state.current_chamber = 0
    reader.state.gun_state = PortalGunState.NO_GUN
    reader.state.button_pressed = False
    reader.state.door_open = False

    fusion = StateFusion(reader)
    
    # Simulate CV false positives
    cv_state = GameState()
    cv_state.portals.blue_active = True
    cv_state.portals.orange_active = True
    cv_state.button_pressed = True
    cv_state.exit_door_open = True

    fused = fusion.fuse_states(cv_state)
    
    # Since player has NO_GUN in Chamber 00, fusion must enforce accurate ground truth!
    assert fused.player.gun_state == PortalGunState.NO_GUN
    assert fused.portals.blue_active is False
    assert fused.portals.orange_active is False
    assert fused.button_pressed is False
    assert fused.exit_door_open is False

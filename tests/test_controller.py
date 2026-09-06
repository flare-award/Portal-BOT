"""Unit tests for Input Controller and Closed-Loop Actuator."""

import pytest
from portal_bot.config import BotConfig, KeyBindings
from portal_bot.controller.closed_loop import ClosedLoopController
from portal_bot.controller.input_controller import InputController
from portal_bot.controller.actions import move_forward, aim_at, emergency_stop
from portal_bot.core.types import ActionType, GameState


def test_input_controller_mock_mode():
    keys = KeyBindings()
    inputs_received = []

    def mock_sink(action_key=None, mouse_dx=0, mouse_dy=0, dt=0.05):
        inputs_received.append((action_key, mouse_dx, mouse_dy))

    ctrl = InputController(keys, mock_mode=True)
    ctrl.set_mock_sink(mock_sink)

    ctrl.key_down("w")
    ctrl.mouse_move_relative(10, -5)
    ctrl.click_mouse("left")
    ctrl.release_all_keys()

    assert len(inputs_received) >= 3


def test_closed_loop_aiming_pid():
    config = BotConfig()
    input_ctrl = InputController(config.keys, mock_mode=True)
    closed_loop = ClosedLoopController(config, input_ctrl)

    # Test aiming at point (800, 400) from center (640, 360)
    centered, dx, dy = closed_loop.aim_towards_screen_point((800, 400), 1280, 720, tolerance_px=10.0)
    assert centered is False
    assert dx > 0  # Should turn right
    assert dy > 0  # Should look down

    # Test when already centered
    centered_now, dx0, dy0 = closed_loop.aim_towards_screen_point((642, 361), 1280, 720, tolerance_px=10.0)
    assert centered_now is True
    assert dx0 == 0 and dy0 == 0


def test_emergency_stop_action():
    config = BotConfig()
    input_ctrl = InputController(config.keys, mock_mode=True)
    closed_loop = ClosedLoopController(config, input_ctrl)

    cmd = emergency_stop()
    result = closed_loop.execute_action(cmd, lambda: GameState())
    assert result.success is True

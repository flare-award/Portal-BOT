"""Unit tests for Stuck Detection and Recovery."""

import time
import pytest
from portal_bot.config import MovementConfig
from portal_bot.core.types import ActionType, GameState
from portal_bot.recovery.death_recovery import DeathRecovery
from portal_bot.recovery.stuck_detector import StuckDetector


def test_stuck_detector_trigger():
    config = MovementConfig(stuck_time_threshold=0.1, min_optical_flow_movement=1.0)
    detector = StuckDetector(config)

    state_idle = GameState(optical_flow_magnitude=0.1)

    # First check: initializes timer
    assert detector.check_stuck(state_idle, last_action_attempted_move=True) is False

    time.sleep(0.15)
    # Second check after elapsed timeout with zero motion
    is_stuck = detector.check_stuck(state_idle, last_action_attempted_move=True)
    assert is_stuck is True

    plan = detector.generate_recovery_plan()
    assert len(plan) > 0
    assert any(cmd.action_type == ActionType.MOVE_BACKWARD for cmd in plan)


def test_death_recovery_trigger():
    recovery = DeathRecovery()

    # Normal state
    normal_state = GameState(death_detected=False)
    actions = recovery.check_and_recover(normal_state)
    assert len(actions) == 0

    # Death state
    death_state = GameState(death_detected=True)
    actions_dead = recovery.check_and_recover(death_state)
    assert len(actions_dead) > 0
    assert any(cmd.action_type == ActionType.QUICK_LOAD for cmd in actions_dead)

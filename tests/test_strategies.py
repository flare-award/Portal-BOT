"""Unit tests for Multi-Stage Action Strategies and Anti-Spin Exploration."""

import pytest
from portal_bot.core.types import (
    ActionType,
    BoundingBox,
    DetectedObject,
    GameState,
    ObjectType,
    PortalGunState,
)
from portal_bot.planner.strategies.button_strategy import ButtonDepositStrategy
from portal_bot.planner.strategies.cube_strategy import CubePickupStrategy
from portal_bot.planner.strategies.exploration_strategy import ChamberExplorationStrategy
from portal_bot.planner.strategies.portal_strategy import PortalGunStrategy
from portal_bot.state.spatial_grid import TrackedObject


def test_cube_pickup_strategy_escalation():
    strat = CubePickupStrategy()
    det = DetectedObject(ObjectType.CUBE, BoundingBox(640, 360, 40, 40))
    tracked = TrackedObject(det)
    state = GameState()

    # Stage 1: Standard alignment
    actions1 = strat.execute(state, tracked, (1280, 720))
    assert len(actions1) > 0

    # Simulate attempt escalation
    strat.attempt_count = 1
    actions2 = strat.execute(state, tracked, (1280, 720))
    assert any(a.action_type == ActionType.ROTATE_CAMERA for a in actions2)

    # Stage 3: Reposition
    strat.attempt_count = 2
    actions3 = strat.execute(state, tracked, (1280, 720))
    assert any(a.action_type == ActionType.MOVE_BACKWARD for a in actions3)


def test_anti_spin_exploration_strategy():
    strat = ChamberExplorationStrategy()
    state = GameState()

    # Steps 1 to 4: Discrete 90 degree scans
    for i in range(4):
        plan = strat.execute(state, (1280, 720))
        assert any(a.action_type == ActionType.ROTATE_CAMERA for a in plan)

    # Step 5: After full 360 scan, MUST relocate forward to break loop!
    plan5 = strat.execute(state, (1280, 720))
    assert any(a.action_type == ActionType.MOVE_FORWARD for a in plan5)
    assert any(a.action_type == ActionType.JUMP for a in plan5)


def test_portal_strategy_respects_gun_state():
    state_no_gun = GameState()
    state_no_gun.player.gun_state = PortalGunState.NO_GUN
    plan_no_gun = PortalGunStrategy.plan_portals(state_no_gun, (1280, 720))
    assert len(plan_no_gun) == 0

    state_single_gun = GameState()
    state_single_gun.player.gun_state = PortalGunState.SINGLE_PORTAL_BLUE
    state_single_gun.portals.blue_active = False
    plan_single = PortalGunStrategy.plan_portals(state_single_gun, (1280, 720))
    assert any(a.action_type == ActionType.FIRE_BLUE_PORTAL for a in plan_single)
    assert not any(a.action_type == ActionType.FIRE_ORANGE_PORTAL for a in plan_single)

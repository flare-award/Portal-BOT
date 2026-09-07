"""Unit tests for Hierarchical Planner and Decision Making Agent."""

import pytest
from portal_bot.config import BotConfig
from portal_bot.core.types import (
    ActionType,
    BoundingBox,
    DetectedObject,
    GameState,
    GoalType,
    ObjectType,
)
from portal_bot.planner.hierarchical import HierarchicalPlanner
from portal_bot.planner.puzzle_solver import DecisionAgent
from portal_bot.state.world_model import WorldModel


def test_goal_decomposition_open_door():
    config = BotConfig()
    wm = WorldModel(config)
    planner = HierarchicalPlanner(wm)
    
    # When exit door is open -> Main subgoal should be REACH_EXIT
    state = GameState(exit_door_open=True)
    door = DetectedObject(object_type=ObjectType.DOOR, bbox=BoundingBox(500, 200, 80, 150), attributes={"is_open": True})
    state.objects.append(door)
    wm.update(state)
    wm.update(state)
    
    subgoals = planner.decompose_main_goal(state)
    assert len(subgoals) > 0
    assert subgoals[0].goal_type == GoalType.REACH_EXIT


def test_goal_decomposition_obtain_cube():
    config = BotConfig()
    wm = WorldModel(config)
    planner = HierarchicalPlanner(wm)
    
    # Door closed, cube visible, button unpressed -> Subgoal should be OBTAIN_CUBE
    state = GameState(exit_door_open=False, button_pressed=False)
    cube = DetectedObject(object_type=ObjectType.CUBE, bbox=BoundingBox(300, 300, 40, 40))
    button = DetectedObject(object_type=ObjectType.BUTTON_FLOOR, bbox=BoundingBox(700, 400, 50, 25))
    state.objects.extend([cube, button])
    wm.update(state)
    wm.update(state)
    
    subgoals = planner.decompose_main_goal(state)
    assert len(subgoals) > 0
    assert subgoals[0].goal_type == GoalType.OBTAIN_CUBE


def test_goal_decomposition_carry_cube():
    config = BotConfig()
    wm = WorldModel(config)
    planner = HierarchicalPlanner(wm)
    
    # Holding cube -> Subgoal should be CARRY_CUBE_TO_BUTTON
    state = GameState(exit_door_open=False)
    state.player.holding_cube = True
    button = DetectedObject(object_type=ObjectType.BUTTON_FLOOR, bbox=BoundingBox(700, 400, 50, 25))
    state.objects.append(button)
    wm.update(state)
    wm.update(state)
    
    subgoals = planner.decompose_main_goal(state)
    assert len(subgoals) > 0
    assert subgoals[0].goal_type == GoalType.CARRY_CUBE_TO_BUTTON


def test_decision_agent_generates_action():
    config = BotConfig()
    wm = WorldModel(config)
    agent = DecisionAgent(config, wm)
    
    state = GameState(exit_door_open=True)
    door = DetectedObject(object_type=ObjectType.DOOR, bbox=BoundingBox(640, 360, 80, 150), attributes={"is_open": True})
    state.objects.append(door)
    
    action = agent.decide_next_action(state)
    assert action is not None
    assert isinstance(action.action_type, ActionType)

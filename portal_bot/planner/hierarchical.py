"""Hierarchical Task Network and Goal Decomposition Planner for Portal 1."""

import time
from typing import List, Optional

from portal_bot.core.types import (
    GameState,
    GoalType,
    ObjectType,
    PortalGunState,
    SubGoal,
)
from portal_bot.state.world_model import WorldModel
from portal_bot.utils.logger import bot_log


class HierarchicalPlanner:
    """Decomposes high-level game objectives into structured subgoals."""

    def __init__(self, world_model: WorldModel):
        self.world_model = world_model
        self.goal_counter = 0

    def _next_goal_id(self, goal_type: GoalType) -> str:
        self.goal_counter += 1
        return f"{goal_type.value}_{self.goal_counter}"

    def decompose_main_goal(self, state: GameState) -> List[SubGoal]:
        """
        Evaluates current chamber state and produces a ranked subgoal hierarchy.
        """
        subgoals: List[SubGoal] = []

        # 1. Chamber Completed -> Walk forward through exit elevator
        if state.level_complete:
            bot_log.goal("Chamber completed! Advancing to next chamber...")
            subgoals.append(SubGoal(
                id=self._next_goal_id(GoalType.COMPLETE_LEVEL),
                goal_type=GoalType.COMPLETE_LEVEL,
                reason="Walk forward through completed chamber exit"
            ))
            return subgoals

        # 2. Exit Door is Open or Button is Pressed -> Proceed to exit!
        if state.exit_door_open or state.button_pressed:
            door_obj = self.world_model.get_exit_door()
            screen_pt = door_obj.screen_pos if door_obj else None
            subgoals.append(SubGoal(
                id=self._next_goal_id(GoalType.REACH_EXIT),
                goal_type=GoalType.REACH_EXIT,
                target_object_type=ObjectType.DOOR,
                target_screen_point=screen_pt,
                reason="Exit door is open, proceed to chamber elevator"
            ))
            return subgoals

        # 3. Holding Cube -> Carry directly to Floor Button
        cube_in_hand = state.player.holding_cube
        button_obj = self.world_model.get_best_button()

        if cube_in_hand:
            if button_obj:
                subgoals.append(SubGoal(
                    id=self._next_goal_id(GoalType.CARRY_CUBE_TO_BUTTON),
                    goal_type=GoalType.CARRY_CUBE_TO_BUTTON,
                    target_object_type=ObjectType.BUTTON_FLOOR,
                    target_screen_point=button_obj.screen_pos,
                    reason="Holding cube, transport and drop it on the floor button"
                ))
            else:
                subgoals.append(SubGoal(
                    id=self._next_goal_id(GoalType.EXPLORE_SURROUNDINGS),
                    goal_type=GoalType.EXPLORE_SURROUNDINGS,
                    reason="Holding cube, scanning chamber to locate floor button"
                ))
            return subgoals

        # 4. Cube is visible in chamber & Button is unpressed -> Obtain Cube
        cube_obj = self.world_model.get_best_cube()

        if cube_obj and not state.button_pressed:
            subgoals.append(SubGoal(
                id=self._next_goal_id(GoalType.OBTAIN_CUBE),
                goal_type=GoalType.OBTAIN_CUBE,
                target_object_type=ObjectType.CUBE,
                target_screen_point=cube_obj.screen_pos,
                reason="Locate and pick up Weighted Storage Cube"
            ))
            return subgoals

        # 5. Default: Systematic Horizontal Exploration
        subgoals.append(SubGoal(
            id=self._next_goal_id(GoalType.EXPLORE_SURROUNDINGS),
            goal_type=GoalType.EXPLORE_SURROUNDINGS,
            reason="Scan room to identify puzzle elements, cubes, buttons, or exit"
        ))
        return subgoals

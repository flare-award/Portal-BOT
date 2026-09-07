"""Decision-making agent and reasoning loop for Portal 1."""

import math
import time
from typing import List, Optional, Tuple

from portal_bot.config import BotConfig
from portal_bot.controller.actions import (
    aim_at,
    move_forward,
    rotate_camera,
    wait,
)
from portal_bot.core.types import (
    ActionCommand,
    ActionResult,
    ActionType,
    GameState,
    GoalType,
    ObjectType,
    SubGoal,
)
from portal_bot.planner.hierarchical import HierarchicalPlanner
from portal_bot.planner.strategies.button_strategy import ButtonDepositStrategy
from portal_bot.planner.strategies.cube_strategy import CubePickupStrategy
from portal_bot.planner.strategies.exploration_strategy import ChamberExplorationStrategy
from portal_bot.planner.strategies.portal_strategy import PortalGunStrategy
from portal_bot.state.world_model import WorldModel
from portal_bot.utils.logger import bot_log


class DecisionAgent:
    """
    Online reasoning agent that maps real-time perceived game state
    and hierarchical goals to concrete multi-strategy action sequences.
    """

    def __init__(self, config: BotConfig, world_model: WorldModel):
        self.config = config
        self.world_model = world_model
        self.hierarchical_planner = HierarchicalPlanner(world_model)

        # Specialized multi-stage strategies
        self.cube_strategy = CubePickupStrategy()
        self.button_strategy = ButtonDepositStrategy()
        self.exploration_strategy = ChamberExplorationStrategy()
        self.portal_strategy = PortalGunStrategy()

        self.current_goal: Optional[SubGoal] = None
        self.action_queue: List[ActionCommand] = []
        self.frame_size = (config.capture.width, config.capture.height)

    def decide_next_action(
        self,
        state: GameState,
        last_action: Optional[ActionCommand] = None,
        last_result: Optional[ActionResult] = None
    ) -> ActionCommand:
        w, h = self.frame_size

        # 1. Update World Model
        self.world_model.update(state)

        # 2. Pop next command from active action sequence
        if self.action_queue:
            next_cmd = self.action_queue.pop(0)
            bot_log.action(f"Executing: {next_cmd.action_type.value} ({next_cmd.reason})")
            return next_cmd

        # 3. Decompose and select current Goal
        subgoals = self.hierarchical_planner.decompose_main_goal(state)
        if not subgoals:
            return wait(0.05, reason="No active goals")

        active_goal = subgoals[0]
        self.current_goal = active_goal
        bot_log.goal(f"Active Subgoal: {active_goal.goal_type.value} - {active_goal.reason}")

        # 4. Synthesize action commands via strategies
        if active_goal.goal_type == GoalType.COMPLETE_LEVEL:
            return move_forward(0.5, reason="Advance through exit elevator")

        elif active_goal.goal_type == GoalType.REACH_EXIT:
            return self._plan_reach_exit(state, active_goal)

        elif active_goal.goal_type == GoalType.CARRY_CUBE_TO_BUTTON:
            button_obj = self.world_model.get_best_button()
            if button_obj:
                plan = self.button_strategy.execute(state, button_obj, self.frame_size)
                if plan:
                    first = plan.pop(0)
                    self.action_queue.extend(plan)
                    return first
            return move_forward(0.3, reason="Approach button area")

        elif active_goal.goal_type == GoalType.OBTAIN_CUBE:
            cube_obj = self.world_model.get_best_cube()
            if cube_obj:
                plan = self.cube_strategy.execute(state, cube_obj, self.frame_size)
                if plan:
                    first = plan.pop(0)
                    self.action_queue.extend(plan)
                    return first
            else:
                self.cube_strategy.reset()
                return rotate_camera(mouse_dx=80, mouse_dy=0, reason="Scan room to find cube")

        elif active_goal.goal_type == GoalType.PLACE_PORTAL_PAIR:
            plan = self.portal_strategy.plan_portals(state, self.frame_size)
            if plan:
                first = plan.pop(0)
                self.action_queue.extend(plan)
                return first
            return wait(0.1, reason="No portal action needed")

        elif active_goal.goal_type == GoalType.EXPLORE_SURROUNDINGS:
            plan = self.exploration_strategy.execute(state, self.frame_size)
            if plan:
                first = plan.pop(0)
                self.action_queue.extend(plan)
                return first

        return wait(0.05, reason="Awaiting state update")

    def _plan_reach_exit(self, state: GameState, goal: SubGoal) -> ActionCommand:
        """Navigates towards the open exit door with purely horizontal steering."""
        w, h = self.frame_size
        door_obj = self.world_model.get_exit_door()

        if door_obj:
            tx, ty = door_obj.screen_pos
            dx = tx - w // 2
            
            if abs(dx) > 30:
                bot_log.action(f"Aligning with Exit Door (offset: {dx}px)")
                return aim_at((tx, ty), reason="Align camera with exit door")
            else:
                bot_log.action("Moving forward toward Exit Door")
                return move_forward(0.4, reason="Walk to exit door")
        else:
            return rotate_camera(mouse_dx=80, mouse_dy=0, reason="Search for exit door")

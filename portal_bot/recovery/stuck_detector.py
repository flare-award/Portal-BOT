"""Stuck condition detector and multi-stage recovery engine."""

import time
from typing import List, Optional

from portal_bot.config import MovementConfig
from portal_bot.controller.actions import (
    jump,
    move_backward,
    quick_load,
    rotate_camera,
    strafe_left,
    strafe_right,
    wait,
)
from portal_bot.core.types import ActionCommand, GameState
from portal_bot.utils.logger import bot_log


class StuckDetector:
    """Monitors progress and executes progressive recovery steps when movement is stalled."""

    def __init__(self, config: MovementConfig):
        self.config = config
        self.last_progress_time = time.time()
        self.is_currently_stuck = False
        self.recovery_stage = 0
        self.consecutive_stuck_triggers = 0

    def reset(self):
        self.last_progress_time = time.time()
        self.is_currently_stuck = False
        self.recovery_stage = 0

    def check_stuck(self, state: GameState, last_action_attempted_move: bool) -> bool:
        now = time.time()
        
        # If moving action was attempted but optical flow was near zero
        if last_action_attempted_move and state.optical_flow_magnitude < self.config.min_optical_flow_movement:
            if now - self.last_progress_time > self.config.stuck_time_threshold:
                self.is_currently_stuck = True
                return True
        else:
            # Significant motion occurred
            self.last_progress_time = now
            self.is_currently_stuck = False
            self.recovery_stage = 0

        return False

    def generate_recovery_plan(self) -> List[ActionCommand]:
        """Generates progressive recovery action sequences."""
        self.recovery_stage += 1
        self.consecutive_stuck_triggers += 1
        bot_log.recovery(f"Stuck detected! Initiating recovery stage {self.recovery_stage}")

        plan: List[ActionCommand] = []

        if self.recovery_stage == 1:
            # Stage 1: Step backward and jump to clear small ledge/prop obstacle
            plan.append(move_backward(0.4, reason="Recovery: Step back from obstacle"))
            plan.append(jump(reason="Recovery: Jump over obstacle"))
            plan.append(wait(0.2))

        elif self.recovery_stage == 2:
            # Stage 2: Strafe sideways and rotate camera
            plan.append(strafe_left(0.4, reason="Recovery: Sidestep left"))
            plan.append(rotate_camera(mouse_dx=80, reason="Recovery: Turn camera to unblock"))
            plan.append(jump(reason="Recovery: Jump"))

        elif self.recovery_stage == 3:
            # Stage 3: Strafe right and rotate 90 degrees
            plan.append(strafe_right(0.5, reason="Recovery: Sidestep right"))
            plan.append(rotate_camera(mouse_dx=-120, reason="Recovery: Turn camera"))
            plan.append(jump(reason="Recovery: Jump"))

        else:
            # Stage 4: Critical stuck -> reload last save
            bot_log.recovery("Severe stuck state reached. Triggering Quickload (F9)")
            plan.append(quick_load(reason="Recovery: Quickload game save"))
            plan.append(wait(1.0))
            self.recovery_stage = 0

        return plan

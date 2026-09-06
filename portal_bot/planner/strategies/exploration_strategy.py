"""Structured chamber exploration strategy that avoids infinite spinning."""

import time
from typing import List, Tuple

from portal_bot.controller.actions import (
    jump,
    move_forward,
    rotate_camera,
    strafe_left,
    strafe_right,
    wait,
)
from portal_bot.core.types import ActionCommand, GameState
from portal_bot.utils.logger import bot_log


class ChamberExplorationStrategy:
    """
    Executes systematic, non-spinning room mapping:
    1. Discrete 90-degree quadrant scans (4 directions max)
    2. Physical relocation into chamber open area (walk forward & jump)
    3. Re-evaluation from new vantage point
    """

    def __init__(self):
        self.quadrant = 0
        self.scans_at_current_spot = 0
        self.last_move_time = time.time()

    def reset(self):
        self.quadrant = 0
        self.scans_at_current_spot = 0
        self.last_move_time = time.time()

    def execute(self, state: GameState, frame_size: Tuple[int, int]) -> List[ActionCommand]:
        actions: List[ActionCommand] = []

        if self.scans_at_current_spot < 4:
            # Discrete 90 degree snap turn
            self.scans_at_current_spot += 1
            self.quadrant = (self.quadrant + 1) % 4
            bot_log.action(f"Scanning quadrant {self.quadrant + 1}/4 (90 deg turn)")
            actions.append(rotate_camera(mouse_dx=100, mouse_dy=0, reason=f"Sweep quadrant {self.quadrant}"))
            actions.append(wait(0.15))
        else:
            # Completed 360 degree sweep at this spot -> PHYSICALLY MOVE to new area!
            bot_log.action("Full sweep complete: Relocating forward to new chamber vantage point")
            self.scans_at_current_spot = 0
            
            # Walk forward and clear geometry
            actions.append(move_forward(0.65, reason="Relocate to chamber center"))
            actions.append(jump(reason="Clear any floor lip or step"))
            actions.append(wait(0.2))

        return actions

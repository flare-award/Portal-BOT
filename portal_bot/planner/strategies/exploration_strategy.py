"""Structured chamber exploration strategy that avoids infinite spinning and erratic jumping."""

import time
from typing import List, Tuple

from portal_bot.controller.actions import (
    move_forward,
    rotate_camera,
    wait,
)
from portal_bot.core.types import ActionCommand, GameState
from portal_bot.utils.logger import bot_log


class ChamberExplorationStrategy:
    """
    Executes smooth, systematic room exploration:
    1. 4 discrete 90-degree quadrant checks at horizontal eye level
    2. Smooth relocation forward into open room space
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
            self.scans_at_current_spot += 1
            self.quadrant = (self.quadrant + 1) % 4
            bot_log.action(f"Scanning quadrant {self.quadrant + 1}/4 (90 deg turn)")
            actions.append(rotate_camera(mouse_dx=90, mouse_dy=0, reason=f"Sweep quadrant {self.quadrant}"))
            actions.append(wait(0.12))
        else:
            bot_log.action("Full sweep complete: Relocating forward to new chamber vantage point")
            self.scans_at_current_spot = 0
            
            # Walk forward smoothly to new position (no jumping, level pitch)
            actions.append(move_forward(0.60, reason="Advance to new vantage point"))
            actions.append(wait(0.15))

        return actions

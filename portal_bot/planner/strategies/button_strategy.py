"""Progressive multi-stage fallback strategy for activating floor buttons with cubes."""

import time
from typing import List, Optional, Tuple

from portal_bot.controller.actions import (
    aim_at,
    interact_use,
    move_backward,
    move_forward,
    rotate_camera,
    wait,
)
from portal_bot.core.types import ActionCommand, GameState
from portal_bot.state.spatial_grid import TrackedObject
from portal_bot.utils.logger import bot_log


class ButtonDepositStrategy:
    """
    Carries held cube directly to floor button and deposits it with zero backward drift.
    """

    def __init__(self):
        self.stage = 0
        self.last_action_time = time.time()
        self.deposited = False

    def reset(self):
        self.stage = 0
        self.last_action_time = time.time()
        self.deposited = False

    def execute(self, state: GameState, button_obj: TrackedObject, frame_size: Tuple[int, int]) -> List[ActionCommand]:
        w, h = frame_size
        bx, by = button_obj.screen_pos
        dx = bx - w // 2
        dy = by - h // 2

        actions: List[ActionCommand] = []

        # If arrived right in front of button pedestal
        if button_obj.bbox.area > 2500 or by > h * 0.70:
            bot_log.action("At button pedestal: dropping cube onto button dome")
            
            # 1. Pitch down towards button dome
            actions.append(rotate_camera(mouse_dx=0, mouse_dy=35, reason="Pitch down to button dome"))
            # 2. Release / drop cube
            actions.append(interact_use(reason="Release cube onto button"))
            actions.append(wait(0.12))
            # 3. Restore camera level back to horizontal eye level
            actions.append(rotate_camera(mouse_dx=0, mouse_dy=-35, reason="Re-level camera back to horizon"))
            # 4. Small single backstep to clear button contact
            actions.append(move_backward(0.20, reason="Short backstep from button"))
            actions.append(wait(0.10))
            
            self.deposited = True
        else:
            # Approach button in straight line
            if abs(dx) > 25:
                actions.append(aim_at((bx, by), reason="Align heading toward button"))
            
            bot_log.action(f"Transporting cube toward button (offset: dx={dx}, dy={dy})")
            actions.append(move_forward(0.40, reason="Step forward towards button"))

        return actions

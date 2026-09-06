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
    Deposits cube onto floor button and levels camera back up to eye-level.
    """

    def __init__(self):
        self.stage = 0
        self.last_drop_time = time.time()

    def reset(self):
        self.stage = 0
        self.last_drop_time = time.time()

    def execute(self, state: GameState, button_obj: TrackedObject, frame_size: Tuple[int, int]) -> List[ActionCommand]:
        w, h = frame_size
        bx, by = button_obj.screen_pos
        dx = bx - w // 2
        dy = by - h // 2

        actions: List[ActionCommand] = []

        # If arrived right at button pedestal
        if button_obj.bbox.area > 3000 or by > h * 0.65:
            bot_log.action("At button pedestal: placing cube onto button")
            # 1. Pitch down to button dome
            actions.append(rotate_camera(mouse_dx=0, mouse_dy=40, reason="Pitch down to button dome"))
            # 2. Drop cube
            actions.append(interact_use(reason="Drop cube on button pedestal"))
            actions.append(wait(0.15))
            # 3. Restore camera level back to horizontal!
            actions.append(rotate_camera(mouse_dx=0, mouse_dy=-40, reason="Level camera back to eye level"))
            # 4. Step backward to clear button switch
            actions.append(move_backward(0.35, reason="Step back to allow button press"))
            actions.append(wait(0.1))
        else:
            if abs(dx) > 25:
                actions.append(aim_at((bx, by), reason="Align heading toward button"))
            actions.append(move_forward(0.35, reason="Transport cube toward button"))

        return actions

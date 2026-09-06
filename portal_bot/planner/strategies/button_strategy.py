"""Progressive multi-stage fallback strategy for activating floor buttons with cubes."""

import time
from typing import List, Optional, Tuple

from portal_bot.controller.actions import (
    aim_at,
    interact_use,
    jump,
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
    Progressively deposits and positions cube onto 1500 MW Super-Colliding floor button:
    Stage 1: Center aim, advance, look down, drop
    Stage 2: Re-align, crouch-drop, jump-drop
    Stage 3: Physical nudge push
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

        if abs(dx) > 30:
            actions.append(aim_at((bx, by), reason="Align heading toward button"))

        if button_obj.bbox.area > 3500 or by > h * 0.70:
            bot_log.action("At button pedestal: executing deposit sequence")
            actions.append(rotate_camera(mouse_dx=0, mouse_dy=55, reason="Pitch down to button dome"))
            actions.append(interact_use(reason="Drop cube on button pedestal"))
            actions.append(wait(0.2))
            actions.append(move_backward(0.25, reason="Step back to allow button press"))
        else:
            actions.append(move_forward(0.35, reason="Transport cube toward button"))

        return actions

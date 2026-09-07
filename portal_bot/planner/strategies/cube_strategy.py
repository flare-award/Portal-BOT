"""Progressive multi-stage fallback strategy for acquiring cubes."""

import math
import time
from typing import List, Optional, Tuple

from portal_bot.controller.actions import (
    aim_at,
    interact_use,
    jump,
    move_backward,
    move_forward,
    rotate_camera,
    strafe_left,
    strafe_right,
    wait,
)
from portal_bot.core.types import ActionCommand, GameState
from portal_bot.state.spatial_grid import TrackedObject
from portal_bot.utils.logger import bot_log


class CubePickupStrategy:
    """
    Progressively escalates grab techniques if simple interaction fails:
    Stage 1: Direct reticle alignment & Use
    Stage 2: Close approach + Look Down + Use + Re-level camera
    Stage 3: Back up, Strafe realignment, Re-approach + Use
    Stage 4: Jump-Grab (for cubes elevated on pedestals)
    """

    def __init__(self):
        self.attempt_count = 0
        self.last_attempt_time = time.time()
        self.last_cube_screen_pos: Optional[Tuple[int, int]] = None

    def reset(self):
        self.attempt_count = 0
        self.last_attempt_time = time.time()
        self.last_cube_screen_pos = None

    def execute(self, state: GameState, cube_obj: TrackedObject, frame_size: Tuple[int, int]) -> List[ActionCommand]:
        w, h = frame_size
        tx, ty = cube_obj.screen_pos
        dx = tx - w // 2
        dy = ty - h // 2
        self.last_cube_screen_pos = (tx, ty)

        now = time.time()
        if now - self.last_attempt_time > 2.5:
            self.attempt_count = (self.attempt_count + 1) % 4
            self.last_attempt_time = now

        bot_log.action(f"Cube Pickup Strategy Stage {self.attempt_count + 1}/4 (Offset: dx={dx}, dy={dy})")

        actions: List[ActionCommand] = []

        if self.attempt_count == 0:
            if abs(dx) > 25 or abs(dy) > 30:
                actions.append(aim_at((tx, ty), reason="Align crosshair directly on Cube"))
            
            if cube_obj.bbox.area > 3000 or ty > h * 0.65:
                actions.append(interact_use(reason="Stage 1: Direct grab (E)"))
            else:
                actions.append(move_forward(0.3, reason="Step closer to cube"))

        elif self.attempt_count == 1:
            bot_log.action("Stage 2: Approach closer, look down, grab and re-level")
            actions.append(move_forward(0.35, reason="Stage 2: Close step"))
            actions.append(rotate_camera(mouse_dx=0, mouse_dy=40, reason="Pitch down toward cube"))
            actions.append(interact_use(reason="Stage 2: Grab with low pitch (E)"))
            actions.append(rotate_camera(mouse_dx=0, mouse_dy=-40, reason="Re-level camera back to eye level"))
            actions.append(wait(0.1))

        elif self.attempt_count == 2:
            bot_log.action("Stage 3: Backstep and reposition angle")
            actions.append(move_backward(0.3, reason="Stage 3: Backstep from cube collision box"))
            actions.append(strafe_left(0.25, reason="Stage 3: Angle adjust"))
            actions.append(aim_at((tx, ty), reason="Re-align with cube"))
            actions.append(move_forward(0.4, reason="Re-approach"))
            actions.append(interact_use(reason="Stage 3: Grab after reposition (E)"))

        else:
            bot_log.action("Stage 4: Jump-approach grab")
            actions.append(move_forward(0.3, reason="Approach ledge"))
            actions.append(jump(reason="Jump towards elevated cube"))
            actions.append(interact_use(reason="Stage 4: Mid-air grab (E)"))
            actions.append(wait(0.15))

        return actions

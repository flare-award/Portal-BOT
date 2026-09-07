"""Portal placement and trajectory planning for Portal 1."""

from typing import List, Optional, Tuple
import numpy as np

from portal_bot.controller.actions import aim_at, fire_blue_portal, fire_orange_portal, wait
from portal_bot.core.types import ActionCommand, GameState, ObjectType, SubGoal
from portal_bot.state.world_model import WorldModel
from portal_bot.utils.logger import bot_log


class PortalPlanner:
    """Plans strategic portal placements to connect disconnected chamber areas."""

    def __init__(self, world_model: WorldModel):
        self.world_model = world_model

    def plan_portal_pair(self, state: GameState, frame_size: Tuple[int, int] = (1280, 720)) -> List[ActionCommand]:
        """
        Creates action sequence to place a pair of portals:
        1. Blue portal on near/left conductive surface.
        2. Orange portal on far/right conductive surface.
        """
        w, h = frame_size
        actions: List[ActionCommand] = []

        # Find portalable walls from detections
        wall_objects = [obj for obj in state.objects if obj.object_type == ObjectType.PORTALABLE_WALL]

        if not wall_objects:
            # Blind aim at left/right walls if no specific wall contour found
            target_blue = (int(w * 0.25), int(h * 0.5))
            target_orange = (int(w * 0.75), int(h * 0.5))
        else:
            # Pick best separated surfaces
            target_blue = (wall_objects[0].bbox.cx, wall_objects[0].bbox.cy)
            target_orange = (wall_objects[-1].bbox.cx, wall_objects[-1].bbox.cy) if len(wall_objects) > 1 else (int(w * 0.8), int(h * 0.5))

        bot_log.goal(f"Planning portal pair: Blue at {target_blue}, Orange at {target_orange}")

        # Place Blue portal
        actions.append(aim_at(target_blue, reason="Aim at surface for Blue Portal"))
        actions.append(wait(0.1))
        actions.append(fire_blue_portal(target_blue, reason="Create entrance portal"))
        actions.append(wait(0.2))

        # Place Orange portal
        actions.append(aim_at(target_orange, reason="Aim at surface for Orange Portal"))
        actions.append(wait(0.1))
        actions.append(fire_orange_portal(target_orange, reason="Create exit portal"))
        actions.append(wait(0.2))

        return actions

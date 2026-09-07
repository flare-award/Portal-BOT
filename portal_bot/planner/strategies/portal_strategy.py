"""Portal Gun placement and trajectory strategy aware of inventory upgrades."""

from typing import List, Optional, Tuple

from portal_bot.controller.actions import aim_at, fire_blue_portal, fire_orange_portal, rotate_camera, wait
from portal_bot.core.types import ActionCommand, GameState, ObjectType, PortalGunState
from portal_bot.utils.logger import bot_log


class PortalGunStrategy:
    """Plans valid portal shots taking gun inventory status and 3D walls into account."""

    @staticmethod
    def plan_portals(state: GameState, frame_size: Tuple[int, int]) -> List[ActionCommand]:
        w, h = frame_size
        actions: List[ActionCommand] = []

        gun_state = state.player.gun_state

        if gun_state == PortalGunState.NO_GUN:
            return []

        # Find portalable wall detections
        wall_objects = [obj for obj in state.objects if obj.object_type == ObjectType.PORTALABLE_WALL]

        target_blue = (wall_objects[0].bbox.cx, wall_objects[0].bbox.cy) if wall_objects else (int(w * 0.35), int(h * 0.50))
        target_orange = (wall_objects[-1].bbox.cx, wall_objects[-1].bbox.cy) if len(wall_objects) > 1 else (int(w * 0.70), int(h * 0.50))

        # 1. SINGLE PORTAL GUN (Blue Only)
        if gun_state == PortalGunState.SINGLE_PORTAL_BLUE:
            if not state.portals.blue_active:
                bot_log.goal(f"Placing Single Blue Portal at wall {target_blue}")
                actions.append(aim_at(target_blue, reason="Aim at conductive surface for Blue Portal"))
                actions.append(wait(0.08))
                actions.append(fire_blue_portal(target_blue, reason="Fire Primary Blue Portal (LMB)"))
                actions.append(wait(0.15))
            return actions

        # 2. DUAL PORTAL GUN (Blue & Orange)
        if gun_state == PortalGunState.DUAL_PORTAL:
            if not state.portals.blue_active:
                bot_log.goal(f"Placing Dual Gun Blue Portal at wall {target_blue}")
                actions.append(aim_at(target_blue, reason="Aim at Blue Portal target"))
                actions.append(wait(0.08))
                actions.append(fire_blue_portal(target_blue, reason="Fire Blue Portal (LMB)"))
                actions.append(wait(0.15))

            if not state.portals.orange_active:
                bot_log.goal(f"Placing Dual Gun Orange Portal at wall {target_orange}")
                actions.append(aim_at(target_orange, reason="Aim at Orange Portal target"))
                actions.append(wait(0.08))
                actions.append(fire_orange_portal(target_orange, reason="Fire Orange Portal (RMB)"))
                actions.append(wait(0.15))

        return actions

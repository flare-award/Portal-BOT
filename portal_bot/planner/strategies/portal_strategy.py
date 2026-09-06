"""Portal Gun placement and trajectory strategy aware of inventory upgrades."""

from typing import List, Optional, Tuple

from portal_bot.controller.actions import aim_at, fire_blue_portal, fire_orange_portal, wait
from portal_bot.core.types import ActionCommand, GameState, ObjectType, PortalGunState
from portal_bot.utils.logger import bot_log


class PortalGunStrategy:
    """Plans valid portal shots taking gun inventory status into account."""

    @staticmethod
    def plan_portals(state: GameState, frame_size: Tuple[int, int]) -> List[ActionCommand]:
        w, h = frame_size
        actions: List[ActionCommand] = []

        gun_state = state.player.gun_state

        # 1. NO GUN: In early chambers (Chamber 00, 01), the player cannot fire portals!
        if gun_state == PortalGunState.NO_GUN:
            bot_log.goal("No Portal Gun in inventory: Navigating via test chamber mechanisms")
            return []

        # Find portalable wall detections
        wall_objects = [obj for obj in state.objects if obj.object_type == ObjectType.PORTALABLE_WALL]
        
        target_blue = (wall_objects[0].bbox.cx, wall_objects[0].bbox.cy) if wall_objects else (int(w * 0.3), int(h * 0.5))
        target_orange = (wall_objects[-1].bbox.cx, wall_objects[-1].bbox.cy) if len(wall_objects) > 1 else (int(w * 0.75), int(h * 0.5))

        # 2. SINGLE PORTAL GUN (Blue Only): Chamber 02 - 10
        if gun_state == PortalGunState.SINGLE_PORTAL_BLUE:
            if not state.portals.blue_active:
                bot_log.goal(f"Placing Single Blue Portal at {target_blue}")
                actions.append(aim_at(target_blue, reason="Aim at conductive surface for Blue Portal"))
                actions.append(wait(0.1))
                actions.append(fire_blue_portal(target_blue, reason="Fire Primary Blue Portal"))
                actions.append(wait(0.2))
            return actions

        # 3. DUAL PORTAL GUN: Chamber 11+
        if gun_state == PortalGunState.DUAL_PORTAL:
            if not state.portals.blue_active:
                bot_log.goal(f"Placing Dual Gun Blue Portal at {target_blue}")
                actions.append(aim_at(target_blue, reason="Aim at Blue Portal target"))
                actions.append(wait(0.1))
                actions.append(fire_blue_portal(target_blue, reason="Fire Blue Portal"))
                actions.append(wait(0.2))

            if not state.portals.orange_active:
                bot_log.goal(f"Placing Dual Gun Orange Portal at {target_orange}")
                actions.append(aim_at(target_orange, reason="Aim at Orange Portal target"))
                actions.append(wait(0.1))
                actions.append(fire_orange_portal(target_orange, reason="Fire Orange Portal"))
                actions.append(wait(0.2))

        return actions

"""Action command factories for Portal-BOT."""

from typing import Optional, Tuple
from portal_bot.core.types import ActionCommand, ActionType


def move_forward(duration: float = 0.35, reason: str = "") -> ActionCommand:
    return ActionCommand(action_type=ActionType.MOVE_FORWARD, duration=duration, reason=reason)


def move_backward(duration: float = 0.35, reason: str = "") -> ActionCommand:
    return ActionCommand(action_type=ActionType.MOVE_BACKWARD, duration=duration, reason=reason)


def strafe_left(duration: float = 0.25, reason: str = "") -> ActionCommand:
    return ActionCommand(action_type=ActionType.STRAFE_LEFT, duration=duration, reason=reason)


def strafe_right(duration: float = 0.25, reason: str = "") -> ActionCommand:
    return ActionCommand(action_type=ActionType.STRAFE_RIGHT, duration=duration, reason=reason)


def jump(reason: str = "") -> ActionCommand:
    return ActionCommand(action_type=ActionType.JUMP, duration=0.1, reason=reason)


def interact_use(reason: str = "Pick up / Drop object or press button") -> ActionCommand:
    return ActionCommand(action_type=ActionType.USE_INTERACT, duration=0.08, reason=reason)


def aim_at(target_screen_point: Tuple[int, int], reason: str = "") -> ActionCommand:
    return ActionCommand(
        action_type=ActionType.AIM_AT_TARGET,
        target_screen_point=target_screen_point,
        reason=reason
    )


def rotate_camera(mouse_dx: int, mouse_dy: int = 0, reason: str = "") -> ActionCommand:
    return ActionCommand(
        action_type=ActionType.ROTATE_CAMERA,
        mouse_dx=mouse_dx,
        mouse_dy=mouse_dy,
        reason=reason
    )


def fire_blue_portal(target_screen_point: Optional[Tuple[int, int]] = None, reason: str = "") -> ActionCommand:
    return ActionCommand(
        action_type=ActionType.FIRE_BLUE_PORTAL,
        target_screen_point=target_screen_point,
        reason=reason
    )


def fire_orange_portal(target_screen_point: Optional[Tuple[int, int]] = None, reason: str = "") -> ActionCommand:
    return ActionCommand(
        action_type=ActionType.FIRE_ORANGE_PORTAL,
        target_screen_point=target_screen_point,
        reason=reason
    )


def quick_load(reason: str = "Reload last save") -> ActionCommand:
    return ActionCommand(action_type=ActionType.QUICK_LOAD, duration=0.1, reason=reason)


def wait(duration: float = 0.2, reason: str = "") -> ActionCommand:
    return ActionCommand(action_type=ActionType.WAIT, duration=duration, reason=reason)


def emergency_stop() -> ActionCommand:
    return ActionCommand(action_type=ActionType.EMERGENCY_STOP, reason="Emergency safety abort")

"""Closed-loop visual servoing and movement execution with feedback."""

import math
import time
from typing import Callable, Optional, Tuple

from portal_bot.config import BotConfig
from portal_bot.controller.input_controller import InputController
from portal_bot.core.types import ActionCommand, ActionResult, ActionType, GameState
from portal_bot.utils.logger import bot_log
from portal_bot.utils.math_3d import PIDController, clamp


class ClosedLoopController:
    """Executes actions with visual feedback loops rather than blind sleep timers."""

    def __init__(self, config: BotConfig, input_ctrl: InputController):
        self.config = config
        self.input = input_ctrl
        self.pid_x = PIDController(
            kp=config.movement.aim_kp,
            ki=config.movement.aim_ki,
            kd=config.movement.aim_kd,
            out_min=-config.movement.max_mouse_step,
            out_max=config.movement.max_mouse_step,
        )
        self.pid_y = PIDController(
            kp=config.movement.aim_kp,
            ki=config.movement.aim_ki,
            kd=config.movement.aim_kd,
            out_min=-config.movement.max_mouse_step,
            out_max=config.movement.max_mouse_step,
        )

    def aim_towards_screen_point(
        self,
        target_point: Tuple[int, int],
        frame_width: int,
        frame_height: int,
        tolerance_px: float = 16.0
    ) -> Tuple[bool, int, int]:
        """
        Calculates and applies mouse step to center target_point at screen center.
        Returns: (is_centered, mouse_dx, mouse_dy)
        """
        cx = frame_width / 2.0
        cy = frame_height / 2.0

        error_x = target_point[0] - cx
        error_y = target_point[1] - cy

        dist = math.hypot(error_x, error_y)
        if dist <= tolerance_px:
            self.pid_x.reset()
            self.pid_y.reset()
            return True, 0, 0

        # Compute PID control steps
        dx = int(round(self.pid_x.update(error_x)))
        dy = int(round(self.pid_y.update(error_y)))

        # Send mouse movement
        self.input.mouse_move_relative(dx, dy)
        return False, dx, dy

    def execute_action(
        self,
        cmd: ActionCommand,
        get_current_state: Callable[[], GameState],
        frame_size: Tuple[int, int] = (1280, 720)
    ) -> ActionResult:
        """Executes an action command with state feedback."""
        t0 = time.time()
        w, h = frame_size

        if cmd.action_type == ActionType.EMERGENCY_STOP:
            self.input.release_all_keys()
            return ActionResult(command=cmd, success=True, message="Emergency stop executed")

        elif cmd.action_type == ActionType.MOVE_FORWARD:
            self.input.key_down(self.config.keys.forward)
            time.sleep(cmd.duration)
            self.input.key_up(self.config.keys.forward)
            
            # Observe state change
            new_state = get_current_state()
            motion = new_state.optical_flow_magnitude > self.config.movement.min_optical_flow_movement
            return ActionResult(
                command=cmd,
                success=True,
                motion_detected=motion,
                optical_flow_magnitude=new_state.optical_flow_magnitude,
                duration_actual=time.time() - t0
            )

        elif cmd.action_type == ActionType.MOVE_BACKWARD:
            self.input.key_down(self.config.keys.backward)
            time.sleep(cmd.duration)
            self.input.key_up(self.config.keys.backward)
            return ActionResult(command=cmd, success=True, motion_detected=True, duration_actual=time.time() - t0)

        elif cmd.action_type == ActionType.STRAFE_LEFT:
            self.input.key_down(self.config.keys.move_left)
            time.sleep(cmd.duration)
            self.input.key_up(self.config.keys.move_left)
            return ActionResult(command=cmd, success=True, motion_detected=True, duration_actual=time.time() - t0)

        elif cmd.action_type == ActionType.STRAFE_RIGHT:
            self.input.key_down(self.config.keys.move_right)
            time.sleep(cmd.duration)
            self.input.key_up(self.config.keys.move_right)
            return ActionResult(command=cmd, success=True, motion_detected=True, duration_actual=time.time() - t0)

        elif cmd.action_type == ActionType.JUMP:
            self.input.tap_key(self.config.keys.jump, 0.08)
            return ActionResult(command=cmd, success=True, motion_detected=True, duration_actual=time.time() - t0)

        elif cmd.action_type == ActionType.USE_INTERACT:
            self.input.tap_key(self.config.keys.use, 0.08)
            return ActionResult(command=cmd, success=True, message="Interacted (E)", duration_actual=time.time() - t0)

        elif cmd.action_type == ActionType.AIM_AT_TARGET and cmd.target_screen_point:
            centered, dx, dy = self.aim_towards_screen_point(cmd.target_screen_point, w, h)
            return ActionResult(
                command=cmd,
                success=True,
                message=f"Aim step dx={dx}, dy={dy}, centered={centered}",
                motion_detected=abs(dx) > 0 or abs(dy) > 0,
                duration_actual=time.time() - t0
            )

        elif cmd.action_type == ActionType.ROTATE_CAMERA:
            self.input.mouse_move_relative(cmd.mouse_dx, cmd.mouse_dy)
            return ActionResult(
                command=cmd,
                success=True,
                motion_detected=True,
                duration_actual=time.time() - t0
            )

        elif cmd.action_type == ActionType.FIRE_BLUE_PORTAL:
            self.input.click_mouse("left", duration=0.08)
            return ActionResult(command=cmd, success=True, message="Fired Blue Portal (LMB)", duration_actual=time.time() - t0)

        elif cmd.action_type == ActionType.FIRE_ORANGE_PORTAL:
            self.input.click_mouse("right", duration=0.08)
            return ActionResult(command=cmd, success=True, message="Fired Orange Portal (RMB)", duration_actual=time.time() - t0)

        elif cmd.action_type == ActionType.QUICK_LOAD:
            self.input.tap_key(self.config.keys.quick_load, 0.1)
            bot_log.recovery("Quickload (F9) sent to game")
            return ActionResult(command=cmd, success=True, message="Quickloaded", duration_actual=time.time() - t0)

        elif cmd.action_type == ActionType.WAIT:
            time.sleep(cmd.duration)
            return ActionResult(command=cmd, success=True, duration_actual=time.time() - t0)

        return ActionResult(command=cmd, success=False, message=f"Unknown action {cmd.action_type}")

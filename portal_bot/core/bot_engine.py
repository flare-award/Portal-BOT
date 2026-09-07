"""Master Engine Coordinator for Portal-BOT with Engine Log Fusion."""

import threading
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from portal_bot.capture.mock_capture import MockScreenCapture
from portal_bot.capture.screen_capture import ScreenCapture
from portal_bot.config import BotConfig, DEFAULT_CONFIG
from portal_bot.controller.actions import emergency_stop
from portal_bot.controller.closed_loop import ClosedLoopController
from portal_bot.controller.input_controller import InputController
from portal_bot.core.events import event_bus
from portal_bot.core.types import (
    ActionCommand,
    ActionResult,
    ActionType,
    BotStatus,
    GameState,
)
from portal_bot.engine_bridge.console_reader import ConsoleLogReader
from portal_bot.engine_bridge.game_fusion import StateFusion
from portal_bot.planner.puzzle_solver import DecisionAgent
from portal_bot.recovery.death_recovery import DeathRecovery
from portal_bot.recovery.stuck_detector import StuckDetector
from portal_bot.state.world_model import WorldModel
from portal_bot.utils.logger import bot_log
from portal_bot.vision.pipeline import VisionPipeline


class BotEngine:
    """Master orchestrator running decoupled Vision, Log Bridge, and Reasoning loops."""

    def __init__(self, config: BotConfig = DEFAULT_CONFIG):
        self.config = config
        self.status = BotStatus.IDLE

        # Components
        if config.capture.use_mock:
            self.capture = MockScreenCapture(config.capture)
        else:
            self.capture = ScreenCapture(config.capture)

        self.vision = VisionPipeline(config.vision)
        self.world_model = WorldModel(config)
        self.input_ctrl = InputController(config.keys, mock_mode=config.capture.use_mock)

        # Hook mock simulator input sink if mock mode
        if config.capture.use_mock and hasattr(self.capture, "step_input"):
            self.input_ctrl.set_mock_sink(self.capture.step_input)

        # Engine Log Bridge & Fusion
        self.console_reader = ConsoleLogReader(config.capture.custom_console_log_path)
        self.state_fusion = StateFusion(self.console_reader)

        self.closed_loop = ClosedLoopController(config, self.input_ctrl)
        self.agent = DecisionAgent(config, self.world_model)
        self.stuck_detector = StuckDetector(config.movement)
        self.death_recovery = DeathRecovery()

        # State storage
        self.latest_state: Optional[GameState] = None
        self.latest_debug_frame: Optional[np.ndarray] = None
        self.state_lock = threading.Lock()

        # Threading
        self._running = False
        self._paused = False
        self._vision_thread: Optional[threading.Thread] = None
        self._reasoning_thread: Optional[threading.Thread] = None

    def start(self):
        """Starts autonomous bot operation."""
        if self._running:
            return

        self._running = True
        self._paused = False
        self.set_status(BotStatus.RUNNING)
        bot_log.info("Bot Engine STARTED")

        self.capture.start()
        self.console_reader.start()

        self._vision_thread = threading.Thread(target=self._vision_loop, daemon=True, name="VisionLoop")
        self._reasoning_thread = threading.Thread(target=self._reasoning_loop, daemon=True, name="ReasoningLoop")

        self._vision_thread.start()
        self._reasoning_thread.start()

    def pause(self):
        """Pauses bot execution without stopping capture."""
        self._paused = True
        self.set_status(BotStatus.PAUSED)
        self.input_ctrl.release_all_keys()
        bot_log.info("Bot PAUSED")

    def resume(self):
        if self._running and self._paused:
            self._paused = False
            self.set_status(BotStatus.RUNNING)
            bot_log.info("Bot RESUMED")

    def stop(self):
        """Stops the bot and releases all inputs."""
        self._running = False
        self._paused = False
        self.input_ctrl.release_all_keys()
        self.capture.stop()
        self.console_reader.stop()
        self.set_status(BotStatus.STOPPED)
        bot_log.info("Bot Engine STOPPED")

    def emergency_stop(self):
        """Immediate hard stop (F8 triggered)."""
        bot_log.recovery("EMERGENCY STOP TRIGGERED (F8)")
        self.stop()

    def set_status(self, status: BotStatus):
        self.status = status
        event_bus.publish("bot_status_changed", status.value)

    def get_state_snapshot(self) -> Optional[GameState]:
        with self.state_lock:
            return self.latest_state

    def get_debug_frame(self) -> Optional[np.ndarray]:
        with self.state_lock:
            if self.latest_debug_frame is not None:
                return self.latest_debug_frame.copy()
        
        # Fallback for live preview when engine is idle/stopped
        try:
            frame = self.capture.capture_frame()
            if frame is not None:
                state, debug = self.vision.process_frame(frame, self.latest_state)
                with self.state_lock:
                    self.latest_state = state
                    self.latest_debug_frame = debug
                return debug
        except Exception:
            pass

        return None

    def _vision_loop(self):
        """High-FPS 60+ visual perception loop."""
        target_delay = 1.0 / max(1, self.config.capture.target_fps)
        
        while self._running:
            t0 = time.time()
            frame = self.capture.capture_frame()
            
            if frame is not None:
                raw_state, debug_frame = self.vision.process_frame(frame, self.latest_state)
                
                # Fuse with Game Engine logs for 100% ground-truth accuracy
                fused_state = self.state_fusion.fuse_states(raw_state)

                with self.state_lock:
                    self.latest_state = fused_state
                    self.latest_debug_frame = debug_frame

                event_bus.publish("frame_processed", {
                    "fps": fused_state.fps,
                    "confidence": fused_state.confidence,
                    "objects_count": len(fused_state.objects),
                    "door_open": fused_state.exit_door_open,
                    "button_pressed": fused_state.button_pressed,
                    "gun_state": fused_state.player.gun_state.value
                })

            elapsed = time.time() - t0
            sleep_time = max(0.001, target_delay - elapsed)
            time.sleep(sleep_time)

    def _reasoning_loop(self):
        """Asynchronous planning and decision-making loop."""
        planning_interval = 1.0 / max(1.0, self.config.planner.planning_fps)
        last_action: Optional[ActionCommand] = None
        last_result: Optional[ActionResult] = None

        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            state = self.get_state_snapshot()
            if state is None:
                time.sleep(0.02)
                continue

            # 1. Check Death Recovery
            death_actions = self.death_recovery.check_and_recover(state)
            if death_actions:
                self.set_status(BotStatus.DEAD_RELOADING)
                for act in death_actions:
                    self.closed_loop.execute_action(act, lambda: self.get_state_snapshot() or state)
                self.set_status(BotStatus.RUNNING)
                continue

            # 2. Check Stuck Condition
            was_moving_cmd = (last_action.action_type in [
                ActionType.MOVE_FORWARD, ActionType.MOVE_BACKWARD,
                ActionType.STRAFE_LEFT, ActionType.STRAFE_RIGHT
            ]) if last_action else False

            is_stuck = self.stuck_detector.check_stuck(state, was_moving_cmd)
            if is_stuck:
                self.set_status(BotStatus.STUCK_RECOVERING)
                recovery_plan = self.stuck_detector.generate_recovery_plan()
                for rec_cmd in recovery_plan:
                    self.closed_loop.execute_action(rec_cmd, lambda: self.get_state_snapshot() or state)
                self.stuck_detector.reset()
                self.set_status(BotStatus.RUNNING)
                continue

            # 3. Formulate next Action via Hierarchical Decision Agent
            t0 = time.time()
            action_cmd = self.agent.decide_next_action(state, last_action, last_result)

            if action_cmd:
                self.set_status(BotStatus.SOLVING_PUZZLE)
                result = self.closed_loop.execute_action(
                    action_cmd,
                    get_current_state=lambda: self.get_state_snapshot() or state,
                    frame_size=(self.config.capture.width, self.config.capture.height)
                )
                last_action = action_cmd
                last_result = result
                self.world_model.action_memory.record_action(action_cmd, result)
            else:
                self.set_status(BotStatus.RUNNING)

            elapsed = time.time() - t0
            sleep_time = max(0.005, planning_interval - elapsed)
            time.sleep(sleep_time)

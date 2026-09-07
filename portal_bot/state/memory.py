"""Action history, failure memory, and experience recording."""

from collections import deque
import time
from typing import Deque, Dict, List, Optional, Tuple

from portal_bot.core.types import ActionCommand, ActionResult, GoalType


class ActionMemory:
    """Records executed actions and evaluates their physical consequences."""

    def __init__(self, max_history: int = 100):
        self.history: Deque[Tuple[ActionCommand, ActionResult]] = deque(maxlen=max_history)
        self.failed_portal_angles: List[Tuple[float, float]] = []  # (yaw, pitch) that failed
        self.completed_goals: List[GoalType] = []
        self.consecutive_failed_actions: int = 0
        self.last_progress_time: float = time.time()

    def record_action(self, command: ActionCommand, result: ActionResult):
        self.history.append((command, result))
        
        if result.success and result.motion_detected:
            self.consecutive_failed_actions = 0
            self.last_progress_time = time.time()
        else:
            self.consecutive_failed_actions += 1

    def record_failed_portal_shot(self, yaw: float, pitch: float):
        self.failed_portal_angles.append((yaw, pitch))
        # Keep only recent 10 failed shots
        if len(self.failed_portal_angles) > 10:
            self.failed_portal_angles.pop(0)

    def is_angle_blacklisted(self, yaw: float, pitch: float, tolerance: float = 8.0) -> bool:
        for fy, fp in self.failed_portal_angles:
            if abs(fy - yaw) < tolerance and abs(fp - pitch) < tolerance:
                return True
        return False

    def mark_goal_completed(self, goal_type: GoalType):
        if goal_type not in self.completed_goals:
            self.completed_goals.append(goal_type)

    def get_recent_actions(self, count: int = 5) -> List[Tuple[ActionCommand, ActionResult]]:
        return list(self.history)[-count:]

"""Unified World Model for Portal-BOT."""

import time
from typing import List, Optional, Tuple

from portal_bot.config import BotConfig
from portal_bot.core.types import DetectedObject, GameState, ObjectType, SubGoal
from portal_bot.state.memory import ActionMemory
from portal_bot.state.spatial_grid import SpatialMemory, TrackedObject


class WorldModel:
    """Maintains consistent spatial representation and world state across frames."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.spatial_memory = SpatialMemory(config.planner.spatial_memory_retention_seconds)
        self.action_memory = ActionMemory()
        self.current_state: Optional[GameState] = None
        self.active_subgoals: List[SubGoal] = []

    def update(self, game_state: GameState):
        self.current_state = game_state
        self.spatial_memory.update(game_state.objects)
        self.spatial_memory.record_location(game_state.player.pos)

    def get_best_cube(self) -> Optional[TrackedObject]:
        cubes = self.spatial_memory.get_objects_by_type(ObjectType.CUBE)
        if not cubes:
            return None
        # Return cube with largest screen area (closest)
        return max(cubes, key=lambda c: c.bbox.area)

    def get_best_button(self) -> Optional[TrackedObject]:
        buttons = self.spatial_memory.get_objects_by_type(ObjectType.BUTTON_FLOOR)
        if not buttons:
            return None
        return max(buttons, key=lambda b: b.bbox.area)

    def get_exit_door(self) -> Optional[TrackedObject]:
        doors = self.spatial_memory.get_objects_by_type(ObjectType.DOOR)
        if not doors:
            return None
        return max(doors, key=lambda d: d.bbox.area)

    def has_both_portals_active(self) -> bool:
        if not self.current_state:
            return False
        return self.current_state.portals.blue_active and self.current_state.portals.orange_active

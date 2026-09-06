"""Spatial memory and object tracking across frames."""

import math
import time
from typing import Dict, List, Optional, Tuple

from portal_bot.core.types import BoundingBox, DetectedObject, ObjectType


class TrackedObject:
    """Represents a temporally filtered tracked object."""

    def __init__(self, detected: DetectedObject):
        self.object_type = detected.object_type
        self.bbox = detected.bbox
        self.screen_pos = detected.center_screen
        self.confidence = detected.confidence
        self.attributes = detected.attributes.copy()
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.seen_count = 1
        self.lost_count = 0

    def update(self, detected: DetectedObject):
        self.last_seen = time.time()
        self.seen_count += 1
        self.lost_count = 0
        
        # Immediate smooth responsive update
        self.screen_pos = detected.center_screen
        self.bbox = detected.bbox
        self.confidence = max(self.confidence, detected.confidence)
        self.attributes.update(detected.attributes)


class SpatialMemory:
    """Manages persistent object tracking and spatial layout memory."""

    def __init__(self, retention_seconds: float = 60.0):
        self.retention_seconds = retention_seconds
        self.tracked_objects: Dict[str, TrackedObject] = {}
        self.visited_locations: List[Tuple[float, float, float]] = []

    def update(self, detected_objects: List[DetectedObject]):
        now = time.time()
        unmatched_detections = list(detected_objects)

        for key, tracked in list(self.tracked_objects.items()):
            best_match = None
            best_dist = float("inf")

            for det in unmatched_detections:
                if det.object_type == tracked.object_type:
                    dist = math.hypot(det.center_screen[0] - tracked.screen_pos[0], det.center_screen[1] - tracked.screen_pos[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_match = det

            if best_match:
                tracked.update(best_match)
                unmatched_detections.remove(best_match)
            else:
                tracked.lost_count += 1
                if now - tracked.last_seen > self.retention_seconds or tracked.lost_count > 15:
                    del self.tracked_objects[key]

        # Handle remaining detections without spawning duplicates
        for det in unmatched_detections:
            if det.object_type != ObjectType.PORTALABLE_WALL:
                same_type = [t for t in self.tracked_objects.values() if t.object_type == det.object_type]
                if same_type:
                    closest = min(
                        same_type,
                        key=lambda t: math.hypot(det.center_screen[0] - t.screen_pos[0], det.center_screen[1] - t.screen_pos[1])
                    )
                    closest.update(det)
                else:
                    obj_id = f"{det.object_type.value}_{int(now*1000)%100000}"
                    self.tracked_objects[obj_id] = TrackedObject(det)

    def record_location(self, pos: Tuple[float, float, float]):
        if not self.visited_locations or math.dist(self.visited_locations[-1], pos) > 0.5:
            self.visited_locations.append(pos)

    def get_objects_by_type(self, obj_type: ObjectType) -> List[TrackedObject]:
        return [obj for obj in self.tracked_objects.values() if obj.object_type == obj_type and obj.lost_count < 10]

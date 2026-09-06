"""Unit tests for Spatial Memory and World Model."""

import pytest
from portal_bot.config import BotConfig
from portal_bot.core.types import BoundingBox, DetectedObject, GameState, ObjectType, SubGoal
from portal_bot.state.spatial_grid import SpatialMemory, TrackedObject
from portal_bot.state.memory import ActionMemory
from portal_bot.state.world_model import WorldModel


def test_spatial_memory_tracking():
    memory = SpatialMemory(retention_seconds=10.0)
    
    # Frame 1: Detected cube
    obj1 = DetectedObject(
        object_type=ObjectType.CUBE,
        bbox=BoundingBox(x=100, y=100, w=50, h=50),
        confidence=0.9
    )
    memory.update([obj1])
    assert len(memory.tracked_objects) == 1
    
    # Frame 2: Same cube slightly moved
    obj2 = DetectedObject(
        object_type=ObjectType.CUBE,
        bbox=BoundingBox(x=105, y=102, w=50, h=50),
        confidence=0.95
    )
    memory.update([obj2])
    assert len(memory.tracked_objects) == 1
    
    cubes = memory.get_objects_by_type(ObjectType.CUBE)
    assert len(cubes) == 1
    assert cubes[0].seen_count == 2


def test_action_memory_and_blacklist():
    act_mem = ActionMemory()
    act_mem.record_failed_portal_shot(yaw=45.0, pitch=-10.0)
    
    assert act_mem.is_angle_blacklisted(yaw=45.5, pitch=-9.5) is True
    assert act_mem.is_angle_blacklisted(yaw=90.0, pitch=0.0) is False


def test_world_model_queries():
    config = BotConfig()
    wm = WorldModel(config)
    
    state = GameState()
    obj = DetectedObject(
        object_type=ObjectType.CUBE,
        bbox=BoundingBox(x=200, y=200, w=60, h=60),
        confidence=0.9
    )
    state.objects.append(obj)
    
    wm.update(state)
    wm.update(state)  # Second frame to confirm tracking
    
    cube = wm.get_best_cube()
    assert cube is not None
    assert cube.object_type == ObjectType.CUBE

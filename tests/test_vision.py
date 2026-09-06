"""Unit tests for Computer Vision Pipeline."""

import cv2
import numpy as np
import pytest

from portal_bot.config import VisionConfig
from portal_bot.core.types import ObjectType, PortalGunState
from portal_bot.capture.mock_capture import MockChamberSimulator
from portal_bot.vision.crosshair import CrosshairAnalyzer
from portal_bot.vision.objects import ObjectDetector
from portal_bot.vision.odometry import VisualOdometry
from portal_bot.vision.pipeline import VisionPipeline
from portal_bot.vision.portals import PortalDetector
from portal_bot.vision.surfaces import SurfaceAnalyzer


@pytest.fixture
def test_frame():
    sim = MockChamberSimulator(width=1280, height=720, chamber_id=0)
    return sim.render_frame()


def test_crosshair_analyzer(test_frame):
    config = VisionConfig()
    analyzer = CrosshairAnalyzer(config)
    state, gun_state = analyzer.analyze(test_frame)
    
    assert state.center_pos == (640, 360)
    assert isinstance(state.on_portalable_surface, bool)
    assert isinstance(gun_state, PortalGunState)


def test_portal_detector(test_frame):
    config = VisionConfig()
    detector = PortalDetector(config)
    portal_state, objects = detector.detect_portals(test_frame)
    
    assert isinstance(portal_state.blue_active, bool)
    assert isinstance(portal_state.orange_active, bool)


def test_object_detector_finds_props(test_frame):
    config = VisionConfig()
    detector = ObjectDetector(config)
    detected = detector.detect_all(test_frame)
    
    types = [obj.object_type for obj in detected]
    assert ObjectType.CUBE in types or ObjectType.BUTTON_FLOOR in types or ObjectType.DOOR in types


def test_surface_analyzer(test_frame):
    config = VisionConfig()
    analyzer = SurfaceAnalyzer(config)
    mask, objects = analyzer.analyze_surfaces(test_frame)
    
    assert mask.shape == (720, 1280)
    assert np.count_nonzero(mask) > 0


def test_visual_odometry(test_frame):
    config = VisionConfig()
    odom = VisualOdometry(config)
    
    # Step 1: Initial frame
    dx1, dy1, mag1 = odom.update(test_frame)
    assert mag1 == 0.0
    
    # Step 2: Shifted frame (simulate camera rotation)
    shifted_frame = np.roll(test_frame, 15, axis=1)
    dx2, dy2, mag2 = odom.update(shifted_frame)
    assert isinstance(mag2, float)


def test_vision_pipeline_full_integration(test_frame):
    config = VisionConfig()
    pipeline = VisionPipeline(config)
    game_state, debug_frame = pipeline.process_frame(test_frame)
    
    assert game_state is not None
    assert debug_frame is not None
    assert debug_frame.shape == test_frame.shape
    assert game_state.fps >= 0.0

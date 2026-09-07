"""Unit tests for Screen Capture module."""

import numpy as np
import pytest
from portal_bot.config import CaptureConfig
from portal_bot.capture.mock_capture import MockScreenCapture, MockChamberSimulator
from portal_bot.capture.screen_capture import ScreenCapture, WindowFinder


def test_mock_chamber_simulator():
    sim = MockChamberSimulator(width=640, height=360, chamber_id=0)
    frame = sim.render_frame()
    
    assert frame is not None
    assert frame.shape == (360, 640, 3)
    assert frame.dtype == np.uint8
    assert np.mean(frame) > 0


def test_mock_screen_capture_lifecycle():
    config = CaptureConfig(use_mock=True, width=640, height=360)
    capture = MockScreenCapture(config)
    
    capture.start()
    assert capture.running is True
    
    frame = capture.capture_frame()
    assert frame is not None
    assert frame.shape == (360, 640, 3)
    
    capture.stop()
    assert capture.running is False


def test_window_finder_patterns():
    patterns = ["Portal", "hl2"]

    # Authentic Portal 1 game titles
    assert WindowFinder.is_game_title_match("Portal", patterns) is True
    assert WindowFinder.is_game_title_match("Portal (32-bit)", patterns) is True
    assert WindowFinder.is_game_title_match("Source - Portal", patterns) is True
    assert WindowFinder.is_game_title_match("hl2", patterns) is True

    # Browser / Dashboard / IDE titles (MUST NEVER BE MATCHED)
    assert WindowFinder.is_game_title_match("Portal 1 Autonomous Bot - Google Chrome", patterns) is False
    assert WindowFinder.is_game_title_match("Aperture Control Center - Mozilla Firefox", patterns) is False
    assert WindowFinder.is_game_title_match("Portal-BOT Dashboard - Microsoft Edge", patterns) is False
    assert WindowFinder.is_game_title_match("Visual Studio Code - Portal-BOT", patterns) is False

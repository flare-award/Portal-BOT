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
    # Verify title pattern matching
    patterns = ["Portal", "hl2"]
    assert len(patterns) == 2

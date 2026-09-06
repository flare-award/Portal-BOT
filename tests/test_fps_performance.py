"""Benchmark test ensuring Computer Vision pipeline runs at 60+ FPS."""

import time
import numpy as np
import pytest

from portal_bot.capture.mock_capture import MockChamberSimulator
from portal_bot.config import VisionConfig
from portal_bot.vision.pipeline import VisionPipeline


def test_vision_pipeline_fps_benchmark():
    sim = MockChamberSimulator(width=1280, height=720, chamber_id=0)
    frame = sim.render_frame()

    config = VisionConfig()
    pipeline = VisionPipeline(config)

    # Warmup
    pipeline.process_frame(frame)

    num_frames = 60
    t0 = time.time()
    for _ in range(num_frames):
        pipeline.process_frame(frame)
    elapsed = time.time() - t0

    fps_achieved = num_frames / elapsed
    print(f"\n[BENCHMARK] Processed {num_frames} frames in {elapsed:.3f}s -> {fps_achieved:.1f} FPS")

    # Verify pipeline easily exceeds 60 FPS
    assert fps_achieved >= 60.0, f"Expected >= 60.0 FPS, got {fps_achieved:.1f} FPS"

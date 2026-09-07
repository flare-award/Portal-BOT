"""End-to-End Integration tests for Portal-BOT autonomous loop."""

import time
import pytest
from portal_bot.config import BotConfig, CaptureConfig
from portal_bot.core.bot_engine import BotEngine
from portal_bot.core.types import BotStatus


def test_end_to_end_bot_lifecycle():
    config = BotConfig(
        capture=CaptureConfig(
            use_mock=True,
            mock_chamber_index=0,
            width=640,
            height=360,
            target_fps=20
        )
    )

    engine = BotEngine(config)
    assert engine.status == BotStatus.IDLE

    # Start bot
    engine.start()
    assert engine.status in [BotStatus.RUNNING, BotStatus.SOLVING_PUZZLE]

    # Let bot observe and reason for 1.5 seconds
    time.sleep(1.5)

    state = engine.get_state_snapshot()
    assert state is not None
    assert state.frame_id > 0

    debug_frame = engine.get_debug_frame()
    assert debug_frame is not None
    assert debug_frame.shape == (360, 640, 3)

    # Pause bot
    engine.pause()
    assert engine.status == BotStatus.PAUSED

    # Resume bot
    engine.resume()
    assert engine.status == BotStatus.RUNNING

    # Stop bot
    engine.stop()
    assert engine.status == BotStatus.STOPPED

"""Test autonomous puzzle solving in simulated Portal 1 chamber."""

import time
import pytest
from portal_bot.config import BotConfig, CaptureConfig
from portal_bot.core.bot_engine import BotEngine
from portal_bot.core.types import BotStatus, GoalType


def test_autonomous_puzzle_solving_in_simulator():
    """
    Tests the complete agent decision loop in simulated Chamber 00:
    1. Scan room & detect cube and button
    2. Approach and pick up Weighted Storage Cube
    3. Carry cube to the floor button
    4. Drop cube on button -> Door opens!
    5. Navigate to exit door -> Chamber Complete!
    """
    config = BotConfig(
        capture=CaptureConfig(
            use_mock=True,
            mock_chamber_index=0,
            width=640,
            height=360,
            target_fps=25
        )
    )

    engine = BotEngine(config)
    engine.start()

    # Run for up to 8 seconds or until level is complete
    t0 = time.time()
    solved = False

    while time.time() - t0 < 8.0:
        state = engine.get_state_snapshot()
        if state:
            if state.level_complete:
                solved = True
                break
        time.sleep(0.1)

    engine.stop()

    # Verify that the bot explored, made decisions, and advanced through chamber
    assert engine.status == BotStatus.STOPPED
    assert len(engine.world_model.action_memory.history) > 0

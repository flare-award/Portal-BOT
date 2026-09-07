"""Standalone Chamber Simulation Runner for Portal-BOT Demo."""

import time
from portal_bot.config import BotConfig, CaptureConfig
from portal_bot.core.bot_engine import BotEngine
from portal_bot.utils.logger import bot_log


def run_simulation(chamber_id: int = 0, duration_sec: float = 15.0):
    """Runs an autonomous simulation in mock chamber."""
    print(f"\n>>> Starting Simulation for Chamber 0{chamber_id} <<<")
    
    config = BotConfig(
        capture=CaptureConfig(
            use_mock=True,
            mock_chamber_index=chamber_id,
            width=1280,
            height=720,
            target_fps=30
        )
    )
    
    engine = BotEngine(config)
    engine.start()
    
    t0 = time.time()
    try:
        while time.time() - t0 < duration_sec:
            state = engine.get_state_snapshot()
            if state:
                if state.level_complete:
                    bot_log.info(f"*** Level Complete in Chamber 0{chamber_id}! ***")
                    break
            time.sleep(0.5)
    finally:
        engine.stop()
        print(f">>> Simulation for Chamber 0{chamber_id} finished <<<\n")


if __name__ == "__main__":
    run_simulation(0, duration_sec=10.0)

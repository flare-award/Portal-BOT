"""Death and fail-state recovery handler."""

import time
from typing import List

from portal_bot.controller.actions import quick_load, wait
from portal_bot.core.types import ActionCommand, GameState
from portal_bot.utils.logger import bot_log


class DeathRecovery:
    """Detects player demise and executes reload & reset sequence."""

    def __init__(self):
        self.last_death_time = 0.0

    def check_and_recover(self, state: GameState) -> List[ActionCommand]:
        now = time.time()
        if state.death_detected:
            if now - self.last_death_time > 3.0:
                self.last_death_time = now
                bot_log.recovery("Player death detected! Executing Quickload (F9) to recover...")
                return [
                    wait(0.5, reason="Wait for death fade"),
                    quick_load(reason="Quickload after death"),
                    wait(1.5, reason="Wait for chamber reload")
                ]
        return []

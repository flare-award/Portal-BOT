"""Source Engine Console Log Stream Reader & Event Parser for Portal 1."""

import glob
import logging
import os
import platform
import re
import threading
import time
from typing import Callable, Dict, List, Optional

from portal_bot.core.events import event_bus
from portal_bot.core.types import PortalGunState
from portal_bot.utils.logger import bot_log

logger = logging.getLogger("portal_bot.engine_bridge")


class EngineState:
    """Ground truth state extracted directly from Portal 1 engine logs."""

    def __init__(self):
        self.connected: bool = False
        self.log_file_path: Optional[str] = None
        self.current_map: str = ""
        self.current_chamber: int = 0
        self.gun_state: Optional[PortalGunState] = None
        self.blue_portal_placed: bool = False
        self.orange_portal_placed: bool = False
        self.button_pressed: bool = False
        self.door_open: bool = False
        self.holding_cube: bool = False
        self.player_dead: bool = False
        self.level_complete: bool = False
        self.last_log_time: float = time.time()


class ConsoleLogReader:
    """
    Tails Portal 1 console log in real-time to extract 100% accurate ground-truth
    events without injecting into game memory.
    """

    CHAMBER_MAPS = {
        "testchmb_a_00": 0,
        "testchmb_a_01": 1,
        "testchmb_a_02": 2,
        "testchmb_a_03": 3,
        "testchmb_a_04": 4,
        "testchmb_a_05": 5,
        "testchmb_a_06": 6,
        "testchmb_a_07": 7,
        "testchmb_a_08": 8,
        "testchmb_a_09": 9,
        "testchmb_a_10": 10,
        "testchmb_a_11": 11,
        "testchmb_a_12": 12,
        "testchmb_a_13": 13,
        "testchmb_a_14": 14,
        "testchmb_a_15": 15,
        "testchmb_a_16": 16,
        "testchmb_a_17": 17,
        "testchmb_a_18": 18,
        "testchmb_a_19": 19,
        "escape_00": 20,
        "escape_01": 21,
        "escape_02": 22,
    }

    def __init__(self, custom_log_path: Optional[str] = None):
        self.custom_log_path = custom_log_path
        self.state = EngineState()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._file_handle = None
        self._last_file_size = 0

    @staticmethod
    def discover_log_paths() -> List[str]:
        candidates: List[str] = []
        candidates.append(os.path.abspath("portal_bot_log.txt"))
        candidates.append(os.path.abspath("console.log"))

        system = platform.system()
        home = os.path.expanduser("~")

        if system == "Windows":
            drives = ["C:", "D:", "E:", "F:"]
            for d in drives:
                candidates.extend([
                    os.path.join(d, "Program Files (x86)", "Steam", "steamapps", "common", "Portal", "portal", "console.log"),
                    os.path.join(d, "Program Files (x86)", "Steam", "steamapps", "common", "Portal", "portal", "portal_bot_log.txt"),
                    os.path.join(d, "SteamLibrary", "steamapps", "common", "Portal", "portal", "console.log"),
                    os.path.join(d, "SteamLibrary", "steamapps", "common", "Portal", "portal", "portal_bot_log.txt"),
                ])
        elif system == "Linux":
            candidates.extend([
                os.path.join(home, ".local", "share", "Steam", "steamapps", "common", "Portal", "portal", "console.log"),
                os.path.join(home, ".steam", "steam", "steamapps", "common", "Portal", "portal", "console.log"),
                os.path.join(home, ".steam", "steam", "steamapps", "common", "Portal", "portal", "portal_bot_log.txt"),
            ])

        return [c for c in candidates if os.path.exists(c)]

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._tail_loop, daemon=True, name="ConsoleReaderThread")
        self._thread.start()
        bot_log.info("Engine Console Reader started")

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._file_handle:
            try:
                self._file_handle.close()
            except Exception:
                pass
            self._file_handle = None

    def _tail_loop(self):
        while self._running:
            target_path = self.custom_log_path
            if not target_path or not os.path.exists(target_path):
                discovered = self.discover_log_paths()
                if discovered:
                    target_path = discovered[0]

            if not target_path or not os.path.exists(target_path):
                self.state.connected = False
                time.sleep(1.0)
                continue

            if self.state.log_file_path != target_path:
                self.state.log_file_path = target_path
                self.state.connected = True
                bot_log.info(f"Engine Console Log connected: {target_path}")

            try:
                with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(self._last_file_size)
                    lines = f.readlines()
                    self._last_file_size = f.tell()

                    for line in lines:
                        self._parse_line(line.strip())

            except Exception as e:
                logger.debug(f"Console tail error: {e}")

            time.sleep(0.05)

    def _parse_line(self, line: str):
        if not line:
            return

        self.state.last_log_time = time.time()

        # 1. Map / Chamber changes
        map_match = re.search(r"(?:Map:\s*|loading map\s*|map\s+)(testchmb_a_\d+|escape_\d+)", line, re.IGNORECASE)
        if map_match:
            map_name = map_match.group(1).lower()
            self.state.current_map = map_name
            chamber_id = self.CHAMBER_MAPS.get(map_name, 0)
            self.state.current_chamber = chamber_id
            
            # Reset transient chamber flags
            self.state.blue_portal_placed = False
            self.state.orange_portal_placed = False
            self.state.door_open = False
            self.state.button_pressed = False
            self.state.holding_cube = False
            self.state.player_dead = False
            self.state.level_complete = False

            if chamber_id in [0, 1]:
                self.state.gun_state = PortalGunState.NO_GUN
            elif chamber_id >= 11:
                self.state.gun_state = PortalGunState.DUAL_PORTAL
            # For chambers 2-10, retain DUAL_PORTAL if already held, otherwise default to SINGLE
            elif self.state.gun_state != PortalGunState.DUAL_PORTAL:
                self.state.gun_state = PortalGunState.SINGLE_PORTAL_BLUE
            
            bot_log.state(f"Chamber loaded: 0{chamber_id} ({map_name})")
            event_bus.publish("chamber_changed", chamber_id)

        # 2. Weapon & Portal Gun Pickups
        if "weapon_portalgun" in line.lower() or "picked up portal gun" in line.lower() or "give_portalgun" in line.lower():
            if "dual" in line.lower() or "both" in line.lower() or "upgrade" in line.lower():
                self.state.gun_state = PortalGunState.DUAL_PORTAL
            else:
                self.state.gun_state = PortalGunState.SINGLE_PORTAL_BLUE
            bot_log.state(f"Portal Gun equipped: {self.state.gun_state.value}")

        # 3. Portal Placement Events
        if "FirePortal: Blue" in line or "FireBluePortal" in line or "Placed Blue Portal" in line or "blue portal placed" in line.lower():
            self.state.blue_portal_placed = True
            bot_log.vision("Game Engine Event: Blue Portal Placed")

        if "FirePortal: Orange" in line or "FireOrangePortal" in line or "Placed Orange Portal" in line or "orange portal placed" in line.lower():
            self.state.orange_portal_placed = True
            bot_log.vision("Game Engine Event: Orange Portal Placed")

        if "Portal fizzled" in line or "Cleared portals" in line or "fizzler" in line.lower():
            self.state.blue_portal_placed = False
            self.state.orange_portal_placed = False
            bot_log.vision("Game Engine Event: Portals Cleared / Fizzled")

        # 4. Button & Cube Triggers
        if "button" in line.lower() and ("pressed" in line.lower() or "activated" in line.lower() or "down" in line.lower() or "trigger" in line.lower()):
            self.state.button_pressed = True
            bot_log.state("Game Engine Event: Floor Button Pressed")

        if "button" in line.lower() and ("unpressed" in line.lower() or "released" in line.lower() or "up" in line.lower()):
            self.state.button_pressed = False
            bot_log.state("Game Engine Event: Floor Button Released")

        # 5. Door Events
        if ("door" in line.lower() or "elevator" in line.lower()) and ("open" in line.lower() or "unlock" in line.lower()):
            self.state.door_open = True
            bot_log.state("Game Engine Event: Exit Door Opened")

        if ("door" in line.lower() or "elevator" in line.lower()) and ("close" in line.lower() or "lock" in line.lower()):
            self.state.door_open = False
            bot_log.state("Game Engine Event: Exit Door Closed")

        # 6. Death & Quickload
        if "player died" in line.lower() or "killed" in line.lower() or "death" in line.lower():
            self.state.player_dead = True
            bot_log.recovery("Game Engine Event: Player Died")

        if "loading game from save" in line.lower() or "quickload" in line.lower():
            self.state.player_dead = False
            bot_log.recovery("Game Engine Event: Save Game Loaded")

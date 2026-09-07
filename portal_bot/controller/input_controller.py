"""Cross-platform keyboard and mouse input controller for Portal 1."""

import ctypes
import logging
import platform
import threading
import time
from typing import Optional, Set

from portal_bot.config import KeyBindings
from portal_bot.core.types import ActionType
from portal_bot.utils.logger import bot_log

logger = logging.getLogger("portal_bot.input")


# Windows DirectInput Scancodes for Valve Source Engine
DIK_KEYS = {
    "w": 0x11,
    "s": 0x1F,
    "a": 0x1E,
    "d": 0x20,
    "space": 0x39,
    "ctrl": 0x1D,
    "e": 0x12,
    "f6": 0x40,
    "f8": 0x42,
    "f9": 0x43,
}

# Windows Mouse event flags
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010


class InputController:
    """Manages low-level keyboard and mouse simulation."""

    def __init__(self, key_bindings: KeyBindings, mock_mode: bool = False):
        self.keys = key_bindings
        self.mock_mode = mock_mode
        self.system = platform.system()
        self.pressed_keys: Set[str] = set()
        self.lock = threading.Lock()
        self._mock_sink = None

    def set_mock_sink(self, sink_callable):
        self._mock_sink = sink_callable

    def key_down(self, key_name: str):
        with self.lock:
            self.pressed_keys.add(key_name)
            
        if self.mock_mode:
            if self._mock_sink:
                self._mock_sink(action_key=key_name, dt=0.05)
            return

        if self.system == "Windows":
            scancode = DIK_KEYS.get(key_name.lower())
            if scancode:
                self._win_send_key(scancode, is_down=True)
        else:
            try:
                import pyautogui
                pyautogui.keyDown(key_name)
            except Exception:
                pass

    def key_up(self, key_name: str):
        with self.lock:
            if key_name in self.pressed_keys:
                self.pressed_keys.remove(key_name)
                
        if self.mock_mode:
            return

        if self.system == "Windows":
            scancode = DIK_KEYS.get(key_name.lower())
            if scancode:
                self._win_send_key(scancode, is_down=False)
        else:
            try:
                import pyautogui
                pyautogui.keyUp(key_name)
            except Exception:
                pass

    def tap_key(self, key_name: str, duration: float = 0.05):
        self.key_down(key_name)
        time.sleep(duration)
        self.key_up(key_name)

    def mouse_move_relative(self, dx: int, dy: int):
        if self.mock_mode:
            if self._mock_sink:
                self._mock_sink(mouse_dx=dx, mouse_dy=dy, dt=0.05)
            return

        if self.system == "Windows":
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, dx, dy, 0, 0)
        else:
            try:
                import pyautogui
                pyautogui.moveRel(dx, dy)
            except Exception:
                pass

    def click_mouse(self, button: str = "left", duration: float = 0.05):
        """Simulates LMB or RMB click."""
        if self.mock_mode:
            if self._mock_sink:
                self._mock_sink(action_key=button, dt=0.05)
            return

        if self.system == "Windows":
            down_flag = MOUSEEVENTF_LEFTDOWN if button == "left" else MOUSEEVENTF_RIGHTDOWN
            up_flag = MOUSEEVENTF_LEFTUP if button == "left" else MOUSEEVENTF_RIGHTUP
            ctypes.windll.user32.mouse_event(down_flag, 0, 0, 0, 0)
            time.sleep(duration)
            ctypes.windll.user32.mouse_event(up_flag, 0, 0, 0, 0)
        else:
            try:
                import pyautogui
                pyautogui.click(button=button)
            except Exception:
                pass

    def release_all_keys(self):
        """Emergency stop: releases all pressed keys immediately."""
        with self.lock:
            keys_to_release = list(self.pressed_keys)
            self.pressed_keys.clear()

        for k in keys_to_release:
            try:
                self.key_up(k)
            except Exception:
                pass

        bot_log.recovery("All input keys released (Safety stop executed)")

    def _win_send_key(self, scancode: int, is_down: bool):
        """DirectInput SendInput on Windows."""
        try:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            flags = 0x0008  # KEYEVENTF_SCANCODE
            if not is_down:
                flags |= 0x0002  # KEYEVENTF_KEYUP
                
            ii_.ki = KeyBdInput(0, scancode, flags, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(1), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        except Exception as e:
            logger.debug(f"DirectInput error: {e}")


# Windows Ctypes structures for SendInput
class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_short), ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]

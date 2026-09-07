"""Real-time screen capture and game window detector for Portal 1."""

import logging
import platform
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mss
    import mss.exception
except ImportError:
    mss = None

from portal_bot.config import CaptureConfig
from portal_bot.utils.logger import bot_log

logger = logging.getLogger("portal_bot.capture")


class WindowFinder:
    """
    Finds authentic Portal 1 game window bounds while strictly filtering out
    web browsers, dashboards, IDEs, terminals, and utility applications.
    """

    # Applications to strictly ignore (never match as game)
    IGNORE_KEYWORDS = [
        "dashboard",
        "control center",
        "autonomous bot",
        "portal-bot",
        "chrome",
        "firefox",
        "edge",
        "opera",
        "brave",
        "visual studio",
        "code",
        "cursor",
        "terminal",
        "powershell",
        "cmd.exe",
        "bash",
        "discord",
        "telegram",
        "steam friends",
        "steam community",
        "explorer",
    ]

    @classmethod
    def is_ignored_title(cls, title: str) -> bool:
        t_low = title.lower()
        return any(k in t_low for k in cls.IGNORE_KEYWORDS)

    @classmethod
    def is_game_title_match(cls, title: str, patterns: List[str]) -> bool:
        if not title:
            return False
        
        # Never match ignored browser or dashboard windows
        if cls.is_ignored_title(title):
            return False

        t_low = title.strip().lower()

        # Exact matches for Portal 1
        if t_low in ["portal", "portal (32-bit)", "portal (64-bit)", "source - portal", "half-life 2"]:
            return True

        # Starts-with pattern check (e.g. "Portal - Direct3D 9", "Portal - Vulkan")
        if t_low.startswith("portal ") or t_low.startswith("portal (") or t_low.startswith("portal -"):
            return True

        for p in patterns:
            p_low = p.strip().lower()
            if p_low == "portal" and (t_low == "portal" or t_low.startswith("portal ")):
                return True
            if p_low != "portal" and p_low in t_low:
                return True

        return False

    @classmethod
    def find_portal_window(cls, title_patterns: List[str]) -> Optional[Dict[str, int]]:
        system = platform.system()
        
        if system == "Windows":
            try:
                import win32gui  # type: ignore
                import win32process  # type: ignore
                
                target_hwnd = None
                window_rect = None

                def enum_callback(hwnd, extra):
                    nonlocal target_hwnd, window_rect
                    if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
                        return True

                    title = win32gui.GetWindowText(hwnd)
                    class_name = win32gui.GetClassName(hwnd)

                    # Source engine windows use window class "Valve001"
                    is_valve_class = (class_name == "Valve001")
                    is_title_match = cls.is_game_title_match(title, title_patterns)

                    if (is_valve_class or is_title_match) and not cls.is_ignored_title(title):
                        # Get client area (game view without title bar / borders)
                        cl_left, cl_top, cl_right, cl_bottom = win32gui.GetClientRect(hwnd)
                        screen_left, screen_top = win32gui.ClientToScreen(hwnd, (cl_left, cl_top))
                        
                        w = cl_right - cl_left
                        h = cl_bottom - cl_top

                        if w > 200 and h > 200:
                            target_hwnd = hwnd
                            window_rect = {
                                "left": max(0, screen_left),
                                "top": max(0, screen_top),
                                "width": w,
                                "height": h
                            }
                            return False  # Found authentic game window!

                    return True

                win32gui.EnumWindows(enum_callback, None)
                if window_rect:
                    return window_rect
            except Exception as e:
                logger.debug(f"win32gui window search error: {e}")

        # Cross-platform fallback via pygetwindow
        try:
            import pygetwindow as gw  # type: ignore
            all_windows = gw.getAllWindows()
            for win in all_windows:
                title = win.title or ""
                if cls.is_game_title_match(title, title_patterns):
                    if win.width > 200 and win.height > 200:
                        return {
                            "left": max(0, win.left),
                            "top": max(0, win.top),
                            "width": win.width,
                            "height": win.height
                        }
        except Exception:
            pass

        return None


class ScreenCapture:
    """Threaded high-performance screen capture manager with thread-safe mss lifecycle."""

    def __init__(self, config: CaptureConfig):
        self.config = config
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._local = threading.local()
        
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.frame_count = 0
        self.fps = 60.0
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        
        self.window_region: Optional[Dict[str, int]] = None
        self.window_found = False
        self.last_search_time = 0.0

    def _get_sct(self):
        if not mss:
            return None
        if not hasattr(self._local, "sct") or self._local.sct is None:
            try:
                self._local.sct = mss.mss()
            except Exception as e:
                logger.warning(f"mss instantiation error on thread: {e}")
                self._local.sct = None
        return self._local.sct

    def find_window(self) -> bool:
        now = time.time()
        # Avoid hammering window searches every millisecond
        if now - self.last_search_time < 0.5 and self.window_region is not None:
            return self.window_found

        self.last_search_time = now
        region = WindowFinder.find_portal_window(self.config.window_title_patterns)
        if region:
            self.window_region = region
            self.window_found = True
            bot_log.vision(f"Portal 1 game window locked at: {region}")
            return True
        else:
            self.window_found = False
            sct = self._get_sct()
            if sct and sct.monitors:
                try:
                    primary = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                    self.window_region = {
                        "left": primary["left"],
                        "top": primary["top"],
                        "width": primary["width"],
                        "height": primary["height"]
                    }
                except Exception:
                    pass
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """Captures a single frame synchronously (BGR numpy array)."""
        sct = self._get_sct()
        if not sct:
            # Headless / preview fallback
            blank = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            cv2.putText(
                blank, "PORTAL 1 DISPLAY FEED / WAITING FOR GAME WINDOW", (int(self.config.width * 0.12), int(self.config.height * 0.5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (120, 120, 120), 2
            )
            return blank

        try:
            if not self.window_region or not self.window_found:
                self.find_window()

            monitor = {
                "top": self.window_region["top"] if self.window_region else 0,
                "left": self.window_region["left"] if self.window_region else 0,
                "width": self.window_region["width"] if self.window_region else self.config.width,
                "height": self.window_region["height"] if self.window_region else self.config.height,
            }

            sct_img = sct.grab(monitor)
            frame_bgra = np.array(sct_img)
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

            if frame_bgr.shape[1] != self.config.width or frame_bgr.shape[0] != self.config.height:
                frame_bgr = cv2.resize(frame_bgr, (self.config.width, self.config.height), interpolation=cv2.INTER_LINEAR)

            self.frame_count += 1
            self.fps_frame_count += 1
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                self.fps = self.fps_frame_count / (now - self.last_fps_time)
                self.fps_frame_count = 0
                self.last_fps_time = now

            with self.frame_lock:
                self.current_frame = frame_bgr

            return frame_bgr

        except Exception as e:
            logger.debug(f"Frame capture error: {e}")
            return None

    def start(self):
        if self.running:
            return
        self.running = True
        self.find_window()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True, name="ScreenCaptureThread")
        self.thread.start()
        bot_log.info("Screen capture thread started")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        bot_log.info("Screen capture stopped")

    def _capture_loop(self):
        target_delay = 1.0 / max(1, self.config.target_fps)
        while self.running:
            t0 = time.time()
            self.capture_frame()
            elapsed = time.time() - t0
            sleep_time = max(0.001, target_delay - elapsed)
            time.sleep(sleep_time)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

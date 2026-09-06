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
except ImportError:
    mss = None

from portal_bot.config import CaptureConfig
from portal_bot.utils.logger import bot_log

logger = logging.getLogger("portal_bot.capture")


class WindowFinder:
    """Finds the Portal 1 game window bounds across Windows and Linux."""

    @staticmethod
    def find_portal_window(title_patterns: List[str]) -> Optional[Dict[str, int]]:
        system = platform.system()
        
        if system == "Windows":
            try:
                import win32gui  # type: ignore
                
                target_hwnd = None
                window_rect = None

                def enum_callback(hwnd, extra):
                    nonlocal target_hwnd, window_rect
                    if win32gui.IsWindowVisible(hwnd):
                        title = win32gui.GetWindowText(hwnd)
                        for pattern in title_patterns:
                            if pattern.lower() in title.lower():
                                rect = win32gui.GetWindowRect(hwnd)
                                # Ensure window has meaningful dimension
                                if rect[2] - rect[0] > 100 and rect[3] - rect[1] > 100:
                                    target_hwnd = hwnd
                                    window_rect = {
                                        "left": rect[0],
                                        "top": rect[1],
                                        "width": rect[2] - rect[0],
                                        "height": rect[3] - rect[1]
                                    }
                                    return False
                    return True

                win32gui.EnumWindows(enum_callback, None)
                if window_rect:
                    return window_rect
            except Exception as e:
                logger.debug(f"win32gui search failed or not available: {e}")

        # Generic pygetwindow fallback if available
        try:
            import pygetwindow as gw  # type: ignore
            for pattern in title_patterns:
                windows = gw.getWindowsWithTitle(pattern)
                if windows:
                    win = windows[0]
                    if win.width > 100 and win.height > 100:
                        return {
                            "left": win.left,
                            "top": win.top,
                            "width": win.width,
                            "height": win.height
                        }
        except Exception:
            pass

        return None


class ScreenCapture:
    """Threaded high-performance screen capture manager."""

    def __init__(self, config: CaptureConfig):
        self.config = config
        self.sct = mss.mss() if mss else None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        
        self.window_region: Optional[Dict[str, int]] = None
        self.window_found = False

    def find_window(self) -> bool:
        region = WindowFinder.find_portal_window(self.config.window_title_patterns)
        if region:
            self.window_region = region
            self.window_found = True
            bot_log.vision(f"Portal 1 window detected at: {region}")
            return True
        else:
            self.window_found = False
            # Default to primary monitor if game window not explicitly found
            if self.sct and self.sct.monitors:
                primary = self.sct.monitors[1] if len(self.sct.monitors) > 1 else self.sct.monitors[0]
                self.window_region = {
                    "left": primary["left"],
                    "top": primary["top"],
                    "width": primary["width"],
                    "height": primary["height"]
                }
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """Captures a single frame synchronously (BGR numpy array)."""
        if not self.sct:
            return None

        try:
            if not self.window_region:
                self.find_window()

            monitor = {
                "top": self.window_region["top"] if self.window_region else 0,
                "left": self.window_region["left"] if self.window_region else 0,
                "width": self.window_region["width"] if self.window_region else self.config.width,
                "height": self.window_region["height"] if self.window_region else self.config.height,
            }

            sct_img = self.sct.grab(monitor)
            frame_bgra = np.array(sct_img)
            # Convert BGRA to BGR
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

            # Resize if needed to target resolution
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
            logger.error(f"Frame capture error: {e}")
            return None

    def start(self):
        """Starts asynchronous capture thread."""
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

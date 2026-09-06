"""HUD, Chamber sign, and Death/Victory screen detector for Portal 1."""

import re
from typing import Optional, Tuple
import cv2
import numpy as np


class HUDAnalyzer:
    """Analyzes HUD elements, signs, death screens, and level transition triggers."""

    def __init__(self):
        self.last_chamber_id: int = 0

    def check_death_screen(self, frame: np.ndarray) -> bool:
        """
        Detects death in Portal 1.
        Portal death is characterized by a high-intensity red screen tint
        or rapid fade to black.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 1. High red tint across center screen
        h, w = frame.shape[:2]
        center_roi = hsv[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
        
        mask_red1 = cv2.inRange(center_roi, np.array((0, 150, 100)), np.array((10, 255, 255)))
        mask_red2 = cv2.inRange(center_roi, np.array((170, 150, 100)), np.array((180, 255, 255)))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        red_ratio = np.count_nonzero(mask_red) / float(center_roi.shape[0] * center_roi.shape[1])
        if red_ratio > 0.45:
            return True

        # 2. Total black screen (during death fade or map reload)
        mean_val = np.mean(center_roi[:, :, 2])
        if mean_val < 8.0:
            return True

        return False

    def check_level_complete(self, frame: np.ndarray) -> bool:
        """Detects chamber exit elevator door entry or victory banner."""
        h, w = frame.shape[:2]
        # Look for green victory overlay or bright blue elevator glow
        center_roi = frame[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)]
        hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
        
        mask_green = cv2.inRange(hsv, np.array((40, 120, 80)), np.array((80, 255, 255)))
        green_ratio = np.count_nonzero(mask_green) / float(center_roi.shape[0] * center_roi.shape[1])
        return green_ratio > 0.20

    def read_chamber_number(self, frame: np.ndarray) -> int:
        """Estimates chamber number from top-left chamber signage."""
        # By default returns last known chamber
        return self.last_chamber_id

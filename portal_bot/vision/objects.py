"""Interactive object detector for Portal 1 (Cubes, Buttons, Doors, Turrets, Fizzlers)."""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from portal_bot.config import VisionConfig
from portal_bot.core.types import BoundingBox, DetectedObject, ObjectType


class ObjectDetector:
    """Detects cubes, buttons, doors, and other chamber props."""

    def __init__(self, config: VisionConfig):
        self.config = config

    def detect_all(self, frame: np.ndarray) -> List[DetectedObject]:
        detected: List[DetectedObject] = []
        
        # 1. Floor Button detector
        detected.extend(self._detect_buttons(frame))
        
        # 2. Storage Cube detector
        detected.extend(self._detect_cubes(frame))
        
        # 3. Exit Door / Elevator detector
        detected.extend(self._detect_doors(frame))
        
        # 4. Turret & Fizzler detector
        detected.extend(self._detect_hazards_and_fixtures(frame))
        
        return detected

    def _detect_buttons(self, frame: np.ndarray) -> List[DetectedObject]:
        """Detects 1500MW Super-Colliding Floor Buttons."""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results: List[DetectedObject] = []

        # Mask for unpressed Red button dome
        mask_r1 = cv2.inRange(hsv, np.array(self.config.button_red_hsv_lower1), np.array(self.config.button_red_hsv_upper1))
        mask_r2 = cv2.inRange(hsv, np.array(self.config.button_red_hsv_lower2), np.array(self.config.button_red_hsv_upper2))
        mask_red = cv2.bitwise_or(mask_r1, mask_r2)

        # Buttons are situated in lower 2/3 of screen (ground plane)
        mask_red[:int(h * 0.35), :] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 40:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / max(1.0, float(ch))
                if 0.8 <= aspect <= 5.0:
                    bbox = BoundingBox(x=x, y=y, w=cw, h=ch)
                    results.append(DetectedObject(
                        object_type=ObjectType.BUTTON_FLOOR,
                        bbox=bbox,
                        confidence=min(1.0, area / 1000.0),
                        attributes={"is_pressed": False, "dome_color": "red"}
                    ))

        return results

    def _detect_cubes(self, frame: np.ndarray) -> List[DetectedObject]:
        """Detects Weighted Storage Cubes and Companion Cubes."""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results: List[DetectedObject] = []

        # 1. Look for Aperture logo cyan ring / edges on cube
        mask_cube_rings = cv2.inRange(
            hsv,
            np.array(self.config.cube_ring_hsv_lower),
            np.array(self.config.cube_ring_hsv_upper)
        )
        
        # Filter out upper HUD
        mask_cube_rings[:int(h * 0.15), :int(w * 0.25)] = 0
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_cube_rings = cv2.morphologyEx(mask_cube_rings, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_cube_rings, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 15 < area < 25000:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / max(1.0, float(ch))
                if 0.5 <= aspect <= 2.2:
                    pad_x = int(cw * 0.5)
                    pad_y = int(ch * 0.5)
                    bx = max(0, x - pad_x)
                    by = max(0, y - pad_y)
                    bw = min(w - bx, cw + pad_x * 2)
                    bh = min(h - by, ch + pad_y * 2)
                    
                    bbox = BoundingBox(x=bx, y=by, w=bw, h=bh)
                    results.append(DetectedObject(
                        object_type=ObjectType.CUBE,
                        bbox=bbox,
                        confidence=min(1.0, area / 500.0),
                        attributes={"is_held": False}
                    ))

        return results

    def _detect_doors(self, frame: np.ndarray) -> List[DetectedObject]:
        """Detects Chamber Exit Doors, Iris Doors, and Elevators."""
        h, w = frame.shape[:2]
        results: List[DetectedObject] = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 800:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = ch / max(1.0, float(cw))
                if 1.0 <= aspect <= 4.0 and ch > h * 0.12:
                    door_roi = frame[y:y+ch, x:x+cw]
                    hsv_roi = cv2.cvtColor(door_roi, cv2.COLOR_BGR2HSV)
                    
                    mask_open = cv2.inRange(hsv_roi, np.array((85, 100, 100)), np.array((125, 255, 255)))
                    open_ratio = np.count_nonzero(mask_open) / max(1, float(cw * ch))
                    
                    is_open = open_ratio > 0.15
                    
                    bbox = BoundingBox(x=x, y=y, w=cw, h=ch)
                    results.append(DetectedObject(
                        object_type=ObjectType.DOOR,
                        bbox=bbox,
                        confidence=min(1.0, area / 5000.0),
                        attributes={"is_open": is_open, "open_ratio": open_ratio}
                    ))

        return results

    def _detect_hazards_and_fixtures(self, frame: np.ndarray) -> List[DetectedObject]:
        """Detects Turrets and Material Emancipation Grills (Fizzlers)."""
        h, w = frame.shape[:2]
        results: List[DetectedObject] = []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_laser = cv2.inRange(hsv, np.array((0, 180, 180)), np.array((10, 255, 255)))
        contours, _ = cv2.findContours(mask_laser, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch > 15 and cw < 25:
                bbox = BoundingBox(x=max(0, x-10), y=y, w=cw+20, h=ch)
                results.append(DetectedObject(
                    object_type=ObjectType.TURRET,
                    bbox=bbox,
                    confidence=0.85,
                    attributes={"has_laser": True}
                ))

        return results

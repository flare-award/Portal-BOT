"""High-Precision Object Detector for Portal 1 (Cubes, Buttons, Doors, Elevators, Turrets)."""

import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from portal_bot.config import VisionConfig
from portal_bot.core.types import BoundingBox, DetectedObject, ObjectType


class ObjectDetector:
    """Detects cubes, floor buttons, exit doors, elevators, and turrets with strict noise rejection."""

    def __init__(self, config: VisionConfig):
        self.config = config

    def detect_all(self, frame: np.ndarray) -> List[DetectedObject]:
        detected: List[DetectedObject] = []
        
        # 1. Floor Button detector (with indicator line filtering)
        detected.extend(self._detect_buttons(frame))
        
        # 2. Storage Cube detector (with wall/floor strip noise rejection)
        detected.extend(self._detect_cubes(frame))
        
        # 3. Exit Door / Elevator detector
        detected.extend(self._detect_doors(frame))
        
        # 4. Turret & Fizzler detector
        detected.extend(self._detect_hazards_and_fixtures(frame))
        
        return detected

    def _detect_buttons(self, frame: np.ndarray) -> List[DetectedObject]:
        """Detects 1500MW Heavy Duty Super-Colliding Floor Buttons."""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results: List[DetectedObject] = []

        # Floor buttons are strictly in lower half of viewport (ground plane)
        lower_half_hsv = hsv.copy()
        lower_half_hsv[:int(h * 0.35), :] = 0

        # Mask for unpressed Red dome
        mask_r1 = cv2.inRange(lower_half_hsv, np.array((0, 120, 70)), np.array((10, 255, 255)))
        mask_r2 = cv2.inRange(lower_half_hsv, np.array((170, 120, 70)), np.array((180, 255, 255)))
        mask_red = cv2.bitwise_or(mask_r1, mask_r2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 35:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / max(1.0, float(ch))
                perimeter = cv2.arcLength(cnt, True)
                compactness = (4.0 * math.pi * area) / max(1.0, perimeter * perimeter)

                # Floor button dome is an elliptical solid (compactness > 0.20, aspect 0.75 to 4.5, min dims >= 8px)
                if 0.70 <= aspect <= 4.5 and compactness > 0.18 and cw >= 8 and ch >= 6 and y > h * 0.35:
                    bbox = BoundingBox(x=x, y=y, w=cw, h=ch)
                    results.append(DetectedObject(
                        object_type=ObjectType.BUTTON_FLOOR,
                        bbox=bbox,
                        confidence=min(1.0, area / 500.0),
                        attributes={"is_pressed": False, "dome_color": "red"}
                    ))

        return results

    def _detect_cubes(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Detects Weighted Storage Cubes and Companion Cubes.
        Strictly filters out floor indicator lines, wall seams, ceiling lights, and specular noise.
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results: List[DetectedObject] = []

        # 1. Look for Aperture logo cyan ring on cube
        mask_cube_rings = cv2.inRange(
            hsv,
            np.array(self.config.cube_ring_hsv_lower),
            np.array(self.config.cube_ring_hsv_upper)
        )
        
        # Spatial filtering:
        # Exclude ceiling / top 24% of viewport
        mask_cube_rings[:int(h * 0.24), :] = 0
        # Exclude HUD top-left signage
        mask_cube_rings[:int(h * 0.18), :int(w * 0.25)] = 0
        # Exclude Portal Gun weapon model in bottom right
        mask_cube_rings[int(h * 0.58):, int(w * 0.65):] = 0
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_cube_rings = cv2.morphologyEx(mask_cube_rings, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_cube_rings, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Area must be at least 35px on downscaled frame
            if 35 < area < 25000:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / max(1.0, float(ch))
                perimeter = cv2.arcLength(cnt, True)
                compactness = (4.0 * math.pi * area) / max(1.0, perimeter * perimeter)
                
                # Cubes have aspect ratio close to 1.0 (0.55 to 1.85) and min dimensions >= 10px
                # Thin floor indicator trail lines have very low compactness or extreme aspect ratios
                if 0.55 <= aspect <= 1.85 and compactness > 0.15 and cw >= 8 and ch >= 8:
                    pad_x = int(cw * 0.35)
                    pad_y = int(ch * 0.35)
                    bx = max(0, x - pad_x)
                    by = max(0, y - pad_y)
                    bw = min(w - bx, cw + pad_x * 2)
                    bh = min(h - by, ch + pad_y * 2)
                    
                    bbox = BoundingBox(x=bx, y=by, w=bw, h=bh)
                    results.append(DetectedObject(
                        object_type=ObjectType.CUBE,
                        bbox=bbox,
                        confidence=min(1.0, area / 400.0),
                        attributes={"is_held": False}
                    ))

        return results

    def _detect_doors(self, frame: np.ndarray) -> List[DetectedObject]:
        """Detects Aperture Iris Doors, Sliding Exit Doors, and Glass Elevators."""
        h, w = frame.shape[:2]
        results: List[DetectedObject] = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 140)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 400:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = ch / max(1.0, float(cw))
                # Doors & Elevators are tall vertical structures (aspect 0.80 to 4.5)
                if 0.80 <= aspect <= 4.5 and ch > h * 0.10 and cw > w * 0.03:
                    door_roi = frame[y:y+ch, x:x+cw]
                    hsv_roi = cv2.cvtColor(door_roi, cv2.COLOR_BGR2HSV)
                    
                    # Blue/cyan illuminated exit doorway or open elevator glow
                    mask_open = cv2.inRange(hsv_roi, np.array((85, 100, 100)), np.array((125, 255, 255)))
                    open_ratio = float(np.count_nonzero(mask_open)) / max(1.0, float(cw * ch))
                    
                    is_open = open_ratio > 0.12
                    
                    bbox = BoundingBox(x=x, y=y, w=cw, h=ch)
                    results.append(DetectedObject(
                        object_type=ObjectType.DOOR,
                        bbox=bbox,
                        confidence=min(1.0, area / 3000.0),
                        attributes={"is_open": is_open, "open_ratio": open_ratio}
                    ))

        return results

    def _detect_hazards_and_fixtures(self, frame: np.ndarray) -> List[DetectedObject]:
        """Detects Sentry Turrets and Material Emancipation Grills (Fizzlers)."""
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

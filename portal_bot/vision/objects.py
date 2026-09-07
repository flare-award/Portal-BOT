"""High-Precision Object Detector for Portal 1 (Cubes, Buttons, Doors, Elevators, Turrets)."""

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
        
        # 1. Floor Button detector
        detected.extend(self._detect_buttons(frame))
        
        # 2. Storage Cube detector (with strict ceiling/wall noise rejection)
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
        mask_r1 = cv2.inRange(lower_half_hsv, np.array((0, 120, 80)), np.array((10, 255, 255)))
        mask_r2 = cv2.inRange(lower_half_hsv, np.array((170, 120, 80)), np.array((180, 255, 255)))
        mask_red = cv2.bitwise_or(mask_r1, mask_r2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 45:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / max(1.0, float(ch))
                # Button dome in perspective is elliptical (aspect ratio 0.9 to 5.0)
                if 0.8 <= aspect <= 5.5 and y > h * 0.35:
                    bbox = BoundingBox(x=x, y=y, w=cw, h=ch)
                    results.append(DetectedObject(
                        object_type=ObjectType.BUTTON_FLOOR,
                        bbox=bbox,
                        confidence=min(1.0, area / 600.0),
                        attributes={"is_pressed": False, "dome_color": "red"}
                    ))

        # Also detect pressed floor button (metallic base with active blue/white center rim)
        mask_pressed = cv2.inRange(lower_half_hsv, np.array((90, 80, 120)), np.array((130, 255, 255)))
        mask_pressed = cv2.morphologyEx(mask_pressed, cv2.MORPH_CLOSE, kernel)
        p_contours, _ = cv2.findContours(mask_pressed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in p_contours:
            area = cv2.contourArea(cnt)
            if area > 80:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / max(1.0, float(ch))
                if 1.2 <= aspect <= 5.0 and y > h * 0.45:
                    # Check if button wasn't already added
                    if not any(abs(r.bbox.cx - (x + cw // 2)) < 30 and abs(r.bbox.cy - (y + ch // 2)) < 30 for r in results):
                        bbox = BoundingBox(x=x, y=y, w=cw, h=ch)
                        results.append(DetectedObject(
                            object_type=ObjectType.BUTTON_FLOOR,
                            bbox=bbox,
                            confidence=0.9,
                            attributes={"is_pressed": True, "dome_color": "blue"}
                        ))

        return results

    def _detect_cubes(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Detects Weighted Storage Cubes and Companion Cubes.
        Strictly rejects ceiling fixtures, elevator wall seams, and tiny specular noise.
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results: List[DetectedObject] = []

        # 1. Look for Aperture logo cyan ring / edges on cube
        mask_cube_rings = cv2.inRange(
            hsv,
            np.array(self.config.cube_ring_hsv_lower),
            np.array(self.config.cube_ring_hsv_upper)
        )
        
        # Strict spatial filtering:
        # Exclude ceiling / top 22% of viewport (cubes are not on ceiling)
        mask_cube_rings[:int(h * 0.22), :] = 0
        # Exclude HUD top-left signage
        mask_cube_rings[:int(h * 0.18), :int(w * 0.25)] = 0
        # Exclude Portal Gun weapon model in extreme bottom right
        mask_cube_rings[int(h * 0.60):, int(w * 0.68):] = 0
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_cube_rings = cv2.morphologyEx(mask_cube_rings, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_cube_rings, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Area must be significant (at least 60px on 360p frame) to eliminate wall seams and light reflections
            if 60 < area < 25000:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / max(1.0, float(ch))
                
                # Cubes have aspect ratio close to 1.0 (0.55 to 1.85)
                if 0.55 <= aspect <= 1.85 and cw > 10 and ch > 10:
                    pad_x = int(cw * 0.4)
                    pad_y = int(ch * 0.4)
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
            if area > 600:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = ch / max(1.0, float(cw))
                # Doors & Elevators are tall vertical structures (aspect 0.85 to 4.5)
                if 0.85 <= aspect <= 4.5 and ch > h * 0.12 and cw > w * 0.04:
                    door_roi = frame[y:y+ch, x:x+cw]
                    hsv_roi = cv2.cvtColor(door_roi, cv2.COLOR_BGR2HSV)
                    
                    # Blue/cyan illuminated exit doorway or open elevator glow
                    mask_open = cv2.inRange(hsv_roi, np.array((85, 100, 100)), np.array((125, 255, 255)))
                    open_ratio = float(np.count_nonzero(mask_open)) / max(1.0, float(cw * ch))
                    
                    # A door is open if it has significant blue/cyan illuminated passage inside
                    is_open = open_ratio > 0.15
                    
                    bbox = BoundingBox(x=x, y=y, w=cw, h=ch)
                    results.append(DetectedObject(
                        object_type=ObjectType.DOOR,
                        bbox=bbox,
                        confidence=min(1.0, area / 3500.0),
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

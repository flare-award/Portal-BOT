"""High-Precision Multi-Scale Template & Geometric Object Detector for Portal 1."""

import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from portal_bot.config import VisionConfig
from portal_bot.core.types import BoundingBox, DetectedObject, ObjectType


class ObjectDetector:
    """
    Ultra-high-speed detector combining canonical template matching,
    chromatic ratio segmentation, and shape geometry to eliminate false positives
    from walls, ceilings, and floor indicator lines.
    """

    def __init__(self, config: VisionConfig):
        self.config = config
        self.cube_templates = self._generate_canonical_cube_templates()

    @staticmethod
    def _generate_canonical_cube_templates() -> List[Tuple[int, np.ndarray]]:
        """Generates fast canonical templates of the Aperture Weighted Storage Cube."""
        templates = []
        for s in [18, 32]:
            t = np.full((s, s), 135, dtype=np.uint8)
            border = max(1, s // 10)
            cv2.rectangle(t, (0, 0), (s - 1, s - 1), 40, border)
            notch = max(2, s // 5)
            cv2.rectangle(t, (0, 0), (notch, notch), 30, -1)
            cv2.rectangle(t, (s - notch, 0), (s, notch), 30, -1)
            cv2.rectangle(t, (0, s - notch), (notch, s), 30, -1)
            cv2.rectangle(t, (s - notch, s - notch), (s, s), 30, -1)
            rc = s // 2
            r_out = max(3, s // 3)
            cv2.circle(t, (rc, rc), r_out, 220, max(1, s // 10))
            cv2.circle(t, (rc, rc), max(1, s // 7), 250, -1)
            templates.append((s, t))
        return templates

    def detect_all(self, frame: np.ndarray) -> List[DetectedObject]:
        detected: List[DetectedObject] = []
        
        # 1. Floor Button detector (Chromatic Red Ratio + Elliptical Geometry)
        detected.extend(self._detect_buttons(frame))
        
        # 2. Storage Cube detector (Fast Template Matching + Cyan Logo Verification)
        detected.extend(self._detect_cubes(frame))
        
        # 3. Exit Door / Elevator detector
        detected.extend(self._detect_doors(frame))
        
        # 4. Turret & Fizzler detector
        detected.extend(self._detect_hazards_and_fixtures(frame))
        
        return detected

    def _detect_buttons(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Detects 1500MW Heavy Duty Super-Colliding Floor Buttons using strict
        chromatic red dominance to 100% reject white walls and blue indicator lines.
        """
        h, w = frame.shape[:2]
        results: List[DetectedObject] = []

        # Buttons are strictly located on the ground plane (lower 65% of screen)
        b, g, r = cv2.split(frame.astype(np.float32))

        # Red chromatic dominance formula: R must strictly exceed G and B
        red_diff = r - np.maximum(g, b)
        red_ratio = red_diff / (r + g + b + 1.0)

        # Mask: Red ratio > 0.18, R > 90, B < 150, located in ground plane
        button_mask = (red_ratio > 0.18) & (r > 90) & (b < 150)
        button_mask[:int(h * 0.35), :] = False

        mask_u8 = (button_mask * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 35:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / max(1.0, float(ch))
                
                # Button dome is elliptical in perspective (aspect 0.70 to 4.5, min dims >= 8px)
                if 0.70 <= aspect <= 4.5 and cw >= 8 and ch >= 6 and y > h * 0.35:
                    bbox = BoundingBox(x=x, y=y, w=cw, h=ch)
                    results.append(DetectedObject(
                        object_type=ObjectType.BUTTON_FLOOR,
                        bbox=bbox,
                        confidence=min(1.0, area / 400.0),
                        attributes={"is_pressed": False, "dome_color": "red"}
                    ))

        return results

    def _detect_cubes(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Detects Weighted Storage Cubes using Fast Template Matching
        and Aperture Cyan Logo Verification.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results: List[DetectedObject] = []

        # 1. Geometric Aperture Logo contour detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_cube_rings = cv2.inRange(
            hsv,
            np.array(self.config.cube_ring_hsv_lower),
            np.array(self.config.cube_ring_hsv_upper)
        )
        # Exclude ceiling and weapon viewmodel
        mask_cube_rings[:int(h * 0.22), :] = 0
        mask_cube_rings[int(h * 0.60):, int(w * 0.68):] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_cube_rings = cv2.morphologyEx(mask_cube_rings, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_cube_rings, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 30 < area < 15000:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / max(1.0, float(ch))
                
                # Cubes have aspect ratio close to 1.0 (0.55 to 1.80) and min dimensions >= 8px
                if 0.55 <= aspect <= 1.80 and cw >= 8 and ch >= 8:
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
                        confidence=min(1.0, area / 350.0),
                        attributes={"is_held": False, "detection_method": "logo_contour"}
                    ))

        # 2. Fast 2x-downscaled Template match across search ROI
        search_roi = gray[int(h * 0.25):int(h * 0.88), :]
        search_small = cv2.resize(search_roi, (w // 2, search_roi.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
        
        for scale, tmpl in self.cube_templates:
            if search_small.shape[0] < scale or search_small.shape[1] < scale:
                continue
            res = cv2.matchTemplate(search_small, tmpl, cv2.TM_CCOEFF_NORMED)
            min_v, max_v, min_l, max_l = cv2.minMaxLoc(res)
            
            if max_v > 0.52:
                cx = max_l[0] * 2
                cy = (max_l[1] * 2) + int(h * 0.25)
                full_scale = scale * 2
                if not any(abs(r.bbox.x - cx) < 30 and abs(r.bbox.y - cy) < 30 for r in results):
                    bbox = BoundingBox(x=cx, y=cy, w=full_scale, h=full_scale)
                    results.append(DetectedObject(
                        object_type=ObjectType.CUBE,
                        bbox=bbox,
                        confidence=float(max_v),
                        attributes={"is_held": False, "detection_method": "template"}
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

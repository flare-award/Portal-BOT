"""Surface segmentation and hazard detection for Portal 1."""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from portal_bot.config import VisionConfig
from portal_bot.core.types import BoundingBox, DetectedObject, ObjectType


class SurfaceAnalyzer:
    """Segments walkable floor, portalable walls, non-portalable surfaces, and hazards."""

    def __init__(self, config: VisionConfig):
        self.config = config

    def analyze_surfaces(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectedObject]]:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_objects: List[DetectedObject] = []

        # 1. Detect Toxic Goo / Acid hazards (murky brownish-green in lower half)
        lower_half_hsv = hsv[int(h * 0.4):, :]
        mask_acid = cv2.inRange(
            lower_half_hsv,
            np.array(self.config.acid_hsv_lower),
            np.array(self.config.acid_hsv_upper)
        )
        
        acid_contours, _ = cv2.findContours(mask_acid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in acid_contours:
            area = cv2.contourArea(cnt)
            if area > 1200:
                x, y, cw, ch = cv2.boundingRect(cnt)
                bbox = BoundingBox(x=x, y=int(h * 0.4) + y, w=cw, h=ch)
                detected_objects.append(DetectedObject(
                    object_type=ObjectType.HAZARD_ACID,
                    bbox=bbox,
                    confidence=min(1.0, area / 5000.0),
                    attributes={"hazard_type": "toxic_goo"}
                ))

        # 2. Portalable Surface Segmentation Mask
        # Portalable concrete has Value (V >= 75) and low Saturation (S <= 65)
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]
        
        portalable_mask = np.zeros((h, w), dtype=np.uint8)
        # Concrete walls: grey/white with low saturation
        portalable_mask[(v_channel >= self.config.portalable_min_brightness) & (s_channel <= self.config.portalable_max_saturation)] = 255
        
        # Exclude HUD / top-left signage from surface mask
        portalable_mask[:int(h * 0.12), :int(w * 0.22)] = 0
        
        # Morphological close to bridge tile seams
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_mask = cv2.morphologyEx(portalable_mask, cv2.MORPH_CLOSE, kernel)

        # Find large contiguous portalable wall patches
        wall_contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in wall_contours:
            area = cv2.contourArea(cnt)
            if area > 2500:
                x, y, cw, ch = cv2.boundingRect(cnt)
                # Wall elevation: spans vertically across mid-screen
                if y < h * 0.85 and (y + ch) > h * 0.20 and cw > 25 and ch > 25:
                    bbox = BoundingBox(x=x, y=y, w=cw, h=ch)
                    detected_objects.append(DetectedObject(
                        object_type=ObjectType.PORTALABLE_WALL,
                        bbox=bbox,
                        confidence=min(1.0, area / 15000.0),
                        attributes={"area": area}
                    ))

        return portalable_mask, detected_objects

    def find_best_portal_target(self, frame: np.ndarray, portalable_mask: np.ndarray, target_side: str = "any") -> Optional[Tuple[int, int]]:
        """
        Finds the center of the largest, clearest portalable surface tile
        for placing a blue or orange portal.
        """
        h, w = frame.shape[:2]
        mask = portalable_mask.copy()
        
        if target_side == "left":
            mask[:, int(w * 0.55):] = 0
        elif target_side == "right":
            mask[:, :int(w * 0.45)] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 1200:
            return None

        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return None

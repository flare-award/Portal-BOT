"""Portal detector for Portal 1 (Blue and Orange portals)."""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from portal_bot.config import VisionConfig
from portal_bot.core.types import BoundingBox, DetectedObject, ObjectType, PortalState


class PortalDetector:
    """Detects active blue and orange portals in the scene."""

    def __init__(self, config: VisionConfig):
        self.config = config

    def detect_portals(self, frame: np.ndarray) -> Tuple[PortalState, List[DetectedObject]]:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        portal_state = PortalState()
        detected_objects: List[DetectedObject] = []

        # 1. Detect Blue Portal (Cyan/Blue HSV Range)
        mask_blue = cv2.inRange(
            hsv,
            np.array(self.config.blue_portal_hsv_lower),
            np.array(self.config.blue_portal_hsv_upper)
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

        blue_contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if blue_contours:
            # Filter contours by aspect ratio (portals are elliptical, height usually >= width)
            valid_blue = []
            for cnt in blue_contours:
                area = cv2.contourArea(cnt)
                if area > 180:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    aspect = ch / max(1.0, float(cw))
                    # Portals usually have aspect ratio between 0.8 and 4.0
                    if 0.6 <= aspect <= 5.0:
                        valid_blue.append((cnt, area, (x, y, cw, ch)))

            if valid_blue:
                best_cnt, best_area, (bx, by, bw, bh) = max(valid_blue, key=lambda item: item[1])
                bbox = BoundingBox(x=bx, y=by, w=bw, h=bh)
                portal_state.blue_active = True
                portal_state.blue_bbox = bbox
                portal_state.blue_screen_pos = (bbox.cx, bbox.cy)

                detected_objects.append(DetectedObject(
                    object_type=ObjectType.PORTAL_BLUE,
                    bbox=bbox,
                    confidence=min(1.0, best_area / 3000.0),
                    attributes={"portal_type": "blue", "aspect_ratio": bh / max(1.0, float(bw))}
                ))

        # 2. Detect Orange Portal (Amber/Orange HSV Range)
        mask_orange = cv2.inRange(
            hsv,
            np.array(self.config.orange_portal_hsv_lower),
            np.array(self.config.orange_portal_hsv_upper)
        )
        mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel)

        orange_contours, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if orange_contours:
            valid_orange = []
            for cnt in orange_contours:
                area = cv2.contourArea(cnt)
                if area > 180:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    aspect = ch / max(1.0, float(cw))
                    if 0.6 <= aspect <= 5.0:
                        valid_orange.append((cnt, area, (x, y, cw, ch)))

            if valid_orange:
                best_cnt, best_area, (ox, oy, ow, oh) = max(valid_orange, key=lambda item: item[1])
                bbox = BoundingBox(x=ox, y=oy, w=ow, h=oh)
                portal_state.orange_active = True
                portal_state.orange_bbox = bbox
                portal_state.orange_screen_pos = (bbox.cx, bbox.cy)

                detected_objects.append(DetectedObject(
                    object_type=ObjectType.PORTAL_ORANGE,
                    bbox=bbox,
                    confidence=min(1.0, best_area / 3000.0),
                    attributes={"portal_type": "orange", "aspect_ratio": oh / max(1.0, float(ow))}
                ))

        return portal_state, detected_objects

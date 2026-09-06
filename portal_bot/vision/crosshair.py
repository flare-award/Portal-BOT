"""High-Precision Reticle, Crosshair, and Portal Gun State Analyzer for Portal 1."""

from typing import Tuple
import cv2
import numpy as np

from portal_bot.config import VisionConfig
from portal_bot.core.types import CrosshairState, PortalGunState


class CrosshairAnalyzer:
    """
    Analyzes the central screen reticle to determine:
    1. Portal Gun state: NO_GUN vs SINGLE_PORTAL_BLUE vs DUAL_PORTAL
    2. Active Portals: Blue placed / Orange placed
    3. Surface conductivity under reticle
    """

    def __init__(self, config: VisionConfig):
        self.config = config

    def analyze(self, frame: np.ndarray) -> Tuple[CrosshairState, PortalGunState]:
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        state = CrosshairState(center_pos=(cx, cy))
        gun_state = PortalGunState.NO_GUN
        
        # Crop 48x48 pixel ROI centered on crosshair
        roi_half = 24
        x1 = max(0, cx - roi_half)
        y1 = max(0, cy - roi_half)
        x2 = min(w, cx + roi_half)
        y2 = min(h, cy + roi_half)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 30 or roi.shape[1] < 30:
            return state, gun_state

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        rh, rw = hsv_roi.shape[:2]
        rcx, rcy = rw // 2, rh // 2

        # 1. Inspect Left Bracket (Blue Portal Bracket)
        # In Portal 1, left bracket is located between 10 and 18 px left of center
        left_bracket_box = hsv_roi[rcy - 12 : rcy + 12, max(0, rcx - 18) : max(0, rcx - 9)]
        
        # Saturated Cyan/Blue glow
        mask_blue = cv2.inRange(
            left_bracket_box,
            np.array((85, 130, 150)),
            np.array((125, 255, 255))
        )
        blue_glow_count = np.count_nonzero(mask_blue)

        # 2. Inspect Right Bracket (Orange Portal Bracket)
        # Located between 9 and 18 px right of center
        right_bracket_box = hsv_roi[rcy - 12 : rcy + 12, min(rw, rcx + 9) : min(rw, rcx + 18)]
        
        # Saturated Amber/Orange glow
        mask_orange = cv2.inRange(
            right_bracket_box,
            np.array((8, 140, 150)),
            np.array((25, 255, 255))
        )
        orange_glow_count = np.count_nonzero(mask_orange)

        # 3. Bracket edge detection to determine unlit bracket presence
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        left_gray = gray_roi[rcy - 10 : rcy + 10, max(0, rcx - 18) : max(0, rcx - 9)]
        right_gray = gray_roi[rcy - 10 : rcy + 10, min(rw, rcx + 9) : min(rw, rcx + 18)]
        
        # Edges in bracket zones
        left_edges = cv2.Canny(left_gray, 40, 120)
        right_edges = cv2.Canny(right_gray, 40, 120)
        
        has_left_outline = (np.count_nonzero(left_edges) >= 4) or (blue_glow_count >= 5)
        has_right_outline = (np.count_nonzero(right_edges) >= 4) or (orange_glow_count >= 5)

        # 4. Infer Portal Gun Inventory Level
        if has_left_outline and has_right_outline:
            gun_state = PortalGunState.DUAL_PORTAL
        elif has_left_outline:
            gun_state = PortalGunState.SINGLE_PORTAL_BLUE
        else:
            gun_state = PortalGunState.NO_GUN

        # 5. Determine active placed portals based strictly on glowing pixels
        # Blue portal is placed only if left bracket glows solid cyan
        if blue_glow_count >= 6:
            state.blue_ring_filled = True
        else:
            state.blue_ring_filled = False

        # Orange portal is placed only if right bracket glows solid orange
        if orange_glow_count >= 6:
            state.orange_ring_filled = True
        else:
            state.orange_ring_filled = False

        # If player has No Gun, portals cannot be active on HUD
        if gun_state == PortalGunState.NO_GUN:
            state.blue_ring_filled = False
            state.orange_ring_filled = False

        # 6. Surface Conductivity under center reticle
        center_spot = hsv_roi[rcy - 3 : rcy + 3, rcx - 3 : rcx + 3]
        if center_spot.size > 0:
            mean_sat = np.mean(center_spot[:, :, 1])
            mean_val = np.mean(center_spot[:, :, 2])
            
            if mean_val >= self.config.portalable_min_brightness and mean_sat <= self.config.portalable_max_saturation:
                state.on_portalable_surface = True
            else:
                state.on_portalable_surface = False

        return state, gun_state

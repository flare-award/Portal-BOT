"""High-Precision Reticle, Crosshair, and Portal Gun State Analyzer for Portal 1."""

from typing import Tuple
import cv2
import numpy as np

from portal_bot.config import VisionConfig
from portal_bot.core.types import CrosshairState, PortalGunState


class CrosshairAnalyzer:
    """
    Analyzes the central screen reticle across any screen resolution to determine:
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
        
        # Adaptive ROI size proportional to screen height
        roi_half = max(28, int(h * 0.065))
        x1 = max(0, cx - roi_half)
        y1 = max(0, cy - roi_half)
        x2 = min(w, cx + roi_half)
        y2 = min(h, cy + roi_half)
        
        raw_roi = frame[y1:y2, x1:x2]
        if raw_roi.size == 0 or raw_roi.shape[0] < 20 or raw_roi.shape[1] < 20:
            return state, gun_state

        # Normalize to standard 96x96 analysis canvas
        norm_roi = cv2.resize(raw_roi, (96, 96), interpolation=cv2.INTER_LINEAR)
        hsv_roi = cv2.cvtColor(norm_roi, cv2.COLOR_BGR2HSV)
        gray_roi = cv2.cvtColor(norm_roi, cv2.COLOR_BGR2GRAY)

        # 1. Left Bracket Analysis (Blue Portal Reticle: x in [14..36], y in [24..72])
        left_hsv = hsv_roi[24:72, 14:36]
        left_gray = gray_roi[24:72, 14:36]
        
        # High-saturation Cyan/Blue glow (active placed blue portal)
        mask_blue = cv2.inRange(
            left_hsv,
            np.array((85, 140, 160)),
            np.array((125, 255, 255))
        )
        blue_glow_count = int(np.count_nonzero(mask_blue))

        # Edge detection for unlit bracket presence
        left_edges = cv2.Canny(left_gray, 40, 120)
        left_edge_count = int(np.count_nonzero(left_edges))
        
        has_left_bracket = (left_edge_count >= 6) or (blue_glow_count >= 8)

        # 2. Right Bracket Analysis (Orange Portal Reticle: x in [60..82], y in [24..72])
        right_hsv = hsv_roi[24:72, 60:82]
        right_gray = gray_roi[24:72, 60:82]
        
        # High-saturation Amber/Orange glow (active placed orange portal)
        mask_orange = cv2.inRange(
            right_hsv,
            np.array((6, 140, 160)),
            np.array((25, 255, 255))
        )
        orange_glow_count = int(np.count_nonzero(mask_orange))

        # Edge detection for unlit right bracket presence
        right_edges = cv2.Canny(right_gray, 40, 120)
        right_edge_count = int(np.count_nonzero(right_edges))
        
        has_right_bracket = (right_edge_count >= 6) or (orange_glow_count >= 8)

        # 3. Determine Portal Gun Inventory State
        if has_left_bracket and has_right_bracket:
            gun_state = PortalGunState.DUAL_PORTAL
        elif has_left_bracket:
            gun_state = PortalGunState.SINGLE_PORTAL_BLUE
        else:
            gun_state = PortalGunState.NO_GUN

        # 4. Placed Portal Status
        if gun_state != PortalGunState.NO_GUN:
            state.blue_ring_filled = (blue_glow_count >= 14)
            
            if gun_state == PortalGunState.DUAL_PORTAL:
                state.orange_ring_filled = (orange_glow_count >= 14)
            else:
                state.orange_ring_filled = False
        else:
            state.blue_ring_filled = False
            state.orange_ring_filled = False

        # 5. Surface Conductivity directly under center crosshair
        center_spot = hsv_roi[44:52, 44:52]
        if center_spot.size > 0:
            mean_sat = float(np.mean(center_spot[:, :, 1]))
            mean_val = float(np.mean(center_spot[:, :, 2]))
            
            if mean_val >= self.config.portalable_min_brightness and mean_sat <= self.config.portalable_max_saturation:
                state.on_portalable_surface = True
            else:
                state.on_portalable_surface = False

        return state, gun_state

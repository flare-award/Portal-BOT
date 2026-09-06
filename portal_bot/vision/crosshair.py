"""Crosshair, reticle, and Portal Gun inventory analyzer for Portal 1."""

from typing import Tuple
import cv2
import numpy as np

from portal_bot.config import VisionConfig
from portal_bot.core.types import CrosshairState, PortalGunState


class CrosshairAnalyzer:
    """
    Analyzes the central screen reticle to determine:
    1. Portal Gun state: NO_GUN vs SINGLE_BLUE vs DUAL_PORTAL
    2. Portal state: Blue portal placed / Orange portal placed
    3. Surface conductivity: Is crosshair aiming at a portal-conductive surface?
    """

    def __init__(self, config: VisionConfig):
        self.config = config

    def analyze(self, frame: np.ndarray) -> Tuple[CrosshairState, PortalGunState]:
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        state = CrosshairState(center_pos=(cx, cy))
        gun_state = PortalGunState.NO_GUN
        
        # Extract 64x64 ROI around screen center
        roi_half = 32
        x1 = max(0, cx - roi_half)
        y1 = max(0, cy - roi_half)
        x2 = min(w, cx + roi_half)
        y2 = min(h, cy + roi_half)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 40 or roi.shape[1] < 40:
            return state, gun_state

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        rh, rw = hsv_roi.shape[:2]
        rcx, rcy = rw // 2, rh // 2

        # 1. Inspect Left Bracket Region (Cyan/Blue Portal Bracket)
        # Position: ~10 to 18 pixels left of center
        left_bracket_roi = hsv_roi[rcy - 14 : rcy + 14, max(0, rcx - 22) : rcx - 8]
        
        mask_blue = cv2.inRange(
            left_bracket_roi,
            np.array(self.config.blue_portal_hsv_lower),
            np.array(self.config.blue_portal_hsv_upper)
        )
        blue_pixels = np.count_nonzero(mask_blue)
        
        # Check grey/bracket outline presence
        left_edge_val = np.mean(left_bracket_roi[:, :, 2])

        # 2. Inspect Right Bracket Region (Amber/Orange Portal Bracket)
        # Position: ~8 to 22 pixels right of center
        right_bracket_roi = hsv_roi[rcy - 14 : rcy + 14, rcx + 8 : min(rw, rcx + 22)]
        
        mask_orange = cv2.inRange(
            right_bracket_roi,
            np.array(self.config.orange_portal_hsv_lower),
            np.array(self.config.orange_portal_hsv_upper)
        )
        orange_pixels = np.count_nonzero(mask_orange)
        right_edge_val = np.mean(right_bracket_roi[:, :, 2])

        # 3. Determine Portal Gun inventory state from reticle structure
        has_left_bracket = (blue_pixels > 6) or (left_edge_val > 120 and np.std(left_bracket_roi[:, :, 2]) > 30)
        has_right_bracket = (orange_pixels > 6) or (right_edge_val > 120 and np.std(right_bracket_roi[:, :, 2]) > 30)

        if has_left_bracket and has_right_bracket:
            gun_state = PortalGunState.DUAL_PORTAL
        elif has_left_bracket:
            gun_state = PortalGunState.SINGLE_PORTAL_BLUE
        else:
            gun_state = PortalGunState.NO_GUN

        # 4. Fill state: only mark portals active if the glowing color pixels are confirmed!
        if blue_pixels >= 8:
            state.blue_ring_filled = True
        else:
            state.blue_ring_filled = False

        if orange_pixels >= 8:
            state.orange_ring_filled = True
        else:
            state.orange_ring_filled = False

        # If player has NO gun, brackets cannot be active
        if gun_state == PortalGunState.NO_GUN:
            state.blue_ring_filled = False
            state.orange_ring_filled = False

        # 5. Check surface conductivity under center crosshair reticle
        center_spot = hsv_roi[rcy - 4 : rcy + 4, rcx - 4 : rcx + 4]
        if center_spot.size > 0:
            mean_sat = np.mean(center_spot[:, :, 1])
            mean_val = np.mean(center_spot[:, :, 2])
            
            # Portalable concrete tiles have high Value and low Saturation
            if mean_val >= self.config.portalable_min_brightness and mean_sat <= self.config.portalable_max_saturation:
                state.on_portalable_surface = True
            else:
                state.on_portalable_surface = False

        return state, gun_state

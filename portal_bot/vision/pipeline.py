"""Master Computer Vision Pipeline for Portal-BOT operating at 60+ FPS."""

import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from portal_bot.config import VisionConfig
from portal_bot.core.types import (
    BoundingBox,
    CrosshairState,
    DetectedObject,
    GameState,
    ObjectType,
    PlayerState,
    PortalGunState,
    PortalState,
)
from portal_bot.vision.crosshair import CrosshairAnalyzer
from portal_bot.vision.hud import HUDAnalyzer
from portal_bot.vision.objects import ObjectDetector
from portal_bot.vision.odometry import VisualOdometry
from portal_bot.vision.portals import PortalDetector
from portal_bot.vision.surfaces import SurfaceAnalyzer


class VisionPipeline:
    """
    High-Performance Vision Pipeline:
    - Downscaled 640x360 multi-cue perception
    - Native High-Res Crosshair & Portal Reticle analysis
    - Precision Cube / Button / Door segmentation
    - Real-time debug visualization overlay
    """

    def __init__(self, config: VisionConfig):
        self.config = config
        self.odometry = VisualOdometry(config)
        self.crosshair_analyzer = CrosshairAnalyzer(config)
        self.surface_analyzer = SurfaceAnalyzer(config)
        self.portal_detector = PortalDetector(config)
        self.object_detector = ObjectDetector(config)
        self.hud_analyzer = HUDAnalyzer(config)

        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 60.0
        self.holding_cube_state = False

    def process_frame(self, frame: np.ndarray, prev_state: Optional[GameState] = None) -> Tuple[GameState, np.ndarray]:
        t0 = time.time()
        self.frame_count += 1
        orig_h, orig_w = frame.shape[:2]

        # Fast downscale to 640x360 for high-FPS processing
        proc_w, proc_h = 640, 360
        scale_x = orig_w / float(proc_w)
        scale_y = orig_h / float(proc_h)
        small_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)

        # 1. Visual Odometry on downscaled frame
        flow_dx, flow_dy, flow_mag = self.odometry.update(small_frame)
        flow_dx *= scale_x
        flow_dy *= scale_y

        # 2. Crosshair Analysis on Native High-Resolution Center ROI
        crosshair_state, gun_state = self.crosshair_analyzer.analyze(frame)

        # 3. Surface & Hazard Analysis
        portalable_mask_small, surface_objects = self.surface_analyzer.analyze_surfaces(small_frame)

        for obj in surface_objects:
            obj.bbox.x = int(obj.bbox.x * scale_x)
            obj.bbox.y = int(obj.bbox.y * scale_y)
            obj.bbox.w = int(obj.bbox.w * scale_x)
            obj.bbox.h = int(obj.bbox.h * scale_y)
            obj.center_screen = (obj.bbox.cx, obj.bbox.cy)

        # 4. Portal Detection in 3D Scene
        portal_state, portal_objects = self.portal_detector.detect_portals(small_frame)
        for obj in portal_objects:
            obj.bbox.x = int(obj.bbox.x * scale_x)
            obj.bbox.y = int(obj.bbox.y * scale_y)
            obj.bbox.w = int(obj.bbox.w * scale_x)
            obj.bbox.h = int(obj.bbox.h * scale_y)
            obj.center_screen = (obj.bbox.cx, obj.bbox.cy)

        # Placed status is determined strictly by reticle HUD brackets / gun state
        portal_state.blue_active = crosshair_state.blue_ring_filled
        portal_state.orange_active = crosshair_state.orange_ring_filled

        # 5. Interactive Object Detection (Cubes, Buttons, Doors)
        interactive_objects = self.object_detector.detect_all(small_frame)
        for obj in interactive_objects:
            obj.bbox.x = int(obj.bbox.x * scale_x)
            obj.bbox.y = int(obj.bbox.y * scale_y)
            obj.bbox.w = int(obj.bbox.w * scale_x)
            obj.bbox.h = int(obj.bbox.h * scale_y)
            obj.center_screen = (obj.bbox.cx, obj.bbox.cy)

        # 6. Held Cube & Player State Analysis
        # A held cube in Portal 1 occupies a massive central-lower view (0.24*w <= x <= 0.76*w, y >= 0.42*h)
        center_hand_roi = small_frame[int(proc_h * 0.42):, int(proc_w * 0.24):int(proc_w * 0.76)]
        is_holding_cube = False
        if center_hand_roi.size > 0:
            hsv_center = cv2.cvtColor(center_hand_roi, cv2.COLOR_BGR2HSV)
            
            # Aperture logo cyan ring mask
            mask_center_cube = cv2.inRange(
                hsv_center,
                np.array(self.config.cube_ring_hsv_lower),
                np.array(self.config.cube_ring_hsv_upper)
            )
            cube_ring_pixels = int(np.count_nonzero(mask_center_cube))

            # Must have substantial Aperture cyan emblem pixels (> 70px) in hand ROI
            if cube_ring_pixels > 70:
                is_holding_cube = True
                self.holding_cube_state = True
            else:
                self.holding_cube_state = False

        # Filter out held-cube artifacts from 3D world interactive object list
        filtered_interactive: List[DetectedObject] = []
        has_pressed_button = False
        has_open_door = False

        for obj in interactive_objects:
            if obj.object_type == ObjectType.CUBE and obj.bbox.cy > orig_h * 0.52 and (orig_w * 0.22 < obj.bbox.cx < orig_w * 0.78) and obj.bbox.area > 20000:
                is_holding_cube = True
                self.holding_cube_state = True
            else:
                filtered_interactive.append(obj)

            if obj.object_type == ObjectType.BUTTON_FLOOR:
                if obj.attributes.get("is_pressed", False):
                    has_pressed_button = True

            if obj.object_type == ObjectType.DOOR:
                if obj.attributes.get("is_open", False):
                    has_open_door = True

        all_objects = surface_objects + portal_objects + filtered_interactive

        # 7. Check HUD flags
        death_detected = self.hud_analyzer.check_death_screen(small_frame)
        level_complete = self.hud_analyzer.check_level_complete(small_frame)

        # 8. Build Comprehensive Game State
        game_state = GameState(
            timestamp=t0,
            frame_id=self.frame_count,
            fps=self.fps,
            player=PlayerState(
                yaw=self.odometry.estimated_yaw,
                pitch=0.0,
                pos=(self.odometry.estimated_pos[0], self.odometry.estimated_pos[1], self.odometry.estimated_pos[2]),
                gun_state=gun_state,
                holding_cube=self.holding_cube_state,
                crosshair=crosshair_state,
            ),
            portals=portal_state,
            objects=all_objects,
            button_pressed=has_pressed_button,
            exit_door_open=has_open_door,
            death_detected=death_detected,
            level_complete=level_complete,
            optical_flow_magnitude=flow_mag,
            confidence=0.95
        )

        # Calculate live FPS
        elapsed = time.time() - t0
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(0.001, elapsed))

        # Render Debug Overlay
        debug_frame = self._render_debug_overlay(frame, game_state)

        return game_state, debug_frame

    def _render_debug_overlay(self, frame: np.ndarray, state: GameState) -> np.ndarray:
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # Draw detected 3D Objects
        for obj in state.objects:
            b = obj.bbox
            color = (0, 255, 0)
            label = obj.object_type.value

            if obj.object_type == ObjectType.CUBE:
                color = (255, 200, 0)
                label = f"CUBE ({obj.confidence:.2f})"
            elif obj.object_type == ObjectType.BUTTON_FLOOR:
                is_p = obj.attributes.get("is_pressed", False)
                color = (0, 255, 100) if is_p else (0, 0, 255)
                label = f"BUTTON: {'PRESSED' if is_p else 'UNPRESSED'}"
            elif obj.object_type == ObjectType.DOOR:
                is_o = obj.attributes.get("is_open", False)
                color = (0, 255, 255) if is_o else (128, 128, 128)
                label = f"EXIT DOOR: {'OPEN' if is_o else 'CLOSED'}"
            elif obj.object_type == ObjectType.PORTAL_BLUE:
                color = (255, 180, 0)
                label = "PORTAL (BLUE)"
            elif obj.object_type == ObjectType.PORTAL_ORANGE:
                color = (0, 140, 255)
                label = "PORTAL (ORANGE)"

            cv2.rectangle(overlay, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 2)
            cv2.putText(overlay, label, (b.x, max(15, b.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Draw Reticle Target
        cx, cy = w // 2, h // 2
        ch_state = state.player.crosshair
        reticle_col = (0, 255, 0) if ch_state.on_portalable_surface else (0, 0, 255)
        
        cv2.circle(overlay, (cx, cy), 5, reticle_col, 1)
        cv2.line(overlay, (cx - 12, cy), (cx + 12, cy), reticle_col, 1)
        cv2.line(overlay, (cx, cy - 12), (cx, cy + 12), reticle_col, 1)

        # HUD Status Bar at top
        cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 20), -1)
        hud_str = f"FPS: {state.fps:.1f} | GUN: {state.player.gun_state.value.upper()} | BLUE: {'YES' if state.portals.blue_active else 'NO'} | ORANGE: {'YES' if state.portals.orange_active else 'NO'} | HAND: {'CUBE' if state.player.holding_cube else 'NONE'}"
        cv2.putText(overlay, hud_str, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1)

        return overlay

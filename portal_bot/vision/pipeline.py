"""High-Performance 60+ FPS Computer Vision Pipeline for Portal-BOT."""

import time
from typing import List, Optional, Tuple
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
    """Ultra-fast optimized Computer Vision pipeline delivering 60+ FPS."""

    def __init__(self, config: VisionConfig):
        self.config = config
        self.crosshair_analyzer = CrosshairAnalyzer(config)
        self.surface_analyzer = SurfaceAnalyzer(config)
        self.portal_detector = PortalDetector(config)
        self.object_detector = ObjectDetector(config)
        self.odometry = VisualOdometry(config)
        self.hud_analyzer = HUDAnalyzer()

        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 60.0
        self.holding_cube_state = False

    def process_frame(self, frame: np.ndarray, prev_state: Optional[GameState] = None) -> Tuple[GameState, np.ndarray]:
        t0 = time.time()
        self.frame_count += 1
        orig_h, orig_w = frame.shape[:2]

        # Fast downscale to 640x360 for high-FPS vector processing
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

        # Scale surface bounding boxes back to native resolution
        for obj in surface_objects:
            obj.bbox.x = int(obj.bbox.x * scale_x)
            obj.bbox.y = int(obj.bbox.y * scale_y)
            obj.bbox.w = int(obj.bbox.w * scale_x)
            obj.bbox.h = int(obj.bbox.h * scale_y)
            obj.center_screen = (obj.bbox.cx, obj.bbox.cy)

        # 4. Portal Detection
        portal_state, portal_objects = self.portal_detector.detect_portals(small_frame)
        for obj in portal_objects:
            obj.bbox.x = int(obj.bbox.x * scale_x)
            obj.bbox.y = int(obj.bbox.y * scale_y)
            obj.bbox.w = int(obj.bbox.w * scale_x)
            obj.bbox.h = int(obj.bbox.h * scale_y)
            obj.center_screen = (obj.bbox.cx, obj.bbox.cy)

        if portal_state.blue_bbox:
            portal_state.blue_bbox.x = int(portal_state.blue_bbox.x * scale_x)
            portal_state.blue_bbox.y = int(portal_state.blue_bbox.y * scale_y)
            portal_state.blue_bbox.w = int(portal_state.blue_bbox.w * scale_x)
            portal_state.blue_bbox.h = int(portal_state.blue_bbox.h * scale_y)
            portal_state.blue_screen_pos = (portal_state.blue_bbox.cx, portal_state.blue_bbox.cy)

        if portal_state.orange_bbox:
            portal_state.orange_bbox.x = int(portal_state.orange_bbox.x * scale_x)
            portal_state.orange_bbox.y = int(portal_state.orange_bbox.y * scale_y)
            portal_state.orange_bbox.w = int(portal_state.orange_bbox.w * scale_x)
            portal_state.orange_bbox.h = int(portal_state.orange_bbox.h * scale_y)
            portal_state.orange_screen_pos = (portal_state.orange_bbox.cx, portal_state.orange_bbox.cy)

        # 5. Interactive Object Detection
        interactive_objects = self.object_detector.detect_all(small_frame)
        for obj in interactive_objects:
            obj.bbox.x = int(obj.bbox.x * scale_x)
            obj.bbox.y = int(obj.bbox.y * scale_y)
            obj.bbox.w = int(obj.bbox.w * scale_x)
            obj.bbox.h = int(obj.bbox.h * scale_y)
            obj.center_screen = (obj.bbox.cx, obj.bbox.cy)

        # Check holding cube state
        holding_cube = False
        hand_roi = small_frame[int(proc_h * 0.65):, int(proc_w * 0.55):]
        if hand_roi.size > 0:
            hsv_hand = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2HSV)
            mask_hand_cube = cv2.inRange(
                hsv_hand,
                np.array(self.config.cube_ring_hsv_lower),
                np.array(self.config.cube_ring_hsv_upper)
            )
            if np.count_nonzero(mask_hand_cube) > 15:
                holding_cube = True
                self.holding_cube_state = True
            else:
                self.holding_cube_state = False

        filtered_interactive = []
        for obj in interactive_objects:
            if obj.object_type == ObjectType.CUBE and obj.bbox.cy > orig_h * 0.70 and obj.bbox.cx > orig_w * 0.55:
                holding_cube = True
                self.holding_cube_state = True
            else:
                filtered_interactive.append(obj)

        all_objects = surface_objects + portal_objects + filtered_interactive

        # 6. Check HUD flags
        death_detected = self.hud_analyzer.check_death_screen(small_frame)
        level_complete = self.hud_analyzer.check_level_complete(small_frame)

        # 7. Player State
        player = PlayerState(
            pos=tuple(self.odometry.estimated_pos),  # type: ignore
            yaw=self.odometry.estimated_yaw,
            pitch=self.odometry.estimated_pitch,
            is_moving=(flow_mag > 0.8),
            is_dead=death_detected,
            holding_cube=self.holding_cube_state,
            crosshair=crosshair_state,
            gun_state=gun_state
        )

        exit_door_open = False
        exit_door_visible = False
        for obj in all_objects:
            if obj.object_type == ObjectType.DOOR:
                exit_door_visible = True
                if obj.attributes.get("is_open", False):
                    exit_door_open = True

        button_pressed = False
        for obj in all_objects:
            if obj.object_type == ObjectType.BUTTON_FLOOR and obj.attributes.get("is_pressed", False):
                button_pressed = True

        hazard_ahead = any(obj.object_type == ObjectType.HAZARD_ACID for obj in all_objects)

        # Instantaneous FPS calculation with exponential smoothing
        now = time.time()
        dt = now - self.last_time
        if dt > 0:
            inst_fps = 1.0 / dt
            self.fps = 0.92 * self.fps + 0.08 * inst_fps
        self.last_time = now

        game_state = GameState(
            frame_id=self.frame_count,
            timestamp=now,
            player=player,
            portals=portal_state,
            objects=all_objects,
            exit_door_open=exit_door_open,
            exit_door_visible=exit_door_visible,
            button_pressed=button_pressed,
            hazard_ahead=hazard_ahead,
            level_complete=level_complete,
            death_detected=death_detected,
            optical_flow_dx=flow_dx,
            optical_flow_dy=flow_dy,
            optical_flow_magnitude=flow_mag,
            fps=self.fps,
            confidence=1.0
        )

        debug_frame = self.draw_debug_overlay(frame, game_state)
        return game_state, debug_frame

    def draw_debug_overlay(self, frame: np.ndarray, state: GameState) -> np.ndarray:
        debug = frame.copy()
        h, w = debug.shape[:2]

        color_map = {
            ObjectType.PORTAL_BLUE: (255, 180, 0),
            ObjectType.PORTAL_ORANGE: (0, 140, 255),
            ObjectType.CUBE: (255, 230, 0),
            ObjectType.BUTTON_FLOOR: (30, 30, 240),
            ObjectType.DOOR: (0, 255, 120),
            ObjectType.HAZARD_ACID: (0, 120, 0),
            ObjectType.TURRET: (0, 0, 255),
            ObjectType.PORTALABLE_WALL: (200, 200, 200)
        }

        for obj in state.objects:
            if obj.object_type == ObjectType.PORTALABLE_WALL:
                continue

            bx, by, bw, bh = obj.bbox.as_tuple()
            color = color_map.get(obj.object_type, (200, 200, 200))
            
            cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), color, 2)
            
            label = f"{obj.object_type.value} ({obj.confidence:.2f})"
            if obj.object_type == ObjectType.DOOR:
                label += " [OPEN]" if obj.attributes.get("is_open") else " [CLOSED]"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(debug, (bx, max(0, by - th - 6)), (bx + tw + 6, by), color, -1)
            cv2.putText(debug, label, (bx + 3, max(12, by - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Crosshair HUD
        cx, cy = w // 2, h // 2
        ch_state = state.player.crosshair
        ch_col = (0, 255, 0) if ch_state.on_portalable_surface else (80, 80, 80)
        cv2.circle(debug, (cx, cy), 4, ch_col, -1)
        cv2.line(debug, (cx - 12, cy), (cx + 12, cy), ch_col, 1)
        cv2.line(debug, (cx, cy - 12), (cx, cy + 12), ch_col, 1)

        # Status Banner
        cv2.rectangle(debug, (0, 0), (w, 36), (20, 20, 25), -1)
        gun_label = state.player.gun_state.value
        held_str = "CUBE" if state.player.holding_cube else "NONE"
        hud_text = f"FPS: {state.fps:.1f} | Gun: {gun_label} | Hand: {held_str} | Blue: {'ON' if state.portals.blue_active else 'OFF'} | Orange: {'ON' if state.portals.orange_active else 'OFF'} | Door: {'OPEN' if state.exit_door_open else 'CLOSED'}"
        cv2.putText(debug, hud_text, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        return debug

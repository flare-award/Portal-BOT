"""Synthetic Portal 1 frame generator and environment simulator for testing."""

import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from portal_bot.config import CaptureConfig


class MockChamberSimulator:
    """
    Simulates a 3D Portal test chamber and renders realistic frames
    from the player's perspective, responding to simulated inputs.
    """

    def __init__(self, width: int = 1280, height: int = 720, chamber_id: int = 0):
        self.width = width
        self.height = height
        self.chamber_id = chamber_id
        
        # Player state
        self.player_pos = [0.0, 0.0, 1.0]  # x, y, z (meters)
        self.player_yaw = 0.0              # degrees (0 = looking forward +Y)
        self.player_pitch = 0.0            # degrees (-89 up, +89 down)
        self.holding_cube = False
        self.is_dead = False
        
        # Chamber elements state
        self.blue_portal_placed: Optional[Tuple[float, float, float]] = None
        self.orange_portal_placed: Optional[Tuple[float, float, float]] = None
        
        # Cube position
        self.cube_pos = [-2.0, 6.0, 0.5]
        # Button position
        self.button_pos = [2.0, 6.0, 0.1]
        self.button_pressed = False
        
        # Exit door position
        self.door_pos = [0.0, 12.0, 1.5]
        self.door_open = False
        
        # Hazard / acid pit region
        self.has_hazard = (chamber_id >= 1)
        self.hazard_y_range = (8.0, 10.0)
        
        # Elevator / chamber complete
        self.level_complete = False
        self.frame_count = 0

    def reset(self, chamber_id: Optional[int] = None):
        if chamber_id is not None:
            self.chamber_id = chamber_id
        self.player_pos = [0.0, 0.0, 1.0]
        self.player_yaw = 0.0
        self.player_pitch = 0.0
        self.holding_cube = False
        self.is_dead = False
        self.blue_portal_placed = None
        self.orange_portal_placed = None
        self.cube_pos = [-2.0, 6.0, 0.5]
        self.button_pos = [2.0, 6.0, 0.1]
        self.button_pressed = False
        self.door_open = False
        self.level_complete = False

    def step_input(self, action_key: Optional[str] = None, mouse_dx: int = 0, mouse_dy: int = 0, dt: float = 0.05):
        """Updates simulator state based on keyboard/mouse inputs."""
        # Mouse rotation
        sensitivity = 0.18
        self.player_yaw += mouse_dx * sensitivity
        self.player_pitch += mouse_dy * sensitivity
        self.player_pitch = max(-80.0, min(80.0, self.player_pitch))
        
        # Movement
        move_speed = 3.5  # m/s
        rad = math.radians(self.player_yaw)
        forward = (-math.sin(rad), math.cos(rad))
        right = (math.cos(rad), math.sin(rad))
        
        dx, dy = 0.0, 0.0
        if action_key == 'w':
            dx += forward[0] * move_speed * dt
            dy += forward[1] * move_speed * dt
        elif action_key == 's':
            dx -= forward[0] * move_speed * dt
            dy -= forward[1] * move_speed * dt
        elif action_key == 'a':
            dx -= right[0] * move_speed * dt
            dy -= right[1] * move_speed * dt
        elif action_key == 'd':
            dx += right[0] * move_speed * dt
            dy += right[1] * move_speed * dt
        elif action_key == 'e':  # Use / interact
            dist_to_cube = math.hypot(self.player_pos[0] - self.cube_pos[0], self.player_pos[1] - self.cube_pos[1])
            if not self.holding_cube and dist_to_cube < 3.2:
                self.holding_cube = True
            elif self.holding_cube:
                self.holding_cube = False
                self.cube_pos[0] = self.player_pos[0] + forward[0] * 1.5
                self.cube_pos[1] = self.player_pos[1] + forward[1] * 1.5
                self.cube_pos[2] = 0.5
        elif action_key == 'left':  # LMB fire blue portal
            self.blue_portal_placed = (
                self.player_pos[0] + forward[0] * 8.0,
                self.player_pos[1] + forward[1] * 8.0,
                1.5
            )
        elif action_key == 'right':  # RMB fire orange portal
            self.orange_portal_placed = (
                self.player_pos[0] + forward[0] * 8.0,
                self.player_pos[1] + forward[1] * 8.0,
                1.5
            )

        self.player_pos[0] += dx
        self.player_pos[1] += dy
        
        # If holding cube, cube follows player
        if self.holding_cube:
            self.cube_pos[0] = self.player_pos[0] + forward[0] * 1.2
            self.cube_pos[1] = self.player_pos[1] + forward[1] * 1.2
            self.cube_pos[2] = 1.0

        # Check button activation (cube near button)
        dist_cube_button = math.hypot(self.cube_pos[0] - self.button_pos[0], self.cube_pos[1] - self.button_pos[1])
        if dist_cube_button < 1.4 and not self.holding_cube:
            self.button_pressed = True
            self.door_open = True
        else:
            dist_player_button = math.hypot(self.player_pos[0] - self.button_pos[0], self.player_pos[1] - self.button_pos[1])
            if dist_player_button < 1.0:
                self.button_pressed = True
                self.door_open = True
            elif not self.holding_cube and dist_cube_button >= 1.4:
                self.button_pressed = False
                self.door_open = False

        # Check exit reach
        dist_door = math.hypot(self.player_pos[0] - self.door_pos[0], self.player_pos[1] - self.door_pos[1])
        if self.door_open and dist_door < 2.5:
            self.level_complete = True

    def render_frame(self) -> np.ndarray:
        self.frame_count += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Sky/ceiling
        frame[0:int(self.height * 0.45), :] = (45, 45, 50)
        # Floor
        frame[int(self.height * 0.45):, :] = (70, 75, 80)
        
        # Wall background
        wall_y1 = int(self.height * 0.25)
        wall_y2 = int(self.height * 0.75)
        cv2.rectangle(frame, (0, wall_y1), (self.width, wall_y2), (185, 190, 195), -1)
        
        grid_step = 60
        for gx in range(0, self.width, grid_step):
            cv2.line(frame, (gx, wall_y1), (gx, wall_y2), (150, 155, 160), 1)
        for gy in range(wall_y1, wall_y2, grid_step):
            cv2.line(frame, (0, gy), (self.width, gy), (150, 155, 160), 1)

        ind_color = (255, 180, 50) if self.door_open else (30, 120, 240)
        cv2.line(frame, (int(self.width * 0.3), int(self.height * 0.8)), (int(self.width * 0.7), int(self.height * 0.5)), ind_color, 4)

        self._render_objects(frame)
        self._render_hud(frame)

        return frame

    def _project_3d(self, x: float, y: float, z: float) -> Optional[Tuple[int, int, float]]:
        dx = x - self.player_pos[0]
        dy = y - self.player_pos[1]
        dz = z - self.player_pos[2]
        
        rad = math.radians(self.player_yaw)
        rot_x = dx * math.cos(rad) - dy * math.sin(rad)
        rot_y = dx * math.sin(rad) + dy * math.cos(rad)
        
        if rot_y <= 0.3:
            return None
            
        pitch_rad = math.radians(self.player_pitch)
        rot_z = dz * math.cos(pitch_rad) + rot_y * math.sin(pitch_rad)
        depth = rot_y * math.cos(pitch_rad) - dz * math.sin(pitch_rad)
        
        if depth <= 0.3:
            return None
            
        fov = math.radians(90.0)
        f = (self.width / 2.0) / math.tan(fov / 2.0)
        
        px = int(self.width / 2.0 + (rot_x * f / depth))
        py = int(self.height / 2.0 - (rot_z * f / depth))
        
        return px, py, depth

    def _render_objects(self, frame: np.ndarray):
        btn_proj = self._project_3d(self.button_pos[0], self.button_pos[1], self.button_pos[2])
        if btn_proj:
            bx, by, depth = btn_proj
            if 0 <= bx < self.width and 0 <= by < self.height:
                radius = max(8, int(80 / depth))
                cv2.ellipse(frame, (bx, by), (radius, radius // 2), 0, 0, 360, (60, 60, 65), -1)
                btn_dome_color = (240, 200, 40) if self.button_pressed else (30, 40, 220)
                cv2.ellipse(frame, (bx, by - radius // 4), (radius * 3 // 4, radius * 3 // 8), 0, 0, 360, btn_dome_color, -1)
                cv2.ellipse(frame, (bx, by - radius // 4), (radius * 3 // 4, radius * 3 // 8), 0, 0, 360, (255, 255, 255), 2)

        if not self.holding_cube:
            cube_proj = self._project_3d(self.cube_pos[0], self.cube_pos[1], self.cube_pos[2])
            if cube_proj:
                cx, cy, depth = cube_proj
                if 0 <= cx < self.width and 0 <= cy < self.height:
                    size = max(10, int(90 / depth))
                    x1 = cx - size // 2
                    y1 = cy - size // 2
                    x2 = cx + size // 2
                    y2 = cy + size // 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (130, 135, 140), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 205, 210), 2)
                    cv2.circle(frame, (cx, cy), max(3, size // 3), (255, 230, 0), max(1, size // 10))
                    cv2.circle(frame, (cx, cy), max(1, size // 6), (255, 255, 255), -1)
        else:
            hc_x = int(self.width * 0.65)
            hc_y = int(self.height * 0.75)
            hc_size = 140
            cv2.rectangle(frame, (hc_x, hc_y), (hc_x + hc_size, hc_y + hc_size), (120, 125, 130), -1)
            cv2.circle(frame, (hc_x + hc_size // 2, hc_y + hc_size // 2), 35, (255, 230, 0), 6)

        if self.blue_portal_placed:
            bp = self._project_3d(self.blue_portal_placed[0], self.blue_portal_placed[1], self.blue_portal_placed[2])
            if bp:
                px, py, depth = bp
                pw = max(10, int(70 / depth))
                ph = max(18, int(130 / depth))
                cv2.ellipse(frame, (px, py), (pw, ph), 0, 0, 360, (255, 190, 0), -1)
                cv2.ellipse(frame, (px, py), (pw * 3 // 4, ph * 3 // 4), 0, 0, 360, (20, 20, 25), -1)
                cv2.ellipse(frame, (px, py), (pw, ph), 0, 0, 360, (255, 255, 200), 2)

        if self.orange_portal_placed:
            op = self._project_3d(self.orange_portal_placed[0], self.orange_portal_placed[1], self.orange_portal_placed[2])
            if op:
                px, py, depth = op
                pw = max(10, int(70 / depth))
                ph = max(18, int(130 / depth))
                cv2.ellipse(frame, (px, py), (pw, ph), 0, 0, 360, (0, 140, 255), -1)
                cv2.ellipse(frame, (px, py), (pw * 3 // 4, ph * 3 // 4), 0, 0, 360, (20, 20, 25), -1)
                cv2.ellipse(frame, (px, py), (pw, ph), 0, 0, 360, (200, 230, 255), 2)

        door_proj = self._project_3d(self.door_pos[0], self.door_pos[1], self.door_pos[2])
        if door_proj:
            dx, dy, depth = door_proj
            dw = max(15, int(110 / depth))
            dh = max(25, int(180 / depth))
            x1 = dx - dw // 2
            y1 = dy - dh // 2
            x2 = dx + dw // 2
            y2 = dy + dh // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 55), -1)
            if self.door_open:
                cv2.rectangle(frame, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), (20, 180, 240), -1)
                cv2.putText(frame, "EXIT", (dx - 18, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                cv2.rectangle(frame, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), (80, 85, 90), -1)
                cv2.line(frame, (dx, y1 + 4), (dx, y2 - 4), (30, 30, 30), 2)
                cv2.putText(frame, "CLOSED", (dx - 28, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    def _render_hud(self, frame: np.ndarray):
        cx = self.width // 2
        cy = self.height // 2
        
        cross_color = (220, 220, 220)
        cv2.circle(frame, (cx, cy), 3, cross_color, -1)
        
        blue_bracket_color = (255, 200, 0) if self.blue_portal_placed else (120, 120, 120)
        cv2.ellipse(frame, (cx - 10, cy), (14, 18), 0, 110, 250, blue_bracket_color, 2)
        
        orange_bracket_color = (0, 150, 255) if self.orange_portal_placed else (120, 120, 120)
        cv2.ellipse(frame, (cx + 10, cy), (14, 18), 0, -70, 70, orange_bracket_color, 2)

        cv2.rectangle(frame, (20, 20), (160, 80), (30, 30, 35), -1)
        cv2.rectangle(frame, (20, 20), (160, 80), (200, 200, 200), 1)
        cv2.putText(frame, f"0{self.chamber_id} / 19", (35, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if self.level_complete:
            cv2.rectangle(frame, (self.width // 2 - 200, self.height // 2 - 40), (self.width // 2 + 200, self.height // 2 + 40), (0, 160, 0), -1)
            cv2.putText(frame, "CHAMBER COMPLETE", (self.width // 2 - 170, self.height // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


class MockScreenCapture:
    """Mock screen capture implementation matching ScreenCapture interface."""

    def __init__(self, config: CaptureConfig):
        self.config = config
        self.sim = MockChamberSimulator(config.width, config.height, config.mock_chamber_index)
        self.running = False
        self.fps = 30.0
        self.frame_count = 0

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def capture_frame(self) -> np.ndarray:
        self.frame_count += 1
        return self.sim.render_frame()

    def get_latest_frame(self) -> np.ndarray:
        return self.sim.render_frame()

    def step_input(self, action_key: Optional[str] = None, mouse_dx: int = 0, mouse_dy: int = 0, dt: float = 0.05):
        self.sim.step_input(action_key, mouse_dx, mouse_dy, dt)

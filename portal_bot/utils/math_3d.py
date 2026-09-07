"""3D Vector, Coordinate Projection, and Spatial Geometry Math for Portal-BOT."""

import math
from typing import Optional, Tuple


def clamp(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(val, max_val))


def normalize_angle_deg(angle: float) -> float:
    """Normalize angle to [-180, 180) degrees."""
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


def distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def distance_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)


def screen_to_angle(
    screen_x: int,
    screen_y: int,
    screen_w: int,
    screen_h: int,
    fov_deg: float = 75.0
) -> Tuple[float, float]:
    cx = screen_w / 2.0
    cy = screen_h / 2.0
    dx = screen_x - cx
    dy = screen_y - cy
    fov_rad = math.radians(fov_deg)
    focal_len = (screen_w / 2.0) / math.tan(fov_rad / 2.0)
    yaw_deg = math.degrees(math.atan2(dx, focal_len))
    pitch_deg = math.degrees(math.atan2(dy, focal_len))
    return yaw_deg, pitch_deg


def estimate_ground_object_3d(
    screen_x: int,
    screen_y: int,
    screen_w: int = 640,
    screen_h: int = 360,
    camera_pitch_deg: float = 0.0,
    eye_height_m: float = 1.6,
    fov_deg: float = 75.0
) -> Tuple[Tuple[float, float, float], float]:
    """
    Inverse Perspective Mapping: Projects a 2D ground-plane pixel (screen_x, screen_y)
    into 3D world relative coordinates (X_right, Y_forward, Z_up) in meters.
    """
    cx = screen_w / 2.0
    cy = screen_h / 2.0
    dx = screen_x - cx
    dy = screen_y - cy

    fov_rad = math.radians(fov_deg)
    focal_len = (screen_w / 2.0) / math.tan(fov_rad / 2.0)

    yaw_deg = math.degrees(math.atan2(dx, focal_len))
    vertical_angle_rad = math.radians(camera_pitch_deg) + math.atan2(dy, focal_len)

    if dy <= 4:
        distance_m = 25.0
    else:
        tan_v = math.tan(vertical_angle_rad)
        if tan_v > 0.05:
            distance_m = max(0.4, min(30.0, eye_height_m / tan_v))
        else:
            distance_m = 25.0

    yaw_rad = math.radians(yaw_deg)
    x_3d = distance_m * math.sin(yaw_rad)
    y_3d = distance_m * math.cos(yaw_rad)
    z_3d = 0.0

    return (round(x_3d, 2), round(y_3d, 2), round(z_3d, 2)), round(distance_m, 2)


def estimate_object_3d(
    screen_x: int,
    screen_y: int,
    bbox_w: int,
    bbox_h: int,
    screen_w: int = 640,
    screen_h: int = 360,
    assumed_real_size_m: float = 0.8,
    camera_pitch_deg: float = 0.0,
    fov_deg: float = 75.0
) -> Tuple[Tuple[float, float, float], float]:
    """
    Projects general 3D objects (walls, doors, portals, cubes) from bounding box apparent size.
    """
    cx = screen_w / 2.0
    cy = screen_h / 2.0
    dx = screen_x - cx
    dy = screen_y - cy

    fov_rad = math.radians(fov_deg)
    focal_len = (screen_w / 2.0) / math.tan(fov_rad / 2.0)

    apparent_px = max(bbox_w, bbox_h)
    if apparent_px > 5:
        distance_m = max(0.4, min(35.0, (assumed_real_size_m * focal_len) / float(apparent_px)))
    else:
        distance_m = 25.0

    yaw_deg = math.degrees(math.atan2(dx, focal_len))
    pitch_deg = math.degrees(math.atan2(dy, focal_len))

    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg + camera_pitch_deg)

    x_3d = distance_m * math.sin(yaw_rad)
    y_3d = distance_m * math.cos(yaw_rad) * math.cos(pitch_rad)
    z_3d = - distance_m * math.sin(pitch_rad)

    return (round(x_3d, 2), round(y_3d, 2), round(z_3d, 2)), round(distance_m, 2)


class PIDController:
    """Proportional-Integral-Derivative controller for visual tracking."""

    def __init__(self, kp: float = 0.22, ki: float = 0.0, kd: float = 0.02, out_min: float = -60.0, out_max: float = 60.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.integral = 0.0
        self.prev_error: Optional[float] = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def update(self, error: float, dt: float = 0.05) -> float:
        p_term = self.kp * error
        d_term = 0.0
        if self.prev_error is not None:
            raw_diff = error - self.prev_error
            raw_diff = clamp(raw_diff, -30.0, 30.0)
            d_term = self.kd * (raw_diff / max(0.01, dt))
            
        self.prev_error = error
        output = p_term + d_term
        return clamp(output, self.out_min, self.out_max)

"""3D Vector, Coordinate Projection, and Control Math for Portal-BOT."""

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
    fov_deg: float = 90.0
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
        # Simple, robust proportional response with damped derivative
        p_term = self.kp * error
        
        d_term = 0.0
        if self.prev_error is not None:
            raw_diff = error - self.prev_error
            # Filter huge jump spikes
            raw_diff = clamp(raw_diff, -30.0, 30.0)
            d_term = self.kd * (raw_diff / max(0.01, dt))
            
        self.prev_error = error
        output = p_term + d_term
        return clamp(output, self.out_min, self.out_max)

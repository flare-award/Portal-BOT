"""Configuration settings for Portal 1 Autonomous Bot."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class CaptureConfig:
    window_title_patterns: List[str] = field(
        default_factory=lambda: ["Portal", "hl2", "Portal (32-bit)", "Source - Portal"]
    )
    target_fps: int = 60
    width: int = 1280
    height: int = 720
    use_mock: bool = False
    mock_chamber_index: int = 0  # 0: Chamber 00, 1: Chamber 01, 2: Chamber 02
    custom_console_log_path: str = ""


@dataclass
class KeyBindings:
    forward: str = "w"
    backward: str = "s"
    move_left: str = "a"
    move_right: str = "d"
    jump: str = "space"
    crouch: str = "ctrl"
    use: str = "e"
    fire_blue: str = "left"  # LMB
    fire_orange: str = "right"  # RMB
    quick_save: str = "f6"
    quick_load: str = "f9"
    emergency_stop: str = "f8"


@dataclass
class VisionConfig:
    # Color HSV ranges for Portal 1 elements
    blue_portal_hsv_lower: Tuple[int, int, int] = (85, 120, 150)
    blue_portal_hsv_upper: Tuple[int, int, int] = (125, 255, 255)

    orange_portal_hsv_lower: Tuple[int, int, int] = (8, 140, 160)
    orange_portal_hsv_upper: Tuple[int, int, int] = (25, 255, 255)

    cube_ring_hsv_lower: Tuple[int, int, int] = (80, 80, 100)
    cube_ring_hsv_upper: Tuple[int, int, int] = (130, 255, 255)

    button_red_hsv_lower1: Tuple[int, int, int] = (0, 120, 100)
    button_red_hsv_upper1: Tuple[int, int, int] = (10, 255, 255)
    button_red_hsv_lower2: Tuple[int, int, int] = (170, 120, 100)
    button_red_hsv_upper2: Tuple[int, int, int] = (180, 255, 255)

    indicator_blue_hsv_lower: Tuple[int, int, int] = (90, 100, 120)
    indicator_blue_hsv_upper: Tuple[int, int, int] = (130, 255, 255)
    indicator_orange_hsv_lower: Tuple[int, int, int] = (10, 120, 120)
    indicator_orange_hsv_upper: Tuple[int, int, int] = (25, 255, 255)

    acid_hsv_lower: Tuple[int, int, int] = (30, 70, 40)
    acid_hsv_upper: Tuple[int, int, int] = (75, 255, 180)

    portalable_min_brightness: int = 75
    portalable_max_saturation: int = 65

    optical_flow_points: int = 100
    optical_flow_min_distance: int = 7
    crosshair_roi_size: float = 0.08


@dataclass
class MovementConfig:
    aim_kp: float = 0.22
    aim_ki: float = 0.0
    aim_kd: float = 0.02
    mouse_sensitivity: float = 1.0
    max_mouse_step: int = 60
    aim_tolerance_px: float = 18.0

    step_duration_short: float = 0.15
    step_duration_normal: float = 0.35
    step_duration_long: float = 0.70

    min_optical_flow_movement: float = 1.2
    stuck_time_threshold: float = 2.5


@dataclass
class PlannerConfig:
    planning_fps: float = 10.0  # reasoning loop rate (Hz)
    max_subgoal_depth: int = 6
    exploration_scan_angles: int = 4
    max_consecutive_failed_actions: int = 4
    spatial_memory_retention_seconds: float = 60.0


@dataclass
class UIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    stream_fps: int = 30
    jpeg_quality: int = 70


@dataclass
class BotConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    keys: KeyBindings = field(default_factory=KeyBindings)
    vision: VisionConfig = field(default_factory=VisionConfig)
    movement: MovementConfig = field(default_factory=MovementConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    ui: UIConfig = field(default_factory=UIConfig)


DEFAULT_CONFIG = BotConfig()

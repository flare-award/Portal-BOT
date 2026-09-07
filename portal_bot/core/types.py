"""Core data structures, enums, and types for Portal-BOT."""

from dataclasses import dataclass, field
from enum import Enum, auto
import time
from typing import Any, Dict, List, Optional, Tuple


class BotStatus(str, Enum):
    IDLE = "Idle"
    RUNNING = "Running"
    PAUSED = "Paused"
    SOLVING_PUZZLE = "Solving Puzzle"
    STUCK_RECOVERING = "Stuck - Recovering"
    DEAD_RELOADING = "Dead - Reloading"
    LEVEL_COMPLETE = "Level Complete"
    STOPPED = "Stopped"
    ERROR = "Error"


class ObjectType(str, Enum):
    PORTAL_BLUE = "portal_blue"
    PORTAL_ORANGE = "portal_orange"
    CUBE = "cube"
    BUTTON_FLOOR = "button_floor"
    BUTTON_PEDESTAL = "button_pedestal"
    DOOR = "door"
    ELEVATOR = "elevator"
    FIZZLER = "fizzler"
    HAZARD_ACID = "hazard_acid"
    TURRET = "turret"
    ENERGY_BALL = "energy_ball"
    ENERGY_CATCHER = "energy_catcher"
    PORTALABLE_WALL = "portalable_wall"
    NON_PORTALABLE_WALL = "non_portalable_wall"
    CHAMBER_SIGN = "chamber_sign"
    CROSSHAIR = "crosshair"
    UNKNOWN = "unknown"


class GoalType(str, Enum):
    COMPLETE_LEVEL = "CompleteLevel"
    REACH_EXIT = "ReachExit"
    OPEN_DOOR = "OpenDoor"
    ACTIVATE_BUTTON = "ActivateButton"
    OBTAIN_CUBE = "ObtainCube"
    CARRY_CUBE_TO_BUTTON = "CarryCubeToButton"
    PLACE_PORTAL_PAIR = "PlacePortalPair"
    PLACE_BLUE_PORTAL = "PlaceBluePortal"
    PLACE_ORANGE_PORTAL = "PlaceOrangePortal"
    TRAVERSE_PORTAL = "TraversePortal"
    EXPLORE_SURROUNDINGS = "ExploreSurroundings"
    AVOID_HAZARD = "AvoidHazard"
    RECOVER_STUCK = "RecoverStuck"
    RESPAWN_RELOAD = "RespawnReload"


class ActionType(str, Enum):
    MOVE_FORWARD = "MoveForward"
    MOVE_BACKWARD = "MoveBackward"
    STRAFE_LEFT = "StrafeLeft"
    STRAFE_RIGHT = "StrafeRight"
    JUMP = "Jump"
    CROUCH = "Crouch"
    USE_INTERACT = "UseInteract"
    FIRE_BLUE_PORTAL = "FireBluePortal"
    FIRE_ORANGE_PORTAL = "FireOrangePortal"
    AIM_AT_TARGET = "AimAtTarget"
    ROTATE_CAMERA = "RotateCamera"
    LOOK_UP = "LookUp"
    LOOK_DOWN = "LookDown"
    WAIT = "Wait"
    QUICK_SAVE = "QuickSave"
    QUICK_LOAD = "QuickLoad"
    EMERGENCY_STOP = "EmergencyStop"


class PortalGunState(str, Enum):
    NO_GUN = "No Gun"
    SINGLE_PORTAL_BLUE = "Single Portal Gun (Blue Only)"
    DUAL_PORTAL = "Dual Portal Gun"


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2

    @property
    def area(self) -> int:
        return self.w * self.h

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class DetectedObject:
    object_type: ObjectType
    bbox: BoundingBox
    confidence: float = 1.0
    center_screen: Tuple[int, int] = (0, 0)
    distance_estimate: float = 1.0  # approximate relative distance (1.0 = normal, 0.2 = close)
    attributes: Dict[str, Any] = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.center_screen == (0, 0):
            self.center_screen = (self.bbox.cx, self.bbox.cy)


@dataclass
class PortalState:
    blue_active: bool = False
    orange_active: bool = False
    blue_bbox: Optional[BoundingBox] = None
    orange_bbox: Optional[BoundingBox] = None
    blue_screen_pos: Optional[Tuple[int, int]] = None
    orange_screen_pos: Optional[Tuple[int, int]] = None
    blue_world_pos: Optional[Tuple[float, float, float]] = None
    orange_world_pos: Optional[Tuple[float, float, float]] = None


@dataclass
class CrosshairState:
    center_pos: Tuple[int, int] = (640, 360)
    on_portalable_surface: bool = True
    blue_ring_filled: bool = False
    orange_ring_filled: bool = False
    interaction_available: bool = False


@dataclass
class PlayerState:
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0    # horizontal angle degrees
    pitch: float = 0.0  # vertical angle degrees
    is_grounded: bool = True
    is_moving: bool = False
    is_stuck: bool = False
    is_dead: bool = False
    holding_cube: bool = False
    crosshair: CrosshairState = field(default_factory=CrosshairState)
    gun_state: PortalGunState = PortalGunState.NO_GUN
    last_motion_time: float = field(default_factory=time.time)


@dataclass
class GameState:
    frame_id: int = 0
    timestamp: float = field(default_factory=time.time)
    player: PlayerState = field(default_factory=PlayerState)
    portals: PortalState = field(default_factory=PortalState)
    objects: List[DetectedObject] = field(default_factory=list)
    
    # High-level state flags
    exit_door_open: bool = False
    exit_door_visible: bool = False
    button_pressed: bool = False
    hazard_ahead: bool = False
    level_complete: bool = False
    death_detected: bool = False
    current_chamber: int = 0
    
    # Visual odometry deltas from last frame
    optical_flow_dx: float = 0.0
    optical_flow_dy: float = 0.0
    optical_flow_magnitude: float = 0.0
    
    fps: float = 0.0
    confidence: float = 1.0


@dataclass
class ActionCommand:
    action_type: ActionType
    duration: float = 0.2
    mouse_dx: int = 0
    mouse_dy: int = 0
    key: Optional[str] = None
    target_screen_point: Optional[Tuple[int, int]] = None
    reason: str = ""
    expected_outcome: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ActionResult:
    command: ActionCommand
    success: bool = True
    message: str = ""
    duration_actual: float = 0.0
    motion_detected: bool = False
    optical_flow_magnitude: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SubGoal:
    id: str
    goal_type: GoalType
    target_object_type: Optional[ObjectType] = None
    target_screen_point: Optional[Tuple[int, int]] = None
    target_world_pos: Optional[Tuple[float, float, float]] = None
    reason: str = ""
    priority: int = 100
    is_completed: bool = False
    is_failed: bool = False
    children: List["SubGoal"] = field(default_factory=list)
    failed_attempts: int = 0
    created_at: float = field(default_factory=time.time)

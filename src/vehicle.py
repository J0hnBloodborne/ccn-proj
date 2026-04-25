"""Vehicle agent with animation state."""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from src.hub import Hub
    from src.router import Router

from src.config import (
    IDM_V0, DATA_GEN_RATE, TURN_SPEED_MULT, TURN_DURATION,
    VEHICLE_LENGTH, VEHICLE_WIDTH, LANE_WIDTH
)
from src.physics import idm_acceleration
from src.animation import (
    bezier_point, bezier_derivative, heading_from_deltas,
    interpolate_angle
)


@dataclass
class TurnState:
    active: bool = False
    path: List[Tuple[float, float]] = field(default_factory=list)
    progress: int = 0
    entry_point: Tuple[float, float] = (0, 0)
    exit_point: Tuple[float, float] = (0, 0)
    turn_type: str = "straight"
    intersection_center: Tuple[float, float] = (0, 0)


@dataclass
class Vehicle:
    osm_id: int
    route: List[int] = field(default_factory=list)
    route_index: int = 0

    x: float = 0.0
    y: float = 0.0
    v: float = 0.0
    a: float = 0.0
    heading: float = 0.0

    lane_offset: float = 0.0
    target_lane_offset: float = 0.0

    state: str = "cruising"
    data_buffer: float = 0.0
    offload_rate: float = 0.0
    connected_hub: Optional['Hub'] = None

    turn: TurnState = field(default_factory=TurnState)
    suspension_phase: float = 0.0

    speed_limit: float = IDM_V0

    def get_color(self) -> Tuple[int, int, int]:
        return {
            "cruising": (50, 200, 100),
            "following": (50, 200, 100),
            "lane_changing": (220, 220, 80),
            "yielding": (255, 150, 50),
            "offloading": (80, 200, 220),
            "turning": (200, 200, 255),
            "approaching_turn": (180, 180, 100),
        }.get(self.state, (220, 220, 230))

    def get_lead_gap_and_dv(self, lead: Optional['Vehicle']) -> Tuple[float, float]:
        if lead is None:
            return 1000.0, 0.0
        dx = lead.x - self.x
        dy = lead.y - self.y
        gap = math.sqrt(dx * dx + dy * dy) - VEHICLE_LENGTH
        dv = self.v - lead.v
        return max(gap, 0.1), dv

    def compute_desired_speed(self) -> float:
        """Compute desired speed based on state."""
        if self.turn.active:
            return self.speed_limit * TURN_SPEED_MULT
        return self.speed_limit

    def update_turn(self):
        """Update turn animation."""
        if not self.turn.active:
            return

        self.turn.progress += 1
        t = self.turn.progress / TURN_DURATION

        if self.turn.progress >= TURN_DURATION:
            self.x = self.turn.exit_point[0]
            self.y = self.turn.exit_point[1]
            self.turn.active = False
            self.turn.path = []
            self.turn.progress = 0
            self.state = "cruising"
            return

        new_pos = bezier_point(
            self.turn.entry_point,
            self.turn.intersection_center,
            self.turn.exit_point,
            t
        )
        self.x = new_pos[0]
        self.y = new_pos[1]

        dx, dy = bezier_derivative(
            self.turn.entry_point,
            self.turn.intersection_center,
            self.turn.exit_point,
            t
        )
        desired_heading = heading_from_deltas(dx, dy)
        self.heading = interpolate_angle(self.heading, desired_heading, 0.2)

    def start_turn(self, path: List[Tuple[float, float]], entry: Tuple[float, float],
                   exit_pt: Tuple[float, float], intersection_center: Tuple[float, float],
                   turn_type: str):
        """Initiate a turn maneuver."""
        self.turn.active = True
        self.turn.path = path
        self.turn.progress = 0
        self.turn.entry_point = entry
        self.turn.exit_point = exit_pt
        self.turn.intersection_center = intersection_center
        self.turn.turn_type = turn_type
        self.state = "turning"

    def update_lane_offset(self):
        """Update lateral position during lane change."""
        diff = self.target_lane_offset - self.lane_offset
        self.lane_offset += diff * 0.1
        if abs(diff) < 0.5:
            self.lane_offset = self.target_lane_offset
            self.target_lane_offset = 0.0

    def update_suspension(self):
        """Update suspension animation phase."""
        self.suspension_phase += self.v * 0.02
        if self.suspension_phase > 2 * math.pi:
            self.suspension_phase -= 2 * math.pi

    def get_brake_active(self) -> bool:
        """Check if brake lights should be on."""
        return self.a < -1.0

    def get_accel_active(self) -> bool:
        """Check if acceleration lines should be shown."""
        return self.a > 1.0
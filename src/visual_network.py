"""Visual network generator - small, dense, guaranteed to work."""

import math
import random
from typing import Dict, List, Tuple, Optional

# Config values (duplicated here for standalone use)
VEHICLE_LENGTH = 4.0
VEHICLE_WIDTH = 1.8
LANE_WIDTH = 3.5
HUB_RADIUS = 150.0
HUB_BANDWIDTH = 100.0
DATA_GEN_RATE = 10.0 / 60.0

IDM_A_MAX = 2.5
IDM_B = 3.0
IDM_V0 = 33.3
IDM_S0 = 2.0
IDM_T = 1.5

MOBIL_P = 0.1
MOBIL_THRESHOLD = 0.01
MOBIL_B_SAFE = 5.0

# Road colors
ROAD_COLORS = {
    "motorway": (60, 80, 160),
    "primary": (180, 140, 100),
    "secondary": (160, 150, 120),
    "residential": (100, 100, 100),
}


@dataclass
class Node:
    id: int
    x: float
    y: float
    in_links: List['Link'] = field(default_factory=list)
    out_links: List['Link'] = field(default_factory=list)
    
    is_signalized: bool = False
    signal_cycle: float = 60.0
    signal_timer: float = 0.0
    signal_phase: int = 0
    priority: int = 1
    
    def __hash__(self):
        return hash(self.id)


@dataclass 
class Link:
    id: int
    from_node: 'Node'
    to_node: 'Node'
    length: float
    num_lanes: int
    free_flow_speed: float
    road_type: str = "residential"
    lanes: List['Lane'] = field(default_factory=list)
    
    density: float = 0.0
    flow: float = 0.0
    is_congested: bool = False
    capacity: float = 0.0
    priority: int = 1
    
    def __hash__(self):
        return hash(self.id)
    
    def __post_init__(self):
        self.lanes = [Lane(idx, self) for idx in range(self.num_lanes)]
        self.capacity = self.free_flow_speed * 0.1
        if self.road_type in ['motorway', 'trunk']:
            self.priority = 3
        elif self.road_type == 'primary':
            self.priority = 2
    
    @property
    def v_c_ratio(self) -> float:
        return self.flow / max(self.capacity, 0.001)
    
    def get_angle(self) -> float:
        return math.atan2(self.to_node.y - self.from_node.y, 
                         self.to_node.x - self.from_node.x)


@dataclass
class Lane:
    index: int
    link: 'Link'
    vehicles: List[int] = field(default_factory=list)


@dataclass
class Vehicle:
    id: int
    link: 'Link'
    lane: int
    x: float
    v: float
    a: float
    
    changing_lane: bool = False
    target_lane: int = -1
    lane_change_timer: float = 0.0
    
    state: str = "moving"
    data_buffer: float = 0.0
    offload_rate: float = 0.0
    connected_hub: Optional['Hub'] = None
    
    route: List[int] = field(default_factory=list)
    route_idx: int = 0
    waiting_at_node: bool = False
    
    def __hash__(self):
        return hash(self.id)
    
    def get_world_pos(self) -> Tuple[float, float]:
        if self.link.length < 0.001:
            return (self.link.from_node.x, self.link.from_node.y)
        
        dx = self.link.to_node.x - self.link.from_node.x
        dy = self.link.to_node.y - self.link.from_node.y
        link_len = self.link.length
        
        dx /= link_len
        dy /= link_len
        
        perp_x = -dy
        perp_y = dx
        
        lane_offset = (self.lane + 0.5) * LANE_WIDTH
        
        world_x = self.link.from_node.x + self.x * dx + perp_x * lane_offset
        world_y = self.link.from_node.y + self.x * dy + perp_y * lane_offset
        
        return (world_x, world_y)


@dataclass
class Hub:
    x: float
    y: float
    radius: float = 150.0
    bandwidth: float = 100.0
    connected: List[Vehicle] = field(default_factory=list)
    
    pulse_phase: float = 0.0
    glow_intensity: float = 1.0
    
    def get_allocated_bw(self) -> float:
        return self.bandwidth / len(self.connected) if self.connected else 0.0
    
    def update_animation(self, dt: float = 1/60):
        self.pulse_phase += dt * 3.0
        if self.pulse_phase > 2 * math.pi:
            self.pulse_phase -= 2 * math.pi
        self.glow_intensity = 0.7 + 0.3 * abs(math.sin(self.pulse_phase)) if self.connected else 0.5
    
    def get_pulse_radius(self, base_radius: float) -> float:
        return base_radius * (1.0 + abs(math.sin(self.pulse_phase)) * 0.1)
    
    def get_ring_alpha(self) -> int:
        return 100 + int(80 * abs(math.sin(self.pulse_phase))) if self.connected else 60


@dataclass
class Event:
    x: float
    y: float
    radius: float = 40.0
    data_amount: float = 5.0
    collected: int = 0


from dataclasses import dataclass, field


class VisualNetwork:
    """Creates a small, dense, visually pleasing network."""
    
    def __init__(self):
        self.nodes: Dict[int, 'Node'] = {}
        self.links: Dict[int, 'Link'] = {}
        self.vehicles: Dict[int, 'Vehicle'] = {}
        
        self.next_node_id = 0
        self.next_link_id = 0
        self.next_vehicle_id = 0
        
        # Generate a small, dense city grid
        self._generate_grid()
    
    def _generate_grid(self):
        """Generate a small, dense 5x5 grid that's guaranteed to work."""
        grid_size = 5
        cell_size = 120  # pixels between nodes
        
        # Create grid of nodes
        for i in range(grid_size):
            for j in range(grid_size):
                x = j * cell_size
                y = i * cell_size
                
                # Every other intersection is signalized
                is_signal = (i + j) % 2 == 0
                priority = 2 if (i == 0 or i == grid_size-1 or j == 0 or j == grid_size-1) else 1
                
                node = Node(
                    id=self.next_node_id,
                    x=x, y=y,
                    priority=priority,
                    is_signalized=is_signal
                )
                node.signal_cycle = 50 + random.uniform(-10, 15)
                
                self.nodes[node.id] = node
                self.next_node_id += 1
        
        # Create horizontal links
        for i in range(grid_size):
            for j in range(grid_size - 1):
                node1 = self.nodes[i * grid_size + j]
                node2 = self.nodes[i * grid_size + j + 1]
                self._add_link(node1, node2, 2, "primary")
        
        # Create vertical links
        for i in range(grid_size - 1):
            for j in range(grid_size):
                node1 = self.nodes[i * grid_size + j]
                node2 = self.nodes[(i + 1) * grid_size + j]
                self._add_link(node1, node2, 2, "primary")
        
        # Add some diagonal shortcuts for more interesting traffic
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                if random.random() < 0.5:
                    node1 = self.nodes[i * grid_size + j]
                    node2 = self.nodes[(i + 1) * grid_size + j + 1]
                    self._add_link(node1, node2, 1, "secondary")
                
                if random.random() < 0.5:
                    node1 = self.nodes[i * grid_size + j + 1]
                    node2 = self.nodes[(i + 1) * grid_size + j]
                    self._add_link(node1, node2, 1, "secondary")
    
    def _add_link(self, from_node: Node, to_node: Node,
                  num_lanes: int, road_type: str):
        """Add a bidirectional link between nodes."""
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        length = math.sqrt(dx * dx + dy * dy)
        
        if length < 1:
            return
        
        if road_type == "primary":
            speed = 13.9  # 50 km/h
        else:
            speed = 8.3   # 30 km/h
        
        # Forward link
        link = Link(
            id=self.next_link_id,
            from_node=from_node,
            to_node=to_node,
            length=length,
            num_lanes=num_lanes,
            free_flow_speed=speed,
            road_type=road_type
        )
        self.links[link.id] = link
        self.next_link_id += 1
        from_node.out_links.append(link)
        to_node.in_links.append(link)
        
        # Reverse link
        rev_link = Link(
            id=self.next_link_id,
            from_node=to_node,
            to_node=from_node,
            length=length,
            num_lanes=num_lanes,
            free_flow_speed=speed,
            road_type=road_type
        )
        self.links[rev_link.id] = rev_link
        self.next_link_id += 1
        to_node.out_links.append(rev_link)
        from_node.in_links.append(rev_link)
    
    def get_link(self, from_id: int, to_id: int) -> Optional[Link]:
        from_node = self.nodes.get(from_id)
        if not from_node:
            return None
        for link in from_node.out_links:
            if link.to_node.id == to_id:
                return link
        return None
    
    def dijkstra(self, start_id: int, end_id: int) -> Optional[List[int]]:
        import heapq
        
        if start_id not in self.nodes or end_id not in self.nodes:
            return None
        if start_id == end_id:
            return [start_id]
        
        dist = {start_id: 0.0}
        prev = {start_id: None}
        pq = [(0.0, start_id)]
        visited = set()
        
        while pq:
            d, current = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)
            
            if current == end_id:
                break
            
            node = self.nodes[current]
            for link in node.out_links:
                neighbor = link.to_node.id
                if neighbor in visited:
                    continue
                
                cost = link.length / max(link.free_flow_speed, 1)
                new_dist = d + cost
                
                if neighbor not in dist or new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = (current, link.id)
                    heapq.heappush(pq, (new_dist, neighbor))
        
        if end_id not in prev:
            return None
        
        path = []
        current = end_id
        while current is not None:
            path.append(current)
            if current in prev and prev[current]:
                current = prev[current][0]
            else:
                current = None
        path.reverse()
        return path
    
    def create_vehicle(self, start_node: Node, end_node: Node) -> Optional[Vehicle]:
        path = self.dijkstra(start_node.id, end_node.id)
        if not path or len(path) < 2:
            return None
        
        first_link = self.get_link(path[0], path[1])
        if not first_link:
            return None
        
        # Find best lane with space
        min_veh = float('inf')
        best_lane = 0
        for i, lane in enumerate(first_link.lanes):
            count = len(lane.vehicles)
            if count < min_veh:
                min_veh = count
                best_lane = i
        
        # Check if there's space
        if min_veh > 0:
            furthest_x = 0
            for vid in first_link.lanes[best_lane].vehicles:
                v = self.vehicles.get(vid)
                if v and v.lane == best_lane and v.x > furthest_x:
                    furthest_x = v.x
            if furthest_x < VEHICLE_LENGTH + 5:
                return None
        
        vehicle = Vehicle(
            id=self.next_vehicle_id,
            link=first_link,
            lane=best_lane,
            x=0,
            v=first_link.free_flow_speed * 0.5,
            a=0,
            route=path,
            route_idx=0
        )
        
        self.vehicles[vehicle.id] = vehicle
        self.next_vehicle_id += 1
        
        first_link.lanes[best_lane].vehicles.append(vehicle.id)
        return vehicle
    
    def remove_vehicle(self, vid: int):
        v = self.vehicles.get(vid)
        if v:
            if 0 <= v.lane < v.link.num_lanes and vid in v.link.lanes[v.lane].vehicles:
                v.link.lanes[v.lane].vehicles.remove(vid)
            del self.vehicles[vid]
    
    def update_flows(self):
        for link in self.links.values():
            total = sum(len(l.vehicles) for l in link.lanes)
            link.density = total / max(link.length, 1)
            
            if total > 0:
                speeds = []
                for l in link.lanes:
                    for vid in l.vehicles:
                        v = self.vehicles.get(vid)
                        if v:
                            speeds.append(v.v)
                link.flow = (sum(speeds) / len(speeds)) * link.density if speeds else 0
            else:
                link.flow = 0
            
            link.is_congested = link.v_c_ratio > 0.5


def create_visual_network() -> VisualNetwork:
    """Create a small, dense visual network."""
    return VisualNetwork()
"""Full-featured traffic simulation graph with lane-based vehicle movement and real-world network support."""

import math
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from src.config import (
    LANE_WIDTH, VEHICLE_LENGTH, IDM_A_MAX, IDM_B, IDM_V0, IDM_S0, IDM_T,
    MOBIL_P, MOBIL_THRESHOLD, MOBIL_B_SAFE, SPEED_LIMITS, JAM_DENSITY, MAX_VEHICLES
)


@dataclass
class Node:
    """Intersection node with right-of-way logic."""
    id: int
    x: float
    y: float
    
    # Connections
    in_links: List['Link'] = field(default_factory=list)
    out_links: List['Link'] = field(default_factory=list)
    
    # Signal state
    is_signalized: bool = False
    signal_cycle: float = 60.0  # seconds
    green_duration: float = 30.0
    signal_timer: float = 0.0
    signal_phase: int = 0  # 0=green, 1=red
    
    # Priority (for unsignalized intersections)
    priority: int = 0  # 0=none, 1=minor, 2=major
    is_yield: bool = False  # Yield sign
    is_stop: bool = False  # Stop sign
    
    # Turn restrictions
    no_left_turn: bool = False
    no_right_turn: bool = False
    
    # Traffic demand
    demand: float = 0.0
    
    # Queue tracking
    approach_queues: Dict[int, List[int]] = field(default_factory=dict)  # link_id -> vehicle ids
    
    def __hash__(self):
        return hash(self.id)
    
    def get_approaching_vehicle_count(self, from_link_id: int) -> int:
        """Get number of vehicles waiting at this approach."""
        return len(self.approach_queues.get(from_link_id, []))
    
    def can_proceed(self, from_link: 'Link', to_link: 'Link', vehicles: Dict) -> Tuple[bool, str]:
        """Check if vehicle can proceed through this node."""
        # Signal check
        if self.is_signalized:
            if self.signal_phase != 0:  # Red
                return False, "red_signal"
        
        # Stop sign - full stop required
        if self.is_stop:
            return True, "stop"
        
        # Yield sign - yield to traffic
        if self.is_yield:
            # Check for conflicting traffic
            for in_link in self.in_links:
                if in_link.id == from_link.id:
                    continue
                if self.get_approaching_vehicle_count(in_link.id) > 0:
                    return False, "yield"
        
        # Right-of-way based on priority
        if self.priority >= 2:
            return True, "priority"
        
        # Minor road - yield to major
        if len(self.in_links) > 1:
            from_priority = sum(l.priority for l in self.in_links if l.road_type in ['motorway', 'trunk', 'primary'])
            if from_priority < 2:
                for in_link in self.in_links:
                    if in_link.road_type in ['motorway', 'trunk', 'primary']:
                        if self.get_approaching_vehicle_count(in_link.id) > 0:
                            return False, "minor_yield"
        
        return True, "clear"
    
    def update_queues(self):
        """Update vehicle queues at this intersection."""
        self.approach_queues.clear()
        for link in self.in_links:
            self.approach_queues[link.id] = []


@dataclass 
class Link:
    """Road segment with lanes and turn properties."""
    id: int
    from_node: 'Node'
    to_node: 'Node'
    length: float
    num_lanes: int
    free_flow_speed: float  # m/s
    
    # Road type
    road_type: str = "residential"
    
    # Lane structure
    lanes: List['Lane'] = field(default_factory=list)
    
    # Traffic state
    density: float = 0.0
    flow: float = 0.0
    is_congested: bool = False
    
    # Capacity
    jam_density: float = 0.15
    capacity: float = 0.0
    
    # Turn properties
    is_right_turn: bool = False
    is_left_turn: bool = False
    is_straight: bool = True
    
    # Priority for right-of-way
    priority: int = 1
    
    def __hash__(self):
        return hash(self.id)
    
    def __post_init__(self):
        self.lanes = [Lane(idx, self) for idx in range(self.num_lanes)]
        
        # Calculate capacity
        self.capacity = self.free_flow_speed * self.jam_density / (1 + self.free_flow_speed * self.jam_density * 0.1)
        
        # Priority based on road type
        if self.road_type in ['motorway', 'trunk']:
            self.priority = 3
        elif self.road_type == 'primary':
            self.priority = 2
        else:
            self.priority = 1
    
    @property
    def v_c_ratio(self) -> float:
        if self.capacity <= 0:
            return 0
        return self.flow / self.capacity
    
    def get_angle(self) -> float:
        dx = self.to_node.x - self.from_node.x
        dy = self.to_node.y - self.from_node.y
        return math.atan2(dy, dx)
    
    def get_turn_angle(self, next_link: 'Link') -> float:
        """Calculate turn angle in degrees (0=straight, positive=right, negative=left)."""
        angle1 = self.get_angle()
        angle2 = next_link.get_angle()
        
        diff = math.degrees(angle2 - angle1)
        # Normalize to -180 to 180
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        
        return diff
    
    def get_turn_speed_factor(self, next_link: 'Link') -> float:
        """Get speed reduction factor for turning."""
        turn_angle = abs(self.get_turn_angle(next_link))
        
        if turn_angle < 15:
            return 1.0  # Straight
        elif turn_angle < 45:
            return 0.8  # Slight turn
        elif turn_angle < 90:
            return 0.6  # Moderate turn
        elif turn_angle < 135:
            return 0.4  # Sharp turn
        else:
            return 0.3  # U-turn


@dataclass
class Lane:
    """Individual lane on a link."""
    index: int
    link: 'Link'
    vehicles: List[int] = field(default_factory=list)
    
    @property
    def is_full(self) -> bool:
        return len(self.vehicles) >= 25


@dataclass
class Vehicle:
    """Vehicle with lane-based movement and turn behavior."""
    id: int
    link: 'Link'
    lane: int
    x: float
    v: float
    a: float
    
    # Lane change
    changing_lane: bool = False
    target_lane: int = -1
    lane_change_timer: float = 0.0
    
    # State
    state: str = "moving"
    
    # Data offloading
    data_buffer: float = 0.0
    offload_rate: float = 0.0
    connected_hub: Optional['Hub'] = None
    
    # Route
    route: List[int] = field(default_factory=list)
    route_idx: int = 0
    
    # Turn behavior
    is_turning: bool = False
    turn_type: str = "straight"  # straight, left, right
    slowed_for_turn: bool = False
    
    # Stopping
    stopping_for_node: bool = False
    waiting_at_node: bool = False
    
    def __hash__(self):
        return hash(self.id)
    
    @property
    def x_end(self) -> float:
        return self.x + VEHICLE_LENGTH
    
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
    
    def get_turn_target_speed(self) -> float:
        """Get speed limit considering turn."""
        if not self.is_turning:
            return self.link.free_flow_speed
        
        # Look ahead for next link
        if self.route_idx < len(self.route) - 1:
            next_node_id = self.route[self.route_idx + 1]
            next_link = self._get_next_link(next_node_id)
            if next_link:
                factor = self.link.get_turn_speed_factor(next_link)
                return next_link.free_flow_speed * factor
        
        return self.link.free_flow_speed * 0.5
    
    def _get_next_link(self, next_node_id: int) -> Optional[Link]:
        """Get the link to the next node."""
        for link in self.link.to_node.out_links:
            if link.to_node.id == next_node_id:
                return link
        return None


@dataclass
class Hub:
    """Wi-Fi access point with animation."""
    x: float
    y: float
    radius: float = 150.0
    bandwidth: float = 100.0
    connected: List[Vehicle] = field(default_factory=list)
    
    pulse_phase: float = 0.0
    glow_intensity: float = 1.0
    
    def get_allocated_bw(self) -> float:
        n = len(self.connected)
        return self.bandwidth / n if n > 0 else 0.0
    
    def update_animation(self, dt: float = 1/60):
        self.pulse_phase += dt * 3.0
        if self.pulse_phase > 2 * math.pi:
            self.pulse_phase -= 2 * math.pi
        
        if self.connected:
            self.glow_intensity = 0.7 + 0.3 * abs(math.sin(self.pulse_phase))
        else:
            self.glow_intensity = 0.5 + 0.2 * abs(math.sin(self.pulse_phase * 0.5))
    
    def get_pulse_radius(self, base_radius: float) -> float:
        pulse = abs(math.sin(self.pulse_phase)) * 0.1
        return base_radius * (1.0 + pulse)
    
    def get_ring_alpha(self) -> int:
        return 100 + int(80 * abs(math.sin(self.pulse_phase))) if self.connected else 60


@dataclass
class Event:
    """Data collection event."""
    x: float
    y: float
    radius: float = 40.0
    data_amount: float = 5.0
    collected: int = 0


class Graph:
    """Road network graph."""
    
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.links: Dict[int, Link] = {}
        self.vehicles: Dict[int, Vehicle] = {}
        
        self.next_node_id = 0
        self.next_link_id = 0
        self.next_vehicle_id = 0
        
        self.node_grid: Dict[Tuple[int, int], List[Node]] = defaultdict(list)
        self.grid_size = 100
    
    def _grid_key(self, x: float, y: float) -> Tuple[int, int]:
        return (int(x / self.grid_size), int(y / self.grid_size))
    
    def add_node(self, x: float, y: float, priority: int = 0, 
                 is_signalized: bool = False, is_yield: bool = False,
                 is_stop: bool = False) -> Node:
        node = Node(
            id=self.next_node_id,
            x=x,
            y=y,
            priority=priority,
            is_signalized=is_signalized,
            is_yield=is_yield,
            is_stop=is_stop
        )
        self.nodes[node.id] = node
        self.next_node_id += 1
        
        key = self._grid_key(x, y)
        self.node_grid[key].append(node)
        
        return node
    
    def add_link(self, from_node: Node, to_node: Node, 
                 num_lanes: int = 1, road_type: str = "residential",
                 is_bidirectional: bool = True) -> Optional[Link]:
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        length = math.sqrt(dx * dx + dy * dy)
        
        if length < 5:
            return None
        
        speed = SPEED_LIMITS.get(road_type, 13.9)
        jam = JAM_DENSITY.get(road_type, 150) / 1000
        
        link = Link(
            id=self.next_link_id,
            from_node=from_node,
            to_node=to_node,
            length=length,
            num_lanes=num_lanes,
            free_flow_speed=speed,
            road_type=road_type,
            jam_density=jam
        )
        
        self.links[link.id] = link
        self.next_link_id += 1
        
        from_node.out_links.append(link)
        to_node.in_links.append(link)
        
        if is_bidirectional and road_type != "motorway":
            rev_link = Link(
                id=self.next_link_id,
                from_node=to_node,
                to_node=from_node,
                length=length,
                num_lanes=num_lanes,
                free_flow_speed=speed,
                road_type=road_type,
                jam_density=jam
            )
            self.links[rev_link.id] = rev_link
            self.next_link_id += 1
            
            to_node.out_links.append(rev_link)
            from_node.in_links.append(rev_link)
        
        return link
    
    def get_link(self, from_id: int, to_id: int) -> Optional[Link]:
        from_node = self.nodes.get(from_id)
        if not from_node:
            return None
        
        for link in from_node.out_links:
            if link.to_node.id == to_id:
                return link
        return None
    
    def get_nearest_node(self, x: float, y: float, max_dist: float = 200) -> Optional[Node]:
        key = self._grid_key(x, y)
        best = None
        best_dist = max_dist
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_key = (key[0] + dx, key[1] + dy)
                for node in self.node_grid.get(check_key, []):
                    dist = math.sqrt((node.x - x) ** 2 + (node.y - y) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best = node
        
        return best
    
    def create_vehicle(self, start_node: Node, end_node: Node) -> Optional[Vehicle]:
        path = self._dijkstra(start_node.id, end_node.id)
        if not path or len(path) < 2:
            return None
        
        first_link = self.get_link(path[0], path[1])
        if not first_link or first_link.num_lanes == 0:
            return None
        
        min_veh = float('inf')
        best_lane = 0
        for i, lane in enumerate(first_link.lanes):
            if len(lane.vehicles) < min_veh:
                min_veh = len(lane.vehicles)
                best_lane = i
        
        if min_veh > 0:
            furthest_x = 0
            for vid in first_link.lanes[best_lane].vehicles:
                v = self.vehicles.get(vid)
                if v and v.lane == best_lane and v.x > furthest_x:
                    furthest_x = v.x
            
            if furthest_x < VEHICLE_LENGTH + 3:
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
        first_link.lanes[best_lane].vehicles.sort(key=lambda vid: self.vehicles[vid].x)
        
        return vehicle
    
    def _dijkstra(self, start_id: int, end_id: int) -> Optional[List[int]]:
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
                
                # Weight based on congestion and length
                congestion = 1.0 + link.v_c_ratio * 3.0
                cost = (link.length / max(link.free_flow_speed, 1)) * congestion
                
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
    
    def remove_vehicle(self, vid: int):
        v = self.vehicles.get(vid)
        if v:
            if 0 <= v.lane < v.link.num_lanes:
                if vid in v.link.lanes[v.lane].vehicles:
                    v.link.lanes[v.lane].vehicles.remove(vid)
            del self.vehicles[vid]
    
    def update_link_flows(self):
        for link in self.links.values():
            total_vehicles = sum(len(l.vehicles) for l in link.lanes)
            link.density = total_vehicles / max(link.length, 1)
            
            if link.length > 0 and total_vehicles > 0:
                speeds = []
                for l in link.lanes:
                    for vid in l.vehicles:
                        v = self.vehicles.get(vid)
                        if v:
                            speeds.append(v.v)
                avg_speed = sum(speeds) / len(speeds) if speeds else 0
                link.flow = avg_speed * link.density
            else:
                link.flow = 0
            
            link.is_congested = link.v_c_ratio > 0.6


class SyntheticNetwork:
    """Generate realistic city network with varying road types."""
    
    def __init__(self, graph: Graph):
        self.graph = graph
    
    def create_realistic_city(self, center_x: float, center_y: float):
        """Create a dense, realistic city network."""
        
        # Major arterials (3 lanes each)
        major_spacing = 400
        major_nodes = {}
        
        # Create main grid
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = center_x + j * major_spacing
                y = center_y + i * major_spacing
                
                is_arterial = (i % 2 == 0) or (j % 2 == 0)
                priority = 3 if is_arterial else 2
                is_signal = random.random() < 0.25
                is_yield = not is_signal and random.random() < 0.15
                
                node = self.graph.add_node(x, y, priority, is_signal, is_yield)
                major_nodes[(i, j)] = node
        
        # Connect major grid
        for i in range(-3, 4):
            for j in range(-3, 3):
                n1 = major_nodes.get((i, j))
                n2 = major_nodes.get((i, j + 1))
                if n1 and n2:
                    lanes = 3 if i % 2 == 0 else 2
                    self.graph.add_link(n1, n2, lanes, "primary")
        
        for j in range(-3, 4):
            for i in range(-3, 3):
                n1 = major_nodes.get((i, j))
                n2 = major_nodes.get((i + 1, j))
                if n1 and n2:
                    lanes = 3 if j % 2 == 0 else 2
                    self.graph.add_link(n1, n2, lanes, "primary")
        
        # Secondary roads (2 lanes) - fill gaps
        for i in range(-3, 3):
            for j in range(-3, 3):
                # Add secondary connections
                for di in [0, 1]:
                    for dj in [0, 1]:
                        n1 = major_nodes.get((i, j))
                        n2 = major_nodes.get((i + di, j + dj))
                        
                        if n1 and n2 and random.random() < 0.4:
                            self.graph.add_link(n1, n2, 2, "secondary")
        
        # Local roads (1 lane) - residential
        local_nodes = {}
        for i in range(-2, 3):
            for j in range(-2, 3):
                for di in range(3):
                    for dj in range(3):
                        # Skip if too close to major nodes
                        x = center_x + j * major_spacing + di * 130 + random.uniform(-30, 30)
                        y = center_y + i * major_spacing + dj * 130 + random.uniform(-30, 30)
                        
                        # Check distance to existing nodes
                        too_close = False
                        for n in self.graph.nodes.values():
                            if math.sqrt((n.x - x) ** 2 + (n.y - y) ** 2) < 80:
                                too_close = True
                                break
                        
                        if not too_close and random.random() < 0.6:
                            node = self.graph.add_node(x, y, 1, False, False)
                            local_nodes[(i, j, di, dj)] = node
                            
                            # Connect to nearest major node
                            nearest = self.graph.get_nearest_node(x, y, 200)
                            if nearest:
                                self.graph.add_link(node, nearest, 1, "residential")
        
        # Add some diagonal shortcuts
        for i in range(-2, 3):
            for j in range(-2, 3):
                n1 = major_nodes.get((i, j))
                n2 = major_nodes.get((i + 1, j + 1))
                if n1 and n2 and random.random() < 0.3:
                    self.graph.add_link(n1, n2, 2, "secondary")
                
                n1 = major_nodes.get((i, j))
                n2 = major_nodes.get((i + 1, j - 1))
                if n1 and n2 and random.random() < 0.3:
                    self.graph.add_link(n1, n2, 2, "secondary")
        
        # Set signal timings
        for node in self.graph.nodes.values():
            if node.is_signalized:
                node.signal_cycle = 45 + random.uniform(-10, 15)
                node.green_duration = node.signal_cycle * 0.5
                node.signal_timer = random.uniform(0, node.signal_cycle)
        
        return self.graph


def create_real_network() -> Graph:
    """Create a realistic city network with proper right-of-way."""
    graph = Graph()
    generator = SyntheticNetwork(graph)
    
    generator.create_realistic_city(0, 0)
    
    print(f"Created network: {len(graph.nodes)} nodes, {len(graph.links)} links")
    
    return graph


def create_dense_network() -> Graph:
    """Legacy function - redirects to real network."""
    return create_real_network()
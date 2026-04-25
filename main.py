"""Islamabad F-6 Traffic Simulation - Full Version with RL."""

import pygame
import math
import random
import json
import heapq
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# ============================================================
# CONFIGURATION
# ============================================================

SCREEN_W, SCREEN_H = 1400, 900
FPS = 60
SIM_DT = 1.0 / FPS

# Traffic Physics
IDM_A_MAX = 2.5      # Max acceleration (m/s^2)
IDM_B = 3.0          # Comfortable deceleration
IDM_V0 = 33.3        # Desired velocity (m/s)
IDM_S0 = 2.0         # Minimum gap (m)
IDM_T = 1.5          # Time headway (s)

# MOBIL Lane Change
MOBIL_BSAFE = 3.0    # Safe deceleration threshold
MOBIL_BTHRESH = 0.1  # Advantage threshold
MOBIL_POLITE = 1.0   # Politeness factor

# Data Offloading
DATA_BASE_RATE = 10.0 / 60.0  # MB/min per vehicle
DATA_EVENT_MULT = 3.0         # Multiplier near events
HUB_RADIUS = 200.0           # meters (increased from 100)
HUB_BASE_BW = 100.0           # Mbps

# Simulation
MAX_VEHICLES = 2000  # increased from 1000
EVENT_SPAWN_INTERVAL = 2  # seconds (very frequent for RL)
VEHICLE_SPAWN_INTERVAL = 5  # frames between spawns (faster)

# Road Colors
ROAD_COLORS = {
    "motorway": (60, 80, 160),
    "primary": (180, 140, 100),
    "secondary": (160, 150, 120),
    "residential": (100, 100, 100),
}

# ============================================================
# PHYSICS FUNCTIONS
# ============================================================

def idm_accel(v: float, dv: float, s: float) -> float:
    """Intelligent Driver Model acceleration."""
    if s <= 0:
        return -IDM_B * 10
    s_star = IDM_S0 + max(0, v * IDM_T + (v * dv) / (2 * math.sqrt(IDM_A_MAX * IDM_B)))
    a = IDM_A_MAX * (1 - (v / IDM_V0) ** 4 - (s_star / max(s, 0.1)) ** 2)
    return max(-IDM_B * 5, min(a, IDM_A_MAX))

def mobil_eval(link, lane, x, v, new_lane, leader, follower):
    """Evaluate lane change utility with MOBIL."""
    if leader is None or follower is None:
        return float('-inf')
    
    gap = leader.x - x - 4.0
    dv = v - leader.v
    a_curr = idm_accel(v, dv, gap)
    
    new_leader = get_leader_in_lane(link, new_lane, x)
    if new_leader:
        new_gap = new_leader.x - x - 4.0
        new_dv = v - new_leader.v
        a_new = idm_accel(v, new_dv, new_gap)
    else:
        a_new = IDM_A_MAX
    
    new_follower_gap = x - follower.x - 4.0
    new_follower_dv = new_leader.v - follower.v if new_leader else 0
    a_follow_new = idm_accel(follower.v, new_follower_dv, new_follower_gap)
    
    gain = (a_new - a_curr) + MOBIL_POLITE * (a_follow_new - follower.a)
    return gain - MOBIL_BTHRESH

def get_leader_in_lane(link, lane, x):
    """Get vehicle ahead in lane."""
    candidates = []
    for vid in link.lanes[lane]:
        if vid in vehicles:
            v = vehicles[vid]
            if v.x > x:
                candidates.append(v)
    return min(candidates, key=lambda v: v.x) if candidates else None

# ============================================================
# DATA STRUCTURES
# ============================================================

vehicles = {}  # Global for MOBIL

@dataclass
class Node:
    id: int
    x: float
    y: float
    in_links: List = field(default_factory=list)
    out_links: List = field(default_factory=list)
    is_signalized: bool = False
    signal_cycle: float = 50.0
    signal_timer: float = 0.0
    signal_phase: int = 0

@dataclass
class Link:
    id: int
    from_node: Node
    to_node: Node
    length: float
    num_lanes: int
    free_flow_speed: float
    road_type: str
    lanes: List = field(default_factory=list)
    
    def __post_init__(self):
        self.lanes = [[] for _ in range(self.num_lanes)]

@dataclass
class Vehicle:
    id: int
    link: 'Link'
    lane: int
    x: float
    v: float
    a: float
    route: List[int]
    route_idx: int
    data_buffer: float = 0.0
    in_event_zone: bool = False
    connected_hub: Optional['Hub'] = None
    waiting: bool = False
    trail: List[Tuple[float, float]] = field(default_factory=list)

@dataclass
class Hub:
    id: int
    x: float
    y: float
    radius: float = HUB_RADIUS
    bandwidth: float = HUB_BASE_BW
    active: bool = True
    connected: List = field(default_factory=list)
    pulse: float = 0.0

@dataclass
class Event:
    id: int
    link: Link
    x_start: float
    x_end: float
    data_multiplier: float = DATA_EVENT_MULT
    active: bool = True
    duration: float = 120.0
    time_left: float = 120.0

# ============================================================
# NETWORK LOADING
# ============================================================

def load_network():
    """Load real Islamabad F-6 network."""
    print("Loading Islamabad F-6 network...")
    
    with open("islamabad_f6_cache.json", 'r') as f:
        data = json.load(f)
    
    nodes_data = data['nodes']
    links_data = data['links']
    
    nodes = {}
    for node_id_str, coords in nodes_data.items():
        node_id = int(node_id_str)
        x, y = coords[0], coords[1]
        is_signal = random.random() < 0.15
        nodes[node_id] = Node(node_id, x, y, is_signalized=is_signal)
    
    links = {}
    next_id = 0
    for link in links_data:
        from_id = int(link[0])
        to_id = int(link[1])
        length = link[2]
        speed = link[3] if len(link) > 3 else 13.9
        lanes = int(link[4]) if len(link) > 4 else 2
        
        if speed >= 25:
            road_type = "motorway"
        elif speed >= 18:
            road_type = "primary"
        elif speed >= 13:
            road_type = "secondary"
        else:
            road_type = "residential"
        
        fn = nodes.get(from_id)
        tn = nodes.get(to_id)
        
        if fn and tn:
            l = Link(next_id, fn, tn, length, lanes, speed, road_type)
            links[next_id] = l
            fn.out_links.append(l)
            tn.in_links.append(l)
            next_id += 1
            
            rl = Link(next_id, tn, fn, length, lanes, speed, road_type)
            links[next_id] = rl
            tn.out_links.append(rl)
            fn.in_links.append(rl)
            next_id += 1
    
    print(f"  {len(nodes)} nodes, {len(links)} links")
    return nodes, links

# ============================================================
# PATHFINDING (A*)
# ============================================================

def heuristic(node_id: int, end_id: int, nodes: Dict) -> float:
    """Euclidean distance heuristic."""
    n1 = nodes[node_id]
    n2 = nodes[end_id]
    return math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)

def astar(nodes, start_id, end_id):
    """A* pathfinding algorithm."""
    if start_id not in nodes or end_id not in nodes:
        return None
    if start_id == end_id:
        return [start_id]
    
    g_score = {start_id: 0.0}
    f_score = {start_id: heuristic(start_id, end_id, nodes)}
    prev = {start_id: None}
    open_set = [(f_score[start_id], start_id)]
    visited = set()
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current in visited:
            continue
        visited.add(current)
        
        if current == end_id:
            path = []
            while current is not None:
                path.append(current)
                current = prev.get(current)
            path.reverse()
            return path
        
        node = nodes[current]
        for link in node.out_links:
            neighbor = link.to_node.id
            cost = link.length / max(link.free_flow_speed, 1)
            new_g = g_score[current] + cost
            
            if neighbor not in g_score or new_g < g_score[neighbor]:
                g_score[neighbor] = new_g
                f_score[neighbor] = new_g + heuristic(neighbor, end_id, nodes)
                prev[neighbor] = current
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None

def get_link(nodes, from_id, to_id):
    """Get link between nodes."""
    node = nodes.get(from_id)
    if not node:
        return None
    for link in node.out_links:
        if link.to_node.id == to_id:
            return link
    return None

# ============================================================
# SIMULATION CLASS
# ============================================================

class TrafficSimulator:
    def __init__(self, nodes, links):
        self.nodes = nodes
        self.links = links
        self.vehicles: Dict[int, Vehicle] = {}
        self.hubs: List[Hub] = []
        self.events: List[Event] = []
        self.next_vehicle_id = 0
        self.next_hub_id = 0
        self.next_event_id = 0
        
        self.time = 0.0
        self.frame = 0
        self.show_heatmap = False
        
        self.total_data_generated = 0.0
        self.total_offloaded = 0.0
        
        global vehicles
        vehicles = self.vehicles
        
        self._create_initial_hubs(20)
        self._spawn_vehicles(2000)
    
    def _create_initial_hubs(self, count):
        """Place initial hubs at intersections."""
        node_list = [n for n in self.nodes.values() if len(n.out_links) >= 2]
        random.shuffle(node_list)
        for n in node_list[:count]:
            hub = Hub(self.next_hub_id, n.x, n.y)
            self.hubs.append(hub)
            self.next_hub_id += 1
    
    def _spawn_vehicles(self, count):
        """Spawn vehicles with routes."""
        node_list = list(self.nodes.values())
        spawned = 0
        attempts = 0
        
        while spawned < count and attempts < count * 50:
            start = random.choice(node_list)
            end = random.choice(node_list)
            if start.id == end.id:
                attempts += 1
                continue
            
            path = astar(self.nodes, start.id, end.id)
            if path and len(path) >= 2:
                link = get_link(self.nodes, path[0], path[1])
                if link:
                    lane = random.randint(0, max(0, link.num_lanes - 1))
                    v = Vehicle(
                        self.next_vehicle_id, link, lane, 0,
                        link.free_flow_speed * 0.5, 0, path, 0
                    )
                    self.vehicles[v.id] = v
                    link.lanes[lane].append(v.id)
                    self.next_vehicle_id += 1
                    spawned += 1
            
            attempts += 1
    
    def _spawn_vehicle(self):
        """Spawn single vehicle."""
        if len(self.vehicles) >= MAX_VEHICLES:
            return
        
        node_list = list(self.nodes.values())
        start = random.choice(node_list)
        end = random.choice(node_list)
        if start.id == end.id:
            return
        
        path = astar(self.nodes, start.id, end.id)
        if path and len(path) >= 2:
            link = get_link(self.nodes, path[0], path[1])
            if link:
                lane = random.randint(0, max(0, link.num_lanes - 1))
                v = Vehicle(
                    self.next_vehicle_id, link, lane, 0,
                    link.free_flow_speed * 0.5, 0, path, 0
                )
                self.vehicles[v.id] = v
                link.lanes[lane].append(v.id)
                self.next_vehicle_id += 1
    
    def _spawn_event(self):
        """Spawn traffic event."""
        links = list(self.links.values())
        link = random.choice(links)
        
        x_start = random.uniform(0, link.length * 0.3)
        x_end = x_start + random.uniform(link.length * 0.1, link.length * 0.2)
        
        event = Event(
            self.next_event_id, link, x_start, x_end,
            data_multiplier=random.uniform(2.0, 5.0)
        )
        self.events.append(event)
        self.next_event_id += 1
    
    def _update_signals(self):
        """Update traffic signals."""
        for node in self.nodes.values():
            if node.is_signalized:
                node.signal_timer += SIM_DT
                if node.signal_timer >= node.signal_cycle:
                    node.signal_timer = 0
                    node.signal_phase = (node.signal_phase + 1) % 2
    
    def _get_leader(self, v: Vehicle) -> Optional[Vehicle]:
        """Get vehicle ahead in same lane."""
        candidates = []
        for vid in v.link.lanes[v.lane]:
            if vid != v.id and vid in self.vehicles:
                other = self.vehicles[vid]
                if other.x > v.x:
                    candidates.append(other)
        return min(candidates, key=lambda x: x.x) if candidates else None
    
    def _get_follower(self, v: Vehicle) -> Optional[Vehicle]:
        """Get vehicle behind in same lane."""
        candidates = []
        for vid in v.link.lanes[v.lane]:
            if vid != v.id and vid in self.vehicles:
                other = self.vehicles[vid]
                if other.x < v.x:
                    candidates.append(other)
        return max(candidates, key=lambda x: x.x) if candidates else None
    
    def _check_lane_change(self, v: Vehicle):
        """MOBIL lane change decision."""
        if v.link.num_lanes <= 1:
            return
        
        leader = self._get_leader(v)
        follower = self._get_follower(v)
        
        if not leader or not follower:
            return
        
        gap = leader.x - v.x - 4.0
        dv = v.v - leader.v
        v.a = idm_accel(v.v, dv, gap)
        
        if v.lane < v.link.num_lanes - 1:
            new_lane = v.lane + 1
            new_leader = get_leader_in_lane(v.link, new_lane, v.x)
            candidates = [vid for vid in v.link.lanes[new_lane] if vid in self.vehicles]
            new_follower = None
            if candidates:
                new_follower = max([self.vehicles[vid] for vid in candidates], key=lambda x: x.x)
            
            if new_leader and new_follower:
                gain = mobil_eval(v.link, v.lane, v.x, v.v, new_lane, new_leader, new_follower)
                if gain > MOBIL_BSAFE:
                    v.link.lanes[v.lane].remove(v.id)
                    v.lane = new_lane
                    v.link.lanes[v.lane].append(v.id)
                    return
        
        if v.lane > 0:
            new_lane = v.lane - 1
            new_leader = get_leader_in_lane(v.link, new_lane, v.x)
            candidates = [vid for vid in v.link.lanes[new_lane] if vid in self.vehicles]
            new_follower = None
            if candidates:
                new_follower = max([self.vehicles[vid] for vid in candidates], key=lambda x: x.x)
            
            if new_leader and new_follower:
                gain = mobil_eval(v.link, v.lane, v.x, v.v, new_lane, new_leader, new_follower)
                if gain > MOBIL_BSAFE:
                    v.link.lanes[v.lane].remove(v.id)
                    v.lane = new_lane
                    v.link.lanes[v.lane].append(v.id)
    
    def _update_vehicle(self, v: Vehicle):
        """Update single vehicle."""
        global vehicles
        vehicles = self.vehicles
        
        # Add trail point every few frames
        if self.frame % 5 == 0:
            pos = self._get_vehicle_pos(v)
            v.trail.append(pos)
            if len(v.trail) > 10:
                v.trail.pop(0)
        
        v.in_event_zone = False
        for event in self.events:
            if event.active and event.link == v.link:
                if event.x_start <= v.x <= event.x_end:
                    v.in_event_zone = True
                    break
        
        # Only event-zone vehicles participate in offloading (generate data)
        if v.in_event_zone:
            data_rate = DATA_BASE_RATE * DATA_EVENT_MULT
            v.data_buffer += data_rate
            self.total_data_generated += data_rate
        # Non-event vehicles don't generate data
        
        self._check_lane_change(v)
        
        leader = self._get_leader(v)
        if leader:
            gap = leader.x - v.x - 4.0
            dv = v.v - leader.v
        else:
            gap = v.link.length - v.x
            dv = 0
        
        can_go = True
        if v.route_idx < len(v.route) - 1:
            next_id = v.route[v.route_idx + 1]
            if v.link.to_node.id == next_id:
                node = v.link.to_node
                if node.is_signalized and node.signal_phase != 0:
                    can_go = False
        
        if can_go and gap > 8:
            a = idm_accel(v.v, dv, gap)
        elif not can_go and v.link.length - v.x < 30:
            a = -IDM_B * 4
        else:
            a = -IDM_B * 2
        
        v.a = a
        v.v = max(0, min(v.v + a * SIM_DT, v.link.free_flow_speed))
        v.x += v.v * SIM_DT * 15
        v.waiting = (v.v < 0.5)
        
        if v.x >= v.link.length:
            if v.id in v.link.lanes[v.lane]:
                v.link.lanes[v.lane].remove(v.id)
            
            v.route_idx += 1
            
            if v.route_idx >= len(v.route):
                del self.vehicles[v.id]
                return
            
            next_link = None
            reverse_link = None
            for link in v.link.to_node.out_links:
                if link.to_node.id == v.link.from_node.id:
                    reverse_link = link
                    break
            
            available = [l for l in v.link.to_node.out_links if l != reverse_link]
            if available:
                next_link = random.choice(available)
            
            if next_link:
                v.link = next_link
                v.x = 0
                v.lane = random.randint(0, max(0, next_link.num_lanes - 1))
                v.trail = []  # Clear trail on link change
                next_link.lanes[v.lane].append(v.id)
        
        pos = self._get_vehicle_pos(v)
        connected = None
        for hub in self.hubs:
            if hub.active:
                dx = pos[0] - hub.x
                dy = pos[1] - hub.y
                if math.sqrt(dx*dx + dy*dy) <= hub.radius:
                    connected = hub
                    if v.id not in hub.connected:
                        hub.connected.append(v.id)
                    break
        
        v.connected_hub = connected
        if connected:
            bw = connected.bandwidth / max(len(connected.connected), 1)
            off = min(bw * SIM_DT * 0.1, v.data_buffer)
            v.data_buffer -= off
            self.total_offloaded += off
    
    def _get_vehicle_pos(self, v: Vehicle) -> Tuple[float, float]:
        """Get vehicle world position with lane offset."""
        t = v.x / max(v.link.length, 1)
        
        bx = v.link.from_node.x + t * (v.link.to_node.x - v.link.from_node.x)
        by = v.link.from_node.y + t * (v.link.to_node.y - v.link.from_node.y)
        
        dx = v.link.to_node.x - v.link.from_node.x
        dy = v.link.to_node.y - v.link.from_node.y
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            dx /= length
            dy /= length
        
        perp_x = -dy
        perp_y = dx
        
        lane_offset = (v.lane - (v.link.num_lanes - 1) / 2) * 3.5
        
        wx = bx + perp_x * lane_offset
        wy = by + perp_y * lane_offset
        return (wx, wy)
    
    def _update_hubs(self):
        """Update hub animations."""
        for hub in self.hubs:
            hub.pulse = (hub.pulse + SIM_DT * 2) % (2 * math.pi)
            hub.connected = [vid for vid in hub.connected if vid in self.vehicles]
    
    def _update_events(self):
        """Update event timers."""
        for event in self.events:
            event.time_left -= SIM_DT
            if event.time_left <= 0:
                event.active = False
        self.events = [e for e in self.events if e.active]
    
    def toggle_heatmap(self):
        """Toggle congestion heatmap."""
        self.show_heatmap = not self.show_heatmap
    
    def get_congestion(self) -> Dict[Tuple[float, float], float]:
        """Get congestion levels for heatmap."""
        congestion = {}
        for link in self.links.values():
            total_vehicles = sum(len(lane) for lane in link.lanes)
            density = total_vehicles / max(link.num_lanes, 1)
            cx = (link.from_node.x + link.to_node.x) / 2
            cy = (link.from_node.y + link.to_node.y) / 2
            congestion[(cx, cy)] = min(density / 5, 1.0)  # Normalize to 0-1
        return congestion
    
    def update(self):
        """Update simulation by one step."""
        self._update_signals()
        self._update_hubs()
        self._update_events()
        
        for v in list(self.vehicles.values()):
            self._update_vehicle(v)
        
        self.time += SIM_DT
        self.frame += 1
        
        if self.frame % VEHICLE_SPAWN_INTERVAL == 0 and len(self.vehicles) < MAX_VEHICLES:
            self._spawn_vehicle()
        
        if self.frame % (EVENT_SPAWN_INTERVAL * FPS) == 0:
            self._spawn_event()
    
    def set_hub_action(self, hub_id: int, active: bool):
        """RL action: set hub active/inactive."""
        for hub in self.hubs:
            if hub.id == hub_id:
                hub.active = active
                if not active:
                    hub.connected = []
                break
    
    def set_hub_bandwidth(self, hub_id: int, bandwidth: float):
        """RL action: set hub bandwidth (0-200 Mbps)."""
        for hub in self.hubs:
            if hub.id == hub_id:
                hub.bandwidth = max(0, min(200, bandwidth))
                break
    
    def get_state(self) -> dict:
        """Get RL state."""
        vs = list(self.vehicles.values())
        
        total_lanes = sum(l.num_lanes for l in self.links.values())
        avg_density = len(vs) / max(total_lanes, 1)
        
        speeds = [v.v for v in vs]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        
        connected = sum(1 for v in vs if v.connected_hub)
        in_event = sum(1 for v in vs if v.in_event_zone)
        
        active_hubs = sum(1 for h in self.hubs if h.active)
        
        total_buffer = sum(v.data_buffer for v in vs)
        offload_ratio = self.total_offloaded / max(self.total_data_generated, 0.1)
        
        return {
            "time": self.time,
            "vehicles": len(vs),
            "avg_speed": avg_speed,
            "connected": connected,
            "in_event": in_event,
            "active_hubs": active_hubs,
            "total_buffer": total_buffer,
            "offload_ratio": offload_ratio,
            "events_active": len(self.events),
        }
    
    def get_rl_state(self) -> List[float]:
        """Get RL state as vector."""
        state = self.get_state()
        
        return [
            state["vehicles"] / MAX_VEHICLES,
            state["avg_speed"] / 33.3,
            state["connected"] / max(state["vehicles"], 1),
            state["in_event"] / max(state["vehicles"], 1),
            state["active_hubs"] / 20,
            state["total_buffer"] / (MAX_VEHICLES * 100),
            state["offload_ratio"],
            len(self.events) / 5,
        ]

# ============================================================
# RENDERER
# ============================================================

class Renderer:
    def __init__(self, sim: TrafficSimulator):
        pygame.init()
        pygame.display.set_caption("Islamabad F-6 - Full Traffic Sim")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Courier New", 12)
        self.big_font = pygame.font.SysFont("Courier New", 14)
        
        self.sim = sim
        
        xs = [n.x for n in sim.nodes.values()]
        ys = [n.y for n in sim.nodes.values()]
        self.min_x, self.max_x = min(xs), max(xs)
        self.min_y, self.max_y = min(ys), max(ys)
        
        pad = 60
        scale_x = (SCREEN_W - 2*pad) / (self.max_x - self.min_x)
        scale_y = (SCREEN_H - 2*pad) / (self.max_y - self.min_y)
        self.scale = min(scale_x, scale_y)
        self.offset_x = (SCREEN_W - (self.max_x - self.min_x) * self.scale) / 2
        self.offset_y = (SCREEN_H - (self.max_y - self.min_y) * self.scale) / 2
        self.center_x = (self.min_x + self.max_x) / 2
        self.center_y = (self.min_y + self.max_y) / 2
    
    def to_screen(self, x, y):
        sx = self.offset_x + (x - self.min_x) * self.scale
        sy = self.offset_y + (y - self.min_y) * self.scale
        return int(sx), int(sy)
    
    def from_screen(self, sx, sy):
        """Convert screen coords to world coords."""
        x = (sx - self.offset_x) / self.scale + self.min_x
        y = (sy - self.offset_y) / self.scale + self.min_y
        return x, y
    
    def zoom_to(self, mouse_x, mouse_y, factor):
        """Zoom toward mouse cursor."""
        wx, wy = self.from_screen(mouse_x, mouse_y)
        
        self.scale *= factor
        self.scale = max(0.1, min(50, self.scale))
        
        self.offset_x = mouse_x - (wx - self.min_x) * self.scale
        self.offset_y = mouse_y - (wy - self.min_y) * self.scale
    
    def draw_lane_markings(self, link):
        """Draw dashed lane markings for a link (only at high zoom)."""
        if self.scale < 2:  # Only show at zoom > 2x
            return
        
        x1, y1 = self.to_screen(link.from_node.x, link.from_node.y)
        x2, y2 = self.to_screen(link.to_node.x, link.to_node.y)
        
        if link.num_lanes <= 1:
            return
        
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            return
        
        perp_x = -dy / length
        perp_y = dx / length
        
        num_dashes = int(length / 15)
        for i in range(1, link.num_lanes):
            offset = (i - (link.num_lanes - 1) / 2) * 3.5 * self.scale
            for j in range(num_dashes):
                t = (j + 0.3) / num_dashes
                t2 = (j + 0.7) / num_dashes
                px = x1 + t * dx + perp_x * offset
                py = y1 + t * dy + perp_y * offset
                px2 = x1 + t2 * dx + perp_x * offset
                py2 = y1 + t2 * dy + perp_y * offset
                pygame.draw.line(self.screen, (180, 180, 180), (int(px), int(py)), (int(px2), int(py2)), 1)
    
    def draw_direction_arrow(self, link):
        """Draw direction arrow on link."""
        if self.scale < 1.5:
            return
        
        x1, y1 = self.to_screen(link.from_node.x, link.from_node.y)
        x2, y2 = self.to_screen(link.to_node.x, link.to_node.y)
        
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            return
        
        dx /= length
        dy /= length
        
        arrow_size = 5
        perp_x = -dy
        perp_y = dx
        
        points = [
            (int(mx + dx * arrow_size), int(my + dy * arrow_size)),
            (int(mx - dx * arrow_size + perp_x * arrow_size * 0.5), int(my - dy * arrow_size + perp_y * arrow_size * 0.5)),
            (int(mx - dx * arrow_size - perp_x * arrow_size * 0.5), int(my - dy * arrow_size - perp_y * arrow_size * 0.5)),
        ]
        pygame.draw.polygon(self.screen, (200, 200, 200), points)
    
    def draw_heatmap(self):
        """Draw congestion heatmap overlay - show on roads."""
        for link in self.sim.links.values():
            total_vehicles = sum(len(lane) for lane in link.lanes)
            density = total_vehicles / max(link.num_lanes, 1)
            level = min(density / 3, 1.0)  # More sensitive threshold
            
            if level < 0.1:
                continue  # Skip low congestion
            
            x1, y1 = self.to_screen(link.from_node.x, link.from_node.y)
            x2, y2 = self.to_screen(link.to_node.x, link.to_node.y)
            
            if level < 0.3:
                color = (0, 100, 0)
            elif level < 0.5:
                color = (100, 100, 0)
            elif level < 0.7:
                color = (150, 80, 0)
            else:
                color = (150, 30, 0)
            
            s = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            width = 4 + int(level * 8)
            pygame.draw.line(s, (*color, int(180 * level)), (int(x1), int(y1)), (int(x2), int(y2)), width)
            self.screen.blit(s, (0, 0))
    
    def draw_minimap(self):
        """Draw minimap in bottom right."""
        minimap_w, minimap_h = 150, 100
        mx, my = SCREEN_W - minimap_w - 10, SCREEN_H - minimap_h - 10
        
        pygame.draw.rect(self.screen, (0, 0, 0), (mx, my, minimap_w, minimap_h))
        pygame.draw.rect(self.screen, (60, 60, 70), (mx, my, minimap_w, minimap_h), 1)
        
        mm_scale_x = minimap_w / (self.max_x - self.min_x)
        mm_scale_y = minimap_h / (self.max_y - self.min_y)
        mm_scale = min(mm_scale_x, mm_scale_y) * 0.9
        
        mm_offset_x = mx + minimap_w / 2
        mm_offset_y = my + minimap_h / 2
        
        for link in self.sim.links.values():
            x1 = mm_offset_x + (link.from_node.x - (self.min_x + self.max_x)/2) * mm_scale
            y1 = mm_offset_y + (link.from_node.y - (self.min_y + self.max_y)/2) * mm_scale
            x2 = mm_offset_x + (link.to_node.x - (self.min_x + self.max_x)/2) * mm_scale
            y2 = mm_offset_y + (link.to_node.y - (self.min_y + self.max_y)/2) * mm_scale
            pygame.draw.line(self.screen, (80, 80, 80), (int(x1), int(y1)), (int(x2), int(y2)), 1)
        
        view_x = mm_offset_x + (self.center_x - self.center_x) * mm_scale
        view_y = mm_offset_y + (self.center_y - self.center_y) * mm_scale
        pygame.draw.rect(self.screen, (255, 255, 0), (int(view_x - 10), int(view_y - 7), 20, 14), 1)
        
        for hub in self.sim.hubs:
            hx = mm_offset_x + (hub.x - (self.min_x + self.max_x)/2) * mm_scale
            hy = mm_offset_y + (hub.y - (self.min_y + self.max_y)/2) * mm_scale
            if 0 <= hx <= SCREEN_W and 0 <= hy <= SCREEN_H:
                pygame.draw.circle(self.screen, (80, 200, 220), (int(hx), int(hy)), 2)
    
    def render(self):
        BLACK = (20, 20, 30)
        GRAY = (60, 60, 70)
        WHITE = (220, 220, 230)
        GREEN = (50, 200, 100)
        CYAN = (80, 200, 220)
        BLUE = (80, 120, 255)
        RED = (255, 60, 60)
        ORANGE = (255, 150, 50)
        YELLOW = (255, 255, 80)
        PURPLE = (150, 80, 200)
        
        self.screen.fill(BLACK)
        
        # Draw heatmap if enabled
        if self.sim.show_heatmap:
            self.draw_heatmap()
        
        # Draw events
        for event in self.sim.events:
            if event.active:
                x1, y1 = self.to_screen(event.link.from_node.x, event.link.from_node.y)
                x2, y2 = self.to_screen(event.link.to_node.x, event.link.to_node.y)
                
                t1 = event.x_start / max(event.link.length, 1)
                t2 = event.x_end / max(event.link.length, 1)
                
                ex1 = x1 + t1 * (x2 - x1)
                ey1 = y1 + t1 * (y2 - y1)
                ex2 = x1 + t2 * (x2 - x1)
                ey2 = y1 + t2 * (y2 - y1)
                
                pygame.draw.line(self.screen, PURPLE, (int(ex1), int(ey1)), (int(ex2), int(ey2)), 8)
        
        # Draw links and lane markings
        for link in self.sim.links.values():
            x1, y1 = self.to_screen(link.from_node.x, link.from_node.y)
            x2, y2 = self.to_screen(link.to_node.x, link.to_node.y)
            
            color = ROAD_COLORS.get(link.road_type, GRAY)
            width = 2 + link.num_lanes
            pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), width)
            
            self.draw_lane_markings(link)
        
        # Draw hub radius
        for hub in self.sim.hubs:
            sx, sy = self.to_screen(hub.x, hub.y)
            sr = int(hub.radius * self.scale)
            
            if hub.active:
                if hub.connected:
                    load_ratio = len(hub.connected) / 10
                    if load_ratio < 0.5:
                        rc = (0, 100, 50, 50)
                    elif load_ratio < 0.8:
                        rc = (100, 100, 0, 50)
                    else:
                        rc = (100, 50, 0, 50)
                else:
                    rc = (0, 50, 80, 30)
                
                s = pygame.Surface((sr*2, sr*2), pygame.SRCALPHA)
                pygame.draw.circle(s, rc, (sr, sr), sr)
                self.screen.blit(s, (sx - sr, sy - sr))
                
                c = CYAN
                if hub.connected:
                    pygame.draw.circle(self.screen, (0, 50, 80), (sx, sy), 12, 1)
            else:
                c = (80, 80, 80)
            
            pulse_r = 6 + 2 * abs(math.sin(hub.pulse))
            pygame.draw.circle(self.screen, c, (sx, sy), int(pulse_r))
        
        # Draw vehicle trails
        for v in self.sim.vehicles.values():
            if len(v.trail) > 1:
                for i, pos in enumerate(v.trail):
                    sx, sy = self.to_screen(pos[0], pos[1])
                    alpha = int(50 * (i / len(v.trail)))
                    if v.connected_hub:
                        trail_color = (80, 200, 220, alpha)
                    else:
                        trail_color = (50, 200, 100, alpha)
                    s = pygame.Surface((4, 4), pygame.SRCALPHA)
                    pygame.draw.circle(s, trail_color, (2, 2), 2)
                    self.screen.blit(s, (sx - 2, sy - 2))
        
        # Draw vehicles
        for v in self.sim.vehicles.values():
            pos = self.sim._get_vehicle_pos(v)
            sx, sy = self.to_screen(pos[0], pos[1])
            
            if v.connected_hub:
                color = CYAN
            elif v.in_event_zone:
                color = YELLOW
            elif v.v < 0.5:
                color = ORANGE
            elif v.waiting:
                color = RED
            else:
                color = GREEN
            
            pygame.draw.circle(self.screen, color, (sx, sy), 3)
        
        # Draw stats panel
        state = self.sim.get_state()
        
        pygame.draw.rect(self.screen, (0, 0, 0), (10, 10, 250, 240), border_radius=8)
        pygame.draw.rect(self.screen, GRAY, (10, 10, 250, 240), 1, border_radius=8)
        
        y = 20
        lines = [
            "ISLAMABAD F-6 TRAFFIC",
            f"Time: {state['time']:.1f}s",
            f"Vehicles: {state['vehicles']}",
            f"Avg Speed: {state['avg_speed']:.1f} m/s",
            f"Connected to Hub: {state['connected']}",
            f"In Event Zone: {state['in_event']}",
            f"Active Hubs: {state['active_hubs']}",
            f"Events: {state['events_active']}",
            f"Heatmap: {'ON' if self.sim.show_heatmap else 'OFF'}",
            f"",
            f"Data Generated: {self.sim.total_data_generated:.0f} MB",
            f"Data Offloaded: {self.sim.total_offloaded:.0f} MB",
            f"Offload Ratio: {state['offload_ratio']*100:.1f}%",
        ]
        
        for i, line in enumerate(lines):
            c = CYAN if i == 0 else WHITE
            if i == 8:  # Heatmap line
                c = YELLOW if self.sim.show_heatmap else GRAY
            t = self.font.render(line, True, c)
            self.screen.blit(t, (20, y))
            y += 16
        
        # Controls hint
        hint = self.font.render("WASD: Pan | Scroll: Zoom | H: Heatmap | Esc: Quit", True, GRAY)
        self.screen.blit(hint, (20, SCREEN_H - 25))
        
        # Draw minimap
        self.draw_minimap()
        
        pygame.display.flip()
    
    def handle_events(self):
        running = True
        pan = 10 / max(self.scale, 0.1)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    self.offset_x = SCREEN_W / 2
                    self.offset_y = SCREEN_H / 2
                elif event.key == pygame.K_h:
                    self.sim.toggle_heatmap()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    self.zoom_to(event.pos[0], event.pos[1], 1.2)
                elif event.button == 5:
                    self.zoom_to(event.pos[0], event.pos[1], 1/1.2)
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.offset_y += pan
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.offset_y -= pan
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.offset_x += pan
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.offset_x -= pan
        
        return running


def main():
    print("=" * 60)
    print("ISLAMABAD F-6 - FULL TRAFFIC SIMULATION")
    print("=" * 60)
    
    nodes, links = load_network()
    sim = TrafficSimulator(nodes, links)
    
    print(f"Vehicles: {len(sim.vehicles)}, Hubs: {len(sim.hubs)}")
    print(f"Events will spawn every {EVENT_SPAWN_INTERVAL}s")
    print()
    print("Features:")
    print("  - A* pathfinding")
    print("  - MOBIL lane changing")
    print("  - Traffic events (purple zones)")
    print("  - Vehicle trails")
    print("  - Lane markings (visible when zoomed in)")
    print("  - Congestion heatmap (H to toggle)")
    print("  - WiFi hub offloading (cyan = connected)")
    print("  - Hub coverage radius")
    print("  - Minimap")
    print()
    print("Controls: WASD pan | Scroll zoom | H: Heatmap | Esc quit")
    
    renderer = Renderer(sim)
    
    running = True
    while running:
        running = renderer.handle_events()
        sim.update()
        renderer.render()
        renderer.clock.tick(FPS)
    
    pygame.quit()


if __name__ == "__main__":
    main()

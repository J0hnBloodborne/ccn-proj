"""Generate network from actual OpenStreetMap data."""

import json
import math
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from src.config import SCALE, OSM_BOUNDS, LANE_WIDTH, VEHICLE_LENGTH, IDM_A_MAX, IDM_B, IDM_V0, IDM_S0, IDM_T
from src.config import SPEED_LIMITS as CONFIG_SPEED_LIMITS, JAM_DENSITY as CONFIG_JAM_DENSITY, HUB_RADIUS


# Speed limits from config
SPEED_LIMITS = {
    'motorway': 27.8,
    'trunk': 22.2,
    'primary': 16.7,
    'secondary': 13.9,
    'tertiary': 11.1,
    'residential': 8.3,
    'service': 5.6,
}

JAM_DENSITY = {
    'motorway': 120,
    'trunk': 130,
    'primary': 140,
    'secondary': 150,
    'tertiary': 160,
    'residential': 170,
}

LANE_COUNTS = {
    "motorway": 3,
    "trunk": 2,
    "primary": 2,
    "secondary": 1,
    "tertiary": 1,
    "residential": 1,
}


@dataclass
class Node:
    """Intersection node."""
    id: int
    x: float
    y: float
    lat: float
    lon: float
    
    in_links: List['Link'] = field(default_factory=list)
    out_links: List['Link'] = field(default_factory=list)
    
    is_signalized: bool = False
    signal_cycle: float = 60.0
    signal_timer: float = 0.0
    signal_phase: int = 0
    
    priority: int = 1
    is_yield: bool = False
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class Link:
    """Road segment."""
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
        if self.capacity <= 0:
            return 0
        return self.flow / self.capacity
    
    def get_angle(self) -> float:
        dx = self.to_node.x - self.from_node.x
        dy = self.to_node.y - self.from_node.y
        return math.atan2(dy, dx)


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


class RealNetworkGraph:
    """Graph built from real OSM data."""
    
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.links: Dict[int, Link] = {}
        self.vehicles: Dict[int, Vehicle] = {}
        
        self.next_node_id = 0
        self.next_link_id = 0
        self.next_vehicle_id = 0
    
    def lat_lon_to_xy(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert lat/lon to x/y coordinates."""
        x = (lon - OSM_BOUNDS['min_lon']) * SCALE * 10  # Scale up for visibility
        y = (OSM_BOUNDS['max_lat'] - lat) * SCALE * 10
        return x, y
    
    def add_node(self, osm_id: int, lat: float, lon: float, 
                 priority: int = 1, is_signalized: bool = False) -> Node:
        x, y = self.lat_lon_to_xy(lat, lon)
        
        node = Node(
            id=self.next_node_id,
            x=x, y=y, lat=lat, lon=lon,
            priority=priority,
            is_signalized=is_signalized
        )
        
        self.nodes[node.id] = node
        self.osm_to_local[osm_id] = node.id
        self.next_node_id += 1
        return node
    
    def add_link(self, from_node: Node, to_node: Node,
                 num_lanes: int = 1, road_type: str = "residential",
                 is_bidirectional: bool = True) -> Optional[Link]:
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        length = math.sqrt(dx * dx + dy * dy)
        
        if length < 10:
            return None
        
        speed = SPEED_LIMITS.get(road_type, 13.9)
        
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
        
        if is_bidirectional and road_type != "motorway":
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
        
        return link
    
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
                
                cost = link.length / max(link.free_flow_speed, 1) * (1 + link.v_c_ratio * 2)
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
            if furthest_x < VEHICLE_LENGTH + 5:
                return None
        
        vehicle = Vehicle(
            id=self.next_vehicle_id,
            link=first_link,
            lane=best_lane,
            x=0,
            v=first_link.free_flow_speed * 0.4,
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
            
            link.is_congested = link.v_c_ratio > 0.6


def load_osm_real(force_refresh: bool = False) -> RealNetworkGraph:
    """Load real OSM data or create fallback."""
    import urllib.request
    import ssl
    
    cache_file = "osm_cache.json"
    
    # Try to load from cache
    if not force_refresh:
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            print("Loaded OSM data from cache")
            return parse_osm_to_graph(data)
        except Exception as e:
            print(f"Cache not found or invalid: {e}")
    
    # Fetch from OSM
    print("Fetching real OSM data from OpenStreetMap...")
    
    query = f"""
[out:json][timeout:90][bbox:{OSM_BOUNDS['min_lat']},{OSM_BOUNDS['min_lon']},{OSM_BOUNDS['max_lat']},{OSM_BOUNDS['max_lon']}];
(
  way["highway"="motorway"](if: length() > 100);
  way["highway"="trunk"](if: length() > 80);
  way["highway"="primary"](if: length() > 60);
  way["highway"="secondary"](if: length() > 50);
  way["highway"="tertiary"](if: length() > 40);
  way["highway"="residential"](if: length() > 30);
);
out body;
>;
out skel qt;
"""
    
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request(
            "https://overpass-api.de/api/interpreter",
            data=query.encode('utf-8'),
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                'User-Agent': 'TrafficSim/1.0 (Research Project)'
            }
        )
        
        print("Requesting data from Overpass API...")
        with urllib.request.urlopen(req, timeout=180, context=ctx) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        elements = data.get('elements', [])
        print(f"Received {len(elements)} OSM elements")
        
        if len(elements) > 50:
            try:
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
                print("Cached OSM data")
            except:
                pass
            return parse_osm_to_graph(data)
        
    except Exception as e:
        print(f"OSM fetch error: {e}")
    
    print("Using fallback: Creating Faizabad Interchange Islamabad network")
    return create_islamabad_network()


def parse_osm_to_graph(data: Dict) -> RealNetworkGraph:
    """Parse OSM JSON data into RealNetworkGraph."""
    graph = RealNetworkGraph()
    graph.osm_to_local = {}
    
    # Extract nodes
    nodes_data = {}
    for element in data.get('elements', []):
        if element.get('type') == 'node':
            nid = element['id']
            lat = element['lat']
            lon = element['lon']
            nodes_data[nid] = (lat, lon)
            graph.add_node(nid, lat, lon)
    
    # Extract ways and create links
    for element in data.get('elements', []):
        if element.get('type') == 'way':
            tags = element.get('tags', {})
            highway = tags.get('highway', 'residential')
            
            if 'lanes' in tags:
                try:
                    lanes = int(tags['lanes'])
                except:
                    lanes = LANE_COUNTS.get(highway, 1)
            else:
                lanes = LANE_COUNTS.get(highway, 1)
            
            node_ids = element.get('nodes', [])
            if len(node_ids) < 2:
                continue
            
            # Create links between consecutive nodes
            for i in range(len(node_ids) - 1):
                n1_id = node_ids[i]
                n2_id = node_ids[i + 1]
                
                if n1_id not in graph.osm_to_local or n2_id not in graph.osm_to_local:
                    continue
                
                n1 = graph.nodes[graph.osm_to_local[n1_id]]
                n2 = graph.nodes[graph.osm_to_local[n2_id]]
                
                is_bidir = tags.get('oneway', 'no') != 'yes' and highway not in ['motorway', 'trunk']
                
                graph.add_link(n1, n2, lanes, highway, is_bidir)
    
    # Add signals to major intersections
    major_intersections = [n for n in graph.nodes.values() if len(n.in_links) >= 2 and len(n.out_links) >= 2]
    for node in random.sample(major_intersections, min(len(major_intersections) // 3, 20)):
        node.is_signalized = True
        node.signal_cycle = 60 + random.uniform(-15, 15)
    
    print(f"Parsed real OSM data: {len(graph.nodes)} nodes, {len(graph.links)} links")
    return graph


def create_islamabad_network() -> RealNetworkGraph:
    """Create a network that mimics Faizabad Interchange, Islamabad."""
    graph = RealNetworkGraph()
    graph.osm_to_local = {}
    
    # Faizabad Interchange area - coordinates centered around 33.66, 73.05
    # This is a major interchange connecting GT Road, Murree Road, and Islamabad
    
    # Create a dense network mimicking the interchange
    node_positions = [
        # Main GT Road (east-west major road)
        (33.655, 73.025, "gt_road"),
        (33.655, 73.035, "gt_road"),
        (33.655, 73.045, "gt_road"),  # Main interchange area
        (33.655, 73.055, "gt_road"),
        (33.655, 73.065, "gt_road"),
        
        # Murree Road (north-south, merging with GT)
        (33.665, 73.045, "murree"),
        (33.660, 73.045, "murree"),
        (33.650, 73.045, "murree"),
        (33.645, 73.045, "murree"),
        
        # Islamabad Link Road
        (33.658, 73.038, "islamabad_link"),
        (33.662, 73.038, "islamabad_link"),
        (33.668, 73.038, "islamabad_link"),
        
        # Service roads
        (33.652, 73.050, "service"),
        (33.658, 73.050, "service"),
        (33.662, 73.050, "service"),
        
        # Roundabout approaches
        (33.653, 73.040, "approach"),
        (33.663, 73.040, "approach"),
        (33.653, 73.052, "approach"),
        (33.663, 73.052, "approach"),
        
        # Additional connections
        (33.648, 73.035, "local"),
        (33.668, 73.035, "local"),
        (33.648, 73.060, "local"),
        (33.668, 73.060, "local"),
        
        # More density for congestion
        (33.657, 73.042, "inner"),
        (33.657, 73.048, "inner"),
        (33.661, 73.042, "inner"),
        (33.661, 73.048, "inner"),
        (33.656, 73.038, "inner"),
        (33.662, 73.055, "inner"),
        
        # Extended network
        (33.640, 73.045, "south"),
        (33.670, 73.045, "north"),
        (33.655, 73.020, "west"),
        (33.655, 73.070, "east"),
    ]
    
    # Create nodes
    for lat, lon, road_type in node_positions:
        priority = 3 if road_type in ["gt_road", "murree"] else 2 if road_type in ["islamabad_link"] else 1
        is_signal = random.random() < 0.3 if priority >= 2 else False
        
        node_id = hash(f"{lat:.4f}{lon:.4f}")
        graph.add_node(node_id, lat, lon, priority=priority, is_signalized=is_signal)
    
    # Create connections
    for i, (lat1, lon1, r1) in enumerate(node_positions):
        x1, y1 = graph.lat_lon_to_xy(lat1, lon1)
        n1 = None
        for n in graph.nodes.values():
            if abs(n.x - x1) < 1 and abs(n.y - y1) < 1:
                n1 = n
                break
        
        if not n1:
            continue
        
        for j, (lat2, lon2, r2) in enumerate(node_positions[i+1:], i+1):
            x2, y2 = graph.lat_lon_to_xy(lat2, lon2)
            n2 = None
            for n in graph.nodes.values():
                if abs(n.x - x2) < 1 and abs(n.y - y2) < 1:
                    n2 = n
                    break
            
            if not n2:
                continue
            
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            # Connect nearby nodes
            same_road = r1 == r2
            nearby = dist < 120
            
            if nearby and (same_road or dist < 80):
                if r1 in ["gt_road", "murree"]:
                    road = "primary"
                    lanes = 3 if r1 == "gt_road" else 2
                elif r1 in ["islamabad_link"]:
                    road = "primary"
                    lanes = 2
                elif r1 == "inner" or r1 == "approach":
                    road = "secondary"
                    lanes = 2
                else:
                    road = "secondary"
                    lanes = 1
                
                if not graph.get_link(n1.id, n2.id):
                    graph.add_link(n1, n2, lanes, road, True)
    
    # Add signal timings
    for node in graph.nodes.values():
        if node.is_signalized:
            node.signal_cycle = 50 + random.uniform(-10, 15)
    
    print(f"Created Islamabad/Faizabad network: {len(graph.nodes)} nodes, {len(graph.links)} links")
    return graph

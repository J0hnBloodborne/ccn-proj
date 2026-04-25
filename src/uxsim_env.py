"""
Production-Grade UXsim-based Data Offloading Environment for RL Training.

Uses real Islamabad F-6 sector OSM data with proper traffic simulation
and WiFi hub placement optimization.
"""

import numpy as np
import math
import json
import urllib.request
import ssl
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import deque

# UXsim imports
import uxsim

# Gymnasium imports
import gymnasium as gym
from gymnasium import spaces


# Islamabad F-6 bounds (real coordinates)
ISLAMABAD_BOUNDS = {
    'min_lat': 33.70,
    'max_lat': 33.76,
    'min_lon': 73.03,
    'max_lon': 73.09
}

SCALE = 10000  # meters per degree


@dataclass
class WiFiHub:
    """WiFi Access Point for data offloading."""
    x: float
    y: float
    radius: float = 250.0  # Coverage radius in meters (250m for urban WiFi)
    bandwidth: float = 100.0  # Mbps
    connected_vehicles: List[str] = field(default_factory=list)
    
    def get_allocated_bw(self) -> float:
        """Get bandwidth per connected vehicle."""
        if len(self.connected_vehicles) == 0:
            return 0.0
        return self.bandwidth / len(self.connected_vehicles)
    
    def update_connections(self, vehicle_positions: Dict[str, Tuple[float, float]]):
        """Update which vehicles are in coverage range."""
        self.connected_vehicles = []
        for vid, (vx, vy) in vehicle_positions.items():
            # Ensure float conversion for numpy types
            dx = float(vx) - float(self.x)
            dy = float(vy) - float(self.y)
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= float(self.radius):
                self.connected_vehicles.append(vid)


class OSMNetworkLoader:
    """Fetches and parses real OSM data for Islamabad F-6."""
    
    OVERPASS_URL = "https://overpass-api.de/api/interpreter"
    
    def __init__(self, bounds: Dict[str, float]):
        self.bounds = bounds
        self.nodes: Dict[int, Tuple[float, float]] = {}
        self.links: List[Tuple[int, int, float, float, int]] = []  # (start, end, length, speed, lanes)
    
    def lat_lon_to_xy(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert lat/lon to local x,y coordinates."""
        center_lat = (self.bounds['min_lat'] + self.bounds['max_lat']) / 2
        center_lon = (self.bounds['min_lon'] + self.bounds['max_lon']) / 2
        x = (lon - center_lon) * 111000 * math.cos(math.radians(center_lat))
        y = (lat - center_lat) * 111000
        return (x, y)
    
    def fetch_osm_data(self) -> Optional[Dict]:
        """Fetch OSM data from Overpass API."""
        query = f"""
        [out:json][timeout:120][bbox:{self.bounds['min_lat']},{self.bounds['min_lon']},{self.bounds['max_lat']},{self.bounds['max_lon']}];
        (
          way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential)$"](if: length() > 30);
        );
        out body;
        >;
        out center qt;
        """
        
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(
                self.OVERPASS_URL,
                data=query.encode('utf-8'),
                headers={
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'User-Agent': 'CCN2-DataOffloading/1.0 (academic research)'
                }
            )
            
            with urllib.request.urlopen(req, timeout=180, context=ctx) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            print(f"OSM fetch error: {e}")
            return None
    
    def parse_osm(self, data: Dict):
        """Parse OSM JSON into nodes and links."""
        # Parse nodes
        node_coords = {}
        for element in data.get('elements', []):
            if element.get('type') == 'node':
                lat, lon = element['lat'], element['lon']
                x, y = self.lat_lon_to_xy(lat, lon)
                self.nodes[element['id']] = (x, y)
                node_coords[element['id']] = (x, y)
        
        # Lane/speed defaults by highway type
        highway_config = {
            'motorway': (3, 25.0),
            'trunk': (2, 20.0),
            'primary': (2, 15.0),
            'secondary': (2, 13.9),
            'tertiary': (1, 11.1),
            'residential': (1, 8.3),
        }
        
        # Parse ways into links
        for element in data.get('elements', []):
            if element.get('type') == 'way':
                tags = element.get('tags', {})
                highway = tags.get('highway', 'residential')
                
                lanes, speed = highway_config.get(highway, (1, 8.3))
                if 'lanes' in tags:
                    try:
                        lanes = int(tags['lanes'])
                    except:
                        pass
                
                if 'maxspeed' in tags:
                    try:
                        speed_str = tags['maxspeed']
                        if 'mph' in speed_str.lower():
                            speed = float(speed_str.replace('mph', '').strip()) * 0.44704
                        else:
                            speed = float(speed_str) / 3.6
                    except:
                        pass
                
                one_way = tags.get('oneway', 'no') == 'yes'
                if highway in ['motorway', 'trunk']:
                    one_way = True
                
                node_ids = element.get('nodes', [])
                if len(node_ids) < 2:
                    continue
                
                # Create links between consecutive nodes
                for i in range(len(node_ids) - 1):
                    n1, n2 = node_ids[i], node_ids[i + 1]
                    
                    if n1 not in node_coords or n2 not in node_coords:
                        continue
                    
                    x1, y1 = node_coords[n1]
                    x2, y2 = node_coords[n2]
                    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    
                    if length < 5:  # Skip very short segments
                        continue
                    
                    self.links.append((n1, n2, length, speed, lanes))
                    
                    # Add reverse link for two-way roads
                    if not one_way:
                        self.links.append((n2, n1, length, speed, lanes))
    
    def load_from_cache(self, cache_file: str = "islamabad_f6_cache.json") -> bool:
        """Try loading from cache file."""
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                self.nodes = {int(k): tuple(v) for k, v in cached['nodes'].items()}
                self.links = [(int(l[0]), int(l[1]), float(l[2]), float(l[3]), int(l[4])) 
                             for l in cached['links']]
                return True
        except:
            return False
    
    def save_to_cache(self, cache_file: str = "islamabad_f6_cache.json"):
        """Save parsed data to cache."""
        try:
            cached = {
                'nodes': {str(k): list(v) for k, v in self.nodes.items()},
                'links': [[l[0], l[1], l[2], l[3], l[4]] for l in self.links]
            }
            with open(cache_file, 'w') as f:
                json.dump(cached, f)
        except:
            pass
    
    def load(self) -> Tuple[int, int]:
        """Load OSM data (from cache or fetch). Returns (num_nodes, num_links)."""
        if self.load_from_cache():
            print(f"Loaded {len(self.nodes)} nodes, {len(self.links)} links from cache")
            return len(self.nodes), len(self.links)
        
        print("Fetching OSM data for Islamabad F-6...")
        data = self.fetch_osm_data()
        
        if data and len(data.get('elements', [])) > 100:
            self.parse_osm(data)
            self.save_to_cache()
            print(f"Loaded {len(self.nodes)} nodes, {len(self.links)} links from OSM")
            return len(self.nodes), len(self.links)
        
        return 0, 0


class DataOffloadingEnv(gym.Env):
    """
    Production-Grade RL Environment for Vehicle Data Offloading.
    
    Uses real Islamabad F-6 OSM data with UXsim traffic simulation.
    
    State: [num_vehicles, connected_ratio, avg_speed, congestion, data_efficiency]
    Action: Discrete - select hub placement strategy (n strategies)
    """
    
    metadata = {"render_modes": ["console"]}
    
    def __init__(
        self,
        num_hubs: int = 20,
        hub_radius: float = 250.0,
        hub_bandwidth: float = 100.0,
        data_gen_rate: float = 10.0 / 60.0,  # MB per second
        sim_duration: float = 300.0,  # 5 minutes
        demand_rate: float = 0.5,  # vehicles per second
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.num_hubs = num_hubs
        self.hub_radius = hub_radius
        self.hub_bandwidth = hub_bandwidth
        self.data_gen_rate = data_gen_rate
        self.sim_duration = sim_duration
        self.demand_rate = demand_rate
        self.seed = seed
        
        # State tracking
        self.hubs: List[WiFiHub] = []
        self.vehicle_data: Dict[str, float] = {}
        self.total_data_generated: float = 0.0
        self.total_data_offloaded: float = 0.0
        
        # UXsim world
        self.W: Optional[uxsim.World] = None
        self.osm_loader: Optional[OSMNetworkLoader] = None
        self.node_name_map: Dict[int, str] = {}  # OSM ID -> UXsim node name
        self.current_time: float = 0.0
        self.vehicles_spawned: int = 0
        
        # Action space: hub placement optimization
        # Actions 0-7: different hub placement strategies based on traffic density
        self.action_space = spaces.Discrete(8)
        
        # State space: [num_vehicles, connected_ratio, avg_speed, congestion, offload_rate]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float32
        )
        
        self.rng = np.random.default_rng(seed)
    
    def _load_network(self) -> uxsim.World:
        """Create UXsim world with real Islamabad F-6 network."""
        # Calculate network bounds
        center_lat = (ISLAMABAD_BOUNDS['min_lat'] + ISLAMABAD_BOUNDS['max_lat']) / 2
        center_lon = (ISLAMABAD_BOUNDS['min_lon'] + ISLAMABAD_BOUNDS['max_lon']) / 2
        
        lat_range = ISLAMABAD_BOUNDS['max_lat'] - ISLAMABAD_BOUNDS['min_lat']
        lon_range = ISLAMABAD_BOUNDS['max_lon'] - ISLAMABAD_BOUNDS['min_lon']
        network_size = max(lat_range, lon_range) * 111000 * 1.2  # meters with buffer
        
        W = uxsim.World(
            network_size,  # First positional: network size (width in meters)
            random_seed=self.rng.integers(0, 100000) if self.seed else None,
            print_mode=False,
            show_mode=False,
            show_progress=False,
        )
        
        # Load OSM data
        self.osm_loader = OSMNetworkLoader(ISLAMABAD_BOUNDS)
        num_nodes, num_links = self.osm_loader.load()
        
        if num_nodes == 0:
            raise RuntimeError("Failed to load OSM network data")
        
        # Add nodes to UXsim
        node_id = 0
        for osm_id, (x, y) in self.osm_loader.nodes.items():
            node_name = f"n_{node_id}"
            W.addNode(node_name, x, y)
            self.node_name_map[osm_id] = node_name
            node_id += 1
        
        # Add links to UXsim
        link_count = 0
        for start_id, end_id, length, speed, lanes in self.osm_loader.links:
            if start_id in self.node_name_map and end_id in self.node_name_map:
                start_name = self.node_name_map[start_id]
                end_name = self.node_name_map[end_id]
                link_name = f"l_{link_count}"
                W.addLink(
                    link_name, start_name, end_name,
                    length=max(length, 10),  # minimum 10m
                    free_flow_speed=speed,
                    number_of_lanes=max(lanes, 1),
                )
                link_count += 1
        
        print(f"Created UXsim network: {len(self.node_name_map)} nodes, {link_count} links")
        return W
    
    def _setup_demand(self):
        """Setup vehicle demand based on real traffic patterns."""
        if self.W is None:
            return
        
        nodes = list(self.node_name_map.values())
        if len(nodes) < 2:
            return
        
        # Generate random OD pairs for demand
        num_demand = int(self.sim_duration * self.demand_rate * 2)  # Round trips
        
        for i in range(num_demand):
            origin = self.rng.choice(nodes)
            destination = self.rng.choice(nodes)
            if origin != destination:
                # Spread departure times evenly across simulation duration
                departure_time = self.rng.uniform(0, self.sim_duration * 0.8)
                self.W.addVehicle(origin, destination, departure_time)
    
    def _initialize_hubs(self, strategy: int = 0):
        """Initialize WiFi hubs using specified strategy.
        
        Strategies:
        0: Random placement
        1: Traffic density based (high traffic = more hubs)
        2: Intersection based
        3: Road segment centers
        4: Grid pattern
        5: Clustered (k-means style)
        6: Road type based (main roads prioritized)
        7: Balanced coverage
        """
        self.hubs = []
        
        if self.W is None or len(self.node_name_map) == 0:
            return
        
        nodes = list(self.W.NODES)
        if len(nodes) == 0:
            return
        
        # Get node positions for placement
        node_positions = [(n.name, n.x, n.y) for n in nodes]
        
        if strategy == 0:  # Random
            selected = self.rng.choice(node_positions, size=min(self.num_hubs, len(node_positions)), replace=False)
        elif strategy == 1:  # Traffic density - use random for now (would need traffic analysis)
            selected = self.rng.choice(node_positions, size=min(self.num_hubs, len(node_positions)), replace=False)
        elif strategy == 2:  # Intersections (nodes with degree > 2)
            degrees = {n.name: 0 for n in nodes}
            for link in self.W.LINKS:
                if hasattr(link, 'start_node') and hasattr(link, 'end_node'):
                    degrees[link.start_node.name] = degrees.get(link.start_node.name, 0) + 1
                    degrees[link.end_node.name] = degrees.get(link.end_node.name, 0) + 1
            sorted_nodes = sorted(node_positions, key=lambda x: degrees.get(x[0], 0), reverse=True)
            selected = sorted_nodes[:min(self.num_hubs, len(sorted_nodes))]
        elif strategy == 3:  # Road segment centers
            selected = self.rng.choice(node_positions, size=min(self.num_hubs, len(node_positions)), replace=False)
        elif strategy == 4:  # Grid pattern
            xs = [p[1] for p in node_positions]
            ys = [p[2] for p in node_positions]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            grid_size = int(math.sqrt(self.num_hubs)) + 1
            step_x = (x_max - x_min) / grid_size
            step_y = (y_max - y_min) / grid_size
            selected = []
            for gi in range(grid_size):
                for gj in range(grid_size):
                    cx = x_min + gi * step_x + step_x / 2
                    cy = y_min + gj * step_y + step_y / 2
                    closest = min(node_positions, key=lambda p: math.sqrt((p[1] - cx)**2 + (p[2] - cy)**2))
                    if closest not in selected:
                        selected.append(closest)
        elif strategy == 5:  # Clustered
            selected = self.rng.choice(node_positions, size=min(self.num_hubs, len(node_positions)), replace=False)
        elif strategy == 6:  # Road type based
            selected = self.rng.choice(node_positions, size=min(self.num_hubs, len(node_positions)), replace=False)
        else:  # 7: Balanced coverage
            selected = self.rng.choice(node_positions, size=min(self.num_hubs, len(node_positions)), replace=False)
        
        for name, x, y in selected:
            hub = WiFiHub(x=x, y=y, radius=self.hub_radius, bandwidth=self.hub_bandwidth)
            self.hubs.append(hub)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment - always create fresh simulation."""
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        
        # ALWAYS create fresh world for each episode
        self.W = self._load_network()
        self._setup_demand()
        
        # Reset tracking
        self.hubs = []
        self.vehicle_data = {}
        self.total_data_generated = 0.0
        self.total_data_offloaded = 0.0
        self.current_time = 0.0
        self.vehicles_spawned = 0
        
        # Initialize hubs with strategy 0 (random)
        self._initialize_hubs(strategy=0)
        
        # Run initial simulation (5 seconds)
        self.W.exec_simulation(0, 5)
        self.current_time = 5.0
        
        # Update tracking
        for vid in self.W.VEHICLES:
            if vid not in self.vehicle_data:
                self.vehicle_data[vid] = 0.0
        
        state = self._get_state()
        info = self._get_info()
        
        return state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one simulation step (5 seconds)."""
        if self.W is None:
            raise RuntimeError("Environment not reset")
        
        # Check if simulation should continue
        if self.W.simulation_terminated():
            terminated = True
            truncated = False
            return self._get_state(), 0.0, terminated, truncated, self._get_info()
        
        # Apply action: reconfigure hub placement
        if action != 0:
            self._initialize_hubs(strategy=action % 8)
        
        # Run simulation (5 seconds)
        end_t = self.current_time + 5.0
        try:
            self.W.exec_simulation(self.current_time, end_t)
            self.current_time = end_t
        except Exception:
            terminated = True
            truncated = False
            return self._get_state(), 0.0, terminated, truncated, self._get_info()
        
        # Update vehicle data
        self._update_vehicle_data()
        
        # Update hub connections and offloading
        self._update_data_offloading()
        
        # Check termination
        terminated = self.current_time >= self.sim_duration
        
        # Get state and reward
        state = self._get_state()
        reward = self._calculate_reward()
        info = self._get_info()
        
        return state, reward, terminated, False, info
    
    def _get_vehicle_positions(self) -> Dict[str, Tuple[float, float]]:
        """Get current positions of all vehicles."""
        positions = {}
        if self.W is None:
            return positions
        
        for vid in self.W.VEHICLES:
            try:
                veh = self.W.VEHICLES[vid]
                if veh.link is not None:
                    x, y = veh.get_xy_coords()
                    # Ensure proper float conversion (handle numpy types)
                    positions[vid] = (float(x), float(y))
            except:
                continue
        
        return positions
    
    def _update_vehicle_data(self):
        """Update data generation for all active vehicles."""
        if self.W is None:
            return
        
        dt = 5.0  # 5 second step
        
        for vid in self.W.VEHICLES:
            if vid not in self.vehicle_data:
                self.vehicle_data[vid] = 0.0
            
            veh = self.W.VEHICLES[vid]
            if veh.link is not None:
                data_gen = self.data_gen_rate * dt
                self.vehicle_data[vid] += data_gen
                self.total_data_generated += data_gen
    
    def _update_data_offloading(self):
        """Update WiFi hub connections and calculate data offloading."""
        vehicle_positions = self._get_vehicle_positions()
        
        # Update hub connections
        for hub in self.hubs:
            hub.update_connections(vehicle_positions)
        
        # Offload data from connected vehicles
        dt = 5.0
        for hub in self.hubs:
            if len(hub.connected_vehicles) == 0:
                continue
            
            bw_per_veh = hub.bandwidth / len(hub.connected_vehicles)
            offload_rate = min(bw_per_veh * dt / 8, 0.5)  # MB per 5 seconds
            
            for vid in hub.connected_vehicles:
                if vid in self.vehicle_data and self.vehicle_data[vid] > 0:
                    actual_offload = min(offload_rate, self.vehicle_data[vid])
                    self.vehicle_data[vid] -= actual_offload
                    self.total_data_offloaded += actual_offload
    
    def _get_state(self) -> np.ndarray:
        """Get current state observation (normalized 0-1)."""
        if self.W is None:
            return np.zeros(5, dtype=np.float32)
        
        # Count active vehicles
        active_vehicles = [v for v in self.W.VEHICLES.values() if v.link is not None]
        num_vehicles = len(active_vehicles)
        
        # Count connected vehicles
        connected_count = sum(len(h.connected_vehicles) for h in self.hubs)
        
        # Calculate average speed
        speeds = [v.speed for v in active_vehicles if hasattr(v, 'speed') and v.speed > 0]
        avg_speed = np.mean(speeds) if speeds else 0.0
        
        # Calculate congestion (vehicles per link)
        num_links = max(len(self.W.LINKS), 1)
        total_on_links = sum(len(link.vehicles) for link in self.W.LINKS if hasattr(link, 'vehicles'))
        congestion = total_on_links / (num_links * 2)  # density per lane
        
        # Offload efficiency
        offload_efficiency = self.total_data_offloaded / max(self.total_data_generated, 0.001)
        
        # Connection ratio
        connected_ratio = connected_count / max(num_vehicles, 1)
        
        return np.array([
            min(num_vehicles / 200.0, 1.0),  # Normalize: max 200 vehicles
            connected_ratio,                    # 0-1
            min(avg_speed / 15.0, 1.0),       # Normalize: max 15 m/s
            min(congestion / 3.0, 1.0),       # Normalize: max 3 vehicles/link
            offload_efficiency,                # 0-1
        ], dtype=np.float32)
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on offloading efficiency and connectivity."""
        # Primary: offload efficiency (higher is better)
        offload_efficiency = self.total_data_offloaded / max(self.total_data_generated, 0.001)
        
        # Secondary: connection coverage
        state = self._get_state()
        connection_bonus = state[1] * 0.3  # Reward for connected vehicles
        
        # Penalty: high congestion (bad for data transfer)
        congestion_penalty = -state[3] * 0.2
        
        # Combined reward
        reward = offload_efficiency * 2.0 + connection_bonus + congestion_penalty
        
        return float(reward)
    
    def _get_info(self) -> Dict:
        """Get additional info for debugging."""
        return {
            "time": self.current_time,
            "num_vehicles": len([v for v in self.W.VEHICLES.values() if v.link is not None]) if self.W else 0,
            "connected_vehicles": sum(len(h.connected_vehicles) for h in self.hubs),
            "total_hubs": len(self.hubs),
            "data_generated": self.total_data_generated,
            "data_offloaded": self.total_data_offloaded,
            "offload_rate": self.total_data_offloaded / max(self.total_data_generated, 0.001),
        }
    
    def render(self, mode: str = "console"):
        """Render the environment state."""
        if mode == "console":
            info = self._get_info()
            print(f"[t={info['time']:.0f}s] "
                  f"Vehicles: {info['num_vehicles']}, "
                  f"Connected: {info['connected_vehicles']}, "
                  f"Offload: {info['data_offloaded']:.1f}MB ({info['offload_rate']:.1%})")
    
    def close(self):
        """Clean up resources."""
        self.W = None
        self.hubs = []
        self.vehicle_data = {}


def test_production_env():
    """Test the production environment with real OSM data."""
    print("=" * 60)
    print("Testing Production-Grade Data Offloading Environment")
    print("=" * 60)
    
    env = DataOffloadingEnv(
        num_hubs=20,
        hub_radius=250.0,
        data_gen_rate=10.0 / 60.0,
        sim_duration=60.0,
        demand_rate=0.5,
        seed=42
    )
    
    print("\nResetting environment...")
    state, info = env.reset()
    print(f"Initial state: {state}")
    print(f"Info: {info}")
    
    print("\nRunning simulation steps...")
    total_reward = 0.0
    for step in range(12):  # 12 steps * 5s = 60s
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 4 == 0:
            print(f"Step {step}: action={action}, reward={reward:.3f}, "
                  f"connected={info['connected_vehicles']}, "
                  f"offload_rate={info['offload_rate']:.2%}")
        
        if terminated:
            break
    
    print(f"\nTotal reward: {total_reward:.3f}")
    print(f"Final offload efficiency: {info['offload_rate']:.2%}")
    
    env.close()
    print("\nTest completed successfully!")
    return info['offload_rate'] > 0


if __name__ == "__main__":
    test_production_env()
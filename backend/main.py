import math
import random
import re
import networkx as nx
import asyncio
import osmnx as ox
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOCATION = "Faizabad Interchange, Islamabad, Pakistan"
WIFI_RANGE_METERS = 100
WIFI_RATE_MB_S = 1.5
DATA_GEN_RATE_MB_STEP = 0.1
NUM_VEHICLES = 50
NUM_HUBS = 15
MAX_BUFFER_MB = 50.0

class ConfigState(BaseModel):
    algorithm: str

class ResetConfig(BaseModel):
    num_hubs: int
    num_vehicles: int
    event_rate: float

def euclidean_dist(lat1, lon1, lat2, lon2):
    lat_mid = (lat1 + lat2) / 2.0
    dx = (lon2 - lon1) * 111320.0 * math.cos(math.radians(lat_mid))
    dy = (lat2 - lat1) * 111320.0
    return math.sqrt(dx*dx + dy*dy)

class Hub:
    def __init__(self, hid, lat, lon):
        self.id = hid
        self.lat = lat
        self.lon = lon
        self.range = WIFI_RANGE_METERS
        self.base_rate = random.choice([1.0, 2.5, 5.0])
        self.active = False
        self.online = True
        self.total_offloaded_mb = 0.0

class Vehicle:
    def __init__(self, vid, start_node, graph):
        self.id = vid
        self.start_node = start_node
        node_data = graph.nodes[start_node]
        self.lat = node_data["y"]
        self.lon = node_data["x"]
        self.target_lat = self.lat
        self.target_lon = self.lon
        self.path = []
        self.path_idx = 0
        
        # IDM Physics
        self.base_speed = random.uniform(0.00004, 0.00008) # degrees max speed
        self.speed = 0.0
        self.a_max = 0.000005 # max acceleration
        self.b_comf = 0.000010 # comfortable braking
        self.s0 = 0.00003 # minimum gap (approx 3m)
        self.T = 1.5 # safe time headway
        
        self.lane = random.choice([0, 1])
        self.display_lat = self.lat
        self.display_lon = self.lon

        # Buffer logic
        self.buffer_mb = 0.0
        self.routine_mb = 0.0
        self.priority_mb = 0.0
        self.events = []
        self.panic = False
        
        # Network log & AIMD TCP State
        self.current_bw_alloc = 0.0
        self.total_packet_drops = 0
        self.cwnd = 1.0          # AIMD Congestion Window (starts at 1MB/step)
        self.ssthresh = 100.0    # Slow-start threshold
        self.connected_hub_id = None
        self.handover_timer = 0  # Steps remaining in delayed handover

    def pick_path(self, graph):
        nodes = list(graph.nodes())
        try:
            self.path = nx.shortest_path(graph, self.path[-1] if self.path else self.start_node, random.choice(nodes), weight='length')
        except:
            self.path = [self.path[-1] if self.path else self.start_node]
        self.path_idx = 1
        if len(self.path) > 1:
            next_node = self.path[1]
            self.target_lat = graph.nodes[next_node]["y"]
            self.target_lon = graph.nodes[next_node]["x"]

class SimulationState:
    def __init__(self):
        self.running = False
        self.step = 0
        self.algorithm = "greedy"
        self.total_generated_mb = 0.0
        self.total_offloaded_mb = 0.0
        self.total_buffer_overflows = 0
        self.priority_latency_sum = 0
        self.priority_events_completed = 0
        self.num_hubs = NUM_HUBS
        self.num_vehicles = NUM_VEHICLES
        self.event_rate = 0.0005
        self.graph = None
        self.hubs = []
        self.vehicles = []
        self.signals = {} # Traffic signals at junctions: {node: {'state': 'GREEN'/'RED', 'timer': 0, 'cycle': int}}
        self.roads = []
        self.recent_events = []
        self.total_network_drops = 0

sim_state = SimulationState()

@app.on_event("startup")
async def startup_event():
    print(f"Loading graph for {LOCATION}...")
    try:
        sim_state.graph = ox.graph_from_place(LOCATION, network_type="drive")
    except Exception as e:
        sim_state.graph = ox.graph_from_point((33.6515, 73.0801), dist=2500, network_type="drive")
    
    roads = []
    for u, v, data in sim_state.graph.edges(data=True):
        highway = data.get('highway', 'unclassified')
        if isinstance(highway, list): 
            highway = highway[0]
            
        if highway in ['motorway', 'trunk', 'motorway_link', 'trunk_link']:
            lanes = 4
        elif highway in ['primary', 'secondary', 'primary_link', 'secondary_link']:
            lanes = 3
        else:
            lanes = 2
            
        if 'geometry' in data:
            coords = list(data['geometry'].coords)
            path_coords = [[pt[1], pt[0]] for pt in coords]
        else:
            u_node = sim_state.graph.nodes[u]
            v_node = sim_state.graph.nodes[v]
            path_coords = [[u_node['y'], u_node['x']], [v_node['y'], v_node['x']]]
            
        roads.append({'path': path_coords, 'lanes': lanes})
    sim_state.roads = roads
    init_sim_entities()
    print("Simulation initialized.")
    asyncio.create_task(simulation_loop())

def init_sim_entities():
    nodes = list(sim_state.graph.nodes(data=True))
    sim_state.hubs = []
    sim_state.vehicles = []
    sim_state.step = 0
    sim_state.total_generated_mb = 0.0
    sim_state.total_offloaded_mb = 0.0
    sim_state.total_buffer_overflows = 0
    sim_state.total_network_drops = 0
    sim_state.priority_latency_sum = 0
    sim_state.priority_events_completed = 0
    sim_state.recent_events = []
    
    # Initialize Traffic Signals at intersections
    sim_state.signals = {}
    for node_id, data in sim_state.graph.nodes(data=True):
        # Intersections can be roughly identified as nodes with more than 2 edges
        edges = list(sim_state.graph.edges(node_id))
        if len(edges) > 2:
            sim_state.signals[node_id] = {
                'state': random.choice(['GREEN', 'RED']),
                'timer': random.randint(0, 100),
                'cycle': random.randint(80, 150) # Ticks to toggle state
            }

    node_ids = list(sim_state.graph.nodes())
    top_right_nodes = [n for n in nodes if n[1]["y"] > 33.6515 + 0.005 and n[1]["x"] > 73.0801 + 0.005]
    if len(top_right_nodes) >= 2 and sim_state.num_hubs >= 2:
        tr_hubs = random.sample(top_right_nodes, 2)
        other_nodes = [n for n in nodes if n not in tr_hubs]
        hub_nodes = tr_hubs + random.sample(other_nodes, sim_state.num_hubs - 2)
    else:
        hub_nodes = random.sample(nodes, sim_state.num_hubs)

    for hid, node in enumerate(hub_nodes):
        sim_state.hubs.append(Hub(hid, node[1]["y"], node[1]["x"]))
    
    for vid in range(sim_state.num_vehicles):
        v = Vehicle(vid, random.choice(node_ids), sim_state.graph)
        v.pick_path(sim_state.graph)
        sim_state.vehicles.append(v)

async def simulation_loop():
    while True:
        if sim_state.running:
            run_step()
        await asyncio.sleep(0.15) # Changed from 0.05 to 0.15 to improve performance

def run_step():
    sim_state.step += 1
    
    # Toggle Traffic Lights
    for sig_node, sig_data in sim_state.signals.items():
        sig_data['timer'] += 1
        if sig_data['timer'] >= sig_data['cycle']:
            sig_data['state'] = 'GREEN' if sig_data['state'] == 'RED' else 'RED'
            sig_data['timer'] = 0

    for hub in sim_state.hubs:
        hub.active = False
        if random.random() < 0.001:
            hub.online = not hub.online

    # Build spatial map for IDM per edge and lane
    edge_vehicles = {}
    for v in sim_state.vehicles:
        if v.path and v.path_idx < len(v.path):
            edge = (v.path[v.path_idx-1] if v.path_idx > 0 else v.path[0], v.path[v.path_idx])
            if edge not in edge_vehicles: edge_vehicles[edge] = []
            edge_vehicles[edge].append(v)
            
    # Sort vehicles on edge by distance to target (closer to target = further ahead)
    for edge, vehs in edge_vehicles.items():
        vehs.sort(key=lambda v: math.sqrt((v.target_lon - v.lon)**2 + (v.target_lat - v.lat)**2))

    EVENT_TYPES = ["Pothole", "Minor Crash", "Debris", "Speed Trap", "Traffic Jam"]

    # --- PHYSICS & IDM ---
    for v in sim_state.vehicles:
        v.current_bw_alloc = 0.0
        
        v.routine_mb += DATA_GEN_RATE_MB_STEP
        sim_state.total_generated_mb += DATA_GEN_RATE_MB_STEP
        if random.random() < sim_state.event_rate:
            evt_size = random.uniform(5.0, 10.0)
            v.events.append({"type": random.choice(EVENT_TYPES), "size": evt_size, "timestamp": sim_state.step})
            v.priority_mb += evt_size
            sim_state.total_generated_mb += evt_size

        total_buf = v.routine_mb + v.priority_mb
        if total_buf > MAX_BUFFER_MB:
            dropped = min(total_buf - MAX_BUFFER_MB, v.routine_mb)
            if dropped > 0:
                v.routine_mb -= dropped
                sim_state.total_buffer_overflows += 1
        
        v.buffer_mb = v.routine_mb + v.priority_mb

        # Default movement (IDM)
        dx = v.target_lon - v.lon
        dy = v.target_lat - v.lat
        dist = math.sqrt(dx*dx + dy*dy)
        
        # IDM leader calculation
        a_idm = v.a_max * (1 - (v.speed / v.base_speed)**4) if v.base_speed > 0 else 0

        # Traffic Signal Logic
        if v.path and v.path_idx < len(v.path):
            next_node = v.path[v.path_idx]
            if next_node in sim_state.signals:
                signal = sim_state.signals[next_node]
                # If RED, treat the junction as a stopped leader at `dist`
                if signal['state'] == 'RED' and dist < v.s0 * 5:
                    s_star = v.s0 + max(0, v.speed * v.T + (v.speed * v.speed) / (2 * math.sqrt(v.a_max * v.b_comf)))
                    a_idm -= v.a_max * (s_star / max(dist, 0.000001))**2

        if v.path and v.path_idx < len(v.path):
            edge = (v.path[v.path_idx-1] if v.path_idx > 0 else v.path[0], v.path[v.path_idx])
            vehs_on_edge = edge_vehicles.get(edge, [])
            idx_in_lane = [veh for veh in vehs_on_edge if veh.lane == v.lane].index(v) if v in vehs_on_edge else -1
            if idx_in_lane > 0:
                leader = [veh for veh in vehs_on_edge if veh.lane == v.lane][idx_in_lane - 1]
                s_dist = math.sqrt((leader.lon - v.lon)**2 + (leader.lat - v.lat)**2)
                delta_v = v.speed - leader.speed
                s_star = v.s0 + max(0, v.speed * v.T + (v.speed * delta_v) / (2 * math.sqrt(v.a_max * v.b_comf)))
                a_idm -= v.a_max * (s_star / max(s_dist, 0.000001))**2
                
                # Multi-lane Overtaking -> If IDM severely brakes us and another lane is freer
                if a_idm < 0 and v.speed < v.base_speed * 0.5:
                    other_lane_vehs = [veh for veh in vehs_on_edge if veh.lane == (1 - v.lane)]
                    safe_to_switch = True
                    for olv in other_lane_vehs:
                        if math.sqrt((olv.lon - v.lon)**2 + (olv.lat - v.lat)**2) < v.s0 * 2:
                            safe_to_switch = False
                            break
                    if safe_to_switch:
                        v.lane = 1 - v.lane

        v.speed = max(0.0, min(v.base_speed, v.speed + a_idm))
        
        if dist < max(v.speed, 0.00001):
            v.lon = v.target_lon
            v.lat = v.target_lat
            v.path_idx += 1
            if v.path_idx < len(v.path):
                next_node = v.path[v.path_idx]
                v.target_lat = sim_state.graph.nodes[next_node]["y"]
                v.target_lon = sim_state.graph.nodes[next_node]["x"]
            else:
                v.pick_path(sim_state.graph)
        else:
            v.lon += (dx/dist) * v.speed
            v.lat += (dy/dist) * v.speed
            
        # Display coordinate with lateral offsets for visual 'lanes'
        # 1 deg ~ 111km. 5m ~ 0.000045 deg
        L_offset = 0.000045
        norm_lon = -dy/max(dist, 1e-6)
        norm_lat = dx/max(dist, 1e-6)
        
        offset = (v.lane - 0.5) * L_offset
        v.display_lon = v.lon + norm_lon * offset
        v.display_lat = v.lat + norm_lat * offset

    # --- NETWORK LAYER (Bandwidth Allocation / Attenuation) ---
    hub_connections = {h.id: [] for h in sim_state.hubs if h.online}
    for v in sim_state.vehicles:
        closest_h = None
        min_d = float('inf')
        for h in sim_state.hubs:
            if h.online:
                d = euclidean_dist(v.lat, v.lon, h.lat, h.lon)
                if d < min_d and d <= h.range:
                    min_d = d
                    closest_h = h
        if closest_h:
            hub_connections[closest_h.id].append({'veh': v, 'dist': min_d})
        else:
            # Dropout or disconnected
            v.connected_hub_id = None
            v.handover_timer = 0
            v.cwnd = 1.0
            v.ssthresh = 100.0
            
    for h in sim_state.hubs:
        if not h.online: continue
        conns = hub_connections[h.id]
        if not conns: continue
        
        # Base Allocation: divide equal share (dynamic allocation)
        share = 1.0 / len(conns)
        for c in conns:
            v = c['veh']
            dist = c['dist']
            
            # --- TCP Handovers Phase ---
            if v.connected_hub_id != h.id:
                if v.connected_hub_id is not None:
                    v.handover_timer = 3  # Ping-pong delay of 3 simulation steps (~150ms drop)
                v.connected_hub_id = h.id
                v.cwnd = 1.0
                v.ssthresh = 100.0
                
            if v.handover_timer > 0:
                v.handover_timer -= 1
                v.current_bw_alloc = 0.0
                continue
            
            # --- TCP AIMD Phase ---
            # Path Loss: linear degradation 100% to 10%
            eff_factor = max(0.1, 1 - (dist / h.range))
            alloc_mb = (h.base_rate * share) * eff_factor
            
            # Random uniform packet drop probability based on physical channel contention and SNR
            congestion_prob = (1.0 - eff_factor) * 0.1
            if len(conns) > 1: congestion_prob += min(0.1, len(conns) * 0.01)
            
            if random.random() < congestion_prob:
                # Multiplicative decrease
                v.ssthresh = max(1.0, v.cwnd / 2.0)
                v.cwnd = max(1.0, v.cwnd * 0.5)
                v.total_packet_drops += 1
                sim_state.total_network_drops += 1
            else:
                # Additive increase -> TCP Slow Start / AIMD Congestion Avoidance
                if v.cwnd < v.ssthresh:
                    v.cwnd += 2.0  # Slow Start
                else:
                    v.cwnd += 1.0 / max(1.0, v.cwnd) # Congestion Avoidance
            
            # Cap TCP transfer limit to physical boundary `alloc_mb`
            tcp_alloc_mb = min(alloc_mb, v.cwnd)
            v.current_bw_alloc = tcp_alloc_mb
            h.active = True
            
            # Transfer
            demand = v.routine_mb + sum(e['size'] for e in v.events)
            amt_to_offload = min(tcp_alloc_mb, demand)
            
            # Predict/Panic
            if sim_state.algorithm == "predictive" and not v.panic and demand < MAX_BUFFER_MB * 0.8:
                skip_offload = False
                future_nodes = v.path[v.path_idx:v.path_idx+10]
                for f_node in future_nodes:
                    ny = sim_state.graph.nodes[f_node]["y"]
                    nx_coord = sim_state.graph.nodes[f_node]["x"]
                    for other_h in sim_state.hubs:
                        if other_h.online and other_h.base_rate > h.base_rate and euclidean_dist(ny, nx_coord, other_h.lat, other_h.lon) <= other_h.range:
                            skip_offload = True
                            break
                    if skip_offload: break
                if skip_offload: continue
                
            v.panic = False
            
            # Execute transfer
            while v.events and amt_to_offload > 0:
                evt = v.events[0]
                if evt["size"] <= amt_to_offload:
                    amt_to_offload -= evt["size"]
                    v.priority_mb -= evt["size"]
                    sim_state.total_offloaded_mb += evt["size"]
                    h.total_offloaded_mb += evt["size"]
                    latency = sim_state.step - evt["timestamp"]
                    sim_state.priority_latency_sum += latency
                    sim_state.priority_events_completed += 1
                    v.events.pop(0)
                    sim_state.recent_events.insert(0, f"[{sim_state.step}] Veh {v.id} offloaded to Hub {h.id} (Lat: {latency}s)")
                else:
                    evt["size"] -= amt_to_offload
                    v.priority_mb -= amt_to_offload
                    sim_state.total_offloaded_mb += amt_to_offload
                    h.total_offloaded_mb += amt_to_offload
                    amt_to_offload = 0
                    
            if amt_to_offload > 0 and v.routine_mb > 0:
                off_routine = min(v.routine_mb, amt_to_offload)
                v.routine_mb -= off_routine
                sim_state.total_offloaded_mb += off_routine
                h.total_offloaded_mb += off_routine
                
            v.buffer_mb = v.routine_mb + v.priority_mb
            sim_state.recent_events = sim_state.recent_events[:30]

            # Simulate Packet dropping (Network Congestion feature)
            if demand > alloc_mb:
                # 1% chance for packets entirely lost in air if congested
                if random.random() < 0.05:
                    dropped_pkts_mb = demand * 0.01
                    v.total_packet_drops += 1
                    sim_state.total_network_drops += dropped_pkts_mb

@app.post("/start")
def start_sim(): sim_state.running = True; return {"status": "started"}
@app.post("/pause")
def pause_sim(): sim_state.running = False; return {"status": "paused"}
@app.post("/step")
def step_sim(): run_step(); return {"status": "stepped"}
@app.post("/algorithm")
def set_algorithm(config: ConfigState): sim_state.algorithm = config.algorithm; return {"status": "success", "algorithm": sim_state.algorithm}
@app.post("/reset")
def reset_sim(config: ResetConfig):
    sim_state.num_hubs = config.num_hubs
    sim_state.num_vehicles = config.num_vehicles
    sim_state.event_rate = config.event_rate
    init_sim_entities()
    return {"status": "reset_successful"}
@app.get("/roads")
def get_roads(): return sim_state.roads
@app.get("/state")
def get_state():
    return {
        "running": sim_state.running, "step": sim_state.step, "algorithm": sim_state.algorithm,
        "total_generated_mb": sim_state.total_generated_mb, "total_offloaded_mb": sim_state.total_offloaded_mb,
        "vehicles": [
            {"id": v.id, "lat": v.display_lat, "lon": v.display_lon, "buffer_mb": v.buffer_mb, "events": v.events,
             "speed": v.speed, "current_bw_alloc": v.current_bw_alloc, "total_packet_drops": v.total_packet_drops}
             for v in sim_state.vehicles],
        "hubs": [
            {"id": h.id, "lat": h.lat, "lon": h.lon, "range": h.range, "rate": h.base_rate, "active": h.active, "online": h.online}
            for h in sim_state.hubs],
        "signals": [
            {"node": n, "lat": sim_state.graph.nodes[n]['y'], "lon": sim_state.graph.nodes[n]['x'], "state": s['state']}
            for n, s in sim_state.signals.items()
        ],
        "recent_events": sim_state.recent_events, "total_network_drops": sim_state.total_network_drops
    }
@app.get("/report")
def export_report():
    avg_latency = sim_state.priority_latency_sum / max(1, sim_state.priority_events_completed)
    return {
        "algorithm": sim_state.algorithm, "total_steps": sim_state.step, "completed_events": sim_state.priority_events_completed,
        "average_priority_latency_steps": round(avg_latency, 2), "total_buffer_overflows": sim_state.total_buffer_overflows,
        "hub_utilization_mb": {f"Hub_{h.id}": round(h.total_offloaded_mb, 2) for h in sim_state.hubs},
        "total_generated_mb": round(sim_state.total_generated_mb, 2), "total_offloaded_mb": round(sim_state.total_offloaded_mb, 2),
        "total_network_drops_mb": round(sim_state.total_network_drops, 2),
        "overall_efficiency": f"{round((sim_state.total_offloaded_mb / max(1, sim_state.total_generated_mb)) * 100, 2)}%"
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import osmnx as ox
import networkx as nx
import random
import math
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
LOCATION = "Faizabad Interchange, Islamabad, Pakistan"
WIFI_RANGE_METERS = 100
WIFI_RATE_MB_S = 1.5      # MB per 0.1s step (effectively 15 MB/s)
DATA_GEN_RATE_MB_STEP = 0.1 # effectively 1 MB/s
NUM_VEHICLES = 50
NUM_HUBS = 15

class ConfigState(BaseModel):
    algorithm: str

class ResetConfig(BaseModel):
    num_hubs: int
    num_vehicles: int
    event_rate: float

# Simulation State globals
class SimulationState:
    def __init__(self):
        self.running = False
        self.step = 0
        self.algorithm = "greedy" # 'greedy' or 'predictive'
        self.total_generated_mb = 0.0
        self.total_offloaded_mb = 0.0
        self.num_hubs = NUM_HUBS
        self.num_vehicles = NUM_VEHICLES
        self.event_rate = 0.0005
        self.graph = None
        self.hubs = []
        self.vehicles = []
        self.roads = []
        self.recent_events = []

sim_state = SimulationState()

def euclidean_dist(lat1, lon1, lat2, lon2):
    # Rough approximation for small distances, returning meters
    # 1 deg lat = 111,320m
    lat_mid = (lat1 + lat2) / 2.0
    dx = (lon2 - lon1) * 111320.0 * math.cos(math.radians(lat_mid))
    dy = (lat2 - lat1) * 111320.0
    return math.sqrt(dx*dx + dy*dy)

@app.on_event("startup")
async def startup_event():
    # Load the graph
    print(f"Loading graph for {LOCATION}...")
    try:
        # Simplify the network type
        sim_state.graph = ox.graph_from_place(LOCATION, network_type="drive")
    except Exception as e:
        print("Failed to download exact place, using bounding box or default fallback...", e)
        # Using a fallback location close to Faizabad interchange (Lat, Lon: 33.6515, 73.0801)
        sim_state.graph = ox.graph_from_point((33.6515, 73.0801), dist=2500, network_type="drive")
    
    # Extract nodes
    nodes = list(sim_state.graph.nodes(data=True))

    # Extract roads for frontend rendering
    roads = []
    for u, v, data in sim_state.graph.edges(data=True):
        if 'geometry' in data:
            coords = list(data['geometry'].coords)
            roads.append([[pt[1], pt[0]] for pt in coords]) # Shapely returns (lon, lat) or (lon, lat, ele)
        else:
            u_node = sim_state.graph.nodes[u]
            v_node = sim_state.graph.nodes[v]
            roads.append([[u_node['y'], u_node['x']], [v_node['y'], v_node['x']]])
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
    sim_state.recent_events = []

    # Initialize Hubs (randomly select nodes as center hubs)
    # Force 2 hubs into the top right quadrant
    center_lat = 33.6515
    center_lon = 73.0801
    top_right_nodes = [n for n in nodes if n[1]["y"] > center_lat + 0.005 and n[1]["x"] > center_lon + 0.005]
    if len(top_right_nodes) >= 2 and sim_state.num_hubs >= 2:
        tr_hubs = random.sample(top_right_nodes, 2)
        other_nodes = [n for n in nodes if n not in tr_hubs]
        hub_nodes = tr_hubs + random.sample(other_nodes, sim_state.num_hubs - 2)
    else:
        hub_nodes = random.sample(nodes, sim_state.num_hubs)

    for hid, node in enumerate(hub_nodes):
        sim_state.hubs.append({
            "id": hid,
            "lat": node[1]["y"],
            "lon": node[1]["x"],
            "range": WIFI_RANGE_METERS,
            "rate": random.choice([1.0, 2.5, 5.0]), # varied network capabilities
            "active": False
        })
    
    # Initialize Vehicles
    node_ids = list(sim_state.graph.nodes())
    for vid in range(sim_state.num_vehicles):
        n1 = random.choice(node_ids)
        n2 = random.choice(node_ids)
        
        try:
            # NetworkX shortest path for proper road following
            path = nx.shortest_path(sim_state.graph, n1, n2, weight='length')
        except:
            path = [n1]

        start_data = sim_state.graph.nodes[n1]
        target_lat = start_data["y"]
        target_lon = start_data["x"]
        
        if len(path) > 1:
            next_node = path[1]
            target_lat = sim_state.graph.nodes[next_node]["y"]
            target_lon = sim_state.graph.nodes[next_node]["x"]

        base_spd = random.uniform(0.00004, 0.00008)
        sim_state.vehicles.append({
            "id": vid,
            "lat": start_data["y"],
            "lon": start_data["x"],
            "buffer_mb": 0.0,
            "target_lat": target_lat,
            "target_lon": target_lon,
            "path": path,
            "path_idx": 1 if len(path) > 1 else 0,
            "base_speed": base_spd,
            "speed": base_spd,
            "events": []
        })

async def simulation_loop():
    while True:
        if sim_state.running:
            run_step()
        await asyncio.sleep(0.05) # 20 steps per second

def run_step():
    sim_state.step += 1
    
    # Reset hub activity
    for hub in sim_state.hubs:
        hub["active"] = False

    node_ids = list(sim_state.graph.nodes())

    # Build congestion map
    edge_traffic = {}
    for v in sim_state.vehicles:
        if v["path"] and v["path_idx"] < len(v["path"]):
            prev = v["path"][v["path_idx"]-1] if v["path_idx"] > 0 else v["path"][0]
            curr = v["path"][v["path_idx"]]
            edge_traffic[(prev, curr)] = edge_traffic.get((prev, curr), 0) + 1

    EVENT_TYPES = ["Pothole", "Minor Crash", "Debris", "Speed Trap", "Traffic Jam"]

    for v in sim_state.vehicles:
        # Generate baseline dashcam data
        v["buffer_mb"] += DATA_GEN_RATE_MB_STEP
        sim_state.total_generated_mb += DATA_GEN_RATE_MB_STEP
        
        # Determine speed reduction from congestion
        if v["path"] and v["path_idx"] < len(v["path"]):
            prev = v["path"][v["path_idx"]-1] if v["path_idx"] > 0 else v["path"][0]
            curr = v["path"][v["path_idx"]]
            vehicles_on_edge = edge_traffic.get((prev, curr), 1)
            v["speed"] = v["base_speed"] / max(1, vehicles_on_edge)

        # Random Events
        if random.random() < sim_state.event_rate:
            evt = random.choice(EVENT_TYPES)
            v["events"].append(evt)
            evt_size = random.uniform(5.0, 10.0)
            v["buffer_mb"] += evt_size
            sim_state.total_generated_mb += evt_size

        # Move vehicle along the calculated path
        dx = v["target_lon"] - v["lon"]
        dy = v["target_lat"] - v["lat"]
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist < v["speed"]:
            # Snap to node and advance target along path
            v["lon"] = v["target_lon"]
            v["lat"] = v["target_lat"]
            v["path_idx"] += 1
            
            if v["path_idx"] < len(v["path"]):
                next_node = v["path"][v["path_idx"]]
                v["target_lat"] = sim_state.graph.nodes[next_node]["y"]
                v["target_lon"] = sim_state.graph.nodes[next_node]["x"]
            else:
                # Reached final destination, pick a new path
                start_node = v["path"][-1] if v["path"] else random.choice(node_ids)
                end_node = random.choice(node_ids)
                try:
                    new_path = nx.shortest_path(sim_state.graph, start_node, end_node, weight='length')
                except:
                    new_path = [start_node]
                
                v["path"] = new_path
                v["path_idx"] = 1
                if len(new_path) > 1:
                    next_node = new_path[1]
                    v["target_lat"] = sim_state.graph.nodes[next_node]["y"]
                    v["target_lon"] = sim_state.graph.nodes[next_node]["x"]
                else:
                    v["target_lat"] = v["lat"]
                    v["target_lon"] = v["lon"]
        else:
            v["lon"] += (dx/dist) * v["speed"]
            v["lat"] += (dy/dist) * v["speed"]
        
        # Offloading Logic
        # Check closest hub
        closest_hub = None
        min_dist = float('inf')
        for hub in sim_state.hubs:
            d = euclidean_dist(v["lat"], v["lon"], hub["lat"], hub["lon"])
            if d < min_dist:
                min_dist = d
                closest_hub = hub
                
        if closest_hub and min_dist <= closest_hub["range"]:
            skip_offload = False
            
            if sim_state.algorithm == "predictive" and v["buffer_mb"] < 30:
                # Look slightly ahead on route
                future_nodes = v["path"][v["path_idx"]:v["path_idx"]+10]
                for f_node in future_nodes:
                    ny = sim_state.graph.nodes[f_node]["y"]
                    nx_coord = sim_state.graph.nodes[f_node]["x"]
                    for h in sim_state.hubs:
                        if h["rate"] > closest_hub["rate"] and euclidean_dist(ny, nx_coord, h["lat"], h["lon"]) <= h["range"]:
                            # Found a stronger hub soon, don't offload yet
                            skip_offload = True
                            break
                    if skip_offload:
                        break
            
            if not skip_offload:
                offload_amt = min(v["buffer_mb"], closest_hub["rate"])
                if offload_amt > 0:
                    v["buffer_mb"] -= offload_amt
                    sim_state.total_offloaded_mb += offload_amt
                    closest_hub["active"] = True
                    
                    # Manage Offloaded Events
                    if v["events"]:
                        for ev in v["events"]:
                            sim_state.recent_events.insert(0, f"[{sim_state.step}] Veh {v['id']} offloaded '{ev}' to Hub {closest_hub['id']}")
                        v["events"] = []
                        sim_state.recent_events = sim_state.recent_events[:30] # Keep recent log size small

@app.post("/start")
def start_sim():
    sim_state.running = True
    return {"status": "started"}

@app.post("/pause")
def pause_sim():
    sim_state.running = False
    return {"status": "paused"}

@app.post("/step")
def step_sim():
    run_step()
    return {"status": "stepped"}

@app.post("/algorithm")
def set_algorithm(config: ConfigState):
    sim_state.algorithm = config.algorithm
    return {"status": "success", "algorithm": sim_state.algorithm}

@app.post("/reset")
def reset_sim(config: ResetConfig):
    sim_state.num_hubs = config.num_hubs
    sim_state.num_vehicles = config.num_vehicles
    sim_state.event_rate = config.event_rate
    init_sim_entities()
    return {"status": "reset_successful"}

@app.get("/roads")
def get_roads():
    return sim_state.roads

@app.get("/state")
def get_state():
    return {
        "running": sim_state.running,
        "step": sim_state.step,
        "algorithm": sim_state.algorithm,
        "total_generated_mb": sim_state.total_generated_mb,
        "total_offloaded_mb": sim_state.total_offloaded_mb,
        "vehicles": sim_state.vehicles,
        "hubs": sim_state.hubs,
        "recent_events": sim_state.recent_events
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

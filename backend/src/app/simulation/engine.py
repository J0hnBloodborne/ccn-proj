import asyncio
import random
import math
import networkx as nx

from app.core.state import sim_state
from app.core.config import DATA_GEN_RATE_MB_STEP, MAX_BUFFER_MB, WIFI_RANGE_METERS
from app.simulation.utils import euclidean_dist

def init_sim_entities():
    nodes = list(sim_state.graph.nodes(data=True))
    sim_state.hubs = []
    sim_state.vehicles = []
    sim_state.step = 0
    sim_state.total_generated_mb = 0.0
    sim_state.total_offloaded_mb = 0.0
    sim_state.total_buffer_overflows = 0
    sim_state.priority_latency_sum = 0
    sim_state.priority_events_completed = 0
    sim_state.recent_events = []

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
            "rate": random.choice([1.0, 2.5, 5.0]),
            "active": False,
            "online": True,
            "total_offloaded_mb": 0.0
        })
    
    node_ids = list(sim_state.graph.nodes())
    for vid in range(sim_state.num_vehicles):
        n1 = random.choice(node_ids)
        n2 = random.choice(node_ids)
        try:
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
            "routine_mb": 0.0,
            "priority_mb": 0.0,
            "panic": False,
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
        await asyncio.sleep(0.05) 

def run_step():
    sim_state.step += 1
    
    for hub in sim_state.hubs:
        hub["active"] = False
        if random.random() < 0.001:
            hub["online"] = not hub["online"]

    node_ids = list(sim_state.graph.nodes())
    edge_traffic = {}
    for v in sim_state.vehicles:
        if v["path"] and v["path_idx"] < len(v["path"]):
            prev = v["path"][v["path_idx"]-1] if v["path_idx"] > 0 else v["path"][0]
            curr = v["path"][v["path_idx"]]
            edge_traffic[(prev, curr)] = edge_traffic.get((prev, curr), 0) + 1

    EVENT_TYPES = ["Pothole", "Minor Crash", "Debris", "Speed Trap", "Traffic Jam"]

    for v in sim_state.vehicles:
        v["routine_mb"] += DATA_GEN_RATE_MB_STEP
        sim_state.total_generated_mb += DATA_GEN_RATE_MB_STEP
        
        if v["path"] and v["path_idx"] < len(v["path"]):
            prev = v["path"][v["path_idx"]-1] if v["path_idx"] > 0 else v["path"][0]
            curr = v["path"][v["path_idx"]]
            vehicles_on_edge = edge_traffic.get((prev, curr), 1)
            v["speed"] = v["base_speed"] / max(1, vehicles_on_edge)

        if random.random() < sim_state.event_rate:
            evt_type = random.choice(EVENT_TYPES)
            evt_size = random.uniform(5.0, 10.0)
            v["events"].append({"type": evt_type, "size": evt_size, "timestamp": sim_state.step})
            v["priority_mb"] += evt_size
            sim_state.total_generated_mb += evt_size

        total_buf = v["routine_mb"] + v["priority_mb"]
        if total_buf > MAX_BUFFER_MB:
            overflow_amt = total_buf - MAX_BUFFER_MB
            dropped_mb = min(overflow_amt, v["routine_mb"])
            if dropped_mb > 0:
                v["routine_mb"] -= dropped_mb
                sim_state.total_buffer_overflows += 1
        
        v["buffer_mb"] = v["routine_mb"] + v["priority_mb"]

        dx = v["target_lon"] - v["lon"]
        dy = v["target_lat"] - v["lat"]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < v["speed"]:
            v["lon"] = v["target_lon"]
            v["lat"] = v["target_lat"]
            v["path_idx"] += 1
            if v["path_idx"] < len(v["path"]):
                next_node = v["path"][v["path_idx"]]
                v["target_lat"] = sim_state.graph.nodes[next_node]["y"]
                v["target_lon"] = sim_state.graph.nodes[next_node]["x"]
            else:
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
        
        closest_hub = None
        min_dist = float('inf')
        for hub in sim_state.hubs:
            d = euclidean_dist(v["lat"], v["lon"], hub["lat"], hub["lon"])
            if d < min_dist:
                min_dist = d
                closest_hub = hub
                
        if closest_hub and min_dist <= closest_hub["range"]:
            if not closest_hub["online"]:
                v["panic"] = True
            else:
                skip_offload = False
                
                if sim_state.algorithm == "predictive" and not v.get("panic", False) and (v["routine_mb"] + v["priority_mb"]) < MAX_BUFFER_MB * 0.8:
                    future_nodes = v["path"][v["path_idx"]:v["path_idx"]+10]
                    for f_node in future_nodes:
                        ny = sim_state.graph.nodes[f_node]["y"]
                        nx_coord = sim_state.graph.nodes[f_node]["x"]
                        for h in sim_state.hubs:
                            if h["online"] and h["rate"] > closest_hub["rate"] and euclidean_dist(ny, nx_coord, h["lat"], h["lon"]) <= h["range"]:
                                skip_offload = True
                                break
                        if skip_offload:
                            break
                
                if not skip_offload:
                    amt_to_offload = closest_hub["rate"]
                    
                    while v["events"] and amt_to_offload > 0:
                        evt = v["events"][0]
                        if evt["size"] <= amt_to_offload:
                            amt_to_offload -= evt["size"]
                            v["priority_mb"] -= evt["size"]
                            sim_state.total_offloaded_mb += evt["size"]
                            closest_hub["total_offloaded_mb"] += evt["size"]
                            latency = sim_state.step - evt["timestamp"]
                            sim_state.priority_latency_sum += latency
                            sim_state.priority_events_completed += 1
                            v["events"].pop(0)
                            sim_state.recent_events.insert(0, f"[{sim_state.step}] Veh {v['id']} offloaded '{evt['type']}' to Hub {closest_hub['id']} (Lat: {latency}s)")
                        else:
                            evt["size"] -= amt_to_offload
                            v["priority_mb"] -= amt_to_offload
                            sim_state.total_offloaded_mb += amt_to_offload
                            closest_hub["total_offloaded_mb"] += amt_to_offload
                            amt_to_offload = 0
                            
                    if amt_to_offload > 0 and v["routine_mb"] > 0:
                        off_routine = min(v["routine_mb"], amt_to_offload)
                        v["routine_mb"] -= off_routine
                        sim_state.total_offloaded_mb += off_routine
                        closest_hub["total_offloaded_mb"] += off_routine
                        
                    closest_hub["active"] = True
                    v["panic"] = False 
            
            v["buffer_mb"] = v["routine_mb"] + v["priority_mb"]
            sim_state.recent_events = sim_state.recent_events[:30]

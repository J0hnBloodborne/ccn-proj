"""Full traffic simulation using REAL OpenStreetMap data."""

import math
import random
from typing import List, Dict, Optional, Tuple

from src.config import (
    SIM_DT, IDM_A_MAX, IDM_B, IDM_V0, IDM_S0, IDM_T,
    MOBIL_P, MOBIL_THRESHOLD, MOBIL_B_SAFE,
    VEHICLE_LENGTH, LANE_WIDTH, MAX_VEHICLES, DATA_GEN_RATE, HUB_RADIUS,
    STOP_LINE_DISTANCE, YIELD_DISTANCE
)
from src.real_network import (
    RealNetworkGraph, Node, Link, Vehicle, Hub, Event, 
    load_osm_real, create_islamabad_network
)


def idm_acceleration(v: float, dv: float, s: float) -> float:
    """IDM car-following acceleration."""
    if s <= 0:
        return -IDM_B * 10
    
    s_star = IDM_S0 + max(0, v * IDM_T + (v * dv) / (2 * math.sqrt(IDM_A_MAX * IDM_B)))
    a = IDM_A_MAX * (1 - (v / IDM_V0) ** 4 - (s_star / max(s, 0.1)) ** 2)
    return max(-IDM_B * 5, min(a, IDM_A_MAX))


class Simulator:
    """Traffic simulation using REAL network data."""
    
    def __init__(self, use_real_osm: bool = True):
        # Load real network (OSM or London-style fallback)
        if use_real_osm:
            self.graph = load_osm_real()
        else:
            self.graph = create_london_network()
        
        self.hubs: List[Hub] = []
        self.events: List[Event] = []
        
        self.time: float = 0.0
        self.frame: int = 0
        self.total_spawned: int = 0
        self.total_completed: int = 0
        
        self.total_data_generated: float = 0.0
        self.total_offloaded: float = 0.0
        self.total_collected: float = 0.0
        self.total_delay: float = 0.0
        
        self._setup_hubs(25)
        self._setup_events(30)
        self._spawn_initial_vehicles(100)
    
    def _setup_hubs(self, count: int):
        """Place hubs at major intersections."""
        nodes = list(self.graph.nodes.values())
        random.shuffle(nodes)
        
        # Place at major intersections
        major = [n for n in nodes if n.priority >= 2][:count]
        for node in major:
            hub = Hub(x=node.x, y=node.y, radius=HUB_RADIUS, bandwidth=100.0)
            self.hubs.append(hub)
        
        # Fill with mid-link hubs
        links = list(self.graph.links.values())
        random.shuffle(links)
        
        remaining = count - len(self.hubs)
        for link in links[:remaining]:
            hub = Hub(
                x=(link.from_node.x + link.to_node.x) / 2,
                y=(link.from_node.y + link.to_node.y) / 2,
                radius=HUB_RADIUS * 0.8,
                bandwidth=80.0
            )
            self.hubs.append(hub)
    
    def _setup_events(self, count: int):
        """Place data collection events."""
        links = list(self.graph.links.values())
        random.shuffle(links)
        
        for link in links[:count]:
            t = random.uniform(0.2, 0.8)
            event = Event(
                x=link.from_node.x + t * (link.to_node.x - link.from_node.x),
                y=link.from_node.y + t * (link.to_node.y - link.from_node.y),
                radius=random.uniform(30, 50),
                data_amount=random.uniform(3, 8)
            )
            self.events.append(event)
    
    def _spawn_initial_vehicles(self, count: int):
        """Spawn initial vehicles."""
        nodes = list(self.graph.nodes.values())
        
        spawned = 0
        attempts = 0
        while spawned < count and attempts < count * 100:
            start = random.choice(nodes)
            end = random.choice(nodes)
            
            if start.id == end.id:
                attempts += 1
                continue
            
            v = self.graph.create_vehicle(start, end)
            if v:
                spawned += 1
                self.total_spawned += 1
            attempts += 1
    
    def _spawn_vehicle(self):
        """Spawn a new vehicle."""
        if len(self.graph.vehicles) >= MAX_VEHICLES:
            return None
        
        nodes = list(self.graph.nodes.values())
        start = random.choice(nodes)
        end = random.choice(nodes)
        
        if start.id == end.id:
            return None
        
        v = self.graph.create_vehicle(start, end)
        if v:
            self.total_spawned += 1
        return v
    
    def _update_signals(self):
        """Update traffic signal states."""
        for node in self.graph.nodes.values():
            if not node.is_signalized:
                continue
            
            node.signal_timer += SIM_DT
            if node.signal_timer >= node.signal_cycle:
                node.signal_timer = 0
                node.signal_phase = (node.signal_phase + 1) % 2
    
    def _check_right_of_way(self, vehicle: Vehicle) -> Tuple[bool, str]:
        """Check if vehicle can proceed based on right-of-way rules."""
        if vehicle.route_idx >= len(vehicle.route) - 1:
            return True, "destination"
        
        next_node_id = vehicle.route[vehicle.route_idx + 1]
        if vehicle.link.to_node.id != next_node_id:
            return True, "not_heading"
        
        node = vehicle.link.to_node
        
        if node.is_signalized and node.signal_phase != 0:
            return False, "red_light"
        
        if node.is_yield:
            for in_link in node.in_links:
                if in_link.id == vehicle.link.id:
                    continue
                for lane in in_link.lanes:
                    for vid in lane.vehicles:
                        v = self.graph.vehicles.get(vid)
                        if v and v.link.length - v.x < YIELD_DISTANCE:
                            return False, "yield"
        
        return True, "clear"
    
    def _get_turn_speed_factor(self, vehicle: Vehicle) -> float:
        """Get speed reduction for turn."""
        if vehicle.route_idx >= len(vehicle.route) - 1:
            return 1.0
        
        next_node_id = vehicle.route[vehicle.route_idx + 1]
        next_link = None
        
        for link in vehicle.link.to_node.out_links:
            if link.to_node.id == next_node_id:
                next_link = link
                break
        
        if not next_link:
            return 1.0
        
        angle1 = vehicle.link.get_angle()
        angle2 = next_link.get_angle()
        diff = math.degrees(angle2 - angle1)
        
        while diff > 180: diff -= 360
        while diff < -180: diff += 360
        
        abs_angle = abs(diff)
        
        if abs_angle < 15: return 1.0
        elif abs_angle < 45: return 0.8
        elif abs_angle < 90: return 0.6
        else: return 0.4
    
    def _get_leader(self, vehicle: Vehicle) -> Optional[Vehicle]:
        """Get vehicle ahead in lane."""
        lane = vehicle.link.lanes[vehicle.lane]
        ahead = []
        for vid in lane.vehicles:
            v = self.graph.vehicles.get(vid)
            if v and v.x > vehicle.x:
                ahead.append(v)
        return min(ahead, key=lambda v: v.x) if ahead else None
    
    def _get_follower(self, vehicle: Vehicle, lane: int) -> Optional[Vehicle]:
        """Get vehicle behind in lane."""
        if lane < 0 or lane >= vehicle.link.num_lanes:
            return None
        
        behind = []
        for vid in vehicle.link.lanes[lane].vehicles:
            v = self.graph.vehicles.get(vid)
            if v and v.x < vehicle.x:
                behind.append(v)
        return max(behind, key=lambda v: v.x) if behind else None
    
    def _update_mobil(self, vehicle: Vehicle):
        """MOBIL lane change."""
        if vehicle.changing_lane:
            vehicle.lane_change_timer += SIM_DT
            if vehicle.lane_change_timer >= 0.8:
                old_lane = vehicle.link.lanes[vehicle.lane]
                if vehicle.id in old_lane.vehicles:
                    old_lane.vehicles.remove(vehicle.id)
                vehicle.lane = vehicle.target_lane
                vehicle.changing_lane = False
                vehicle.target_lane = -1
                vehicle.lane_change_timer = 0
                new_lane = vehicle.link.lanes[vehicle.lane]
                new_lane.vehicles.append(vehicle.id)
            return
        
        dist_to_intersection = vehicle.link.length - vehicle.x
        if dist_to_intersection < 50:
            return
        
        if vehicle.v > vehicle.link.free_flow_speed * 0.7:
            return
        
        for target_lane in [vehicle.lane - 1, vehicle.lane + 1]:
            if target_lane < 0 or target_lane >= vehicle.link.num_lanes:
                continue
            
            follower = self._get_follower(vehicle, target_lane)
            if follower:
                gap = vehicle.x - follower.x - VEHICLE_LENGTH
                if gap < VEHICLE_LENGTH * 2:
                    continue
            
            leader = self._get_leader(vehicle)
            if leader:
                current_accel = idm_acceleration(vehicle.v, vehicle.v - leader.v, 
                                                  leader.x - vehicle.x - VEHICLE_LENGTH)
            else:
                current_accel = IDM_A_MAX
            
            ahead = [self.graph.vehicles[vid] for vid in vehicle.link.lanes[target_lane].vehicles
                    if vid in self.graph.vehicles and self.graph.vehicles[vid].x > vehicle.x]
            target_leader = min(ahead, key=lambda v: v.x) if ahead else None
            
            if target_leader:
                target_accel = idm_acceleration(vehicle.v, vehicle.v - target_leader.v,
                                                 target_leader.x - vehicle.x - VEHICLE_LENGTH)
            else:
                target_accel = IDM_A_MAX
            
            if follower:
                fol_gap = vehicle.x - follower.x - VEHICLE_LENGTH
                fol_accel = idm_acceleration(follower.v, follower.v - vehicle.v, fol_gap)
                if fol_accel < -MOBIL_B_SAFE:
                    continue
            
            gain = target_accel - current_accel
            if follower:
                gain -= MOBIL_P * IDM_A_MAX
            
            threshold = MOBIL_THRESHOLD if vehicle.link.is_congested else 0.1
            if gain > threshold:
                vehicle.changing_lane = True
                vehicle.target_lane = target_lane
                vehicle.lane_change_timer = 0
                break
    
    def _update_vehicle(self, vehicle: Vehicle):
        """Update single vehicle."""
        if vehicle.state == "lane_change":
            self._update_mobil(vehicle)
        
        leader = self._get_leader(vehicle)
        
        if leader:
            gap = leader.x - vehicle.x - VEHICLE_LENGTH
            dv = vehicle.v - leader.v
        else:
            gap = vehicle.link.length - vehicle.x
            dv = 0
        
        can_proceed, reason = self._check_right_of_way(vehicle)
        turn_factor = self._get_turn_speed_factor(vehicle)
        
        if not can_proceed:
            if reason == "red_light":
                dist_to_stop = vehicle.link.length - vehicle.x - 5
                if 0 < dist_to_stop < STOP_LINE_DISTANCE:
                    a = -IDM_B * 3
                elif dist_to_stop <= 5:
                    a = -IDM_B * 5
                else:
                    a = -IDM_B * 2
            else:
                a = -IDM_B * 2
        elif gap < VEHICLE_LENGTH * 2:
            a = -IDM_B * 5
        else:
            a = idm_acceleration(vehicle.v, dv, gap)
            if vehicle.link.length - vehicle.x < 60:
                a = min(a, (vehicle.link.free_flow_speed * turn_factor - vehicle.v) * 0.5)
        
        max_speed = vehicle.link.free_flow_speed * turn_factor
        if vehicle.v > max_speed:
            a = min(a, -1.5)
        
        if vehicle.link.is_congested:
            a *= 0.7
        
        vehicle.v += a * SIM_DT
        vehicle.v = max(0, min(vehicle.v, max_speed))
        vehicle.x += vehicle.v * SIM_DT * 50
        
        if vehicle.v < vehicle.link.free_flow_speed * 0.3:
            self.total_delay += SIM_DT
        
        if vehicle.x >= vehicle.link.length:
            self._transfer_vehicle(vehicle)
        
        if vehicle.v < 0.5 and not can_proceed:
            vehicle.state = "stopped"
        elif vehicle.changing_lane:
            vehicle.state = "lane_change"
        else:
            vehicle.state = "moving"
    
    def _transfer_vehicle(self, vehicle: Vehicle):
        """Transfer to next link."""
        if vehicle.route_idx >= len(vehicle.route) - 1:
            self.graph.remove_vehicle(vehicle.id)
            self.total_completed += 1
            return
        
        if 0 <= vehicle.lane < vehicle.link.num_lanes:
            lane = vehicle.link.lanes[vehicle.lane]
            if vehicle.id in lane.vehicles:
                lane.vehicles.remove(vehicle.id)
        
        vehicle.route_idx += 1
        next_node_id = vehicle.route[vehicle.route_idx]
        
        next_link = None
        for link in self.graph.nodes[next_node_id].in_links:
            if link.from_node.id == vehicle.link.to_node.id:
                next_link = link
                break
        
        if not next_link:
            incoming = [l for l in self.graph.nodes[next_node_id].in_links]
            if incoming:
                next_link = incoming[0]
            else:
                self.graph.remove_vehicle(vehicle.id)
                self.total_completed += 1
                return
        
        vehicle.link = next_link
        vehicle.x = 0
        vehicle.lane = min(vehicle.lane, max(0, next_link.num_lanes - 1))
        vehicle.changing_lane = False
        
        lane = next_link.lanes[vehicle.lane]
        lane.vehicles.append(vehicle.id)
    
    def _update_data_offloading(self):
        """Update data offloading."""
        for vehicle in self.graph.vehicles.values():
            vehicle.data_buffer += DATA_GEN_RATE
            self.total_data_generated += DATA_GEN_RATE
            
            world_x, world_y = vehicle.get_world_pos()
            connected_to = None
            
            for hub in self.hubs:
                dx = world_x - hub.x
                dy = world_y - hub.y
                dist = math.sqrt(dx * dx + dy * dy)
                
                if dist <= hub.radius:
                    connected_to = hub
                    if vehicle not in hub.connected:
                        hub.connected.append(vehicle)
                    break
            
            vehicle.connected_hub = connected_to
            
            if connected_to:
                bw = connected_to.get_allocated_bw()
                vehicle.offload_rate = bw
                offload = min(bw * SIM_DT * 0.1, vehicle.data_buffer)
                vehicle.data_buffer -= offload
                self.total_offloaded += offload
            else:
                vehicle.offload_rate = 0
    
    def _update_event_detection(self):
        """Update event detection."""
        for event in self.events:
            for vehicle in self.graph.vehicles.values():
                wx, wy = vehicle.get_world_pos()
                dist = math.sqrt((wx - event.x) ** 2 + (wy - event.y) ** 2)
                
                if dist <= event.radius:
                    collected = min(event.data_amount * 0.1, vehicle.link.free_flow_speed * 0.5)
                    vehicle.data_buffer += collected
                    event.collected += 1
                    self.total_collected += collected
    
    def update(self):
        """Main update."""
        self._update_signals()
        self.graph.update_flows()
        
        for vehicle in list(self.graph.vehicles.values()):
            self._update_vehicle(vehicle)
        
        self._update_data_offloading()
        self._update_event_detection()
        
        for hub in self.hubs:
            hub.update_animation(SIM_DT)
        
        self.time += SIM_DT
        self.frame += 1
        
        if self.frame % 20 == 0 and len(self.graph.vehicles) < MAX_VEHICLES:
            spawn_count = min(8, MAX_VEHICLES - len(self.graph.vehicles))
            for _ in range(spawn_count):
                self._spawn_vehicle()
    
    def get_state(self) -> dict:
        """Get simulation state."""
        vehicles = list(self.graph.vehicles.values())
        total = len(vehicles)
        
        if total == 0:
            return {
                "time": self.time, "vehicles": 0, "connected": 0,
                "avg_speed": 0, "congestion": 0, "stopped": 0
            }
        
        speeds = [v.v for v in vehicles]
        connected = sum(1 for v in vehicles if v.connected_hub)
        stopped = sum(1 for v in vehicles if v.v < 0.5)
        
        total_congestion = sum(l.v_c_ratio for l in self.graph.links.values())
        congestion = total_congestion / max(len(self.graph.links), 1)
        
        return {
            "time": self.time,
            "vehicles": total,
            "stopped": stopped,
            "connected": connected,
            "avg_speed": sum(speeds) / total,
            "max_speed": max(speeds),
            "congestion": congestion,
            "avg_buffer": sum(v.data_buffer for v in vehicles) / total,
            "total_generated": self.total_data_generated,
            "total_offloaded": self.total_offloaded,
            "total_collected": self.total_collected,
            "total_delay": self.total_delay,
            "spawned": self.total_spawned,
            "completed": self.total_completed,
        }


def create_simulator() -> Simulator:
    """Create simulator with real network."""
    return Simulator(use_real_osm=True)
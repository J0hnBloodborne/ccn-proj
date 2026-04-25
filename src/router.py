"""Router with A* pathfinding and bezier turn planning."""

import heapq
import math
from typing import List, Tuple, Optional, Dict
from src.graph import Graph
from src.config import INTERSECTION_RADIUS, TURN_DURATION, LANE_WIDTH


TURN_TYPES = {
    "straight": 0,
    "slight_left": 1,
    "slight_right": -1,
    "left": 2,
    "right": -2,
}


def dijkstra(graph: Graph, start: int, end: int) -> Optional[List[int]]:
    """Find shortest path using Dijkstra's algorithm."""
    if start == end:
        return [start]

    dist = {start: 0}
    prev = {start: None}
    pq = [(0, start)]
    visited = set()

    while pq:
        d, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)

        if current == end:
            break

        for neighbor_id, edge_id in graph.adjacency.get(current, []):
            if neighbor_id in visited:
                continue

            edge = graph.edges.get(edge_id)
            if not edge:
                continue

            new_dist = d + edge.length
            if neighbor_id not in dist or new_dist < dist[neighbor_id]:
                dist[neighbor_id] = new_dist
                prev[neighbor_id] = (current, edge_id)
                heapq.heappush(pq, (new_dist, neighbor_id))

    if end not in prev:
        return None

    path = []
    current = end
    while current is not None:
        path.append(current)
        if current in prev:
            current = prev[current][0] if isinstance(prev[current], tuple) else prev[current]
        else:
            current = None

    path.reverse()
    return path


def get_turn_direction(current_node: int, next_node: int, target_node: int, graph: Graph) -> str:
    """Determine turn type at intersection."""
    if target_node not in graph.nodes:
        return "straight"

    curr = graph.nodes[current_node]
    next_n = graph.nodes[next_node]
    target = graph.nodes[target_node]

    dx1 = next_n.x - curr.x
    dy1 = next_n.y - curr.y
    dx2 = target.x - next_n.x
    dy2 = target.y - next_n.y

    angle1 = math.atan2(dy1, dx1)
    angle2 = math.atan2(dy2, dx2)

    angle_diff = angle2 - angle1
    while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    while angle_diff < -math.pi:
        angle_diff += 2 * math.pi

    angle_deg = math.degrees(angle_diff)

    if abs(angle_deg) < 30:
        return "straight"
    elif -60 < angle_deg < -30:
        return "slight_right"
    elif 30 < angle_deg < 60:
        return "slight_left"
    elif angle_deg <= -60:
        return "right"
    else:
        return "left"


def compute_turn_path(
    entry_point: Tuple[float, float],
    intersection_center: Tuple[float, float],
    exit_point: Tuple[float, float],
    turn_type: str
) -> List[Tuple[float, float]]:
    """Compute bezier control point based on turn type."""
    cx, cy = intersection_center
    ex, ey = exit_point

    if turn_type == "straight":
        offset_x, offset_y = 0, 0
    elif turn_type == "slight_left":
        offset_x, offset_y = 15, -10
    elif turn_type == "slight_right":
        offset_x, offset_y = 15, 10
    elif turn_type == "left":
        offset_x, offset_y = 0, -25
    elif turn_type == "right":
        offset_x, offset_y = 0, 25
    else:
        offset_x, offset_y = 0, 0

    control = (cx + offset_x, cy + offset_y)

    points = []
    for i in range(TURN_DURATION + 1):
        t = i / TURN_DURATION
        u = 1 - t
        x = u * u * entry_point[0] + 2 * u * t * control[0] + t * t * exit_point[0]
        y = u * u * entry_point[1] + 2 * u * t * control[1] + t * t * exit_point[1]
        points.append((x, y))

    return points


def get_entry_and_exit_points(
    current_node: int,
    next_node: int,
    target_node: int,
    graph: Graph
) -> Tuple[Tuple[float, float], Tuple[float, float], str]:
    """Get entry point, exit point, and turn type for a turn maneuver."""
    curr = graph.nodes[current_node]
    next_n = graph.nodes[next_node]
    target = graph.nodes[target_node]

    dx = next_n.x - curr.x
    dy = next_n.y - curr.y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 0.001:
        dx, dy = 1, 0
        dist = 1

    dx /= dist
    dy /= dist

    entry_offset = INTERSECTION_RADIUS * 1.5
    entry_point = (next_n.x - dx * entry_offset, next_n.y - dy * entry_offset)

    tdx = target.x - next_n.x
    tdy = target.y - next_n.y
    tdist = math.sqrt(tdx * tdx + tdy * tdy)
    if tdist < 0.001:
        tdx, tdy = dx, dy
        tdist = 1

    tdx /= tdist
    tdy /= tdist

    exit_offset = INTERSECTION_RADIUS * 1.5
    exit_point = (next_n.x + tdx * exit_offset, next_n.y + tdy * exit_offset)

    turn_type = get_turn_direction(current_node, next_node, target_node, graph)

    return entry_point, exit_point, turn_type


class Router:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.path_cache: Dict[Tuple[int, int], List[int]] = {}

    def get_path(self, start: int, end: int) -> Optional[List[int]]:
        """Get path with caching."""
        key = (start, end)
        if key in self.path_cache:
            return self.path_cache[key]

        path = dijkstra(self.graph, start, end)
        if path:
            self.path_cache[key] = path
        return path

    def get_next_maneuver(self, current_node: int, path: List[int], path_index: int) -> Optional[dict]:
        """Get next turn maneuver info."""
        if path_index >= len(path) - 1:
            return None

        next_node = path[path_index + 1]
        target_node = path[path_index + 2] if path_index + 2 < len(path) else None

        entry, exit_pt, turn_type = get_entry_and_exit_points(
            current_node, next_node, target_node if target_node else next_node, self.graph
        )

        return {
            "next_node": next_node,
            "target_node": target_node,
            "entry_point": entry,
            "exit_point": exit_pt,
            "turn_type": turn_type,
            "intersection_center": (self.graph.nodes[next_node].x, self.graph.nodes[next_node].y) if next_node in self.graph.nodes else entry,
        }
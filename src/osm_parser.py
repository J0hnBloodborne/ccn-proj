"""OSM Parser - UXsim-inspired OpenStreetMap data fetcher and parser."""

import json
import math
import time
import random
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import deque

from src.config import SCALE, OSM_BOUNDS, OVERPASS_URL


LANE_COUNTS = {
    "motorway": 3,
    "trunk": 2,
    "primary": 2,
    "secondary": 1,
    "tertiary": 1,
    "residential": 1,
    "unclassified": 1,
    "living_street": 1,
    "pedestrian": 1,
    "track": 1,
    "road": 1,
}

SPEED_LIMITS = {
    "motorway": 22.2,
    "trunk": 13.9,
    "primary": 11.1,
    "secondary": 8.3,
    "tertiary": 8.3,
    "residential": 5.6,
    "unclassified": 8.3,
    "living_street": 2.8,
    "pedestrian": 2.8,
    "track": 4.2,
    "road": 8.3,
}

ONE_WAY_ROADS = {"motorway", "trunk"}


@dataclass
class OSMNode:
    osm_id: int
    lat: float
    lon: float
    x: float
    y: float
    in_links: List[int] = field(default_factory=list)
    out_links: List[int] = field(default_factory=list)


@dataclass
class OSMWay:
    osm_id: int
    highway: str
    node_ids: List[int]
    lanes: int
    one_way: bool
    maxspeed: float
    length: float = 0.0
    way_points: List[Tuple[float, float]] = field(default_factory=list)


class OSMGraph:
    def __init__(self):
        self.nodes: Dict[int, OSMNode] = {}
        self.ways: Dict[int, OSMWay] = {}
        self.node_coords: Dict[int, Tuple[float, float]] = {}

    def lat_lon_to_xy(self, lat: float, lon: float) -> Tuple[float, float]:
        x = (lon + 0.1) * SCALE
        y = (51.51 - lat) * SCALE
        return (x, y)

    def add_node(self, osm_id: int, lat: float, lon: float):
        x, y = self.lat_lon_to_xy(lat, lon)
        self.nodes[osm_id] = OSMNode(osm_id=osm_id, lat=lat, lon=lon, x=x, y=y)
        self.node_coords[osm_id] = (x, y)

    def add_way(self, osm_id: int, highway: str, node_ids: List[int],
                lanes: int, one_way: bool, maxspeed: float):
        if len(node_ids) < 2:
            return

        way_points = []
        total_length = 0.0
        for i, nid in enumerate(node_ids):
            if nid in self.node_coords:
                way_points.append(self.node_coords[nid])
                if i > 0:
                    prev = self.node_coords[node_ids[i - 1]]
                    curr = self.node_coords[nid]
                    total_length += math.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)

        self.ways[osm_id] = OSMWay(
            osm_id=osm_id,
            highway=highway,
            node_ids=node_ids,
            lanes=lanes,
            one_way=one_way,
            maxspeed=maxspeed,
            length=total_length,
            way_points=way_points
        )

    def get_jam_density(self, lanes: int) -> float:
        return 0.2 * lanes


def fetch_osm_data() -> Optional[Dict]:
    """Fetch OSM data from Overpass API with improved query."""
    import urllib.request
    import ssl

    query = f"""
[out:json][timeout:60][bbox:{OSM_BOUNDS['min_lat']},{OSM_BOUNDS['min_lon']},{OSM_BOUNDS['max_lat']},{OSM_BOUNDS['max_lon']}];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential)$"](if: length() > 50);
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
            "https://overpass-api.de/api/interpreter",
            data=query.encode('utf-8'),
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                'User-Agent': 'VehicularNetworkSim/1.0'
            }
        )

        with urllib.request.urlopen(req, timeout=120, context=ctx) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data
    except Exception as e:
        print(f"Overpass API error: {e}")
        return None


def parse_osm_data(data: Dict) -> OSMGraph:
    """Parse OSM JSON data into OSMGraph."""
    graph = OSMGraph()

    nodes_data = {}
    for element in data.get('elements', []):
        if element.get('type') == 'node':
            nodes_data[element['id']] = {'lat': element['lat'], 'lon': element['lon']}
            graph.add_node(element['id'], element['lat'], element['lon'])

    for element in data.get('elements', []):
        if element.get('type') == 'way':
            tags = element.get('tags', {})
            highway = tags.get('highway', 'unclassified')

            if 'lanes' in tags:
                try:
                    lanes = int(tags['lanes'])
                except:
                    lanes = LANE_COUNTS.get(highway, 1)
            else:
                lanes = LANE_COUNTS.get(highway, 1)

            one_way = tags.get('oneway', 'no') == 'yes'
            if highway in ONE_WAY_ROADS:
                one_way = True

            maxspeed_str = tags.get('maxspeed', '')
            if maxspeed_str:
                try:
                    if 'mph' in maxspeed_str.lower():
                        maxspeed = float(maxspeed_str.replace('mph', '').strip()) * 0.44704
                    else:
                        maxspeed = float(maxspeed_str) / 3.6
                except:
                    maxspeed = SPEED_LIMITS.get(highway, 8.3)
            else:
                maxspeed = SPEED_LIMITS.get(highway, 8.3)

            center = element.get('center')
            if center:
                lat, lon = center.get('lat', 0), center.get('lon', 0)
                if element['id'] not in graph.nodes:
                    graph.add_node(element['id'], lat, lon)

            node_ids = element.get('nodes', [])
            if node_ids:
                first_node = node_ids[0]
                last_node = node_ids[-1]
                if first_node in nodes_data:
                    graph.add_way(element['id'], highway, node_ids, lanes, one_way, maxspeed)

    return graph


def create_realistic_city_grid() -> OSMGraph:
    """Create a realistic procedural city grid with varying road types and irregular layout."""
    graph = OSMGraph()

    random.seed(42)

    nodes = {}
    node_id = 1000

    main_roads_x = [150, 450, 850, 1300, 1750]
    main_roads_y = [150, 400, 750, 1150, 1550]

    secondary_roads_x = [300, 650, 1050, 1450]
    secondary_roads_y = [280, 580, 980, 1350]

    for x in main_roads_x:
        for y in main_roads_y:
            lat = 51.51 - y / SCALE
            lon = x / SCALE - 0.1 + random.uniform(-20, 20) / SCALE
            graph.add_node(node_id, lat, lon)
            nodes[node_id] = (x, y)
            node_id += 1

    for x in secondary_roads_x:
        for y in main_roads_y:
            lat = 51.51 - y / SCALE
            lon = x / SCALE - 0.1 + random.uniform(-15, 15) / SCALE
            graph.add_node(node_id, lat, lon)
            nodes[node_id] = (x, y)
            node_id += 1

    for x in main_roads_x:
        for y in secondary_roads_y:
            lat = 51.51 - y / SCALE
            lon = x / SCALE - 0.1 + random.uniform(-15, 15) / SCALE
            graph.add_node(node_id, lat, lon)
            nodes[node_id] = (x, y)
            node_id += 1

    for x in secondary_roads_x:
        for y in secondary_roads_y:
            lat = 51.51 - y / SCALE
            lon = x / SCALE - 0.1 + random.uniform(-10, 10) / SCALE
            graph.add_node(node_id, lat, lon)
            nodes[node_id] = (x, y)
            node_id += 1

    def add_road(n1, n2, road_type, speed, two_way=True):
        if n1 in graph.nodes and n2 in graph.nodes:
            lanes = 3 if road_type == "motorway" else 2 if road_type == "primary" else 1

            graph.add_way(
                osm_id=n1 * 10000 + n2,
                highway=road_type,
                node_ids=[n1, n2],
                lanes=lanes,
                one_way=False,
                maxspeed=speed
            )

            if two_way and road_type != "motorway":
                graph.add_way(
                    osm_id=n2 * 10000 + n1,
                    highway=road_type,
                    node_ids=[n2, n1],
                    lanes=lanes,
                    one_way=False,
                    maxspeed=speed
                )

    all_node_ids = list(nodes.keys())
    for i, n1 in enumerate(all_node_ids):
        for j, n2 in enumerate(all_node_ids):
            if i >= j:
                continue

            x1, y1 = nodes[n1]
            x2, y2 = nodes[n2]

            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if dist < 180 and dist > 80:
                is_horizontal = abs(y2 - y1) < 50
                is_vertical = abs(x2 - x1) < 50

                if is_horizontal and abs(y1 - y2) < 30:
                    row_matches = sum(1 for ny in [y1, y2] if any(abs(ny - my) < 50 for my in main_roads_y))
                    if row_matches >= 1:
                        rt = "primary" if x1 in main_roads_x or x2 in main_roads_x else "secondary"
                        spd = SPEED_LIMITS.get(rt, 8.3)
                        add_road(n1, n2, rt, spd)

                elif is_vertical and abs(x1 - x2) < 30:
                    col_matches = sum(1 for nx in [x1, x2] if any(abs(nx - mx) < 50 for mx in main_roads_x))
                    if col_matches >= 1:
                        rt = "primary" if y1 in main_roads_y or y2 in main_roads_y else "secondary"
                        spd = SPEED_LIMITS.get(rt, 8.3)
                        add_road(n1, n2, rt, spd)

    for i, n1 in enumerate(all_node_ids):
        for j, n2 in enumerate(all_node_ids):
            if i >= j:
                continue

            x1, y1 = nodes[n1]
            x2, y2 = nodes[n2]
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if 100 < dist < 200:
                is_diag = abs(x2 - x1) > 50 and abs(y2 - y1) > 50

                same_row = any(abs(y1 - ry) < 30 and abs(y2 - ry) < 30 for ry in main_roads_y)
                same_col = any(abs(x1 - cx) < 30 and abs(x2 - cx) < 30 for cx in main_roads_x)

                if is_diag and (same_row or same_col) and random.random() < 0.3:
                    add_road(n1, n2, "tertiary", 8.3, False)

    local_nodes_x = [180, 320, 480, 620, 780, 920, 1080, 1220, 1380, 1520, 1680, 1820]
    local_nodes_y = [180, 320, 480, 620, 780, 920, 1080, 1280, 1480]

    for x in local_nodes_x:
        for y in local_nodes_y:
            near_main = any(abs(x - mx) < 100 or abs(y - my) < 100 for mx in main_roads_x for my in main_roads_y)
            if near_main and random.random() < 0.7:
                lat = 51.51 - y / SCALE
                lon = x / SCALE - 0.1
                graph.add_node(node_id, lat, lon)
                nodes[node_id] = (x, y)

                parent = min(nodes.keys(), key=lambda nid: math.sqrt((nodes[nid][0] - x) ** 2 + (nodes[nid][1] - y) ** 2))
                add_road(node_id, parent, "residential", 5.6, True)

                node_id += 1

    return graph


def load_or_fetch_graph(cache_file: str = "osm_cache.json") -> OSMGraph:
    """Load from cache or fetch from Overpass API."""
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
        print("Loaded OSM data from cache")
        return parse_osm_data(data)
    except:
        pass

    print("Fetching OSM data from Overpass API...")
    data = fetch_osm_data()
    if data and len(data.get('elements', [])) > 100:
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            print("Cached OSM data")
        except:
            pass
        return parse_osm_data(data)

    print("Warning: Using realistic procedural city grid")
    return create_realistic_city_grid()
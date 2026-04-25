"""Configuration for full traffic simulation."""

SCALE = 111000
FPS = 60
SIM_DT = 1.0 / FPS

SCREEN_W = 1400
SCREEN_H = 900

# Vehicle dimensions
VEHICLE_LENGTH = 4.0  # meters
VEHICLE_WIDTH = 1.8  # meters

# Road geometry
LANE_WIDTH = 3.5  # meters (standard)
ROAD_MARKING_WIDTH = 0.15  # meters

# Hub parameters
HUB_RADIUS = 150  # meters
HUB_BANDWIDTH = 100.0  # Mbps
DATA_GEN_RATE = 10.0 / 60.0  # MB per second

# IDM Car-Following Parameters
IDM_A_MAX = 2.5      # Maximum acceleration (m/s^2)
IDM_B = 3.0          # Comfortable deceleration (m/s^2)
IDM_V0 = 33.3        # Desired velocity (m/s) = 120 km/h
IDM_S0 = 2.0         # Minimum gap (m)
IDM_T = 1.5          # Time headway (s)

# MOBIL Lane Changing
MOBIL_P = 0.1        # Politeness factor (0-1, lower = more aggressive)
MOBIL_THRESHOLD = 0.01  # Minimum advantage to trigger lane change
MOBIL_B_SAFE = 5.0    # Safe deceleration threshold

# Traffic Simulation Parameters
SPAWN_RATE = 0.8  # vehicles per second
MAX_VEHICLES = 400  # MORE vehicles for congestion
QUEUE_LENGTH_MAX = 60  # max vehicles per lane
STOP_LINE_DISTANCE = 35  # meters before intersection
YIELD_DISTANCE = 25     # meters to check for yielding

# Turn behavior
TURN_SPEED_FACTOR_STRAIGHT = 1.0
TURN_SPEED_FACTOR_SLIGHT = 0.8
TURN_SPEED_FACTOR_MODERATE = 0.6
TURN_SPEED_FACTOR_SHARP = 0.4

# Right-of-way
STOP_SIGN_PENALTY = 0.3  # Time penalty for stop sign
YIELD_CHECK_DISTANCE = 50  # meters to check for yielding vehicles

# Visualization
HUB_PULSE_SPEED = 0.08  # Pulse animation speed
HUB_GLOW_INTENSITY = 60  # Glow strength
VEHICLE_SPAWN_RATE = 3  # Vehicles to spawn per interval

# Speed limits by road type (m/s)
SPEED_LIMITS = {
    'motorway': 27.8,      # 100 km/h
    'trunk': 22.2,         # 80 km/h
    'primary': 16.7,       # 60 km/h
    'secondary': 13.9,     # 50 km/h
    'tertiary': 11.1,      # 40 km/h
    'residential': 8.3,    # 30 km/h
    'service': 5.6,        # 20 km/h
}

# Jam density (vehicles per km per lane)
JAM_DENSITY = {
    'motorway': 120,
    'trunk': 130,
    'primary': 140,
    'secondary': 150,
    'tertiary': 160,
    'residential': 170,
    'service': 180,
}

# Colors
BLACK = (20, 20, 30)
GRAY = (60, 60, 70)
LIGHT_GRAY = (120, 120, 130)
WHITE = (220, 220, 230)
GREEN = (50, 200, 100)
YELLOW = (255, 255, 80)
ORANGE = (255, 150, 50)
RED = (255, 60, 60)
CYAN = (80, 200, 220)
BLUE = (80, 120, 255)
PURPLE = (180, 80, 200)
PINK = (255, 100, 150)

ROAD_COLORS = {
    "motorway": (60, 80, 160),
    "trunk": (80, 100, 180),
    "primary": (180, 140, 100),
    "secondary": (160, 150, 120),
    "tertiary": (130, 130, 130),
    "residential": (100, 100, 100),
    "service": (80, 80, 80),
}

# OSM bounds for F-6, Islamabad (small area ~400x300 pixels)
OSM_BOUNDS = {
    "min_lat": 33.697,
    "max_lat": 33.699,
    "min_lon": 73.038,
    "max_lon": 73.042,
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_QUERY = f"""
[out:json][timeout:60];
(
  way["highway"="motorway"]({OSM_BOUNDS['min_lat']},{OSM_BOUNDS['min_lon']},{OSM_BOUNDS['max_lat']},{OSM_BOUNDS['max_lon']});
  way["highway"="trunk"]({OSM_BOUNDS['min_lat']},{OSM_BOUNDS['min_lon']},{OSM_BOUNDS['max_lat']},{OSM_BOUNDS['max_lon']});
  way["highway"="primary"]({OSM_BOUNDS['min_lat']},{OSM_BOUNDS['min_lon']},{OSM_BOUNDS['max_lat']},{OSM_BOUNDS['max_lon']});
  way["highway"="secondary"]({OSM_BOUNDS['min_lat']},{OSM_BOUNDS['min_lon']},{OSM_BOUNDS['max_lat']},{OSM_BOUNDS['max_lon']});
  way["highway"="tertiary"]({OSM_BOUNDS['min_lat']},{OSM_BOUNDS['min_lon']},{OSM_BOUNDS['max_lat']},{OSM_BOUNDS['max_lon']});
  way["highway"="residential"]({OSM_BOUNDS['min_lat']},{OSM_BOUNDS['min_lon']},{OSM_BOUNDS['max_lat']},{OSM_BOUNDS['max_lon']});
  way["highway"="living_street"]({OSM_BOUNDS['min_lat']},{OSM_BOUNDS['min_lon']},{OSM_BOUNDS['max_lat']},{OSM_BOUNDS['max_lon']});
);
out body;
>;
out skel qt;
"""
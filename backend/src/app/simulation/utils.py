import math

def euclidean_dist(lat1, lon1, lat2, lon2):
    # Rough approximation for small distances, returning meters
    # 1 deg lat = 111,320m
    lat_mid = (lat1 + lat2) / 2.0
    dx = (lon2 - lon1) * 111320.0 * math.cos(math.radians(lat_mid))
    dy = (lat2 - lat1) * 111320.0
    return math.sqrt(dx*dx + dy*dy)

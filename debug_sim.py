"""Debug the visual simulation - check what loads."""
import sys

# Add verbose output
print("Starting debug...", flush=True)

try:
    import pygame
    print("Pygame imported", flush=True)
except Exception as e:
    print(f"Pygame error: {e}", flush=True)
    sys.exit(1)

# Test basic pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
print("Screen created", flush=True)

# Import main simulation components
print("Importing simulation...", flush=True)

# Import the key parts from main.py
import math
import random

# Simulate what load_real_osm_network does
import json
SCALE = 111000

try:
    with open("osm_cache.json", 'r') as f:
        data = json.load(f)
    print(f"Loaded cache: {len(data.get('elements', []))} elements", flush=True)
    
    # Parse nodes
    nodes = []
    for element in data.get('elements', []):
        if element.get('type') == 'node':
            lat = element['lat']
            lon = element['lon']
            x = (lon - 73.038) * SCALE * 10
            y = (33.699 - lat) * SCALE * 10
            nodes.append((x, y))
    
    print(f"Parsed {len(nodes)} nodes", flush=True)
    print(f"Node coordinates: min_x={min(n[0] for n in nodes):.0f}, max_x={max(n[0] for n in nodes):.0f}", flush=True)
    print(f"Node coordinates: min_y={min(n[1] for n in nodes):.0f}, max_y={max(n[1] for n in nodes):.0f}", flush=True)
    
except Exception as e:
    print(f"Error loading cache: {e}", flush=True)

# Draw something simple on screen
screen.fill((20, 20, 30))

# Draw a grid
for i in range(10):
    pygame.draw.line(screen, (60, 60, 70), (50 + i*70, 50), (50 + i*70, 550), 1)
for i in range(8):
    pygame.draw.line(screen, (60, 60, 70), (50, 50 + i*70), (650, 50 + i*70), 1)

# Draw a title
font = pygame.font.SysFont("Arial", 24)
text = font.render("Traffic Sim Debug", True, (80, 200, 220))
screen.blit(text, (280, 20))

# Draw nodes if we have them
if nodes:
    for x, y in nodes[:20]:  # First 20
        # Scale to screen
        sx = 50 + (x / 500) * 600
        sy = 50 + (y / 500) * 500
        pygame.draw.circle(screen, (255, 255, 80), (int(sx), int(sy)), 5)

pygame.display.flip()

# Wait for user to close
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

pygame.quit()
print("Done!", flush=True)
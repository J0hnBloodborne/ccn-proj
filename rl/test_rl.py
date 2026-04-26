"""Minimal RL test."""
from main import load_network, TrafficSimulator, Hub

print("Starting quick RL test...")

nodes, links = load_network()
print(f"Network loaded: {len(nodes)} nodes")

sim = TrafficSimulator(nodes, links)
print(f"Sim created: {len(sim.vehicles)} vehicles, {len(sim.hubs)} hubs")

# Quick run
print("Running 1 frame...")
sim.update()
print(f"After 1 frame: {len(sim.vehicles)} vehicles")

state = sim.get_state()
print(f"State: {state}")

print("Test PASSED!")

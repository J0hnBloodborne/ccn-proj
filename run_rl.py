"""Minimal RL training - just 5 frames, 3 episodes."""
from main import load_network, TrafficSimulator, Hub
import json
import os

LOG_DIR = "rl_logs"
os.makedirs(LOG_DIR, exist_ok=True)

print("Minimal RL training...")

nodes, links = load_network()
sim = TrafficSimulator(nodes, links)

rewards = []
for ep in range(1, 4):
    sim.hubs = []
    sim.next_hub_id = 0
    sim.vehicles = {}
    sim.events = []
    sim.time = 0
    sim.frame = 0
    sim.total_data_generated = 0
    sim.total_offloaded = 0
    
    # Place 5 random hubs
    for i in range(5):
        hub = Hub(sim.next_hub_id, float(i * 100), float(i * 50))
        hub.active = True
        hub.bandwidth = 100
        sim.hubs.append(hub)
        sim.next_hub_id += 1
    
    # Run 5 frames
    for _ in range(5):
        sim.update()
    
    state = sim.get_state()
    reward = state['offload_ratio'] * 100
    rewards.append(reward)
    print(f"Episode {ep}: reward={reward:.4f}")

# Save
log_file = f"{LOG_DIR}/mini_test.json"
with open(log_file, 'w') as f:
    json.dump({"episodes": 3, "rewards": rewards}, f)

print(f"Saved to {log_file}")

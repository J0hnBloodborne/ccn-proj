"""Quick RL test with fewer episodes."""
import json
import os
import numpy as np
from datetime import datetime

# Import from train_rl
from main import load_network, TrafficSimulator
from main import Hub

LOG_DIR = "rl_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Quick test config
QUICK_EPISODES = 20
NUM_HUBS = 20

print("Quick RL Test - 20 episodes only")

# Setup
nodes, links = load_network()
sim = TrafficSimulator(nodes, links)

# Get candidate locations
candidates = []
for node_id, node in sim.nodes.items():
    if len(node.out_links) >= 2:
        candidates.append((node_id, node.x, node.y))

print(f"Found {len(candidates)} candidate locations")

# Simple random policy for quick test
rewards = []

for ep in range(1, QUICK_EPISODES + 1):
    # Reset sim
    sim.hubs = []
    sim.next_hub_id = 0
    sim.vehicles = {}
    sim.events = []
    sim.time = 0
    sim.frame = 0
    sim.total_data_generated = 0
    sim.total_offloaded = 0
    
    # Place 20 random hubs
    selected = np.random.choice(len(candidates), NUM_HUBS, replace=False)
    for idx in selected:
        _, x, y = candidates[idx]
        hub = Hub(sim.next_hub_id, x, y)
        hub.active = True
        hub.bandwidth = 100
        sim.hubs.append(hub)
        sim.next_hub_id += 1
    
    # Simulate 60 seconds
    for _ in range(60 * 60):
        sim.update()
    
    # Get metrics
    state = sim.get_state()
    reward = state['offload_ratio'] * 100 - len(sim.hubs) * 0.5
    rewards.append(reward)
    
    print(f"Episode {ep}: Reward={reward:.2f}, Offload={state['offload_ratio']*100:.1f}%, Connected={state['connected']}")

# Save logs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"{LOG_DIR}/quick_test_{timestamp}.json"
with open(log_file, 'w') as f:
    json.dump({
        "episodes": QUICK_EPISODES,
        "rewards": rewards,
        "avg_reward": float(np.mean(rewards)),
        "max_reward": float(np.max(rewards))
    }, f, indent=2)

print(f"\nAvg reward: {np.mean(rewards):.2f}")
print(f"Max reward: {np.max(rewards):.2f}")
print(f"Logs saved to: {log_file}")

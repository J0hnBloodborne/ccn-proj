"""
PyTorch DQN Training for CCN2 Data Offloading
Optimized: 100 episodes, 10s each (~10 min total)
Fixed GPU tensor handling
"""

import pygame
import sys
import json
import random
import numpy as np
from datetime import datetime
from collections import deque
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import the simulator
from main import (
    TrafficSimulator, load_network, MAX_VEHICLES, EVENT_SPAWN_INTERVAL,
    HUB_BASE_BW, HUB_RADIUS, DATA_EVENT_MULT, DATA_BASE_RATE, SIM_DT, Hub
)

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# RL Hyperparameters
EPISODE_DURATION = 10.0  # seconds per episode (reduced)
NUM_EPISODES = 100
NUM_HUBS = 20
BATCH_SIZE = 32
MEMORY_SIZE = 10000
GAMMA = 0.95
LR = 0.001
EPSILON_START = 1.0
EPSILON_MIN = 0.1

# Logging
LOG_DIR = "rl_logs"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class DQN(nn.Module):
    """Deep Q-Network."""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Experience replay buffer."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class HubManagementEnv:
    """Gym-style environment for hub management."""
    
    def __init__(self, sim):
        self.sim = sim
        self.state_size = 8
        # Actions: 4 per hub (off, low, mid, high bandwidth)
        self.action_size = NUM_HUBS * 4
        
        # Pre-place hubs
        self._setup_hubs()
    
    def _setup_hubs(self):
        """Setup hubs at candidate locations."""
        self.sim.hubs = []
        self.sim.next_hub_id = 0
        
        # Get candidate locations
        candidates = []
        for node_id, node in self.sim.nodes.items():
            if len(node.out_links) >= 2:
                candidates.append((node_id, node.x, node.y))
        
        # Place NUM_HUBS at random locations
        if len(candidates) >= NUM_HUBS:
            selected = random.sample(candidates, NUM_HUBS)
            for _, x, y in selected:
                hub = Hub(self.sim.next_hub_id, x, y)
                hub.active = True
                hub.bandwidth = HUB_BASE_BW
                self.sim.hubs.append(hub)
                self.sim.next_hub_id += 1
    
    def reset(self):
        """Reset environment."""
        self._reset_sim()
        return self._get_state()
    
    def _reset_sim(self):
        """Reset simulation state."""
        # Clear vehicles but keep hubs
        self.sim.vehicles = {}
        self.sim.events = []
        self.sim.time = 0
        self.sim.frame = 0
        self.sim.total_data_generated = 0
        self.sim.total_offloaded = 0
        
        # Reset hub states
        for hub in self.sim.hubs:
            hub.active = True
            hub.bandwidth = HUB_BASE_BW
            hub.connected = []
    
    def _get_state(self):
        """Get current state as tensor."""
        state = self.sim.get_rl_state()
        return torch.FloatTensor(state).to(device)
    
    def step(self, actions):
        """Apply actions and simulate."""
        # Decode multi-hot action
        for i, hub in enumerate(self.sim.hubs):
            if i * 4 < len(actions):
                action_idx = actions[i * 4:(i + 1) * 4]
                action = torch.argmax(action_idx).item()
                
                if action == 0:
                    hub.active = False
                    hub.bandwidth = 0
                elif action == 1:
                    hub.active = True
                    hub.bandwidth = 50
                elif action == 2:
                    hub.active = True
                    hub.bandwidth = 100
                else:
                    hub.active = True
                    hub.bandwidth = 200
        
        # Record initial state
        initial_offload = self.sim.total_offloaded
        initial_data = self.sim.total_data_generated
        
        # Simulate episode
        frames = int(EPISODE_DURATION * 60)
        for _ in range(frames):
            self.sim.update()
        
        # Calculate reward
        metrics = self.sim.get_state()
        
        offloaded = self.sim.total_offloaded - initial_offload
        energy_penalty = sum(h.bandwidth for h in self.sim.hubs) / (NUM_HUBS * 200)
        
        reward = offloaded * 10 - energy_penalty * 2
        
        return self._get_state(), reward, False, metrics


def train():
    """Train DQN agent."""
    print("\n" + "=" * 60)
    print("PYTORCH DQN TRAINING FOR CCN2")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Episode duration: {EPISODE_DURATION}s")
    print(f"State size: 8, Action size: {NUM_HUBS * 4}")
    
    # Setup
    nodes, links = load_network()
    sim = TrafficSimulator(nodes, links)
    
    env = HubManagementEnv(sim)
    
    # Create networks
    policy_net = DQN(env.state_size, env.action_size).to(device)
    target_net = DQN(env.state_size, env.action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)
    
    epsilon = EPSILON_START
    rewards_history = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{LOG_DIR}/dqn_train_{timestamp}.jsonl"
    
    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        
        # Epsilon-greedy action
        if random.random() < epsilon:
            actions = torch.randn(env.action_size).to(device)
        else:
            with torch.no_grad():
                actions = policy_net(state)
        
        next_state, reward, done, metrics = env.step(actions.cpu())
        
        memory.push(state, actions, reward, next_state, done)
        
        # Train on GPU
        if len(memory) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            
            # Stack tensors on GPU
            states = torch.stack([b[0] for b in batch]).to(device)
            actions = torch.stack([b[1] for b in batch]).to(device)
            rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(device)
            next_states = torch.stack([b[3] for b in batch]).to(device)
            
            with torch.no_grad():
                next_q = target_net(next_states).max(1)[0]
                targets = rewards + GAMMA * next_q
            
            current_q = policy_net(states)
            loss = F.mse_loss(current_q.mean(), targets.mean())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epsilon = max(EPSILON_MIN, epsilon * 0.995)
        rewards_history.append(reward)
        
        # Log
        log_entry = {
            "episode": episode,
            "reward": reward,
            "offload_ratio": metrics.get('offload_ratio', 0),
            "connected": metrics.get('connected', 0),
            "epsilon": epsilon
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Progress
        if episode % 10 == 0:
            avg = np.mean(rewards_history[-10:])
            print(f"Episode {episode}/{NUM_EPISODES} | Reward: {reward:.2f} | Avg(10): {avg:.2f} | Eps: {epsilon:.3f}")
        
        # Update target network
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    # Save model
    model_file = f"{CHECKPOINT_DIR}/dqn_final_{timestamp}.pt"
    torch.save(policy_net.state_dict(), model_file)
    
    print(f"\nTraining complete!")
    print(f"Model saved: {model_file}")
    print(f"Log: {log_file}")
    print(f"Avg reward: {np.mean(rewards_history):.2f}")
    print(f"Max reward: {np.max(rewards_history):.2f}")
    
    return rewards_history


if __name__ == "__main__":
    train()

"""
Two-Stage RL Training for CCN2 Data Offloading
- Stage 1: Hub Placement RL (select K locations from N candidates)
- Stage 2: Hub Management RL (ON/OFF + bandwidth control)
"""

import pygame
import sys
import json
import random
import numpy as np
from datetime import datetime
from collections import deque
import os

# Import the simulator
from main import (
    TrafficSimulator, load_network, MAX_VEHICLES, EVENT_SPAWN_INTERVAL,
    HUB_BASE_BW, HUB_RADIUS, DATA_EVENT_MULT, DATA_BASE_RATE, SIM_DT
)

# RL Hyperparameters
EPISODE_DURATION = 60.0  # seconds per episode
NUM_EPISODES = 500
NUM_HUBS = 20
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.1
BATCH_SIZE = 32
MEMORY_SIZE = 10000
PLACEMENT_LR = 0.001
PLACEMENT_GAMMA = 0.95

# Logging
LOG_DIR = "rl_logs"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class RLLogger:
    """Logger for RL training metrics."""
    
    def __init__(self, stage_name):
        self.stage_name = stage_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"{LOG_DIR}/{stage_name}_{timestamp}.jsonl"
        self.episode_summaries = []
        self.best_reward = -float('inf')
        self.best_model = None
    
    def log_episode(self, episode, reward, metrics):
        """Log episode results."""
        summary = {
            "episode": episode,
            "reward": reward,
            **metrics,
            "timestamp": datetime.now().isoformat()
        }
        self.episode_summaries.append(summary)
        
        # Append to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(summary) + '\n')
        
        # Save best model
        if reward > self.best_reward:
            self.best_reward = reward
            self.save_checkpoint(episode, "best")
            print(f"  New best reward: {reward:.2f}")
    
    def save_checkpoint(self, episode, suffix):
        """Save model checkpoint."""
        checkpoint_file = f"{CHECKPOINT_DIR}/{self.stage_name}_ep{episode}_{suffix}.json"
        data = {
            "episode": episode,
            "best_reward": self.best_reward,
            "q_network": self.q_network.state_dict() if hasattr(self, 'q_network') else None
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f)
    
    def print_progress(self, episode, avg_reward, recent_rewards):
        """Print training progress."""
        recent_avg = np.mean(recent_rewards[-100:]) if len(recent_rewards) > 0 else 0
        print(f"  Episode {episode}/{NUM_EPISODES} | "
              f"Reward: {reward:.2f} | "
              f"Avg(100): {recent_avg:.2f} | "
              f"Best: {self.best_reward:.2f}")


class DQNAgent:
    """Simple DQN Agent for RL."""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        
        # Simple Q-network (in real implementation would use torch/tf)
        self.q_values = np.zeros((100, action_size))  # Discretized Q-table
        self.q_network = None  # Placeholder for actual network
        
        self.learning_rate = LEARNING_RATE
        self.gamma = GAMMA
    
    def act(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Simple argmax (would use neural network in real impl)
        state_idx = int(hash(str(state[:3])) % 100)
        return int(np.argmax(self.q_values[state_idx]))
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on batch of experiences."""
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        
        for state, action, reward, next_state, done in batch:
            state_idx = int(hash(str(state[:3])) % 100)
            if done:
                target = reward
            else:
                next_idx = int(hash(str(next_state[:3])) % 100)
                target = reward + self.gamma * np.max(self.q_values[next_idx])
            
            self.q_values[state_idx, action] += self.learning_rate * (target - self.q_values[state_idx, action])
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * 0.995)


class HubPlacementEnv:
    """Stage 1: Hub Placement RL Environment."""
    
    def __init__(self, sim):
        self.sim = sim
        self.candidate_locations = self._get_candidates()
        self.num_candidates = len(self.candidate_locations)
        self.num_hubs_to_place = NUM_HUBS
        
        self.state_size = self.num_candidates + 4  # candidates + metrics
        self.action_size = self.num_candidates  # Select which location to activate
    
    def _get_candidates(self):
        """Get candidate hub locations (intersection nodes)."""
        candidates = []
        for node_id, node in self.sim.nodes.items():
            if len(node.out_links) >= 2:
                candidates.append((node_id, node.x, node.y))
        return candidates
    
    def reset(self):
        """Reset environment."""
        self.placed_hubs = set()
        self._reset_hubs()
        return self._get_state()
    
    def _reset_hubs(self):
        """Clear all hubs."""
        self.sim.hubs = []
        self.sim.next_hub_id = 0
    
    def _get_state(self):
        """Get current state."""
        metrics = self.sim.get_state()
        state = np.zeros(self.state_size)
        
        # Mark placed hubs
        for hub_id in self.placed_hubs:
            if hub_id < self.num_candidates:
                state[hub_id] = 1.0
        
        # Add metrics
        state[self.num_candidates] = metrics['in_event'] / 50.0
        state[self.num_candidates + 1] = metrics['connected'] / 20.0
        state[self.num_candidates + 2] = metrics['offload_ratio']
        state[self.num_candidates + 3] = len(self.placed_hubs) / self.num_hubs_to_place
        
        return state
    
    def step(self, action):
        """Place hub at candidate location."""
        reward = 0
        done = False
        
        if action < self.num_candidates and action not in self.placed_hubs:
            if len(self.placed_hubs) < self.num_hubs_to_place:
                # Place hub
                node_id, x, y = self.candidate_locations[action]
                from main import Hub
                hub = Hub(self.sim.next_hub_id, x, y)
                self.sim.hubs.append(hub)
                self.sim.next_hub_id += 1
                self.placed_hubs.add(action)
                
                # Simulate briefly
                for _ in range(int(EPISODE_DURATION * 60)):
                    self.sim.update()
                
                # Calculate reward
                state = self._get_state()
                metrics = self.sim.get_state()
                reward = metrics['offload_ratio'] * 10 - len(self.placed_hubs) * 0.1
                
                if len(self.placed_hubs) >= self.num_hubs_to_place:
                    done = True
            else:
                done = True
        else:
            reward = -0.1  # Penalty for invalid action
        
        return self._get_state(), reward, done, {}
    
    def get_placed_locations(self):
        """Get indices of placed hub locations."""
        return list(self.placed_hubs)


class HubManagementEnv:
    """Stage 2: Hub Management RL Environment."""
    
    def __init__(self, sim, hub_locations):
        self.sim = sim
        self.hub_locations = hub_locations  # Pre-placed locations from Stage 1
        
        self.state_size = 8  # State features
        self.action_size = NUM_HUBS * 4  # 4 actions per hub (on/off + 2 bandwidth levels)
        
        # Setup pre-placed hubs
        self._setup_hubs()
    
    def _setup_hubs(self):
        """Setup hubs at pre-placed locations."""
        self.sim.hubs = []
        self.sim.next_hub_id = 0
        
        from main import Hub
        for idx, loc_idx in enumerate(self.hub_locations):
            from main import load_network
            # Use candidate locations
            candidates = []
            for node_id, node in self.sim.nodes.items():
                if len(node.out_links) >= 2:
                    candidates.append((node_id, node.x, node.y))
            
            if idx < len(candidates):
                _, x, y = candidates[loc_idx]
                hub = Hub(self.sim.next_hub_id, x, y)
                hub.active = False  # Start all off
                self.sim.hubs.append(hub)
                self.sim.next_hub_id += 1
    
    def reset(self):
        """Reset environment."""
        self._reset_hubs()
        
        # Reset sim for new episode
        self.sim.vehicles = {}
        self.sim.events = []
        self.sim.time = 0
        self.sim.frame = 0
        self.sim.total_data_generated = 0
        self.sim.total_offloaded = 0
        
        from main import Hub
        for hub in self.sim.hubs:
            hub.active = False
            hub.connected = []
        
        return self._get_state()
    
    def _reset_hubs(self):
        """Reset hub states."""
        for hub in self.sim.hubs:
            hub.active = False
            hub.connected = []
    
    def _get_state(self):
        """Get current state."""
        state = self.sim.get_rl_state()
        return np.array(state)
    
    def step(self, actions):
        """Apply hub management actions and simulate."""
        # Decode actions: for each hub, action determines state
        # Actions 0-3 per hub: 0=off, 1=low_bw, 2=mid_bw, 3=high_bw
        for i, hub in enumerate(self.sim.hubs):
            action = actions[i] if i < len(actions) else 0
            
            if action == 0:
                hub.active = False
                hub.bandwidth = 0
            elif action == 1:
                hub.active = True
                hub.bandwidth = 50
            elif action == 2:
                hub.active = True
                hub.bandwidth = 100
            elif action == 3:
                hub.active = True
                hub.bandwidth = 200
        
        # Simulate episode
        initial_offload = self.sim.total_offloaded
        initial_data = self.sim.total_data_generated
        
        for _ in range(int(EPISODE_DURATION * 60)):
            self.sim.update()
        
        # Calculate reward
        metrics = self.sim.get_state()
        
        offloaded = self.sim.total_offloaded - initial_offload
        energy_cost = sum(hub.bandwidth for hub in self.sim.hubs) / (NUM_HUBS * 200)
        offload_reward = offloaded * 10
        energy_penalty = energy_cost * 2
        
        reward = offload_reward - energy_penalty
        
        return self._get_state(), reward, False, metrics
    
    def get_hub_states(self):
        """Get current hub states for visualization."""
        return [(h.active, h.bandwidth) for h in self.sim.hubs]


def train_stage1():
    """Train Stage 1: Hub Placement RL."""
    print("\n" + "=" * 60)
    print("STAGE 1: HUB PLACEMENT RL")
    print("=" * 60)
    
    # Setup
    nodes, links = load_network()
    sim = TrafficSimulator(nodes, links)
    
    env = HubPlacementEnv(sim)
    agent = DQNAgent(env.state_size, env.action_size)
    logger = RLLogger("stage1_placement")
    
    rewards_history = []
    
    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        total_reward = 0
        
        for step in range(env.num_hubs_to_place):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        
        # Log and save
        metrics = {
            "hubs_placed": len(env.placed_hubs),
            "total_reward": total_reward,
            "epsilon": agent.epsilon
        }
        logger.log_episode(episode, total_reward, metrics)
        
        # Progress every 50 episodes
        if episode % 50 == 0:
            recent = np.mean(rewards_history[-50:])
            print(f"  Placement Progress: Episode {episode} | Avg Reward(50): {recent:.2f}")
    
    # Get final placements
    placements = env.get_placed_locations()
    print(f"\n  Final placements: {len(placements)} hubs")
    print(f"  Locations: {placements[:5]}... (showing first 5)")
    
    return placements


def train_stage2(hub_locations):
    """Train Stage 2: Hub Management RL."""
    print("\n" + "=" * 60)
    print("STAGE 2: HUB MANAGEMENT RL")
    print("=" * 60)
    
    # Setup
    nodes, links = load_network()
    sim = TrafficSimulator(nodes, links)
    
    env = HubManagementEnv(sim, hub_locations)
    agent = DQNAgent(env.state_size, env.action_size)
    logger = RLLogger("stage2_management")
    
    rewards_history = []
    
    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        total_reward = 0
        
        # Generate actions for all hubs
        actions = []
        for i in range(len(env.sim.hubs)):
            actions.append(agent.act(state))
        
        next_state, reward, done, metrics = env.step(actions)
        
        agent.remember(state, actions[0], reward, next_state, done)
        agent.replay()
        agent.decay_epsilon()
        
        total_reward = reward
        rewards_history.append(total_reward)
        
        # Log episode
        log_metrics = {
            "reward": total_reward,
            "offload_ratio": metrics.get('offload_ratio', 0),
            "active_hubs": metrics.get('active_hubs', 0),
            "connected": metrics.get('connected', 0),
            "epsilon": agent.epsilon
        }
        logger.log_episode(episode, total_reward, log_metrics)
        
        # Progress every 50 episodes
        if episode % 50 == 0:
            recent = np.mean(rewards_history[-50:])
            print(f"  Management Progress: Episode {episode} | Avg Reward(50): {recent:.2f}")
    
    # Get optimal hub states
    optimal_states = env.get_hub_states()
    print(f"\n  Optimal hub states learned")
    active_count = sum(1 for a, _ in optimal_states if a)
    print(f"  Active hubs: {active_count}/{len(optimal_states)}")
    
    return optimal_states


def save_training_summary(placements, management_states):
    """Save final training summary."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "stage1_placements": placements,
        "stage2_hub_states": [
            {"active": active, "bandwidth": bw} 
            for active, bw in management_states
        ],
        "num_hubs": NUM_HUBS,
        "episodes_stage1": NUM_EPISODES,
        "episodes_stage2": NUM_EPISODES,
        "episode_duration": EPISODE_DURATION
    }
    
    summary_file = f"{LOG_DIR}/training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  Summary saved to: {summary_file}")
    return summary_file


def main():
    """Main training function."""
    print("=" * 60)
    print("TWO-STAGE RL TRAINING FOR CCN2")
    print("=" * 60)
    print(f"Stage 1: Hub Placement ({NUM_HUBS} hubs from candidates)")
    print(f"Stage 2: Hub Management (on/off + bandwidth)")
    print(f"Episodes: {NUM_EPISODES} per stage")
    print(f"Episode Duration: {EPISODE_DURATION}s")
    
    # Stage 1: Learn optimal hub locations
    placements = train_stage1()
    
    # Stage 2: Learn optimal hub management
    management_states = train_stage2(placements)
    
    # Save final summary
    summary_file = save_training_summary(placements, management_states)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Logs saved to: {LOG_DIR}/")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}/")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()

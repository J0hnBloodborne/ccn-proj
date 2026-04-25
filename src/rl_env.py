"""RL Environment for vehicular data offloading optimization."""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Optional, Any
import random

from src.simulator import Simulator
from src.config import HUB_RADIUS


class DataOffloadEnv(gym.Env):
    """
    RL environment for vehicular data offloading optimization.
    
    State: Vehicle positions, buffers, hub connections, network congestion
    Action: Hub bandwidth allocation levels (per hub)
    Reward: Data offloaded - latency penalty - congestion cost
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        self.config = config or {}
        
        # Number of hubs (action space dimension)
        self.num_hubs = self.config.get("num_hubs", 15)
        
        # State dimensions
        # Vehicle states: speed, buffer, connection status, position
        self.max_vehicles = self.config.get("max_vehicles", 50)
        self.state_dim = self.max_vehicles * 4 + self.num_hubs * 2  # vehicle features + hub features
        
        # Action: hub bandwidth levels (0-1 normalized)
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.num_hubs,), dtype=np.float32
        )
        
        # Observation: vehicle speeds, buffers, connections, congestion
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )
        
        self.simulator: Optional[Simulator] = None
        self.episode_steps = 0
        self.max_steps = 1000
        
        # Reward tracking
        self.total_offloaded = 0.0
        self.total_latency = 0.0
        self.total_congestion = 0.0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Create new simulator
        from src.simulator import Simulator
        self.simulator = Simulator()
        
        # Limit hubs to match action space
        self.simulator.hubs = self.simulator.hubs[:self.num_hubs]
        
        self.episode_steps = 0
        self.total_offloaded = 0.0
        self.total_latency = 0.0
        self.total_congestion = 0.0
        
        state = self._get_state()
        info = self._get_info()
        
        return state, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return next state."""
        # Apply action: scale hub bandwidths
        self._apply_action(action)
        
        # Update simulation
        for _ in range(10):  # Run 10 steps per RL step
            self.simulator.update()
        
        # Get new state
        state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Track metrics
        state_dict = self.simulator.get_state()
        self.total_offloaded += state_dict.get("total_offloaded", 0)
        self.total_congestion += state_dict.get("congestion", 0)
        
        self.episode_steps += 1
        
        # Check termination
        terminated = self.episode_steps >= self.max_steps
        truncated = False
        
        info = self._get_info()
        
        return state, reward, terminated, truncated, info
    
    def _apply_action(self, action: np.ndarray):
        """Apply action to modify hub bandwidths."""
        action = np.clip(action, 0.0, 1.0)
        
        for i, hub in enumerate(self.simulator.hubs):
            if i < len(action):
                # Scale bandwidth between 20% and 100%
                hub.bandwidth = 20.0 + action[i] * 80.0
    
    def _get_state(self) -> np.ndarray:
        """Get current state as normalized vector."""
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        if self.simulator is None:
            return state
        
        vehicles = list(self.simulator.graph.vehicles.values())
        
        # Vehicle features (up to max_vehicles)
        for i, vehicle in enumerate(vehicles[:self.max_vehicles]):
            idx = i * 4
            
            # Speed (normalized to 30 m/s max)
            state[idx] = min(vehicle.v / 30.0, 1.0)
            
            # Buffer (normalized to 100 MB max)
            state[idx + 1] = min(vehicle.data_buffer / 100.0, 1.0)
            
            # Connection status (0 or 1)
            state[idx + 2] = 1.0 if vehicle.connected_hub else 0.0
            
            # Distance to nearest hub (normalized to 200m max)
            wx, wy = vehicle.get_world_pos()
            min_dist = min(
                np.sqrt((hub.x - wx)**2 + (hub.y - wy)**2)
                for hub in self.simulator.hubs
            )
            state[idx + 3] = min(min_dist / 200.0, 1.0)
        
        # Hub features
        hub_offset = self.max_vehicles * 4
        for i, hub in enumerate(self.simulator.hubs[:self.num_hubs]):
            idx = hub_offset + i * 2
            
            # Number of connections (normalized to 10 max)
            state[idx] = min(len(hub.connected) / 10.0, 1.0)
            
            # Current bandwidth allocation (normalized)
            state[idx + 1] = hub.bandwidth / 100.0
        
        # Network congestion as additional feature
        state[-1] = self.simulator.graph.get_congestion_level() if hasattr(self.simulator.graph, 'get_congestion_level') else 0.0
        
        return state
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on offloading success."""
        state = self.simulator.get_state()
        
        # Data offloading reward
        offloaded = state.get("total_offloaded", 0) - self.total_offloaded
        offload_reward = offloaded * 0.1
        
        # Latency penalty (vehicles with high buffers)
        avg_buffer = state.get("avg_buffer", 0)
        latency_penalty = -avg_buffer * 0.05
        
        # Congestion penalty
        congestion = state.get("congestion", 0)
        congestion_penalty = -congestion * 2.0
        
        # Connection reward
        connected_ratio = state.get("connected", 0) / max(state.get("vehicles", 1), 1)
        connection_reward = connected_ratio * 0.5
        
        total_reward = offload_reward + latency_penalty + congestion_penalty + connection_reward
        
        return total_reward
    
    def _get_info(self) -> Dict:
        """Get additional info dict."""
        if self.simulator is None:
            return {}
        
        state = self.simulator.get_state()
        return {
            "vehicles": state.get("vehicles", 0),
            "connected": state.get("connected", 0),
            "avg_speed": state.get("avg_speed", 0),
            "congestion": state.get("congestion", 0),
            "total_offloaded": state.get("total_offloaded", 0),
            "total_generated": state.get("total_generated", 0),
        }
    
    def render(self):
        """Render not implemented for this environment."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass


class MultiAgentOffloadEnv(gym.Env):
    """
    Multi-agent variant where each hub makes decisions independently.
    Each hub considers local vehicle distribution.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        self.config = config or {}
        self.num_agents = self.config.get("num_hubs", 15)
        
        # Each agent observes nearby vehicles and makes local decision
        obs_per_agent = 10 * 4 + 2  # 10 nearest vehicles + hub state
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_per_agent,), dtype=np.float32
        )
        
        # Each agent controls its own bandwidth
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        self.simulator: Optional[Simulator] = None
        self.episode_steps = 0
        self.max_steps = 1000
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        from src.simulator import Simulator
        self.simulator = Simulator()
        self.simulator.hubs = self.simulator.hubs[:self.num_agents]
        
        self.episode_steps = 0
        
        obs = self._get_all_observations()
        info = {}
        
        return obs, info
    
    def step(self, actions: Dict[int, float]) -> Tuple[Dict, Dict, bool, bool, Dict]:
        """Execute multi-agent actions."""
        # Apply actions to respective hubs
        for hub_idx, action in actions.items():
            if 0 <= hub_idx < len(self.simulator.hubs):
                bandwidth = 20.0 + action[0] * 80.0
                self.simulator.hubs[hub_idx].bandwidth = bandwidth
        
        # Update simulation
        for _ in range(10):
            self.simulator.update()
        
        # Get observations
        obs = self._get_all_observations()
        
        # Calculate rewards
        rewards = self._get_all_rewards(actions)
        
        self.episode_steps += 1
        terminated = self.episode_steps >= self.max_steps
        
        info = self._get_info()
        
        return obs, rewards, terminated, False, info
    
    def _get_all_observations(self) -> Dict[int, np.ndarray]:
        """Get observation for each agent (hub)."""
        obs = {}
        
        for i, hub in enumerate(self.simulator.hubs):
            local_obs = np.zeros(42, dtype=np.float32)  # 10 vehicles * 4 + hub state 2
            
            # Find nearest vehicles to this hub
            vehicles = list(self.simulator.graph.vehicles.values())
            vehicles.sort(key=lambda v: self._distance_to_hub(v, hub))
            
            for j, vehicle in enumerate(vehicles[:10]):
                idx = j * 4
                local_obs[idx] = min(vehicle.v / 30.0, 1.0)
                local_obs[idx + 1] = min(vehicle.data_buffer / 100.0, 1.0)
                local_obs[idx + 2] = 1.0 if vehicle.connected_hub == hub else 0.0
                local_obs[idx + 3] = self._distance_to_hub(vehicle, hub) / 200.0
            
            # Hub state
            local_obs[40] = len(hub.connected) / 10.0
            local_obs[41] = hub.bandwidth / 100.0
            
            obs[i] = local_obs
        
        return obs
    
    def _distance_to_hub(self, vehicle, hub) -> float:
        wx, wy = vehicle.get_world_pos()
        return np.sqrt((hub.x - wx)**2 + (hub.y - wy)**2)
    
    def _get_all_rewards(self, actions: Dict[int, float]) -> Dict[int, float]:
        """Get reward for each agent."""
        rewards = {}
        
        state = self.simulator.get_state()
        offloaded_delta = state.get("total_offloaded", 0)
        
        # Distribute reward based on hub contribution
        total_connections = sum(len(h.connected) for h in self.simulator.hubs)
        
        for i, hub in enumerate(self.simulator.hubs):
            # Reward based on connections and bandwidth usage
            connections = len(hub.connected)
            bandwidth_util = hub.bandwidth / 100.0
            
            reward = connections * 0.1 + bandwidth_util * 0.2
            
            # Penalty if this hub is underutilized
            if connections == 0 and hub.bandwidth > 50:
                reward -= 0.5
            
            rewards[i] = reward
        
        return rewards
    
    def _get_info(self) -> Dict:
        return self.simulator.get_state() if self.simulator else {}


def create_env(env_type: str = "single") -> gym.Env:
    """Factory function to create RL environment."""
    if env_type == "single":
        return DataOffloadEnv()
    elif env_type == "multi":
        return MultiAgentOffloadEnv()
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
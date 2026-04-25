"""
Quick verification test - runs a smaller simulation to show it works.
"""

from stable_baselines3 import PPO
from src.uxsim_env import DataOffloadingEnv

print("=" * 60)
print("Quick RL Training Test - Smaller Network")
print("=" * 60)

# Use smaller duration for quick test
env = DataOffloadingEnv(
    num_hubs=10,
    hub_radius=250.0,
    data_gen_rate=10.0 / 60.0,
    sim_duration=60.0,  # 60 seconds total
    demand_rate=0.2,
    seed=42
)

print("\nResetting environment...")
state, info = env.reset()
print(f"Network: {len(env.node_name_map)} nodes, {len(env.osm_loader.links)} links")
print(f"Initial state: {state}")
print(f"Info: {info}")

print("\nCreating PPO model...")
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=32,
    batch_size=16,
    n_epochs=5,
    gamma=0.99,
    verbose=1
)

print("\nTraining for 128 steps...")
model.learn(total_timesteps=128, progress_bar=True)

print("\nTraining complete!")

# Test the trained model
print("\n" + "=" * 60)
print("Testing Trained Policy")
print("=" * 60)

obs, info = env.reset()
total_reward = 0.0

for step in range(12):  # 12 steps * 5s = 60s
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if step % 4 == 0:
        print(f"[t={info['time']:.0f}s] Action: {action}, "
              f"Vehicles: {info['num_vehicles']}, "
              f"Connected: {info['connected_vehicles']}, "
              f"Offload: {info['offload_rate']:.1%}")
    
    if terminated:
        break

print(f"\nTotal episode reward: {total_reward:.3f}")
print(f"Final offload efficiency: {info['offload_rate']:.1%}")
print(f"Total data offloaded: {info['data_offloaded']:.1f} MB")

env.close()
print("\nQuick test complete!")
print("=" * 60)
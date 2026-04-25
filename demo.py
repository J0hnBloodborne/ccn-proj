"""
Minimal demonstration - trains and tests in under 60 seconds.
"""

from stable_baselines3 import PPO
from src.uxsim_env import DataOffloadingEnv

print("=" * 50)
print("Islamabad F-6 Data Offloading - RL Demo")
print("=" * 50)

env = DataOffloadingEnv(
    num_hubs=10,
    hub_radius=250.0,
    data_gen_rate=10.0 / 60.0,
    sim_duration=30.0,  # 30 seconds
    demand_rate=0.2,
    seed=42
)

print("\nResetting...")
state, info = env.reset()
print(f"Network: {len(env.node_name_map)} nodes, {len(env.osm_loader.links)} links")

print("\nTraining PPO (64 steps)...")
model = PPO("MlpPolicy", env, n_steps=32, batch_size=16, n_epochs=3, verbose=0)
model.learn(total_timesteps=64)

print("\nTesting trained policy...")
obs, info = env.reset()
total_reward = 0.0

for step in range(6):  # 6 steps * 5s = 30s
    action, _ = model.predict(obs)
    obs, reward, terminated, _, info = env.step(action)
    total_reward += reward
    print(f"  t={info['time']:.0f}s: reward={reward:.2f}, connected={info['connected_vehicles']}, offload={info['offload_rate']:.0%}")
    if terminated:
        break

print(f"\nTotal reward: {total_reward:.2f}")
print(f"Final offload efficiency: {info['offload_rate']:.0%}")
env.close()
print("\nDemo complete!")
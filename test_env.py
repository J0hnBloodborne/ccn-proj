"""Quick test of the environment."""
import sys

print("Step 1: Importing...", file=sys.stderr)
from src.uxsim_env import DataOffloadingEnv

print("Step 2: Creating environment...", file=sys.stderr)
env = DataOffloadingEnv(num_hubs=3, sim_duration=10, demand_rate=0.1, seed=42)

print("Step 3: Loading network...", file=sys.stderr)
env.W = env._load_network()

print("Step 4: Setup demand...", file=sys.stderr)
env._setup_demand()

print("Step 5: Init hubs...", file=sys.stderr)
env._initialize_hubs(0)

print("Step 6: Run simulation...", file=sys.stderr)
env.W.exec_simulation(0, 10)

print("Step 7: Results...", file=sys.stderr)
num_veh = len(env.W.VEHICLES)
active = sum(1 for v in env.W.VEHICLES.values() if v.link is not None)

print(f"Total vehicles: {num_veh}", file=sys.stderr)
print(f"Active vehicles: {active}", file=sys.stderr)
print(f"SUCCESS: Environment works!", file=sys.stderr)

env.close()
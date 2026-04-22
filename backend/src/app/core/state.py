from app.core.config import NUM_HUBS, NUM_VEHICLES

class SimulationState:
    def __init__(self):
        self.running = False
        self.step = 0
        self.algorithm = "greedy" # 'greedy' or 'predictive'
        self.total_generated_mb = 0.0
        self.total_offloaded_mb = 0.0
        self.total_buffer_overflows = 0
        self.priority_latency_sum = 0
        self.priority_events_completed = 0
        self.num_hubs = NUM_HUBS
        self.num_vehicles = NUM_VEHICLES
        self.event_rate = 0.0005
        self.graph = None
        self.hubs = []
        self.vehicles = []
        self.roads = []
        self.recent_events = []

# Single global instance
sim_state = SimulationState()

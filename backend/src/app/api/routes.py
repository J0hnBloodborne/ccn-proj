from fastapi import APIRouter
from app.core.state import sim_state
from app.models.schemas import ConfigState, ResetConfig
from app.simulation.engine import run_step, init_sim_entities

router = APIRouter()

@router.post("/start")
def start_sim():
    sim_state.running = True
    return {"status": "started"}

@router.post("/pause")
def pause_sim():
    sim_state.running = False
    return {"status": "paused"}

@router.post("/step")
def step_sim():
    run_step()
    return {"status": "stepped"}

@router.post("/algorithm")
def set_algorithm(config: ConfigState):
    sim_state.algorithm = config.algorithm
    return {"status": "success", "algorithm": sim_state.algorithm}

@router.post("/reset")
def reset_sim(config: ResetConfig):
    sim_state.num_hubs = config.num_hubs
    sim_state.num_vehicles = config.num_vehicles
    sim_state.event_rate = config.event_rate
    init_sim_entities()
    return {"status": "reset_successful"}

@router.get("/roads")
def get_roads():
    return sim_state.roads

@router.get("/state")
def get_state():
    return {
        "running": sim_state.running,
        "step": sim_state.step,
        "algorithm": sim_state.algorithm,
        "total_generated_mb": sim_state.total_generated_mb,
        "total_offloaded_mb": sim_state.total_offloaded_mb,
        "vehicles": sim_state.vehicles,
        "hubs": sim_state.hubs,
        "recent_events": sim_state.recent_events
    }

@router.get("/report")
def export_report():
    avg_latency = 0
    if sim_state.priority_events_completed > 0:
        avg_latency = sim_state.priority_latency_sum / sim_state.priority_events_completed
        
    hub_utilization = {f"Hub_{h['id']}": round(h["total_offloaded_mb"], 2) for h in sim_state.hubs}
    
    return {
        "algorithm": sim_state.algorithm,
        "total_steps": sim_state.step,
        "completed_events": sim_state.priority_events_completed,
        "average_priority_latency_steps": round(avg_latency, 2),
        "total_buffer_overflows": sim_state.total_buffer_overflows,
        "hub_utilization_mb": hub_utilization,
        "total_generated_mb": round(sim_state.total_generated_mb, 2),
        "total_offloaded_mb": round(sim_state.total_offloaded_mb, 2),
        "overall_efficiency": f"{round((sim_state.total_offloaded_mb / max(1, sim_state.total_generated_mb)) * 100, 2)}%"
    }

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import osmnx as ox

from app.api.routes import router as api_router
from app.core.state import sim_state
from app.core.config import LOCATION
from app.simulation.engine import init_sim_entities, simulation_loop

app = FastAPI(title="Simulation Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    print(f"Loading graph for {LOCATION}...")
    try:
        sim_state.graph = ox.graph_from_place(LOCATION, network_type="drive")
    except Exception as e:
        print("Failed to download exact place, using bounding box or default fallback...", e)
        sim_state.graph = ox.graph_from_point((33.6515, 73.0801), dist=2500, network_type="drive")
    
    # Extract roads for frontend rendering
    roads = []
    for u, v, data in sim_state.graph.edges(data=True):
        if 'geometry' in data:
            coords = list(data['geometry'].coords)
            roads.append([[pt[1], pt[0]] for pt in coords])
        else:
            u_node = sim_state.graph.nodes[u]
            v_node = sim_state.graph.nodes[v]
            roads.append([[u_node['y'], u_node['x']], [v_node['y'], v_node['x']]])
    sim_state.roads = roads
    
    init_sim_entities()

    print("Simulation initialized.")
    asyncio.create_task(simulation_loop())

if __name__ == "__main__":
    import uvicorn
    # Make sure to run the file inside `src/` to correctly access `app.main:app` module.
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

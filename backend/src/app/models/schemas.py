from pydantic import BaseModel

class ConfigState(BaseModel):
    algorithm: str

class ResetConfig(BaseModel):
    num_hubs: int
    num_vehicles: int
    event_rate: float

import asyncio
import logging
from fastapi import FastAPI
from app.rl.train import start_training_loop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RL Training Headless Space")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting headless RL training server...")
    # Launch the RL training loop as a background task
    asyncio.create_task(start_training_loop())

@app.get("/health")
def health_check():
    """
    Minimal health check endpoint required by Hugging Face Spaces on port 7860.
    """
    return {"status": "ok", "message": "Training is running"}

if __name__ == "__main__":
    import uvicorn
    # Hugging Face spaces expect port 7860
    uvicorn.run("app.train_main:app", host="0.0.0.0", port=7860)

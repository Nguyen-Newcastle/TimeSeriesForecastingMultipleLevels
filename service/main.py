from fastapi import FastAPI, BackgroundTasks
from models import (
    processing_and_training_pipeline_for_easy_level,
    processing_and_training_pipeline_for_medium_level,
    processing_and_training_pipeline_for_hard_level,
)
import train_easy_model, train_medium_model, train_hard_model

app = FastAPI()

@app.post("/process/easy")
async def process_easy():
    train_easy_model.main()  # Directly calling the function
    return {"message": "Easy level processing completed."}

@app.post("/process/medium")
async def process_medium():
    train_medium_model.main()  # Directly calling the function
    return {"message": "Medium level processing completed."}

@app.post("/process/hard")
async def process_hard():
    train_hard_model.main()  # Directly calling the function
    return {"message": "Hard level processing completed."}

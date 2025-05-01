import enum
import math
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ScamLLM model pipeline
classifier = pipeline(
    task="text-classification",
    model="phishbot/ScamLLM",
    top_k=None,
    device=(0 if torch.cuda.is_available() else -1)
)

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    is_phishing: bool
    score: float

@app.get("/")
async def root():
    return {"message": "Welcome to the Phishing Detection API!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(input: TextInput):
    result = classifier(input.text)[0]
    is_phishing = result[1]['score'] > 0.5
    score = result[1]['score'] if is_phishing else 1 - result[1]['score']
    result = PredictionOutput(is_phishing=is_phishing, score=score)
    return JSONResponse(
        content=jsonable_encoder(result),
        status_code=200,
    )
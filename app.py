import enum
import math
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import pipeline
from model import load_and_infer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    prediction, probability = load_and_infer(input.text)

    is_phishing = True if prediction == 'Spam' else False   
    result = PredictionOutput(is_phishing=is_phishing, score=probability)
    return JSONResponse(
        content=jsonable_encoder(result),
        status_code=200,
    )
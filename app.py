from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from contextlib import asynccontextmanager

import os

# Model path
MODEL_PATH = './model'

# Global model/tokenizer
model = None
tokenizer = None

# Lifespan context for model loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        # Force use of slow tokenizer to avoid protobuf dependency issues
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError("Model loading failed")

    yield  # App runs here

app = FastAPI(
    title="Nepali Sentiment Analysis API",
    description="API for analyzing sentiment in Nepali text using BERT",
    version="1.0.0",
    lifespan=lifespan
)

# Set up the templates
templates = Jinja2Templates(directory="templates")

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    score: float

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    try:
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)
            # Apply softmax to get proper probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities).item()

        sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
        sentiment = sentiment_map[prediction.item()]

        return SentimentResponse(
            text=request.text,
            sentiment=sentiment,
            score=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

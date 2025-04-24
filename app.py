from fastapi import FastAPI, HTTPException
from fastapi import Request
from pydantic import BaseModel
import torch
from transformers import BertForSequenceClassification, AutoTokenizer
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
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
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

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    score: float

@app.get("/")
async def root():
    return {
        "message": "Nepali Sentiment Analysis API",
        "usage": "POST /predict with JSON body: {'text': 'तपाईंको फिल्म राम्रो छ'}"
    }

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
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(outputs.logits, dim=1)
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

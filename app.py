"""
Nepali Sentiment Analysis API
Run with: uvicorn app:app --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertForSequenceClassification, AutoTokenizer
import os

# Initialize FastAPI app
app = FastAPI(
    title="Nepali Sentiment Analysis API",
    description="API for analyzing sentiment in Nepali text using BERT",
    version="1.0.0"
)

# Model loading
MODEL_PATH = './trained_model'

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    score: float

# Global variables for model and tokenizer
model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, tokenizer
    try:
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        model.eval()  # Set to evaluation mode
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Model loading failed")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Nepali Sentiment Analysis API",
        "usage": "POST /predict with JSON body: {'text': 'तपाईंको फिल्म राम्रो छ'}"
    }

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    """
    Predict sentiment for Nepali text

    Args:
        request (SentimentRequest): Request body containing text

    Returns:
        SentimentResponse: Prediction results
    """
    try:
        # Tokenize input
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(outputs.logits, dim=1)
            confidence = torch.max(probabilities).item()

        # Map prediction to sentiment
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
    """Health check endpoint"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}
#!/usr/bin/env python3

"""
Nepali Sentiment Analysis Inference Script
Usage: python predict.py "तपाईंको फिल्म राम्रो छ"
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./model"  # Update this if your model is in a different path


def load_model(model_path=MODEL_PATH):
    """Load the fine-tuned model and tokenizer from disk"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)  # force slow tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load model from '{model_path}': {e}")
        sys.exit(1)


def predict_sentiment(text, model, tokenizer):
    """Return sentiment label for input Nepali text"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    sentiment_map = {
        0: "Negative",
        1: "Positive",
        2: "Neutral"
    }

    return sentiment_map.get(predicted_class, "Unknown")

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py \"Nepali sentence here\"")
        sys.exit(1)

    text = sys.argv[1]

    print("Loading model...")
    model, tokenizer = load_model()

    print("\nAnalyzing...")
    sentiment = predict_sentiment(text, model, tokenizer)

    print("\nResult:")
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {sentiment}")

if __name__ == "__main__":
    main()

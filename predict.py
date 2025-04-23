#!/usr/bin/env python3
"""
Nepali Sentiment Analysis Inference Script
Usage: python predict.py "तपाईंको फिल्म राम्रो छ"
"""

import sys
import torch
from transformers import BertForSequenceClassification, AutoTokenizer

def load_model(model_path='./trained_model'):
    """Load the trained model and tokenizer"""
    try:
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

def predict_sentiment(text, model, tokenizer):
    """Predict sentiment for given Nepali text"""
    # Prepare input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1)

    # Map prediction to sentiment
    sentiment_map = {0: "Negative 😞", 1: "Positive 😊", 2: "Neutral 😐"}
    return sentiment_map[prediction.item()]

def main():
    """Main function"""
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python predict.py \"तपाईंको फिल्म राम्रो छ\"")
        sys.exit(1)

    # Get input text
    text = sys.argv[1]

    # Load model
    print("Loading model...")
    model, tokenizer = load_model()

    # Make prediction
    print("\nAnalyzing text...")
    sentiment = predict_sentiment(text, model, tokenizer)

    # Print results
    print("\nResults:")
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
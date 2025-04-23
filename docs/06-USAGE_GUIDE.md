# Usage Guide - Nepali Sentiment Analysis 🚀

This guide explains how to use the trained Nepali sentiment analysis model for making predictions and integrating it into your applications.

## Quick Start 🏃‍♂️

### 1. Basic Usage
```python
from transformers import BertForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained('./trained_model')
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

# Prepare text
text = "तपाईंको फिल्म राम्रो छ"  # "Your movie is good"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Make prediction
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1)

# Map prediction to sentiment
sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
result = sentiment_map[prediction.item()]
```

### 2. Batch Processing
```python
def predict_batch(texts, model, tokenizer, device='cpu'):
    # Tokenize all texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    # Get predictions
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)

    return predictions.cpu().numpy()
```

## Integration Examples 💡

### 1. FastAPI Web Service
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class SentimentRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_sentiment(request: SentimentRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1)

    return {
        "text": request.text,
        "sentiment": sentiment_map[prediction.item()]
    }
```

### 2. Command Line Tool
```python
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description='Analyze sentiment of Nepali text'
    )
    parser.add_argument('text', help='Nepali text to analyze')
    args = parser.parse_args()

    # Process text
    inputs = tokenizer(
        args.text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1)

    print(f"Text: {args.text}")
    print(f"Sentiment: {sentiment_map[prediction.item()]}")

if __name__ == '__main__':
    main()
```

## Model Loading Options 🔄

### 1. Local Model
```python
# Load from local directory
model = BertForSequenceClassification.from_pretrained('./trained_model')
```

### 2. Compressed Model
```python
import zipfile

# Extract model from zip
with zipfile.ZipFile('my-model.zip', 'r') as zip_ref:
    zip_ref.extractall('extracted_model')

# Load extracted model
model = BertForSequenceClassification.from_pretrained('./extracted_model')
```

## Performance Optimization 🚀

### 1. GPU Acceleration
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### 2. Batch Processing for Speed
```python
# Process multiple texts at once
texts = [
    "पहिलो सिजन राम्रो थियो",
    "यो फिल्म मन परेन",
    "ठिकै छ"
]

# Tokenize batch
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# Get predictions
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=1)
```

## Best Practices 👌

### 1. Input Validation
```python
def validate_input(text):
    if not text or not isinstance(text, str):
        raise ValueError("Input must be a non-empty string")

    # Check for Nepali characters
    if not any('\u0900' <= char <= '\u097F' for char in text):
        raise ValueError("Text must contain Nepali characters")
```

### 2. Error Handling
```python
def safe_predict(text):
    try:
        # Validate input
        validate_input(text)

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Predict
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1)

        return sentiment_map[prediction.item()]

    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return None
```

### 3. Model Management
```python
def load_model_safely(model_path):
    try:
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None
```

## Common Issues and Solutions 🔧

### 1. Memory Issues
- Use batch processing for large datasets
- Clear GPU cache regularly
- Use model half-precision if needed

### 2. Performance Issues
- Use GPU when available
- Optimize batch size
- Cache tokenizer outputs

### 3. Text Processing
- Handle special characters
- Normalize text if needed
- Validate input length

## Examples by Use Case 📝

### 1. Social Media Analysis
```python
def analyze_social_posts(posts):
    results = []
    for post in posts:
        sentiment = safe_predict(post)
        results.append({
            'post': post,
            'sentiment': sentiment
        })
    return results
```

### 2. Customer Feedback Analysis
```python
def analyze_feedback(feedback_list):
    sentiments = predict_batch(feedback_list, model, tokenizer)

    # Count sentiments
    sentiment_counts = {
        'positive': (sentiments == 1).sum(),
        'neutral': (sentiments == 2).sum(),
        'negative': (sentiments == 0).sum()
    }

    return sentiment_counts
```

## API Reference 📚

### Model Methods
- `model.eval()`: Set model to evaluation mode
- `model.to(device)`: Move model to specific device
- `model.save_pretrained()`: Save model to directory

### Tokenizer Methods
- `tokenizer.encode()`: Convert text to tokens
- `tokenizer.decode()`: Convert tokens back to text
- `tokenizer.batch_encode_plus()`: Process multiple texts

## Resources 📚

### Documentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### Tutorials
- [BERT Fine-tuning Tutorial](https://www.youtube.com/watch?v=hinZO--TEk4)
- [Sentiment Analysis with BERT](https://www.youtube.com/watch?v=8N-nM3QW7O0)

### Community
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [PyTorch Forums](https://discuss.pytorch.org/)

## Next Steps 🎯

1. Explore [Theory and Concepts](07-THEORY_AND_CONCEPTS.md)
2. Review [Technical Details](03-TECHNICAL_DETAILS.md)
3. Check [Project Structure](02-PROJECT_STRUCTURE.md)
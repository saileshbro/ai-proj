# Model Training Guide 🚂

This guide provides a detailed walkthrough of the model training process for the Nepali Sentiment Analysis project, making it easy for undergraduate students to understand and replicate the training.

## Training Overview 📋

### Training Pipeline
```mermaid
graph LR
    A[Data Preparation] --> B[Model Setup]
    B --> C[Training Loop]
    C --> D[Evaluation]
    D --> E[Model Saving]
```

## 1. Environment Setup 🔧

### Hardware Requirements
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Memory Considerations
- Batch size: 16 samples
- Sequence length: 512 tokens
- Model size: ~1GB
- Required RAM: ~8GB minimum

## 2. Model Configuration ⚙️

### BERT Model Setup
```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=3  # For our three sentiment classes
)
```

### Why These Parameters?
- **bert-base-multilingual-cased**:
  - Pre-trained on 104 languages
  - Includes Nepali text support
  - Case-sensitive for better accuracy
- **num_labels=3**:
  - 0: Negative
  - 1: Positive
  - 2: Neutral

## 3. Training Configuration 🎛️

### Hyperparameters
```python
# Key training parameters
batch_size = 16
learning_rate = 2e-5
epochs = 10
max_length = 512
```

Why these values?
- **batch_size=16**:
  - Balances memory usage and training speed
  - Works well on most GPUs
  - Can be reduced to 8 if memory limited

- **learning_rate=2e-5**:
  - Standard for BERT fine-tuning
  - Prevents catastrophic forgetting
  - Stable training dynamics

- **epochs=10**:
  - Sufficient for convergence
  - Prevents overfitting
  - Can be adjusted based on validation performance

## 4. Training Loop Implementation 🔄

### Setting Up the Training Loop
```python
from tqdm import tqdm
import torch.nn as nn

# Training preparation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

# Initialize tracking
loss_values = []
accuracy_values = []
```

### Training Process
```python
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    # Progress bar
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}')

    for batch in progress_bar:
        # Move batch to device
        batch = tuple(t.to(device) for t in batch)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=batch[0],
            attention_mask=batch[1],
            labels=batch[2]
        )

        # Calculate loss
        loss = outputs.loss
        total_loss += loss.item()

        # Calculate accuracy
        predictions = torch.argmax(outputs.logits, dim=1)
        correct_predictions += (predictions == batch[2]).sum().item()
        total_predictions += batch[0].size(0)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update parameters
        optimizer.step()

        # Update progress bar
        progress_bar.set_postfix({'Training Loss': loss.item()})
```

## 5. Model Evaluation 📊

### Training Metrics
```python
def plot_training_metrics(loss_values, accuracy_values):
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), loss_values, 'b-')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), accuracy_values, 'r-')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
```

### Interpreting Results
- Loss should decrease over time
- Accuracy should increase
- Watch for:
  - Sudden spikes (learning rate too high)
  - Plateaus (learning saturation)
  - Oscillations (unstable training)

## 6. Model Saving and Loading 💾

### Saving the Model
```python
# Save the model
output_dir = './trained_model'
model.save_pretrained(output_dir)

# Create distributable version
import shutil
shutil.make_archive("my-model", 'zip', "my-model")
```

### Loading for Inference
```python
# Load the model
loaded_model = BertForSequenceClassification.from_pretrained('./trained_model')
loaded_model.eval()  # Set to evaluation mode
```

## 7. Troubleshooting Guide 🔍

### Common Issues

1. **Out of Memory (OOM)**
```python
# Solutions:
# 1. Reduce batch size
batch_size = 8  # Instead of 16

# 2. Reduce sequence length
max_length = 256  # Instead of 512

# 3. Enable gradient accumulation
accumulation_steps = 2
```

2. **Slow Training**
```python
# 1. Check if using GPU
print(f"Using GPU: {next(model.parameters()).is_cuda}")

# 2. Optimize data loading
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=4  # Parallel data loading
)
```

3. **Poor Performance**
```python
# 1. Learning rate adjustment
learning_rate = 1e-5  # Try smaller learning rate

# 2. Increase epochs
epochs = 15  # Train for longer

# 3. Add weight decay
optimizer = AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.01
)
```

## 8. Performance Optimization 🚀

### Memory Optimization
```python
# Clear cache between epochs
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### Training Speed
```python
# Enable cuDNN autotuner
torch.backends.cudnn.benchmark = True
```

### Model Size
```python
def get_model_size():
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
```

## Resources 📚

### PyTorch & Training
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [BERT Fine-tuning Tutorial](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)

### Videos
- [Training Neural Networks](https://www.youtube.com/watch?v=q8SA3rM6ckI)
- [PyTorch Tutorial for Beginners](https://www.youtube.com/watch?v=QyUTJxCQHow)

### Papers
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Fine-tuning Best Practices](https://arxiv.org/abs/1905.05583)

## Next Steps 🎯

1. Try the model with the [Usage Guide](06-USAGE_GUIDE.md)
2. Explore [Theory and Concepts](07-THEORY_AND_CONCEPTS.md)
3. Contribute improvements to the project
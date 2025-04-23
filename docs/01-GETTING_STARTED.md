# Getting Started with Nepali Sentiment Analysis 🚀

This guide will help you set up and run the Nepali Sentiment Analysis project. We'll go through each step in detail, explaining what each component does and why it's needed.

## Prerequisites 📋

### 1. Python Environment (Python 3.7+)
Python is our main programming language. We use version 3.7 or higher because:
- It supports all required machine learning libraries
- Has improved type hinting features
- Better async/await support
- [Learn Python basics here](https://www.learnpython.org/)

### 2. Understanding of Core Libraries

#### PyTorch 🔥
- Deep learning framework used for model training
- [PyTorch Tutorial for Beginners](https://pytorch.org/tutorials/beginner/basics/intro.html)
- [PyTorch Video Course](https://www.youtube.com/watch?v=QyUTJxCQHow)

#### Transformers 🤗
- Hugging Face's library for state-of-the-art NLP
- Provides pre-trained BERT models
- [Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Transformers Course](https://huggingface.co/course/chapter1/1)

#### Pandas 🐼
- Data manipulation and analysis library
- Used for dataset handling
- [Pandas in 10 Minutes](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html)
- [Pandas Video Tutorial](https://www.youtube.com/watch?v=vmEHCJofslg)

## Installation Steps 💻

1. **Create a Virtual Environment**
   ```bash
   # Create a new virtual environment
   python -m venv venv

   # Activate the environment
   # On Windows:
   venv\\Scripts\\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install Required Packages**
   ```bash
   # Install PyTorch
   # For CPU only:
   pip install torch torchvision torchaudio
   # For CUDA support (GPU):
   # Visit https://pytorch.org/ for the correct command for your system

   # Install other dependencies
   pip install transformers pandas numpy matplotlib tqdm scikit-learn
   ```

3. **Verify Installation**
   ```python
   # Run these commands in Python to verify installation
   import torch
   import transformers
   import pandas as pd
   print(f"PyTorch version: {torch.__version__}")
   print(f"Transformers version: {transformers.__version__}")
   print(f"Pandas version: {pd.__version__}")
   ```

## Project Setup 🛠️

1. **Clone the Repository**
   ```bash
   git clone [your-repo-url]
   cd [repository-name]
   ```

2. **Download Pre-trained Model (Optional)**
   - Download from [HuggingFace Hub](https://huggingface.co/bert-base-multilingual-cased)
   - Or use the provided trained model

3. **Prepare Your Data**
   - Place your training data in CSV format
   - Required columns: 'text' (Nepali text) and 'label' (0, 1, or 2)

## Hardware Requirements 💽

### Minimum Requirements
- 8GB RAM
- Any modern CPU
- 10GB free disk space

### Recommended Requirements
- 16GB RAM
- NVIDIA GPU with 6GB+ VRAM
- CUDA-compatible system
- 20GB+ free disk space

## Common Issues and Solutions 🔧

### CUDA Out of Memory
```python
# Reduce batch size in model_training.ipynb
batch_size = 8  # Default is 16
```

### CPU/GPU Selection
```python
# Force CPU usage
device = torch.device('cpu')

# Check GPU availability
print(torch.cuda.is_available())
```

## Additional Resources 📚

### Nepali Language Resources
- [Nepali Word Embeddings](https://github.com/rabindralamsal/Word2Vec-Embeddings-for-Nepali-Language)
- [Nepali NLP Tools](https://github.com/sushil79g/Nepali_nlp)

### Deep Learning Resources
- [Deep Learning with PyTorch](https://pytorch.org/tutorials/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Sentiment Analysis Basics](https://towardsdatascience.com/sentiment-analysis-concept-analysis-and-applications-6c94d6f58c17)

## Next Steps 👣

After setting up, proceed to:
1. [Project Structure](02-PROJECT_STRUCTURE.md) to understand the codebase
2. [Dataset Guide](04-DATASET_GUIDE.md) to learn about data preparation
3. [Model Training](05-MODEL_TRAINING.md) to start training your model

## Need Help? 🤝

- Create an issue on GitHub
- Check our FAQ section
- Join our community Discord
- Email: [support-email]
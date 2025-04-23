#!/usr/bin/env python3
"""
Environment Setup Checker for Nepali Sentiment Analysis
This script validates your environment and helps identify potential issues.
"""

import sys
import torch
import transformers
import pandas as pd
import numpy as np
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("\n📌 Checking Python version...")
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    if sys.version_info >= (3, 7):
        print("✅ Python version is compatible")
    else:
        print("❌ Python 3.7 or higher is required")

def check_gpu():
    """Check GPU availability and CUDA setup"""
    print("\n📌 Checking GPU setup...")
    if torch.cuda.is_available():
        print("✅ CUDA is available")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        # Test GPU memory allocation
        try:
            test_tensor = torch.rand(1000, 1000).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("✅ GPU memory allocation test passed")
        except Exception as e:
            print(f"❌ GPU memory test failed: {str(e)}")
    else:
        print("ℹ️ No GPU detected - will use CPU (training will be slower)")

def check_libraries():
    """Check required libraries and versions"""
    print("\n📌 Checking required libraries...")

    libraries = {
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'pandas': pd.__version__,
        'numpy': np.__version__
    }

    for lib, version in libraries.items():
        print(f"✅ {lib}: {version}")

def check_data_files():
    """Check if required data files exist"""
    print("\n📌 Checking data files...")

    required_files = ['train.csv', 'test.csv']
    model_path = Path('./trained_model')

    for file in required_files:
        if Path(file).exists():
            print(f"✅ Found {file}")
            # Check file size
            size_mb = Path(file).stat().st_size / (1024 * 1024)
            print(f"  📊 Size: {size_mb:.2f} MB")
            # Check row count
            try:
                df = pd.read_csv(file)
                print(f"  📊 Rows: {len(df)}")
            except Exception as e:
                print(f"  ❌ Error reading {file}: {str(e)}")
        else:
            print(f"❌ Missing {file}")

    if model_path.exists() and model_path.is_dir():
        print("✅ Found trained_model directory")
        # Check model files
        model_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
        for file in model_files:
            if (model_path / file).exists():
                print(f"  ✅ Found {file}")
            else:
                print(f"  ❌ Missing {file}")
    else:
        print("ℹ️ No trained model found (needed for inference)")

def check_memory():
    """Check available system memory"""
    print("\n📌 Checking system memory...")
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Total Memory: {memory.total / (1024**3):.2f} GB")
        print(f"Available Memory: {memory.available / (1024**3):.2f} GB")
        if memory.available / (1024**3) < 8:
            print("⚠️ Warning: Less than 8GB available memory might cause issues")
        else:
            print("✅ Sufficient memory available")
    except ImportError:
        print("ℹ️ Could not check memory (psutil not installed)")

def main():
    """Run all checks"""
    print("🔍 Starting Environment Check")
    print("=" * 50)

    check_python_version()
    check_gpu()
    check_libraries()
    check_data_files()
    check_memory()

    print("\n" + "=" * 50)
    print("✨ Environment check completed!")

if __name__ == "__main__":
    main()
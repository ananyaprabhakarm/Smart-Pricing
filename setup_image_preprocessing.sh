#!/bin/bash

# Image Preprocessing Pipeline Setup Script
# This script sets up the environment for running the image preprocessing pipeline

echo "Setting up Image Preprocessing Pipeline for ML Challenge 2025..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed. Please install pip3."
    exit 1
fi

echo "✓ pip3 found"

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Python dependencies installed successfully"
else
    echo "Error: Failed to install Python dependencies"
    exit 1
fi

# Test PyTorch installation
echo "Testing PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
if [ $? -eq 0 ]; then
    echo "✓ PyTorch is working"
else
    echo "Error: PyTorch test failed"
    exit 1
fi

# Test Torchvision
echo "Testing Torchvision..."
python3 -c "import torchvision; print(f'Torchvision version: {torchvision.__version__}')"
if [ $? -eq 0 ]; then
    echo "✓ Torchvision is working"
else
    echo "Error: Torchvision test failed"
    exit 1
fi

# Test Transformers
echo "Testing Transformers..."
python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
if [ $? -eq 0 ]; then
    echo "✓ Transformers is working"
else
    echo "Error: Transformers test failed"
    exit 1
fi

# Test Scikit-learn
echo "Testing Scikit-learn..."
python3 -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"
if [ $? -eq 0 ]; then
    echo "✓ Scikit-learn is working"
else
    echo "Error: Scikit-learn test failed"
    exit 1
fi

# Create output directories
echo "Creating output directories..."
mkdir -p ../image_features/images
echo "✓ Output directories created"

# Check GPU availability
echo "Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA GPU available: {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠ No CUDA GPU detected - will use CPU (slower)')
"

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "To run the Image Preprocessing Pipeline:"
echo "  cd src"
echo "  python3 image_preprocessing.py          # Process 1000 images"
echo "  python3 test_image_preprocessing.py     # Test with 10 images"
echo "  python3 run_image_preprocessing.py 500 0  # Process 500 images starting from index 0"
echo ""
echo "Results will be saved in: ../image_features/"
echo "=========================================="

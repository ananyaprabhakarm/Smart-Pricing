#!/bin/bash

# Image Preprocessing Pipeline Setup Script
# This script sets up the environment for running the image preprocessing pipeline

echo "Setting up Image Preprocessing Pipeline for ML Challenge 2025..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.7 or higher."
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

# Check if Tesseract is installed
if ! command -v tesseract &> /dev/null; then
    echo "Warning: Tesseract OCR is not installed."
    echo "Please install Tesseract OCR:"
    echo "  Ubuntu/Debian: sudo apt-get install tesseract-ocr"
    echo "  macOS: brew install tesseract"
    echo "  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
    echo ""
    echo "The pipeline will not work without Tesseract OCR."
    exit 1
fi

echo "✓ Tesseract OCR found"

# Create output directories
echo "Creating output directories..."
mkdir -p ../ocr_output/images
echo "✓ Output directories created"

# Test Tesseract
echo "Testing Tesseract OCR..."
tesseract --version
if [ $? -eq 0 ]; then
    echo "✓ Tesseract OCR is working"
else
    echo "Error: Tesseract OCR test failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "To run the OCR pipeline:"
echo "  cd src"
echo "  python3 ocr_pipeline.py          # Process 1000 images"
echo "  python3 test_ocr.py              # Test with 10 images"
echo "  python3 run_ocr.py 500 0         # Process 500 images starting from index 0"
echo ""
echo "Results will be saved in: ../ocr_output/"
echo "=========================================="

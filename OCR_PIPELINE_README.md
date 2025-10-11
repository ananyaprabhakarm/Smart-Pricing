# OCR Pipeline for Product Image Text Extraction

This pipeline extracts text from product images using PyTesseract OCR for the ML Challenge 2025 Smart Product Pricing dataset.

## Features

- Downloads product images from URLs
- Extracts text using PyTesseract OCR
- Handles image preprocessing and error recovery
- Provides confidence scores and statistics
- Saves results in JSON format with detailed metadata
- Progress tracking and intermediate saves

## Installation

1. Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
cd src
python ocr_pipeline.py
```

### Custom Parameters
```bash
cd src
python run_ocr.py 1000 0  # Process 1000 images starting from index 0
```

### Programmatic Usage
```python
from ocr_pipeline import OCRPipeline

# Initialize pipeline
pipeline = OCRPipeline()

# Process first 1000 images
results = pipeline.process_images(num_images=1000, start_index=0)

# Save results
pipeline.save_results(results, "my_results.json")

# Generate summary report
summary = pipeline.create_summary_report(results)
```

## Output Structure

The pipeline creates the following outputs:

### Directory Structure
```
ocr_output/
├── images/                    # Downloaded images
├── ocr_results_final_1000_images.json  # Main results
├── ocr_summary_report.json    # Summary statistics
└── ocr_results_intermediate_*.json  # Intermediate saves
```

### Result Format
Each OCR result contains:
```json
{
  "sample_id": 12345,
  "extracted_text": "Product Name and Description...",
  "confidence_score": 85.5,
  "word_count": 25,
  "status": "success",
  "image_dimensions": [800, 600],
  "image_url": "https://...",
  "catalog_content": "Original catalog text...",
  "price": 29.99,
  "image_path": "/path/to/image.jpg"
}
```

## Configuration

### OCRPipeline Parameters
- `dataset_folder`: Path to dataset folder (default: '../dataset/')
- `output_folder`: Path to save OCR results (default: '../ocr_output/')

### Process Parameters
- `num_images`: Number of images to process (default: 1000)
- `start_index`: Starting index in dataset (default: 0)
- `max_retries`: Maximum download retry attempts (default: 3)

## Performance Notes

- Images are downloaded with retry mechanism and exponential backoff
- Intermediate results are saved every 100 images
- Failed downloads/OCR are logged but don't stop the pipeline
- Memory usage is optimized by processing images individually

## Error Handling

The pipeline handles various error scenarios:
- Network timeouts and connection errors
- Invalid image URLs
- Corrupted image files
- OCR processing failures
- File system errors

All errors are logged with appropriate detail levels.

## Requirements

- Python 3.7+
- Tesseract OCR engine
- Internet connection for image downloads
- Sufficient disk space for images and results

## Troubleshooting

### Common Issues

1. **Tesseract not found**: Ensure Tesseract is installed and in PATH
2. **Permission errors**: Check write permissions for output directories
3. **Memory issues**: Process smaller batches of images
4. **Network timeouts**: Increase timeout values or retry counts

### Logging

The pipeline uses Python logging. To increase verbosity:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

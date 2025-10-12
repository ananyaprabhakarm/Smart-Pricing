# Image Preprocessing Pipeline for ML Challenge 2025

This pipeline implements comprehensive image preprocessing and feature extraction using state-of-the-art pretrained models for the Smart Product Pricing Challenge.

## Features

### 🔧 **Advanced Image Processing**
- **Automatic Download**: Downloads product images from URLs with retry mechanism
- **Standardized Resizing**: All images resized to 224×224 pixels
- **Robust Preprocessing**: Handles various image formats and quality issues
- **Batch Processing**: Efficient processing with configurable batch sizes

### 🤖 **Multiple Pretrained Models**
- **EfficientNet-B0**: Lightweight CNN with excellent accuracy-efficiency tradeoff
- **ResNet50**: Deep residual network with proven performance
- **ViT-B/16**: Vision Transformer with state-of-the-art results
- **Frozen Weights**: Uses pretrained weights without fine-tuning for feature extraction

### 📊 **Feature Extraction**
- **High-Dimensional Embeddings**: Extracts 512-2048 dimensional feature vectors
- **Multiple Model Support**: Combines features from different architectures
- **Normalization**: StandardScaler normalization for better model performance
- **Efficient Storage**: Saves features in optimized formats (JSON + Pickle)

## Installation

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM recommended
- 5GB+ disk space for models and features

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'Torchvision version: {torchvision.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

## Usage

### Quick Start
```bash
cd src

# Test with small sample (10 images)
python test_image_preprocessing.py

# Process 1000 images
python image_preprocessing.py

# Custom processing
python run_image_preprocessing.py 500 0  # 500 images starting from index 0
```

### Programmatic Usage
```python
from image_preprocessing import ImagePreprocessingPipeline

# Initialize pipeline
pipeline = ImagePreprocessingPipeline()

# Process images
results = pipeline.process_images(num_images=1000, start_index=0)

# Create feature matrices
feature_matrices = pipeline.create_feature_matrix(results)
normalized_matrices = pipeline.normalize_features(feature_matrices)

# Save everything
pipeline.save_feature_matrices(feature_matrices, normalized_matrices)
summary = pipeline.create_summary_report(results)
```

## Output Structure

### Directory Layout
```
image_features/
├── images/                           # Downloaded and resized images
├── image_features_final_1000_images.json    # Complete results
├── feature_extraction_summary.json          # Performance statistics
├── raw_features.pkl                          # Raw feature matrices
├── normalized_features.pkl                   # Normalized feature matrices
└── image_features_intermediate_*.json        # Intermediate saves
```

### Feature Dimensions
- **EfficientNet-B0**: 1280 features
- **ResNet50**: 2048 features  
- **ViT-B/16**: 768 features

### Result Format
Each result contains:
```json
{
  "sample_id": 12345,
  "image_url": "https://...",
  "catalog_content": "Original text...",
  "price": 29.99,
  "image_path": "/path/to/image.jpg",
  "features": {
    "efficientnet": [0.1, 0.2, ...],  // 1280 features
    "resnet": [0.3, 0.4, ...],        // 2048 features
    "vit": [0.5, 0.6, ...]            // 768 features
  },
  "status": "success"
}
```

## Model Details

### EfficientNet-B0
- **Architecture**: Compound scaling CNN
- **Parameters**: 5.3M
- **Input Size**: 224×224×3
- **Output Features**: 1280D
- **Advantages**: Best accuracy-efficiency tradeoff

### ResNet50
- **Architecture**: Deep residual network
- **Parameters**: 25.6M
- **Input Size**: 224×224×3
- **Output Features**: 2048D
- **Advantages**: Proven performance, robust features

### ViT-B/16
- **Architecture**: Vision Transformer
- **Parameters**: 86M
- **Input Size**: 224×224×3
- **Output Features**: 768D
- **Advantages**: State-of-the-art performance, attention mechanisms

## Performance Characteristics

### Expected Performance
- **Processing Speed**: 
  - GPU: ~50-100 images/minute
  - CPU: ~10-20 images/minute
- **Success Rate**: 85-95% (depends on image quality)
- **Memory Usage**: 
  - GPU: ~2-4GB VRAM
  - CPU: ~4-8GB RAM
- **Storage**: ~100-200MB per 1000 images (features only)

### Optimization Features
- **Model Caching**: Models loaded once and reused
- **Batch Processing**: Efficient tensor operations
- **Memory Management**: Automatic cleanup and garbage collection
- **Progress Tracking**: Real-time progress bars and logging

## Integration with ML Pipeline

### Feature Usage
The extracted features can be used for:
1. **Multimodal Learning**: Combine with text features
2. **Ensemble Methods**: Use features from multiple models
3. **Transfer Learning**: Fine-tune on extracted features
4. **Similarity Matching**: Find visually similar products

### Example Integration
```python
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load normalized features
with open('image_features/normalized_features.pkl', 'rb') as f:
    normalized_features = pickle.load(f)

# Use EfficientNet features for training
efficientnet_features = normalized_features['efficientnet']['features']
sample_ids = normalized_features['efficientnet']['sample_ids']

# Train model (example)
X = efficientnet_features
y = get_prices_for_samples(sample_ids)  # Your price data
model = RandomForestRegressor()
model.fit(X, y)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use CPU instead of GPU
   - Process fewer images at once

2. **Model Download Failures**:
   - Check internet connection
   - Increase timeout values
   - Use offline model files

3. **Image Download Issues**:
   - Increase retry attempts
   - Check URL validity
   - Handle network timeouts

4. **Feature Extraction Errors**:
   - Verify image format compatibility
   - Check model initialization
   - Handle corrupted images

### Performance Optimization

1. **GPU Acceleration**:
   ```python
   # Ensure CUDA is available
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

2. **Batch Processing**:
   ```python
   # Process multiple images together
   pipeline.process_images(batch_size=64)
   ```

3. **Memory Management**:
   ```python
   # Clear cache periodically
   torch.cuda.empty_cache()
   ```

## Advanced Configuration

### Custom Models
```python
# Add custom model
pipeline.models['custom'] = your_custom_model
pipeline.models['custom'].eval()
pipeline.models['custom'].to(device)
```

### Custom Transforms
```python
# Modify preprocessing
pipeline.transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Feature Selection
```python
# Use only specific models
selected_models = ['efficientnet', 'resnet']
for model_name in selected_models:
    features = pipeline.extract_features(image_tensor, model_name)
```

## Next Steps

### Immediate Actions
1. **Install Dependencies**: Run `pip install -r requirements.txt`
2. **Test Pipeline**: Execute `python test_image_preprocessing.py`
3. **Full Processing**: Run `python image_preprocessing.py`
4. **Analyze Results**: Review feature quality and performance

### Potential Enhancements
1. **Data Augmentation**: Add rotation, flipping, color jittering
2. **Model Ensembling**: Combine features from multiple models
3. **Dimensionality Reduction**: PCA or t-SNE for visualization
4. **Feature Selection**: Identify most important features
5. **Custom Architectures**: Train domain-specific models

## Conclusion

This image preprocessing pipeline provides a robust foundation for extracting high-quality visual features from product images. The combination of multiple pretrained models ensures comprehensive feature representation, while the efficient processing pipeline enables scalable feature extraction for large datasets.

The extracted features can significantly enhance the pricing model's performance by providing rich visual information about product characteristics, quality, and brand recognition.

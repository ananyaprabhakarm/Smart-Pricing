# Combined Dataset for ML Challenge 2025

This document describes the comprehensive dataset created by combining `new_sample.csv` with OCR-extracted text features for the Smart Product Pricing Challenge.

## Overview

The combined dataset integrates multiple data sources to create a rich, multimodal feature set for price prediction:

1. **Original Product Data**: Item names, descriptions, quantities, and prices
2. **Catalog Content**: Detailed product descriptions and specifications
3. **OCR Features**: Text extracted from product images using PyTesseract
4. **Engineered Features**: Computed features for enhanced ML performance

## Dataset Files

### Core Dataset Files
1. **`basic_combined_dataset.csv`** - Basic merge of new_sample.csv and OCR data
2. **`enhanced_dataset.csv`** - Enhanced with OCR text features
3. **`ml_ready_dataset.csv`** - ML-ready dataset with engineered features
4. **`sample_data.csv`** - Sample of 10 rows for quick inspection
5. **`feature_summary.json`** - Comprehensive feature analysis and statistics

## Dataset Statistics

### Basic Information
- **Total Samples**: 100 products
- **Total Features**: 34 columns
- **Price Range**: $0.94 - $298.00
- **Average Price**: $29.30
- **OCR Success Rate**: 70.0%
- **Average Confidence**: 76.7%
- **Average Word Count**: 15.8 words per extraction

### Data Quality
- **Complete Records**: 100% (no missing sample_ids)
- **OCR Coverage**: 70% of images had successful text extraction
- **Feature Completeness**: High coverage across all feature categories

## Feature Categories

### 1. Basic Product Information (6 features)
- `sample_id`: Unique product identifier
- `item_name`: Product name
- `item_description`: Product category/description
- `item_quantity`: Product quantity
- `unit`: Unit of measurement
- `price`: Product price (target variable)

### 2. Image and OCR Features (7 features)
- `image_url`: Original image URL
- `image_path`: Local path to downloaded image
- `extracted_text`: OCR-extracted text from image
- `confidence_score`: OCR confidence (0-100)
- `word_count`: Number of words extracted
- `status`: OCR processing status
- `image_dimensions`: Image size (width, height)

### 3. Catalog Content Features (3 features)
- `catalog_content`: Detailed product description
- `catalog_text_length`: Character count of catalog content
- `catalog_word_count`: Word count of catalog content

### 4. OCR Text Analysis Features (9 features)
- `text_length`: Length of extracted text
- `char_count`: Character count (excluding spaces)
- `line_count`: Number of lines in extracted text
- `has_numbers`: Boolean - contains digits
- `has_currency`: Boolean - contains currency symbols
- `has_brand_keywords`: Boolean - contains brand-related terms
- `has_size_info`: Boolean - contains size/dimension info
- `has_material_info`: Boolean - contains material information
- `has_color_info`: Boolean - contains color information

### 5. Engineered Features (9 features)
- `has_extracted_text`: Boolean - successful OCR extraction
- `text_quality_score`: Confidence score × extraction success
- `item_name_length`: Character count of item name
- `item_description_length`: Character count of item description
- `has_quantity`: Boolean - quantity information available
- `quantity_log`: Log-transformed quantity
- `price_log`: Log-transformed price
- `price_per_unit`: Price divided by quantity
- `category`: Lowercase item description for categorization

## Usage Examples

### Loading the Dataset
```python
import pandas as pd

# Load ML-ready dataset
df = pd.read_csv('combined_dataset/ml_ready_dataset.csv')

# Basic information
print(f"Dataset shape: {df.shape}")
print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
```

### Feature Engineering for ML
```python
# Select features for ML model
feature_columns = [
    'item_quantity', 'confidence_score', 'word_count',
    'catalog_text_length', 'catalog_word_count',
    'text_length', 'char_count', 'line_count',
    'has_numbers', 'has_currency', 'has_brand_keywords',
    'has_size_info', 'has_material_info', 'has_color_info',
    'has_extracted_text', 'text_quality_score',
    'item_name_length', 'item_description_length',
    'has_quantity', 'quantity_log', 'price_log', 'price_per_unit'
]

X = df[feature_columns]
y = df['price']

# Handle missing values
X = X.fillna(0)
```

### Analyzing OCR Performance
```python
# OCR success analysis
ocr_success = df['has_extracted_text'].sum()
total_samples = len(df)
print(f"OCR Success Rate: {ocr_success/total_samples*100:.1f}%")

# Confidence score analysis
high_confidence = df[df['confidence_score'] > 80]
print(f"High confidence extractions: {len(high_confidence)}")

# Text quality analysis
quality_scores = df['text_quality_score']
print(f"Average text quality score: {quality_scores.mean():.2f}")
```

### Price Analysis
```python
# Price distribution analysis
price_stats = df['price'].describe()
print("Price Statistics:")
print(price_stats)

# Price by category
category_prices = df.groupby('item_description')['price'].agg(['mean', 'count'])
print("\nPrice by Category:")
print(category_prices)
```

## Data Quality Insights

### OCR Performance
- **70% Success Rate**: Good coverage for image text extraction
- **76.7% Average Confidence**: Reasonable quality of extracted text
- **15.8 Average Words**: Sufficient text content for analysis

### Feature Completeness
- **High Coverage**: Most features have complete data
- **Missing Values**: Primarily in OCR features for failed extractions
- **Data Types**: Appropriate types for ML algorithms

### Price Distribution
- **Wide Range**: $0.94 to $298.00 covers various product categories
- **Reasonable Average**: $29.30 suggests diverse product mix
- **Log Transformation**: Applied for better ML performance

## Integration with ML Pipeline

### Recommended Feature Sets

#### 1. Basic Features (for simple models)
```python
basic_features = [
    'item_quantity', 'catalog_text_length', 'catalog_word_count',
    'item_name_length', 'item_description_length', 'has_quantity'
]
```

#### 2. OCR Features (for text-aware models)
```python
ocr_features = [
    'confidence_score', 'word_count', 'text_length', 'char_count',
    'has_numbers', 'has_currency', 'has_brand_keywords',
    'has_size_info', 'has_material_info', 'has_color_info'
]
```

#### 3. Engineered Features (for advanced models)
```python
engineered_features = [
    'text_quality_score', 'quantity_log', 'price_log', 'price_per_unit',
    'has_extracted_text', 'text_quality_score'
]
```

### Model Training Example
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Prepare features
feature_columns = basic_features + ocr_features + engineered_features
X = df[feature_columns].fillna(0)
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: ${mae:.2f}")
print(f"R²: {r2:.3f}")
```

## Next Steps

### Immediate Actions
1. **Explore Data**: Use `sample_data.csv` for quick inspection
2. **Feature Selection**: Choose relevant features for your model
3. **Model Training**: Implement and train price prediction models
4. **Validation**: Cross-validate model performance

### Potential Enhancements
1. **Feature Engineering**: Create additional derived features
2. **Text Processing**: Apply NLP techniques to extracted text
3. **Image Features**: Integrate with image preprocessing pipeline
4. **Ensemble Methods**: Combine multiple model types

### Advanced Analysis
1. **Category-Specific Models**: Train separate models for different product categories
2. **Confidence Weighting**: Weight predictions by OCR confidence scores
3. **Missing Value Imputation**: Develop strategies for failed OCR extractions
4. **Feature Importance**: Analyze which features contribute most to price prediction

## File Descriptions

### `basic_combined_dataset.csv`
- **Purpose**: Basic merge of original and OCR data
- **Size**: 100 rows × 14 columns
- **Use Case**: Simple analysis and data inspection

### `enhanced_dataset.csv`
- **Purpose**: Enhanced with OCR text features
- **Size**: 100 rows × 23 columns
- **Use Case**: Text analysis and feature exploration

### `ml_ready_dataset.csv`
- **Purpose**: Complete dataset ready for ML
- **Size**: 100 rows × 34 columns
- **Use Case**: Model training and evaluation

### `feature_summary.json`
- **Purpose**: Comprehensive feature analysis
- **Content**: Statistics, data types, missing values
- **Use Case**: Data understanding and feature selection

### `sample_data.csv`
- **Purpose**: Quick data inspection
- **Size**: 10 rows × 34 columns
- **Use Case**: Rapid data exploration

## Conclusion

The combined dataset provides a comprehensive foundation for price prediction in the ML Challenge 2025. With 34 features spanning product information, catalog content, OCR-extracted text, and engineered features, it enables sophisticated multimodal analysis.

The 70% OCR success rate and 76.7% average confidence provide substantial text-based features for analysis, while the wide price range ($0.94-$298.00) offers good diversity for model training.

This dataset is ready for immediate use in machine learning pipelines and provides multiple pathways for feature selection and model development.


import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import ViTImageProcessor, ViTModel
import requests
from io import BytesIO
import time
from tqdm import tqdm
import json
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImagePreprocessingPipeline:
    def __init__(self, dataset_folder='../dataset/', output_folder='../image_features/'):
        """
        Initialize Image Preprocessing Pipeline for extracting visual features
        
        Args:
            dataset_folder: Path to dataset folder
            output_folder: Path to save image features
        """
        self.dataset_folder = dataset_folder
        self.output_folder = output_folder
        self.images_folder = os.path.join(output_folder, 'images')
        
        # Create output directories
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.images_folder, exist_ok=True)
        
        # Load dataset
        self.train_df = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
        logger.info(f"Loaded training dataset with {len(self.train_df)} samples")
        
        # Initialize models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.models = {}
        self.processors = {}
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize pretrained models"""
        logger.info("Initializing pretrained models...")
        
        # EfficientNet-B0
        try:
            self.models['efficientnet'] = models.efficientnet_b0(pretrained=True)
            self.models['efficientnet'].classifier = nn.Identity()  # Remove classifier
            self.models['efficientnet'].eval()
            self.models['efficientnet'].to(self.device)
            logger.info("✓ EfficientNet-B0 loaded")
        except Exception as e:
            logger.warning(f"Failed to load EfficientNet: {e}")
            
        # ResNet50
        try:
            self.models['resnet'] = models.resnet50(pretrained=True)
            self.models['resnet'].fc = nn.Identity()  # Remove final layer
            self.models['resnet'].eval()
            self.models['resnet'].to(self.device)
            logger.info("✓ ResNet50 loaded")
        except Exception as e:
            logger.warning(f"Failed to load ResNet50: {e}")
            
        # ViT-B/16
        try:
            self.processors['vit'] = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            self.models['vit'] = ViTModel.from_pretrained('google/vit-base-patch16-224')
            self.models['vit'].eval()
            self.models['vit'].to(self.device)
            logger.info("✓ ViT-B/16 loaded")
        except Exception as e:
            logger.warning(f"Failed to load ViT: {e}")
            
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def download_image(self, image_url, sample_id, max_retries=3):
        """
        Download image from URL with retry mechanism
        
        Args:
            image_url: URL of the image
            sample_id: Sample ID for naming
            max_retries: Maximum number of retry attempts
            
        Returns:
            str: Path to downloaded image or None if failed
        """
        if not isinstance(image_url, str) or pd.isna(image_url):
            return None
            
        filename = f"{sample_id}_{Path(image_url).name}"
        image_path = os.path.join(self.images_folder, filename)
        
        # Skip if already downloaded
        if os.path.exists(image_path):
            return image_path
            
        for attempt in range(max_retries):
            try:
                response = requests.get(image_url, timeout=15)
                response.raise_for_status()
                
                # Save image
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                
                logger.debug(f"Downloaded image for sample {sample_id}")
                return image_path
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for sample {sample_id}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        logger.error(f"Failed to download image for sample {sample_id} after {max_retries} attempts")
        return None
    
    def preprocess_image(self, image_path):
        """
        Preprocess image: resize to 224x224 and normalize
        
        Args:
            image_path: Path to the image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {str(e)}")
            return None
    
    def extract_features(self, image_tensor, model_name):
        """
        Extract features using specified model
        
        Args:
            image_tensor: Preprocessed image tensor
            model_name: Name of the model to use
            
        Returns:
            np.ndarray: Extracted features
        """
        if model_name not in self.models:
            return None
            
        try:
            with torch.no_grad():
                if model_name == 'vit':
                    # ViT requires different preprocessing
                    inputs = self.processors['vit'](image_tensor.squeeze(0), return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.models[model_name](**inputs)
                    features = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
                else:
                    # CNN models (EfficientNet, ResNet)
                    features = self.models[model_name](image_tensor)
                
                return features.cpu().numpy().flatten()
                
        except Exception as e:
            logger.error(f"Failed to extract features with {model_name}: {str(e)}")
            return None
    
    def process_images(self, num_images=1000, start_index=0, batch_size=32):
        """
        Process images and extract features using multiple models
        
        Args:
            num_images: Number of images to process
            start_index: Starting index in the dataset
            batch_size: Batch size for processing
            
        Returns:
            list: List of feature extraction results
        """
        logger.info(f"Starting image preprocessing for {num_images} images starting from index {start_index}")
        
        # Get subset of data
        end_index = min(start_index + num_images, len(self.train_df))
        subset_df = self.train_df.iloc[start_index:end_index].copy()
        
        logger.info(f"Processing {len(subset_df)} samples")
        
        results = []
        successful_downloads = 0
        successful_extractions = 0
        
        for idx, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc="Processing images"):
            sample_id = row['sample_id']
            image_url = row['image_link']
            
            # Download image
            image_path = self.download_image(image_url, sample_id)
            if image_path:
                successful_downloads += 1
            
            # Initialize result
            result = {
                'sample_id': sample_id,
                'image_url': image_url,
                'catalog_content': row['catalog_content'],
                'price': row['price'],
                'image_path': image_path,
                'features': {},
                'status': 'failed_no_image' if not image_path else 'processing'
            }
            
            if image_path:
                # Preprocess image
                image_tensor = self.preprocess_image(image_path)
                
                if image_tensor is not None:
                    # Extract features from all available models
                    for model_name in self.models.keys():
                        features = self.extract_features(image_tensor, model_name)
                        if features is not None:
                            result['features'][model_name] = features.tolist()
                    
                    if result['features']:
                        result['status'] = 'success'
                        successful_extractions += 1
                    else:
                        result['status'] = 'failed_feature_extraction'
                else:
                    result['status'] = 'failed_preprocessing'
            
            results.append(result)
            
            # Save intermediate results every 100 images
            if (idx - start_index + 1) % 100 == 0:
                self.save_results(results, f"image_features_intermediate_{idx - start_index + 1}.json")
        
        logger.info(f"Image Preprocessing completed:")
        logger.info(f"  - Images processed: {len(subset_df)}")
        logger.info(f"  - Successful downloads: {successful_downloads}")
        logger.info(f"  - Successful extractions: {successful_extractions}")
        logger.info(f"  - Success rate: {successful_extractions/len(subset_df)*100:.2f}%")
        
        return results
    
    def save_results(self, results, filename=None):
        """
        Save feature extraction results to JSON file
        
        Args:
            results: List of feature extraction results
            filename: Optional custom filename
        """
        if filename is None:
            filename = f"image_features_{len(results)}_images.json"
        
        filepath = os.path.join(self.output_folder, filename)
        
        # Clean results for JSON serialization
        clean_results = []
        for result in results:
            clean_result = {
                'sample_id': result['sample_id'],
                'image_url': result['image_url'],
                'catalog_content': result['catalog_content'],
                'price': result['price'],
                'image_path': result['image_path'],
                'features': result['features'],
                'status': result['status']
            }
            clean_results.append(clean_result)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {filepath}")
    
    def create_feature_matrix(self, results):
        """
        Create feature matrices for each model
        
        Args:
            results: List of feature extraction results
            
        Returns:
            dict: Feature matrices for each model
        """
        feature_matrices = {}
        
        for model_name in self.models.keys():
            features_list = []
            sample_ids = []
            
            for result in results:
                if result['status'] == 'success' and model_name in result['features']:
                    features_list.append(result['features'][model_name])
                    sample_ids.append(result['sample_id'])
            
            if features_list:
                feature_matrix = np.array(features_list)
                feature_matrices[model_name] = {
                    'features': feature_matrix,
                    'sample_ids': sample_ids,
                    'shape': feature_matrix.shape
                }
                
                logger.info(f"{model_name} feature matrix shape: {feature_matrix.shape}")
        
        return feature_matrices
    
    def normalize_features(self, feature_matrices):
        """
        Normalize features using StandardScaler
        
        Args:
            feature_matrices: Dictionary of feature matrices
            
        Returns:
            dict: Normalized feature matrices
        """
        normalized_matrices = {}
        
        for model_name, matrix_data in feature_matrices.items():
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(matrix_data['features'])
            
            normalized_matrices[model_name] = {
                'features': normalized_features,
                'sample_ids': matrix_data['sample_ids'],
                'scaler': scaler,
                'shape': normalized_features.shape
            }
            
            logger.info(f"{model_name} normalized features shape: {normalized_features.shape}")
        
        return normalized_matrices
    
    def save_feature_matrices(self, feature_matrices, normalized_matrices):
        """
        Save feature matrices and scalers
        
        Args:
            feature_matrices: Raw feature matrices
            normalized_matrices: Normalized feature matrices
        """
        # Save raw features
        raw_features_path = os.path.join(self.output_folder, 'raw_features.pkl')
        with open(raw_features_path, 'wb') as f:
            pickle.dump(feature_matrices, f)
        
        # Save normalized features
        normalized_features_path = os.path.join(self.output_folder, 'normalized_features.pkl')
        with open(normalized_features_path, 'wb') as f:
            pickle.dump(normalized_matrices, f)
        
        logger.info(f"Feature matrices saved to {self.output_folder}")
    
    def create_summary_report(self, results):
        """
        Create a summary report of feature extraction results
        
        Args:
            results: List of feature extraction results
        """
        # Calculate statistics
        total_images = len(results)
        successful_extractions = sum(1 for r in results if r['status'] == 'success')
        
        # Count features per model
        model_counts = {}
        for model_name in self.models.keys():
            model_counts[model_name] = sum(1 for r in results 
                                         if r['status'] == 'success' and model_name in r['features'])
        
        # Create summary
        summary = {
            'total_images_processed': total_images,
            'successful_feature_extractions': successful_extractions,
            'success_rate_percentage': (successful_extractions / total_images) * 100,
            'model_performance': model_counts,
            'available_models': list(self.models.keys()),
            'device_used': str(self.device),
            'sample_results': results[:3]  # First 3 results as examples
        }
        
        # Count status distribution
        status_distribution = {}
        for result in results:
            status = result['status']
            status_distribution[status] = status_distribution.get(status, 0) + 1
        summary['status_distribution'] = status_distribution
        
        # Save summary
        summary_path = os.path.join(self.output_folder, 'feature_extraction_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary report saved to {summary_path}")
        return summary

def main():
    """
    Main function to run the image preprocessing pipeline
    """
    # Initialize pipeline
    pipeline = ImagePreprocessingPipeline()
    
    # Process first 1000 images
    logger.info("Starting image preprocessing for first 1000 images...")
    results = pipeline.process_images(num_images=1000, start_index=0)
    
    # Save final results
    pipeline.save_results(results, "image_features_final_1000_images.json")
    
    # Create feature matrices
    feature_matrices = pipeline.create_feature_matrix(results)
    normalized_matrices = pipeline.normalize_features(feature_matrices)
    
    # Save feature matrices
    pipeline.save_feature_matrices(feature_matrices, normalized_matrices)
    
    # Create summary report
    summary = pipeline.create_summary_report(results)
    
    # Print summary
    print("\n" + "="*60)
    print("IMAGE PREPROCESSING PIPELINE SUMMARY")
    print("="*60)
    print(f"Total images processed: {summary['total_images_processed']}")
    print(f"Successful extractions: {summary['successful_feature_extractions']}")
    print(f"Success rate: {summary['success_rate_percentage']:.2f}%")
    print(f"Device used: {summary['device_used']}")
    print(f"Available models: {', '.join(summary['available_models'])}")
    print("\nModel performance:")
    for model, count in summary['model_performance'].items():
        print(f"  {model}: {count} successful extractions")
    print("\nStatus distribution:")
    for status, count in summary['status_distribution'].items():
        print(f"  {status}: {count}")
    print("="*60)

if __name__ == "__main__":
    main()

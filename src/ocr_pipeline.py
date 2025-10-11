import os
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import requests
from io import BytesIO
import time
from tqdm import tqdm
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRPipeline:
    def __init__(self, dataset_folder='../dataset/', output_folder='../ocr_output/'):
        """
        Initialize OCR Pipeline for extracting text from product images
        
        Args:
            dataset_folder: Path to dataset folder
            output_folder: Path to save OCR results
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
                response = requests.get(image_url, timeout=10)
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
    
    def extract_text_from_image(self, image_path, sample_id):
        """
        Extract text from image using PyTesseract OCR
        
        Args:
            image_path: Path to the image file
            sample_id: Sample ID for logging
            
        Returns:
            dict: OCR results including text and confidence scores
        """
        if not image_path or not os.path.exists(image_path):
            return {
                'sample_id': sample_id,
                'extracted_text': '',
                'confidence_score': 0.0,
                'word_count': 0,
                'status': 'failed_no_image'
            }
        
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text with confidence scores
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Extract text
            extracted_text = pytesseract.image_to_string(image)
            
            # Calculate average confidence (excluding -1 values)
            confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Count words
            word_count = len(extracted_text.split())
            
            return {
                'sample_id': sample_id,
                'extracted_text': extracted_text.strip(),
                'confidence_score': avg_confidence,
                'word_count': word_count,
                'status': 'success',
                'image_dimensions': image.size,
                'ocr_data': ocr_data
            }
            
        except Exception as e:
            logger.error(f"OCR failed for sample {sample_id}: {str(e)}")
            return {
                'sample_id': sample_id,
                'extracted_text': '',
                'confidence_score': 0.0,
                'word_count': 0,
                'status': f'failed_ocr: {str(e)}'
            }
    
    def process_images(self, num_images=1000, start_index=0):
        """
        Process images and extract text using OCR
        
        Args:
            num_images: Number of images to process
            start_index: Starting index in the dataset
            
        Returns:
            list: List of OCR results
        """
        logger.info(f"Starting OCR pipeline for {num_images} images starting from index {start_index}")
        
        # Get subset of data
        end_index = min(start_index + num_images, len(self.train_df))
        subset_df = self.train_df.iloc[start_index:end_index].copy()
        
        logger.info(f"Processing {len(subset_df)} samples")
        
        ocr_results = []
        successful_downloads = 0
        successful_ocr = 0
        
        for idx, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc="Processing images"):
            sample_id = row['sample_id']
            image_url = row['image_link']
            
            # Download image
            image_path = self.download_image(image_url, sample_id)
            if image_path:
                successful_downloads += 1
            
            # Extract text using OCR
            ocr_result = self.extract_text_from_image(image_path, sample_id)
            
            # Add metadata
            ocr_result.update({
                'image_url': image_url,
                'catalog_content': row['catalog_content'],
                'price': row['price'],
                'image_path': image_path
            })
            
            if ocr_result['status'] == 'success':
                successful_ocr += 1
            
            ocr_results.append(ocr_result)
            
            # Save intermediate results every 100 images
            if (idx - start_index + 1) % 100 == 0:
                self.save_results(ocr_results, f"ocr_results_intermediate_{idx - start_index + 1}.json")
        
        logger.info(f"OCR Pipeline completed:")
        logger.info(f"  - Images processed: {len(subset_df)}")
        logger.info(f"  - Successful downloads: {successful_downloads}")
        logger.info(f"  - Successful OCR: {successful_ocr}")
        logger.info(f"  - Success rate: {successful_ocr/len(subset_df)*100:.2f}%")
        
        return ocr_results
    
    def save_results(self, ocr_results, filename=None):
        """
        Save OCR results to JSON file
        
        Args:
            ocr_results: List of OCR results
            filename: Optional custom filename
        """
        if filename is None:
            filename = f"ocr_results_{len(ocr_results)}_images.json"
        
        filepath = os.path.join(self.output_folder, filename)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean results for JSON serialization
        clean_results = []
        for result in ocr_results:
            clean_result = {}
            for key, value in result.items():
                if key == 'ocr_data':
                    # Skip detailed OCR data for JSON (too large)
                    continue
                clean_result[key] = convert_numpy(value)
            clean_results.append(clean_result)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {filepath}")
    
    def create_summary_report(self, ocr_results):
        """
        Create a summary report of OCR results
        
        Args:
            ocr_results: List of OCR results
        """
        # Calculate statistics
        total_images = len(ocr_results)
        successful_ocr = sum(1 for r in ocr_results if r['status'] == 'success')
        avg_confidence = np.mean([r['confidence_score'] for r in ocr_results if r['confidence_score'] > 0])
        avg_word_count = np.mean([r['word_count'] for r in ocr_results if r['word_count'] > 0])
        
        # Create summary
        summary = {
            'total_images_processed': total_images,
            'successful_ocr_extractions': successful_ocr,
            'success_rate_percentage': (successful_ocr / total_images) * 100,
            'average_confidence_score': avg_confidence,
            'average_words_per_image': avg_word_count,
            'status_distribution': {},
            'sample_results': ocr_results[:5]  # First 5 results as examples
        }
        
        # Count status distribution
        for result in ocr_results:
            status = result['status']
            summary['status_distribution'][status] = summary['status_distribution'].get(status, 0) + 1
        
        # Save summary
        summary_path = os.path.join(self.output_folder, 'ocr_summary_report.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary report saved to {summary_path}")
        return summary

def main():
    """
    Main function to run the OCR pipeline
    """
    # Initialize pipeline
    pipeline = OCRPipeline()
    
    # Process first 1000 images
    logger.info("Starting OCR extraction from first 1000 images...")
    ocr_results = pipeline.process_images(num_images=1000, start_index=0)
    
    # Save final results
    pipeline.save_results(ocr_results, "ocr_results_final_1000_images.json")
    
    # Create summary report
    summary = pipeline.create_summary_report(ocr_results)
    
    # Print summary
    print("\n" + "="*50)
    print("OCR PIPELINE SUMMARY")
    print("="*50)
    print(f"Total images processed: {summary['total_images_processed']}")
    print(f"Successful OCR extractions: {summary['successful_ocr_extractions']}")
    print(f"Success rate: {summary['success_rate_percentage']:.2f}%")
    print(f"Average confidence score: {summary['average_confidence_score']:.2f}")
    print(f"Average words per image: {summary['average_words_per_image']:.1f}")
    print("\nStatus distribution:")
    for status, count in summary['status_distribution'].items():
        print(f"  {status}: {count}")
    print("="*50)

if __name__ == "__main__":
    main()

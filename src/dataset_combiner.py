import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetCombiner:
    def __init__(self, 
                 new_sample_path='../dataset/new_sample.csv',
                 ocr_csv_path='../ocr_csv/ocr_results_combined.csv',
                 ocr_text_features_path='../ocr_csv/ocr_results_final_1000_images_text_features.csv',
                 output_folder='../combined_dataset/'):
        """
        Initialize Dataset Combiner
        
        Args:
            new_sample_path: Path to new_sample.csv
            ocr_csv_path: Path to OCR combined CSV
            ocr_text_features_path: Path to OCR text features CSV
            output_folder: Path to save combined datasets
        """
        self.new_sample_path = new_sample_path
        self.ocr_csv_path = ocr_csv_path
        self.ocr_text_features_path = ocr_text_features_path
        self.output_folder = output_folder
        
        # Create output directory
        os.makedirs(self.output_folder, exist_ok=True)
        
        logger.info(f"Dataset Combiner initialized")
        logger.info(f"Output folder: {self.output_folder}")
    
    def load_datasets(self):
        """
        Load all datasets
        
        Returns:
            tuple: (new_sample_df, ocr_df, ocr_features_df)
        """
        logger.info("Loading datasets...")
        
        # Load new_sample.csv
        try:
            new_sample_df = pd.read_csv(self.new_sample_path)
            logger.info(f"Loaded new_sample.csv: {len(new_sample_df)} rows")
        except Exception as e:
            logger.error(f"Failed to load new_sample.csv: {e}")
            return None, None, None
        
        # Load OCR combined CSV
        try:
            ocr_df = pd.read_csv(self.ocr_csv_path)
            logger.info(f"Loaded OCR combined CSV: {len(ocr_df)} rows")
        except Exception as e:
            logger.error(f"Failed to load OCR CSV: {e}")
            return None, None, None
        
        # Load OCR text features CSV
        try:
            ocr_features_df = pd.read_csv(self.ocr_text_features_path)
            logger.info(f"Loaded OCR text features CSV: {len(ocr_features_df)} rows")
        except Exception as e:
            logger.error(f"Failed to load OCR text features CSV: {e}")
            return None, None, None
        
        return new_sample_df, ocr_df, ocr_features_df
    
    def create_basic_combined_dataset(self, new_sample_df, ocr_df):
        """
        Create basic combined dataset with new_sample and OCR data
        
        Args:
            new_sample_df: New sample dataframe
            ocr_df: OCR dataframe
            
        Returns:
            pd.DataFrame: Combined dataset
        """
        logger.info("Creating basic combined dataset...")
        
        # Merge on sample_id
        combined_df = pd.merge(
            new_sample_df, 
            ocr_df, 
            on='sample_id', 
            how='left',
            suffixes=('_new_sample', '_ocr')
        )
        
        # Handle duplicate columns
        if 'catalog_content_new_sample' in combined_df.columns and 'catalog_content_ocr' in combined_df.columns:
            # Use new_sample catalog_content as primary, OCR as backup
            combined_df['catalog_content'] = combined_df['catalog_content_new_sample'].fillna(combined_df['catalog_content_ocr'])
            combined_df = combined_df.drop(['catalog_content_new_sample', 'catalog_content_ocr'], axis=1)
        
        if 'price_new_sample' in combined_df.columns and 'price_ocr' in combined_df.columns:
            # Use new_sample price as primary
            combined_df['price'] = combined_df['price_new_sample']
            combined_df = combined_df.drop(['price_new_sample', 'price_ocr'], axis=1)
        
        if 'image_link' in combined_df.columns and 'image_url' in combined_df.columns:
            # Use new_sample image_link as primary
            combined_df['image_url'] = combined_df['image_link']
            combined_df = combined_df.drop('image_link', axis=1)
        
        logger.info(f"Basic combined dataset created: {len(combined_df)} rows")
        return combined_df
    
    def create_enhanced_dataset(self, basic_df, ocr_features_df):
        """
        Create enhanced dataset with OCR text features
        
        Args:
            basic_df: Basic combined dataframe
            ocr_features_df: OCR text features dataframe
            
        Returns:
            pd.DataFrame: Enhanced dataset
        """
        logger.info("Creating enhanced dataset with OCR text features...")
        
        # Merge with OCR text features
        enhanced_df = pd.merge(
            basic_df,
            ocr_features_df,
            on='sample_id',
            how='left',
            suffixes=('', '_features')
        )
        
        # Remove duplicate columns
        duplicate_cols = [col for col in enhanced_df.columns if col.endswith('_features')]
        enhanced_df = enhanced_df.drop(duplicate_cols, axis=1)
        
        logger.info(f"Enhanced dataset created: {len(enhanced_df)} rows")
        return enhanced_df
    
    def create_ml_ready_dataset(self, enhanced_df):
        """
        Create ML-ready dataset with engineered features
        
        Args:
            enhanced_df: Enhanced dataframe
            
        Returns:
            pd.DataFrame: ML-ready dataset
        """
        logger.info("Creating ML-ready dataset...")
        
        ml_df = enhanced_df.copy()
        
        # Handle missing values
        ml_df['extracted_text'] = ml_df['extracted_text'].fillna('')
        ml_df['confidence_score'] = ml_df['confidence_score'].fillna(0)
        ml_df['word_count'] = ml_df['word_count'].fillna(0)
        
        # Create additional features
        ml_df['has_extracted_text'] = ml_df['extracted_text'].str.len() > 0
        ml_df['text_quality_score'] = ml_df['confidence_score'] * ml_df['has_extracted_text'].astype(int)
        
        # Create catalog text features
        ml_df['catalog_text_length'] = ml_df['catalog_content'].str.len()
        ml_df['catalog_word_count'] = ml_df['catalog_content'].str.split().str.len()
        
        # Create item features
        ml_df['item_name_length'] = ml_df['item_name'].str.len()
        ml_df['item_description_length'] = ml_df['item_description'].str.len()
        
        # Create quantity features
        ml_df['has_quantity'] = ml_df['item_quantity'].notna()
        ml_df['quantity_log'] = np.log1p(ml_df['item_quantity'].fillna(0))
        
        # Create price features
        ml_df['price_log'] = np.log1p(ml_df['price'])
        ml_df['price_per_unit'] = ml_df['price'] / ml_df['item_quantity'].replace(0, np.nan)
        
        # Create category features
        ml_df['category'] = ml_df['item_description'].str.lower()
        
        logger.info(f"ML-ready dataset created: {len(ml_df)} rows, {len(ml_df.columns)} columns")
        return ml_df
    
    def create_feature_summary(self, ml_df):
        """
        Create feature summary for the ML dataset
        
        Args:
            ml_df: ML-ready dataframe
            
        Returns:
            dict: Feature summary
        """
        logger.info("Creating feature summary...")
        
        summary = {
            'dataset_info': {
                'total_samples': len(ml_df),
                'total_features': len(ml_df.columns),
                'missing_values': ml_df.isnull().sum().to_dict(),
                'data_types': ml_df.dtypes.to_dict()
            },
            'feature_categories': {
                'basic_info': ['sample_id', 'item_name', 'item_description', 'item_quantity', 'unit', 'price'],
                'catalog_features': ['catalog_content', 'catalog_text_length', 'catalog_word_count'],
                'ocr_features': ['extracted_text', 'confidence_score', 'word_count', 'has_extracted_text', 'text_quality_score'],
                'text_features': ['text_length', 'char_count', 'line_count', 'has_numbers', 'has_currency', 
                                 'has_brand_keywords', 'has_size_info', 'has_material_info', 'has_color_info'],
                'engineered_features': ['item_name_length', 'item_description_length', 'has_quantity', 
                                       'quantity_log', 'price_log', 'price_per_unit', 'category']
            },
            'statistics': {
                'price_stats': ml_df['price'].describe().to_dict(),
                'confidence_stats': ml_df['confidence_score'].describe().to_dict(),
                'word_count_stats': ml_df['word_count'].describe().to_dict(),
                'text_length_stats': ml_df['text_length'].describe().to_dict() if 'text_length' in ml_df.columns else {}
            }
        }
        
        return summary
    
    def save_datasets(self, basic_df, enhanced_df, ml_df, feature_summary):
        """
        Save all datasets and summary
        
        Args:
            basic_df: Basic combined dataframe
            enhanced_df: Enhanced dataframe
            ml_df: ML-ready dataframe
            feature_summary: Feature summary
        """
        logger.info("Saving datasets...")
        
        # Save basic combined dataset
        basic_path = os.path.join(self.output_folder, 'basic_combined_dataset.csv')
        basic_df.to_csv(basic_path, index=False, encoding='utf-8')
        logger.info(f"Basic dataset saved: {basic_path}")
        
        # Save enhanced dataset
        enhanced_path = os.path.join(self.output_folder, 'enhanced_dataset.csv')
        enhanced_df.to_csv(enhanced_path, index=False, encoding='utf-8')
        logger.info(f"Enhanced dataset saved: {enhanced_path}")
        
        # Save ML-ready dataset
        ml_path = os.path.join(self.output_folder, 'ml_ready_dataset.csv')
        ml_df.to_csv(ml_path, index=False, encoding='utf-8')
        logger.info(f"ML-ready dataset saved: {ml_path}")
        
        # Save feature summary
        summary_path = os.path.join(self.output_folder, 'feature_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(feature_summary, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Feature summary saved: {summary_path}")
        
        # Save sample data for inspection
        sample_path = os.path.join(self.output_folder, 'sample_data.csv')
        ml_df.head(10).to_csv(sample_path, index=False, encoding='utf-8')
        logger.info(f"Sample data saved: {sample_path}")
    
    def process_all(self):
        """
        Process all datasets and create combined versions
        """
        logger.info("Starting dataset combination process...")
        
        # Load datasets
        new_sample_df, ocr_df, ocr_features_df = self.load_datasets()
        
        if new_sample_df is None or ocr_df is None or ocr_features_df is None:
            logger.error("Failed to load datasets")
            return
        
        # Create basic combined dataset
        basic_df = self.create_basic_combined_dataset(new_sample_df, ocr_df)
        
        # Create enhanced dataset
        enhanced_df = self.create_enhanced_dataset(basic_df, ocr_features_df)
        
        # Create ML-ready dataset
        ml_df = self.create_ml_ready_dataset(enhanced_df)
        
        # Create feature summary
        feature_summary = self.create_feature_summary(ml_df)
        
        # Save all datasets
        self.save_datasets(basic_df, enhanced_df, ml_df, feature_summary)
        
        logger.info("Dataset combination process completed successfully!")
        
        return {
            'basic_df': basic_df,
            'enhanced_df': enhanced_df,
            'ml_df': ml_df,
            'feature_summary': feature_summary
        }

def main():
    """
    Main function to combine datasets
    """
    # Initialize combiner
    combiner = DatasetCombiner()
    
    # Process all datasets
    results = combiner.process_all()
    
    if results:
        # Print summary
        print("\n" + "="*60)
        print("DATASET COMBINATION SUMMARY")
        print("="*60)
        print(f"Basic dataset: {len(results['basic_df'])} rows, {len(results['basic_df'].columns)} columns")
        print(f"Enhanced dataset: {len(results['enhanced_df'])} rows, {len(results['enhanced_df'].columns)} columns")
        print(f"ML-ready dataset: {len(results['ml_df'])} rows, {len(results['ml_df'].columns)} columns")
        print(f"Output folder: {combiner.output_folder}")
        print("="*60)
        
        # Show sample of ML-ready dataset
        print("\nSample of ML-ready dataset:")
        sample_cols = ['sample_id', 'item_name', 'price', 'confidence_score', 'word_count', 'has_extracted_text']
        available_cols = [col for col in sample_cols if col in results['ml_df'].columns]
        print(results['ml_df'][available_cols].head().to_string())

if __name__ == "__main__":
    main()


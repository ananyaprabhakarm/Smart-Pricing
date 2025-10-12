import os
import pandas as pd
import json
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRToCSVConverter:
    def __init__(self, ocr_output_folder='../ocr_output/', csv_output_folder='../ocr_csv/'):
        """
        Initialize OCR to CSV converter
        
        Args:
            ocr_output_folder: Path to OCR output folder
            csv_output_folder: Path to save CSV files
        """
        self.ocr_output_folder = ocr_output_folder
        self.csv_output_folder = csv_output_folder
        
        # Create output directory
        os.makedirs(self.csv_output_folder, exist_ok=True)
        
        logger.info(f"OCR to CSV Converter initialized")
        logger.info(f"OCR input folder: {self.ocr_output_folder}")
        logger.info(f"CSV output folder: {self.csv_output_folder}")
    
    def load_ocr_results(self, json_file_path):
        """
        Load OCR results from JSON file
        
        Args:
            json_file_path: Path to JSON file
            
        Returns:
            list: OCR results
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Loaded {len(results)} OCR results from {json_file_path}")
            return results
        except Exception as e:
            logger.error(f"Failed to load {json_file_path}: {str(e)}")
            return []
    
    def convert_to_main_csv(self, results, output_filename):
        """
        Convert OCR results to main CSV format
        
        Args:
            results: List of OCR results
            output_filename: Output CSV filename
            
        Returns:
            str: Path to saved CSV file
        """
        # Prepare data for CSV
        csv_data = []
        
        for result in results:
            row = {
                'sample_id': result.get('sample_id', ''),
                'image_url': result.get('image_url', ''),
                'catalog_content': result.get('catalog_content', ''),
                'price': result.get('price', ''),
                'image_path': result.get('image_path', ''),
                'extracted_text': result.get('extracted_text', ''),
                'confidence_score': result.get('confidence_score', 0.0),
                'word_count': result.get('word_count', 0),
                'status': result.get('status', ''),
                'image_dimensions': str(result.get('image_dimensions', '')),
            }
            csv_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(csv_data)
        
        # Save to CSV
        output_path = os.path.join(self.csv_output_folder, output_filename)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Main CSV saved: {output_path} ({len(df)} rows)")
        return output_path
    
    def convert_to_text_features_csv(self, results, output_filename):
        """
        Convert OCR results to text features CSV for ML analysis
        
        Args:
            results: List of OCR results
            output_filename: Output CSV filename
            
        Returns:
            str: Path to saved CSV file
        """
        # Prepare data for text analysis
        csv_data = []
        
        for result in results:
            if result.get('status') == 'success' and result.get('extracted_text'):
                extracted_text = result.get('extracted_text', '')
                
                # Basic text features
                row = {
                    'sample_id': result.get('sample_id', ''),
                    'price': result.get('price', ''),
                    'extracted_text': extracted_text,
                    'text_length': len(extracted_text),
                    'word_count': result.get('word_count', 0),
                    'confidence_score': result.get('confidence_score', 0.0),
                    'char_count': len(extracted_text.replace(' ', '')),
                    'line_count': len(extracted_text.split('\n')),
                    'has_numbers': any(char.isdigit() for char in extracted_text),
                    'has_currency': any(symbol in extracted_text for symbol in ['$', '€', '£', '¥', '₹']),
                    'has_brand_keywords': any(keyword in extracted_text.lower() 
                                            for keyword in ['brand', 'made by', 'manufacturer', 'company']),
                    'has_size_info': any(keyword in extracted_text.lower() 
                                       for keyword in ['size', 'dimension', 'inch', 'cm', 'mm', 'oz', 'lb']),
                    'has_material_info': any(keyword in extracted_text.lower() 
                                           for keyword in ['material', 'fabric', 'metal', 'plastic', 'wood', 'leather']),
                    'has_color_info': any(keyword in extracted_text.lower() 
                                        for keyword in ['color', 'colour', 'black', 'white', 'red', 'blue', 'green']),
                }
                csv_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(csv_data)
        
        # Save to CSV
        output_path = os.path.join(self.csv_output_folder, output_filename)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Text features CSV saved: {output_path} ({len(df)} rows)")
        return output_path
    
    def convert_to_summary_csv(self, results, output_filename):
        """
        Convert OCR results to summary statistics CSV
        
        Args:
            results: List of OCR results
            output_filename: Output CSV filename
            
        Returns:
            str: Path to saved CSV file
        """
        # Calculate summary statistics
        successful_results = [r for r in results if r.get('status') == 'success']
        
        if not successful_results:
            logger.warning("No successful OCR results found for summary")
            return None
        
        # Extract metrics
        confidence_scores = [r.get('confidence_score', 0) for r in successful_results]
        word_counts = [r.get('word_count', 0) for r in successful_results]
        text_lengths = [len(r.get('extracted_text', '')) for r in successful_results]
        
        summary_data = {
            'metric': [
                'total_samples',
                'successful_extractions',
                'success_rate_percentage',
                'avg_confidence_score',
                'median_confidence_score',
                'min_confidence_score',
                'max_confidence_score',
                'avg_word_count',
                'median_word_count',
                'min_word_count',
                'max_word_count',
                'avg_text_length',
                'median_text_length',
                'min_text_length',
                'max_text_length'
            ],
            'value': [
                len(results),
                len(successful_results),
                (len(successful_results) / len(results)) * 100,
                np.mean(confidence_scores),
                np.median(confidence_scores),
                np.min(confidence_scores),
                np.max(confidence_scores),
                np.mean(word_counts),
                np.median(word_counts),
                np.min(word_counts),
                np.max(word_counts),
                np.mean(text_lengths),
                np.median(text_lengths),
                np.min(text_lengths),
                np.max(text_lengths)
            ]
        }
        
        # Create DataFrame
        df = pd.DataFrame(summary_data)
        
        # Save to CSV
        output_path = os.path.join(self.csv_output_folder, output_filename)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Summary CSV saved: {output_path}")
        return output_path
    
    def convert_all_ocr_files(self):
        """
        Convert all OCR JSON files to CSV format
        """
        logger.info("Starting conversion of all OCR files to CSV...")
        
        # Find all JSON files in OCR output folder
        json_files = []
        for file in os.listdir(self.ocr_output_folder):
            if file.endswith('.json') and 'ocr_results' in file:
                json_files.append(file)
        
        if not json_files:
            logger.warning("No OCR JSON files found")
            return
        
        logger.info(f"Found {len(json_files)} OCR JSON files to convert")
        
        converted_files = []
        
        for json_file in json_files:
            logger.info(f"Processing {json_file}...")
            
            # Load results
            json_path = os.path.join(self.ocr_output_folder, json_file)
            results = self.load_ocr_results(json_path)
            
            if not results:
                continue
            
            # Generate output filenames
            base_name = json_file.replace('.json', '')
            
            # Convert to different CSV formats
            main_csv = self.convert_to_main_csv(results, f"{base_name}_main.csv")
            text_features_csv = self.convert_to_text_features_csv(results, f"{base_name}_text_features.csv")
            summary_csv = self.convert_to_summary_csv(results, f"{base_name}_summary.csv")
            
            converted_files.extend([main_csv, text_features_csv, summary_csv])
        
        logger.info(f"Conversion completed! {len(converted_files)} CSV files created")
        return converted_files
    
    def create_combined_csv(self):
        """
        Create a combined CSV with all OCR results
        """
        logger.info("Creating combined CSV with all OCR results...")
        
        # Find all main CSV files
        csv_files = []
        for file in os.listdir(self.csv_output_folder):
            if file.endswith('_main.csv'):
                csv_files.append(file)
        
        if not csv_files:
            logger.warning("No main CSV files found for combination")
            return
        
        # Combine all CSV files
        combined_data = []
        for csv_file in csv_files:
            csv_path = os.path.join(self.csv_output_folder, csv_file)
            df = pd.read_csv(csv_path)
            combined_data.append(df)
        
        # Concatenate all dataframes
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Remove duplicates based on sample_id
        combined_df = combined_df.drop_duplicates(subset=['sample_id'], keep='first')
        
        # Save combined CSV
        combined_path = os.path.join(self.csv_output_folder, 'ocr_results_combined.csv')
        combined_df.to_csv(combined_path, index=False, encoding='utf-8')
        
        logger.info(f"Combined CSV saved: {combined_path} ({len(combined_df)} unique samples)")
        return combined_path

def main():
    """
    Main function to convert OCR results to CSV
    """
    # Initialize converter
    converter = OCRToCSVConverter()
    
    # Convert all OCR files
    converted_files = converter.convert_all_ocr_files()
    
    # Create combined CSV
    combined_csv = converter.create_combined_csv()
    
    # Print summary
    print("\n" + "="*60)
    print("OCR TO CSV CONVERSION SUMMARY")
    print("="*60)
    print(f"CSV files created: {len(converted_files) if converted_files else 0}")
    print(f"Combined CSV: {combined_csv}")
    print(f"Output folder: {converter.csv_output_folder}")
    print("="*60)
    
    # List created files
    if os.path.exists(converter.csv_output_folder):
        csv_files = [f for f in os.listdir(converter.csv_output_folder) if f.endswith('.csv')]
        print(f"\nCreated CSV files:")
        for file in sorted(csv_files):
            print(f"  - {file}")

if __name__ == "__main__":
    main()

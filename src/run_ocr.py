#!/usr/bin/env python3
"""
Simple runner script for OCR pipeline
Usage: python run_ocr.py [num_images] [start_index]
"""

import sys
import os
from ocr_pipeline import OCRPipeline

def main():
    # Parse command line arguments
    num_images = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    start_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    print(f"Running OCR pipeline for {num_images} images starting from index {start_index}")
    
    # Initialize and run pipeline
    pipeline = OCRPipeline()
    ocr_results = pipeline.process_images(num_images=num_images, start_index=start_index)
    
    # Save results
    pipeline.save_results(ocr_results, f"ocr_results_{num_images}_images.json")
    summary = pipeline.create_summary_report(ocr_results)
    
    print(f"\nOCR pipeline completed successfully!")
    print(f"Results saved in: {pipeline.output_folder}")
    print(f"Images downloaded to: {pipeline.images_folder}")

if __name__ == "__main__":
    main()

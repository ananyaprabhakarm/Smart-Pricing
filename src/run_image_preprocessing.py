#!/usr/bin/env python3
"""
Runner script for Image Preprocessing Pipeline
Usage: python run_image_preprocessing.py [num_images] [start_index]
"""

import sys
import os
from image_preprocessing import ImagePreprocessingPipeline

def main():
    # Parse command line arguments
    num_images = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    start_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    print(f"Running Image Preprocessing Pipeline for {num_images} images starting from index {start_index}")
    
    # Initialize and run pipeline
    pipeline = ImagePreprocessingPipeline()
    results = pipeline.process_images(num_images=num_images, start_index=start_index)
    
    # Save results
    pipeline.save_results(results, f"image_features_{num_images}_images.json")
    
    # Create feature matrices
    feature_matrices = pipeline.create_feature_matrix(results)
    normalized_matrices = pipeline.normalize_features(feature_matrices)
    
    # Save feature matrices
    pipeline.save_feature_matrices(feature_matrices, normalized_matrices)
    
    # Create summary
    summary = pipeline.create_summary_report(results)
    
    print(f"\nImage preprocessing pipeline completed successfully!")
    print(f"Results saved in: {pipeline.output_folder}")
    print(f"Images downloaded to: {pipeline.images_folder}")
    print(f"Success rate: {summary['success_rate_percentage']:.2f}%")

if __name__ == "__main__":
    main()

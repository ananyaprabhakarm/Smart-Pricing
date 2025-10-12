#!/usr/bin/env python3
"""
Test script for Image Preprocessing Pipeline - processes a small sample first
"""

import os
import sys
from image_preprocessing import ImagePreprocessingPipeline

def test_pipeline():
    """
    Test the image preprocessing pipeline with a small sample (10 images)
    """
    print("Testing Image Preprocessing Pipeline with 10 images...")
    
    # Initialize pipeline
    pipeline = ImagePreprocessingPipeline()
    
    # Process first 10 images as a test
    print("Processing first 10 images...")
    results = pipeline.process_images(num_images=10, start_index=0)
    
    # Save test results
    pipeline.save_results(results, "test_image_features_10_images.json")
    
    # Create feature matrices
    feature_matrices = pipeline.create_feature_matrix(results)
    normalized_matrices = pipeline.normalize_features(feature_matrices)
    
    # Save feature matrices
    pipeline.save_feature_matrices(feature_matrices, normalized_matrices)
    
    # Create summary
    summary = pipeline.create_summary_report(results)
    
    # Print detailed results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Images processed: {summary['total_images_processed']}")
    print(f"Successful extractions: {summary['successful_feature_extractions']}")
    print(f"Success rate: {summary['success_rate_percentage']:.1f}%")
    print(f"Device used: {summary['device_used']}")
    print(f"Available models: {', '.join(summary['available_models'])}")
    
    print("\nModel performance:")
    for model, count in summary['model_performance'].items():
        print(f"  {model}: {count} successful extractions")
    
    print("\nSample feature shapes:")
    for result in results[:3]:
        if result['status'] == 'success' and result['features']:
            print(f"\nSample {result['sample_id']}:")
            for model_name, features in result['features'].items():
                print(f"  {model_name}: {len(features)} features")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print(f"Results saved in: {pipeline.output_folder}")
    print("="*60)

if __name__ == "__main__":
    test_pipeline()

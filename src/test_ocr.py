#!/usr/bin/env python3
"""
Test script for OCR pipeline - processes a small sample first
"""

import os
import sys
from ocr_pipeline import OCRPipeline

def test_pipeline():
    """
    Test the OCR pipeline with a small sample (10 images)
    """
    print("Testing OCR Pipeline with 10 images...")
    
    # Initialize pipeline
    pipeline = OCRPipeline()
    
    # Process first 10 images as a test
    print("Processing first 10 images...")
    ocr_results = pipeline.process_images(num_images=10, start_index=0)
    
    # Save test results
    pipeline.save_results(ocr_results, "test_ocr_results_10_images.json")
    
    # Create summary
    summary = pipeline.create_summary_report(ocr_results)
    
    # Print detailed results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Images processed: {summary['total_images_processed']}")
    print(f"Successful OCR: {summary['successful_ocr_extractions']}")
    print(f"Success rate: {summary['success_rate_percentage']:.1f}%")
    print(f"Average confidence: {summary['average_confidence_score']:.1f}")
    print(f"Average words per image: {summary['average_words_per_image']:.1f}")
    
    print("\nSample extracted texts:")
    for i, result in enumerate(ocr_results[:3]):
        if result['status'] == 'success' and result['extracted_text']:
            print(f"\nSample {i+1} (ID: {result['sample_id']}):")
            print(f"Text: {result['extracted_text'][:200]}...")
            print(f"Confidence: {result['confidence_score']:.1f}")
            print(f"Words: {result['word_count']}")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print(f"Results saved in: {pipeline.output_folder}")
    print("="*60)

if __name__ == "__main__":
    test_pipeline()

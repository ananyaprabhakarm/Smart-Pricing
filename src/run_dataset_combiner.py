#!/usr/bin/env python3
"""
Simple runner script for dataset combination
Usage: python run_dataset_combiner.py
"""

import sys
import os
from dataset_combiner import DatasetCombiner

def main():
    print("Combining new_sample.csv with OCR CSV data...")
    
    # Initialize combiner
    combiner = DatasetCombiner()
    
    # Process all datasets
    results = combiner.process_all()
    
    if results:
        print(f"\nDataset combination completed successfully!")
        print(f"Combined datasets saved in: {combiner.output_folder}")
        print(f"ML-ready dataset: {len(results['ml_df'])} rows, {len(results['ml_df'].columns)} columns")
    else:
        print("Dataset combination failed!")

if __name__ == "__main__":
    main()


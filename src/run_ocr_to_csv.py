#!/usr/bin/env python3
"""
Simple runner script for OCR to CSV conversion
Usage: python run_ocr_to_csv.py
"""

import sys
import os
from ocr_to_csv_converter import OCRToCSVConverter

def main():
    print("Converting OCR results to CSV format...")
    
    # Initialize converter
    converter = OCRToCSVConverter()
    
    # Convert all OCR files
    converted_files = converter.convert_all_ocr_files()
    
    # Create combined CSV
    combined_csv = converter.create_combined_csv()
    
    print(f"\nConversion completed successfully!")
    print(f"CSV files saved in: {converter.csv_output_folder}")
    
    if converted_files:
        print(f"Total CSV files created: {len(converted_files)}")
    
    if combined_csv:
        print(f"Combined CSV: {combined_csv}")

if __name__ == "__main__":
    main()

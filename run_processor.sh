#!/bin/bash

# Simple script to run the universal CSV processor on INPUT_CSV folder

echo "========================================"
echo "Universal CSV Processor - Batch Mode"
echo "========================================"
echo ""

# Check if INPUT_CSV directory exists
if [ ! -d "INPUT_CSV" ]; then
    echo "❌ Error: INPUT_CSV directory not found!"
    echo "Please create an INPUT_CSV folder and add your CSV files to it."
    exit 1
fi

# Count CSV files
csv_count=$(ls -1 INPUT_CSV/*.csv 2>/dev/null | wc -l)
if [ "$csv_count" -eq 0 ]; then
    echo "❌ No CSV files found in INPUT_CSV folder!"
    exit 1
fi

echo "✅ Found $csv_count CSV files in INPUT_CSV folder"
echo ""

# Run the processor
python3 universal_csv_processor.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Processing completed successfully!"
    echo "Check the OUTPUT_PROCESSED_* folder for results."
else
    echo ""
    echo "❌ Processing failed. Please check the error messages above."
fi
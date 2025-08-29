#!/usr/bin/env python3
"""
Process only critical CSV files for equipment-vendor mapping
"""

import os
import csv
import json
from datetime import datetime
from pathlib import Path

# Critical files needed for equipment-vendor mapping
CRITICAL_FILES = [
    "Equipment.csv",
    "Equipment_Material.csv", 
    "Functional_Locations.csv",
    "Vendors.csv"
]

def simple_process(csv_path, output_path):
    """Simple processing without API calls - just structure the data"""
    print(f"Processing {csv_path.name}...")
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    results = []
    for i, row in enumerate(rows, 1):
        # Create semantic statement from row data
        semantic_parts = []
        for key, value in row.items():
            if value and value != 'null':
                semantic_parts.append(f"{key} is {value}")
        
        semantic_statement = f"Record with {', '.join(semantic_parts[:3])}..." if semantic_parts else "Record"
        
        # Create combined text
        combined_text = " | ".join([f"{k}: {v}" for k, v in row.items() if v])
        
        results.append({
            "row_number": i,
            "json_data": row,
            "semantic_statement": semantic_statement,
            "combined_text": semantic_statement + " | " + combined_text
        })
    
    # Save results
    output_data = {
        "metadata": {
            "total_records": len(results),
            "processing_time": 0,
            "api_calls": 0,
            "batch_size": 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "records": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"  ✓ Saved {len(results)} records")
    return True

def main():
    # Use existing output directory or create new one
    output_dirs = sorted(Path(".").glob("OUTPUT_INPUT_CSV_*"))
    if output_dirs:
        output_dir = output_dirs[-1]
        print(f"Using existing output directory: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"OUTPUT_INPUT_CSV_{timestamp}")
        output_dir.mkdir(exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    print("="*60)
    
    # Process critical files
    for filename in CRITICAL_FILES:
        csv_path = Path("INPUT_CSV") / filename
        if csv_path.exists():
            output_path = output_dir / f"{csv_path.stem}_processed.json"
            
            # Skip if already processed
            if output_path.exists():
                print(f"✓ {filename} already processed")
                continue
            
            simple_process(csv_path, output_path)
        else:
            print(f"⚠️ {filename} not found")
    
    print("\n" + "="*60)
    print("✅ Critical files processed!")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
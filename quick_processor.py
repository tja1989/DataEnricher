#!/usr/bin/env python3
"""
Quick CSV Processor - Robust processing for all CSV files
"""

import os
import csv
import json
import time
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = OpenAI()

def process_csv_file(csv_path, output_path):
    """Process a single CSV file"""
    print(f"Processing {csv_path.name}...")
    
    results = []
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Process in small batches
        batch_size = 5
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            
            # Create batch prompt
            batch_data = []
            for j, row in enumerate(batch):
                batch_data.append(f"Row {i+j+1}: {json.dumps(row)}")
            
            prompt = f"""Convert these CSV rows to semantic descriptions.
For each row, provide:
1. A semantic_statement describing the data
2. The original json_data
3. A combined_text (semantic + structured)

Data:
{chr(10).join(batch_data)}

Return a JSON array with objects containing: row_number, json_data, semantic_statement, combined_text"""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                batch_results = json.loads(content).get('results', [])
                
                # Add to results
                for result in batch_results:
                    results.append(result)
                    
                print(f"  Processed rows {i+1}-{min(i+batch_size, len(rows))}")
                
            except Exception as e:
                print(f"  Error processing batch at row {i+1}: {e}")
                # Add raw data for failed batch
                for j, row in enumerate(batch):
                    results.append({
                        "row_number": i+j+1,
                        "json_data": row,
                        "semantic_statement": f"Data for {csv_path.stem} row {i+j+1}",
                        "combined_text": " | ".join([f"{k}: {v}" for k, v in row.items() if v])
                    })
        
        # Save results
        output_data = {
            "metadata": {
                "total_records": len(results),
                "processing_time": 0,
                "api_calls": len(results) // batch_size + 1,
                "batch_size": batch_size,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "records": results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"  ✓ Saved {len(results)} records to {output_path.name}")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"OUTPUT_INPUT_CSV_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Process all CSV files
    csv_files = list(Path("INPUT_CSV").glob("*.csv"))
    
    successful = 0
    failed = 0
    
    for csv_file in csv_files:
        output_file = output_dir / f"{csv_file.stem}_processed.json"
        
        if process_csv_file(csv_file, output_file):
            successful += 1
        else:
            failed += 1
    
    # Create summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_folder": "INPUT_CSV",
        "output_folder": str(output_dir),
        "total_files": len(csv_files),
        "successful": successful,
        "failed": failed
    }
    
    with open(output_dir / "processing_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print(f"✅ Processing complete!")
    print(f"   Successful: {successful}/{len(csv_files)} files")
    print(f"   Output: {output_dir}")

if __name__ == "__main__":
    main()
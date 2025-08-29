#!/usr/bin/env python3
"""
Analyze EMSTEEL CSV files and test automatic batch sizing effectiveness
"""

import pandas as pd
import os
from pathlib import Path
from simple_csv_processor import SimpleCsvProcessor

def analyze_csv_characteristics(file_path):
    """Analyze characteristics of a CSV file"""
    df = pd.read_csv(file_path)
    
    # Basic stats
    num_rows = len(df)
    num_cols = len(df.columns)
    
    # Data density
    total_cells = num_rows * num_cols
    null_cells = df.isnull().sum().sum()
    null_percentage = (null_cells / total_cells * 100) if total_cells > 0 else 0
    
    # Content complexity
    sample_size = min(5, num_rows)
    sample = df.head(sample_size)
    
    total_chars = 0
    max_chars_row = 0
    
    for _, row in sample.iterrows():
        row_chars = 0
        for value in row.values:
            if pd.notna(value):
                chars = len(str(value))
                row_chars += chars
                total_chars += chars
        max_chars_row = max(max_chars_row, row_chars)
    
    avg_chars_per_row = total_chars / sample_size if sample_size > 0 else 0
    
    # Column types
    text_cols = len(df.select_dtypes(include=['object']).columns)
    numeric_cols = len(df.select_dtypes(include=['int64', 'float64']).columns)
    
    return {
        'file_name': Path(file_path).name,
        'rows': num_rows,
        'columns': num_cols,
        'null_percentage': round(null_percentage, 1),
        'avg_chars_per_row': round(avg_chars_per_row, 0),
        'max_chars_per_row': max_chars_row,
        'text_columns': text_cols,
        'numeric_columns': numeric_cols,
        'estimated_tokens_per_row': round((avg_chars_per_row / 4) + 50, 0)
    }

def test_batch_sizing():
    """Test automatic batch sizing on EMSTEEL files"""
    
    csv_dir = Path("EMSTEEL_PoC_Synthetic_CSVs")
    
    if not csv_dir.exists():
        print("❌ EMSTEEL_PoC_Synthetic_CSVs directory not found")
        return
    
    # Get all CSV files
    csv_files = list(csv_dir.glob("*.csv"))
    
    print("="*80)
    print("EMSTEEL CSV FILES ANALYSIS & BATCH SIZING")
    print("="*80)
    
    # Initialize processor for batch size testing
    processor = SimpleCsvProcessor(batch_size=None)  # Auto mode
    
    results = []
    
    for csv_file in sorted(csv_files):
        print(f"\n📁 Analyzing: {csv_file.name}")
        print("-" * 40)
        
        # Analyze file characteristics
        stats = analyze_csv_characteristics(csv_file)
        
        # Get auto-detected batch size
        df = pd.read_csv(csv_file)
        detected_batch_size = processor._determine_batch_size(df)
        stats['auto_batch_size'] = detected_batch_size
        
        # Calculate API efficiency
        api_calls_needed = (stats['rows'] + detected_batch_size - 1) // detected_batch_size
        stats['api_calls'] = api_calls_needed
        stats['rows_per_call'] = detected_batch_size
        
        results.append(stats)
        
        # Display results
        print(f"📊 Data Profile:")
        print(f"   • Rows: {stats['rows']}")
        print(f"   • Columns: {stats['columns']}")
        print(f"   • Null %: {stats['null_percentage']}%")
        print(f"   • Avg chars/row: {stats['avg_chars_per_row']:.0f}")
        print(f"   • Est. tokens/row: {stats['estimated_tokens_per_row']:.0f}")
        
        print(f"\n⚙️  Auto Batch Sizing:")
        print(f"   • Detected batch size: {detected_batch_size}")
        print(f"   • API calls needed: {api_calls_needed}")
        print(f"   • Efficiency: {detected_batch_size} rows/call")
        
        # Provide assessment
        if stats['estimated_tokens_per_row'] * detected_batch_size > 3000:
            print(f"   ⚠️  Warning: High token usage per batch")
        elif detected_batch_size == 1:
            print(f"   ⚠️  Warning: Very complex data, single row batches")
        else:
            print(f"   ✅ Batch size appropriate for data complexity")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_rows = sum(r['rows'] for r in results)
    total_api_calls = sum(r['api_calls'] for r in results)
    avg_batch_size = sum(r['auto_batch_size'] * r['rows'] for r in results) / total_rows if total_rows > 0 else 0
    
    print(f"\n📈 Overall Statistics:")
    print(f"   • Total files: {len(results)}")
    print(f"   • Total rows: {total_rows}")
    print(f"   • Total API calls: {total_api_calls}")
    print(f"   • Average batch size: {avg_batch_size:.1f}")
    print(f"   • API efficiency: {total_rows/total_api_calls:.1f} rows/call")
    
    print("\n📊 Per-File Batch Sizes:")
    for r in sorted(results, key=lambda x: x['auto_batch_size']):
        print(f"   • {r['file_name']:<25} → Batch size: {r['auto_batch_size']:2d} "
              f"(for {r['rows']:4d} rows = {r['api_calls']:3d} API calls)")
    
    # Cost estimation
    print("\n💰 API Cost Optimization:")
    print(f"   • Without batching: {total_rows} API calls")
    print(f"   • With auto-batching: {total_api_calls} API calls")
    print(f"   • Reduction: {((total_rows - total_api_calls) / total_rows * 100):.1f}%")
    print(f"   • Cost savings: ~{((total_rows - total_api_calls) / total_rows * 100):.0f}% reduction")

if __name__ == "__main__":
    test_batch_sizing()
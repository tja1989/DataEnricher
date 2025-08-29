#!/usr/bin/env python3
"""
Test automatic batch sizing feature
"""

import pandas as pd
import tempfile
import os
from pathlib import Path

# Create test CSV files with different characteristics
def create_test_files():
    """Create test CSV files with various data characteristics"""
    
    # 1. Small simple file (should use small batch)
    small_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['A', 'B', 'C', 'D', 'E'],
        'value': [10, 20, 30, 40, 50]
    }
    pd.DataFrame(small_data).to_csv('test_small.csv', index=False)
    print("✅ Created test_small.csv (5 rows, 3 columns)")
    
    # 2. Medium complexity file
    medium_data = {
        'equipment_id': [f'EQ-{i:04d}' for i in range(1, 26)],
        'description': [f'Equipment {i} with moderate description text' for i in range(1, 26)],
        'location': [f'Building {i%5 + 1}, Floor {i%3 + 1}' for i in range(1, 26)],
        'status': ['Active', 'Maintenance', 'Inactive'] * 8 + ['Active'],
        'installed_date': ['2023-01-15'] * 25,
        'last_service': ['2024-10-01'] * 25
    }
    pd.DataFrame(medium_data).to_csv('test_medium.csv', index=False)
    print("✅ Created test_medium.csv (25 rows, 6 columns)")
    
    # 3. Large complex file with many columns
    import random
    large_data = {}
    for col_num in range(15):
        col_name = f'field_{col_num+1}'
        if col_num < 3:
            # Text fields with longer content
            large_data[col_name] = [
                f'This is a longer text field with more detailed information about item {i}' 
                for i in range(1, 51)
            ]
        elif col_num < 8:
            # Numeric fields
            large_data[col_name] = [random.randint(100, 9999) for _ in range(50)]
        else:
            # Mixed with some nulls
            values = [f'Value_{i}' if i % 3 != 0 else None for i in range(1, 51)]
            large_data[col_name] = values
    
    pd.DataFrame(large_data).to_csv('test_large.csv', index=False)
    print("✅ Created test_large.csv (50 rows, 15 columns)")
    
    # 4. Sparse data file (many nulls)
    sparse_data = {
        'id': list(range(1, 31)),
        'primary_value': [f'PV-{i}' for i in range(1, 31)],
        'optional_1': [f'Opt1-{i}' if i % 3 == 0 else None for i in range(1, 31)],
        'optional_2': [f'Opt2-{i}' if i % 4 == 0 else None for i in range(1, 31)],
        'optional_3': [f'Opt3-{i}' if i % 5 == 0 else None for i in range(1, 31)],
        'notes': [None] * 25 + ['Some notes'] * 5
    }
    pd.DataFrame(sparse_data).to_csv('test_sparse.csv', index=False)
    print("✅ Created test_sparse.csv (30 rows, 6 columns, ~60% null)")


def test_batch_detection():
    """Test automatic batch size detection"""
    from simple_csv_processor import SimpleCsvProcessor
    
    print("\n" + "="*60)
    print("Testing Automatic Batch Size Detection")
    print("="*60 + "\n")
    
    # Create processor with auto batch sizing
    processor = SimpleCsvProcessor(batch_size=None)  # Auto mode
    
    test_files = [
        'test_small.csv',
        'test_medium.csv', 
        'test_large.csv',
        'test_sparse.csv'
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\n📊 Analyzing: {test_file}")
            df = pd.read_csv(test_file)
            
            # Test batch size detection
            detected_size = processor._determine_batch_size(df)
            
            print(f"   ➜ Auto-detected batch size: {detected_size}")
            
            # Show rationale
            num_rows = len(df)
            num_cols = len(df.columns)
            null_pct = (df.isnull().sum().sum() / (num_rows * num_cols) * 100)
            
            print(f"   📈 Data profile:")
            print(f"      - Rows: {num_rows}")
            print(f"      - Columns: {num_cols}")
            print(f"      - Null percentage: {null_pct:.1f}%")
            
            # Estimate efficiency
            api_calls = (num_rows + detected_size - 1) // detected_size  # Ceiling division
            print(f"   💰 API efficiency:")
            print(f"      - Total API calls needed: {api_calls}")
            print(f"      - Rows per API call: {detected_size}")


if __name__ == "__main__":
    print("🚀 Auto Batch Size Testing\n")
    
    # Create test files
    create_test_files()
    
    # Test batch detection
    test_batch_detection()
    
    print("\n" + "="*60)
    print("✅ Testing complete!")
    print("\nTo process a file with auto batch sizing:")
    print("  python simple_csv_processor.py <file.csv>")
    print("\nTo use fixed batch size:")
    print("  python simple_csv_processor.py <file.csv> -b 15")
    print("="*60)
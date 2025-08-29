#!/usr/bin/env python3
"""
Test the folder processing capability of simple_csv_processor.py
"""

import os
import pandas as pd
from pathlib import Path
import shutil

def create_test_csv_files():
    """Create test CSV files in INPUT_CSV folder"""
    
    # Create INPUT_CSV folder if it doesn't exist
    input_dir = Path("INPUT_CSV")
    input_dir.mkdir(exist_ok=True)
    
    print("📁 Creating test CSV files in INPUT_CSV folder...")
    
    # Test file 1: Simple data
    test1_data = {
        'id': [1, 2, 3],
        'name': ['Item A', 'Item B', 'Item C'],
        'value': [100, 200, 300]
    }
    pd.DataFrame(test1_data).to_csv(input_dir / "test_simple.csv", index=False)
    print("   ✅ Created test_simple.csv (3 rows)")
    
    # Test file 2: Medium data  
    test2_data = {
        'product_id': [f'P{i:03d}' for i in range(1, 8)],
        'description': [f'Product description {i}' for i in range(1, 8)],
        'price': [10.5 * i for i in range(1, 8)],
        'category': ['Electronics', 'Clothing', 'Electronics', 'Food', 'Clothing', 'Food', 'Electronics']
    }
    pd.DataFrame(test2_data).to_csv(input_dir / "test_products.csv", index=False)
    print("   ✅ Created test_products.csv (7 rows)")
    
    # Test file 3: Equipment-like data
    test3_data = {
        'equipment_id': ['EQ-001', 'EQ-002', 'EQ-003', 'EQ-004', 'EQ-005'],
        'name': ['Pump A', 'Motor B', 'Valve C', 'Sensor D', 'Controller E'],
        'location': ['Building 1', 'Building 2', 'Building 1', 'Building 3', 'Building 2'],
        'status': ['Active', 'Maintenance', 'Active', 'Active', 'Inactive']
    }
    pd.DataFrame(test3_data).to_csv(input_dir / "test_equipment.csv", index=False)
    print("   ✅ Created test_equipment.csv (5 rows)")
    
    print(f"\n✅ Created 3 test CSV files in {input_dir}")
    print(f"   Total rows across all files: 15")
    
    return str(input_dir)

def test_folder_processing():
    """Test processing all CSVs in INPUT_CSV folder"""
    print("\n" + "="*60)
    print("TESTING FOLDER PROCESSING")
    print("="*60)
    
    # Check if simple_csv_processor.py exists
    if not Path("simple_csv_processor.py").exists():
        print("❌ Error: simple_csv_processor.py not found")
        return
    
    # Create test files
    input_folder = create_test_csv_files()
    
    print("\n📊 Testing folder processing capabilities:")
    print("\n1. Default processing (no arguments - uses INPUT_CSV):")
    print("   python simple_csv_processor.py")
    
    print("\n2. Specific folder:")
    print("   python simple_csv_processor.py INPUT_CSV/")
    
    print("\n3. With output directory:")
    print("   python simple_csv_processor.py INPUT_CSV/ -o OUTPUT_TEST/")
    
    print("\n4. Process EMSTEEL folder:")
    print("   python simple_csv_processor.py EMSTEEL_PoC_Synthetic_CSVs/")
    
    print("\n" + "="*60)
    print("✅ Test files created successfully!")
    print("\nYou can now test the folder processing with:")
    print("   python simple_csv_processor.py")
    print("\nThis will process all CSV files in the INPUT_CSV folder")
    print("="*60)

def cleanup_test_files():
    """Clean up test files"""
    input_dir = Path("INPUT_CSV")
    
    if input_dir.exists():
        response = input("Remove test CSV files from INPUT_CSV? (y/n): ")
        if response.lower() == 'y':
            for csv_file in input_dir.glob("test_*.csv"):
                csv_file.unlink()
                print(f"   Removed {csv_file.name}")
            print("✅ Test files cleaned up")
        else:
            print("Test files kept in INPUT_CSV")

if __name__ == "__main__":
    test_folder_processing()
    
    # Optional cleanup
    print("\nCleanup:")
    cleanup_test_files()
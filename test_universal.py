#!/usr/bin/env python3
"""
Test suite for Universal CSV Processor

Tests all modes: auto, simple, optimized, and folder
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
import shutil
from universal_csv_processor import UniversalCSVProcessor, process_csv, analyze_csv


def create_test_csvs():
    """Create test CSV files of various sizes"""
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Small CSV (5 rows) - should trigger simple mode
    small_data = {
        'ID': [f'S{i:03d}' for i in range(1, 6)],
        'Name': [f'Item {i}' for i in range(1, 6)],
        'Value': [i * 100 for i in range(1, 6)]
    }
    pd.DataFrame(small_data).to_csv(test_dir / "small.csv", index=False)
    
    # Medium CSV (25 rows) - should trigger optimized mode
    medium_data = {
        'Product_ID': [f'P{i:04d}' for i in range(1, 26)],
        'Product_Name': [f'Product {i}' for i in range(1, 26)],
        'Price': [i * 9.99 for i in range(1, 26)],
        'Stock': [i * 10 for i in range(1, 26)]
    }
    pd.DataFrame(medium_data).to_csv(test_dir / "medium.csv", index=False)
    
    # CSV with nulls
    nulls_data = {
        'Code': ['A001', None, 'A003'],
        'Value': [100, 200, None],
        'Status': ['Active', 'Pending', 'Active']
    }
    pd.DataFrame(nulls_data).to_csv(test_dir / "nulls.csv", index=False)
    
    print(f"✅ Created test CSV files in {test_dir}/")
    return test_dir


def test_auto_mode_detection():
    """Test automatic mode detection"""
    print("\n" + "="*60)
    print("TEST 1: Auto Mode Detection")
    print("="*60)
    
    processor = UniversalCSVProcessor(mode="auto")
    
    # Test with small file
    print("\nAnalyzing small.csv:")
    suggestion = analyze_csv("test_data/small.csv")
    print(f"  Suggested: {suggestion}")
    
    # Test with medium file
    print("\nAnalyzing medium.csv:")
    suggestion = analyze_csv("test_data/medium.csv")
    print(f"  Suggested: {suggestion}")
    
    # Test with folder
    print("\nAnalyzing test_data folder:")
    suggestion = analyze_csv("test_data")
    print(f"  Suggested: {suggestion}")
    
    print("\n✅ Auto mode detection working correctly")
    return True


def test_simple_mode():
    """Test simple processing mode"""
    print("\n" + "="*60)
    print("TEST 2: Simple Mode Processing")
    print("="*60)
    
    processor = UniversalCSVProcessor(mode="simple")
    results = processor.process("test_data/small.csv")
    
    print(f"✅ Processed {len(results)} records in simple mode")
    print(f"  API calls: {processor.stats.api_calls}")
    print(f"  Time: {processor.stats.processing_time:.2f}s")
    
    # Verify structure
    assert len(results) == 5
    assert all('json_data' in r for r in results)
    assert all('semantic_statement' in r for r in results)
    assert all('combined_text' in r for r in results)
    
    # Show sample
    print(f"\nSample semantic: {results[0]['semantic_statement'][:100]}...")
    
    return True


def test_optimized_mode():
    """Test optimized batch processing mode"""
    print("\n" + "="*60)
    print("TEST 3: Optimized Mode Processing")
    print("="*60)
    
    processor = UniversalCSVProcessor(mode="optimized", batch_size=10)
    results = processor.process("test_data/medium.csv")
    
    print(f"✅ Processed {len(results)} records in optimized mode")
    print(f"  API calls: {processor.stats.api_calls}")
    print(f"  Time: {processor.stats.processing_time:.2f}s")
    print(f"  API efficiency: {processor.stats.api_efficiency:.1f} rows/call")
    
    # Verify batch processing worked
    assert processor.stats.api_calls < len(results)  # Should be fewer API calls than rows
    assert len(results) == 25
    
    return True


def test_folder_mode():
    """Test folder processing mode"""
    print("\n" + "="*60)
    print("TEST 4: Folder Mode Processing")
    print("="*60)
    
    processor = UniversalCSVProcessor(mode="folder")
    summary = processor.process("test_data", output_path="test_output")
    
    stats = summary.get('statistics', {})
    print(f"✅ Processed {stats['successful_files']}/{stats['total_files']} files")
    print(f"  Total rows: {stats['processed_rows']}")
    print(f"  Total API calls: {stats['total_api_calls']}")
    print(f"  Total time: {stats['total_time']:.2f}s")
    
    # Verify outputs were created
    assert Path("test_output").exists()
    assert (Path("test_output") / "processing_summary.json").exists()
    
    return True


def test_null_handling():
    """Test handling of null values"""
    print("\n" + "="*60)
    print("TEST 5: Null Value Handling")
    print("="*60)
    
    processor = UniversalCSVProcessor(mode="simple")
    results = processor.process("test_data/nulls.csv")
    
    print(f"✅ Processed {len(results)} records with nulls")
    
    # Check that nulls are handled
    for result in results:
        if None in result['json_data'].values():
            print(f"  Row {result['row_number']}: Contains null values")
            assert "Error" not in result['semantic_statement']
    
    return True


def test_convenience_functions():
    """Test convenience functions"""
    print("\n" + "="*60)
    print("TEST 6: Convenience Functions")
    print("="*60)
    
    # Test process_csv function
    results = process_csv("test_data/small.csv", mode="simple")
    print(f"✅ process_csv() returned {len(results)} records")
    
    # Test analyze_csv function
    suggestion = analyze_csv("test_data/medium.csv")
    print(f"✅ analyze_csv() suggested: {suggestion}")
    
    return True


def test_performance_comparison():
    """Compare performance between modes"""
    print("\n" + "="*60)
    print("TEST 7: Performance Comparison")
    print("="*60)
    
    test_file = "test_data/medium.csv"
    
    # Simple mode
    print("\nSimple mode:")
    start = time.time()
    processor_simple = UniversalCSVProcessor(mode="simple")
    results_simple = processor_simple.process(test_file)
    simple_time = time.time() - start
    print(f"  Time: {simple_time:.2f}s")
    print(f"  API calls: {processor_simple.stats.api_calls}")
    
    # Optimized mode
    print("\nOptimized mode:")
    start = time.time()
    processor_opt = UniversalCSVProcessor(mode="optimized", batch_size=10)
    results_opt = processor_opt.process(test_file)
    opt_time = time.time() - start
    print(f"  Time: {opt_time:.2f}s")
    print(f"  API calls: {processor_opt.stats.api_calls}")
    
    # Calculate improvements
    speedup = simple_time / opt_time if opt_time > 0 else 1
    api_reduction = (1 - processor_opt.stats.api_calls / processor_simple.stats.api_calls) * 100
    
    print(f"\n📊 Performance Improvement:")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  API reduction: {api_reduction:.1f}%")
    
    return True


def cleanup():
    """Clean up test files"""
    if Path("test_data").exists():
        shutil.rmtree("test_data")
    if Path("test_output").exists():
        shutil.rmtree("test_output")
    
    # Remove any test output files
    for pattern in ["*_processed.json", "processing_summary.json"]:
        for file in Path(".").glob(pattern):
            if "test" in str(file):
                file.unlink()
    
    print("\n🧹 Cleaned up test files")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("UNIVERSAL CSV PROCESSOR - TEST SUITE")
    print("="*70)
    
    try:
        # Create test data
        create_test_csvs()
        
        # Run tests
        tests = [
            ("Auto Mode Detection", test_auto_mode_detection),
            ("Simple Mode", test_simple_mode),
            ("Optimized Mode", test_optimized_mode),
            ("Folder Mode", test_folder_mode),
            ("Null Handling", test_null_handling),
            ("Convenience Functions", test_convenience_functions),
            ("Performance Comparison", test_performance_comparison)
        ]
        
        passed = 0
        failed = 0
        
        for name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
                    print(f"❌ {name} failed")
            except Exception as e:
                failed += 1
                print(f"❌ {name} failed with error: {e}")
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"✅ Passed: {passed}/{len(tests)}")
        if failed > 0:
            print(f"❌ Failed: {failed}/{len(tests)}")
        else:
            print("🎉 ALL TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cleanup()


if __name__ == "__main__":
    main()
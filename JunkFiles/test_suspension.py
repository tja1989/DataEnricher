#!/usr/bin/env python3
"""
Test script for suspension and resume functionality
"""

import os
import sys
from pathlib import Path
from universal_csv_processor import UniversalCSVProcessor, ProcessingSuspendedException, APIConnectionError

def test_suspension():
    """Test the suspension and resume capability"""
    
    # Test file path
    test_file = "INPUT_CSV/Equipment.csv"
    output_file = "test_output.json"
    checkpoint_file = "processing_state.json"
    
    # Check if test file exists
    if not os.path.exists(test_file):
        print(f"❌ Test file {test_file} not found")
        print("Please ensure INPUT_CSV folder has CSV files")
        return
    
    print("="*60)
    print("Testing Suspension and Resume Functionality")
    print("="*60)
    
    try:
        # Create processor
        processor = UniversalCSVProcessor(mode="simple")
        
        # Test 1: Check API connection
        print("\n1. Testing API connection check...")
        if processor.check_api_connection():
            print("   ✅ API connection successful")
        else:
            print("   ❌ API connection failed")
            print("   Testing wait for connection...")
            if processor.wait_for_api_connection(max_wait_seconds=30):
                print("   ✅ Connection restored")
            else:
                print("   ❌ Could not establish connection")
                return
        
        # Test 2: Process with suspension capability
        print("\n2. Testing process with suspension...")
        try:
            results = processor.process_with_suspension(
                test_file, 
                output_file,
                checkpoint_file
            )
            print(f"   ✅ Processed {len(results)} rows successfully")
            
            # Check if checkpoint was cleaned up
            if not os.path.exists(checkpoint_file):
                print("   ✅ Checkpoint file cleaned up after success")
            else:
                print("   ⚠️ Checkpoint file still exists")
                
        except ProcessingSuspendedException as e:
            print(f"   ⚠️ Processing suspended: {e}")
            print(f"   Checkpoint file exists: {os.path.exists(checkpoint_file)}")
            
            # Test resume
            print("\n3. Testing resume from checkpoint...")
            processor2 = UniversalCSVProcessor(mode="simple")
            try:
                results = processor2.process_with_suspension(
                    test_file,
                    output_file,
                    checkpoint_file
                )
                print(f"   ✅ Resume successful, total {len(results)} rows processed")
            except Exception as e:
                print(f"   ❌ Resume failed: {e}")
        
        except APIConnectionError as e:
            print(f"   ❌ API Connection Error: {e}")
        
        # Test 3: Check output
        if os.path.exists(output_file):
            import json
            with open(output_file, 'r') as f:
                output_data = json.load(f)
                print(f"\n4. Output validation:")
                print(f"   Total records: {output_data['metadata']['total_records']}")
                print(f"   Processing mode: {output_data['metadata']['mode']}")
                print(f"   API calls made: {output_data['metadata']['api_calls']}")
                
                # Check if all records have semantic statements
                records_with_semantic = sum(
                    1 for r in output_data['records'] 
                    if r.get('semantic_statement') and not r['semantic_statement'].startswith('Error')
                )
                print(f"   Records with valid semantic: {records_with_semantic}/{len(output_data['records'])}")
        
        # Clean up test files
        print("\n5. Cleaning up test files...")
        for file in [output_file, checkpoint_file]:
            if os.path.exists(file):
                os.remove(file)
                print(f"   ✅ Removed {file}")
        
        print("\n" + "="*60)
        print("✅ All tests completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()

def test_api_failure_simulation():
    """Test behavior with simulated API failure"""
    print("\n" + "="*60)
    print("Testing API Failure Handling")
    print("="*60)
    
    # Create processor with invalid API key
    print("\n1. Testing with invalid API key...")
    os.environ['OPENAI_API_KEY'] = 'invalid_key_for_testing'
    
    try:
        processor = UniversalCSVProcessor(mode="simple")
        result = processor.check_api_connection()
        if not result:
            print("   ✅ Correctly detected invalid API key")
        else:
            print("   ❌ Failed to detect invalid API key")
            
    except Exception as e:
        print(f"   ✅ Raised exception for invalid key: {type(e).__name__}")
    
    finally:
        # Restore real API key if it exists
        from dotenv import load_dotenv
        load_dotenv()

if __name__ == "__main__":
    # Run tests
    test_suspension()
    
    # Optional: test API failure handling
    if "--test-failure" in sys.argv:
        test_api_failure_simulation()
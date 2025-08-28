"""
Test ultra-simple LLM discovery on all 20 equipment.
"""

import csv
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from llm_discovery_ultra_simple import UltraSimpleLLMDiscovery
from vector_relationship_builder import FAISSVectorDB

load_dotenv()


def load_equipment_list(csv_path: str) -> list:
    """Load all equipment from CSV."""
    equipment = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            equipment.append({
                'id': row['Equipment_ID'],
                'name': row['Equipment_Name']
            })
    return equipment


def load_ground_truth():
    """Load ground truth for validation."""
    # Equipment-Material relationships
    equipment_materials = {}
    with open('EMSTEEL_PoC_Synthetic_CSVs/Equipment_Material.csv', 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            if line.strip():
                parts = line.strip().split(',')
                eq_id = parts[0]
                mat_id = parts[1]
                if eq_id not in equipment_materials:
                    equipment_materials[eq_id] = []
                equipment_materials[eq_id].append(mat_id)
    
    # Material-Vendor relationships
    material_vendors = {}
    with open('EMSTEEL_PoC_Synthetic_CSVs/Materials.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Approved_Suppliers']:
                material_vendors[row['Material_ID']] = row['Approved_Suppliers'].split(',')
    
    return equipment_materials, material_vendors


def test_all_equipment():
    """Test discovery on all equipment."""
    
    print("Loading vector database...")
    vector_db = FAISSVectorDB()
    vector_db.load(Path("faiss_vector_db"))
    
    discoverer = UltraSimpleLLMDiscovery(vector_db)
    
    equipment_list = load_equipment_list("EMSTEEL_PoC_Synthetic_CSVs/Equipment.csv")
    equipment_materials, material_vendors = load_ground_truth()
    
    print(f"Testing {len(equipment_list)} equipment\n")
    
    all_results = []
    statistics = {
        'total': len(equipment_list),
        'successful': 0,
        'failed': 0,
        'materials': {
            'found_when_expected': 0,
            'total_expected': 0,
            'found_without_source': 0,
            'total_found': 0
        },
        'vendors': {
            'equipment_with_vendors': 0,
            'expected_with_vendors': 0,
            'total_found': 0
        }
    }
    
    for i, eq in enumerate(equipment_list, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(equipment_list)}] {eq['name']} ({eq['id']})")
        
        try:
            # Run discovery
            result = discoverer.discover_all(eq['id'], eq['name'])
            all_results.append(result)
            
            if result['success']:
                statistics['successful'] += 1
                
                # Analyze materials
                found_materials = [m['material_id'] for m in result.get('materials', [])]
                expected_materials = equipment_materials.get(eq['id'], [])
                
                if expected_materials:
                    statistics['materials']['total_expected'] += len(expected_materials)
                    found_expected = [m for m in found_materials if m in expected_materials]
                    statistics['materials']['found_when_expected'] += len(found_expected)
                    print(f"  Materials: Found {len(found_expected)}/{len(expected_materials)} expected")
                else:
                    if found_materials:
                        statistics['materials']['found_without_source'] += len(found_materials)
                        print(f"  Materials: Found {len(found_materials)} (no source data)")
                    else:
                        print(f"  Materials: None found (none expected)")
                
                statistics['materials']['total_found'] += len(found_materials)
                
                # Analyze vendors
                found_vendors = result.get('vendors', [])
                if found_vendors:
                    statistics['vendors']['equipment_with_vendors'] += 1
                    statistics['vendors']['total_found'] += len(found_vendors)
                    print(f"  Vendors: Found {len(found_vendors)}")
                    for v in found_vendors:
                        print(f"    - {v['vendor_id']}: {v.get('supplies_materials', [])}")
                
                # Check if vendors were expected
                expected_vendors = set()
                for mat_id in expected_materials:
                    if mat_id in material_vendors:
                        expected_vendors.update(material_vendors[mat_id])
                if expected_vendors:
                    statistics['vendors']['expected_with_vendors'] += 1
                    found_vendor_ids = [v['vendor_id'] for v in found_vendors]
                    matched = len(set(found_vendor_ids) & expected_vendors)
                    print(f"  Vendor accuracy: {matched}/{len(expected_vendors)} expected vendors found")
                
                print(f"  Summary: {result.get('summary', '')[:150]}...")
                
            else:
                statistics['failed'] += 1
                print(f"  ❌ Failed: {result.get('summary', 'Unknown error')}")
                
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            statistics['failed'] += 1
            all_results.append({
                'equipment_id': eq['id'],
                'equipment_name': eq['name'],
                'success': False,
                'summary': str(e)
            })
        
        # Rate limiting
        time.sleep(0.5)
    
    # Final statistics
    print(f"\n{'='*60}")
    print("FINAL STATISTICS")
    print(f"{'='*60}")
    
    print(f"Success Rate: {statistics['successful']}/{statistics['total']} ({statistics['successful']/statistics['total']*100:.1f}%)")
    
    if statistics['materials']['total_expected'] > 0:
        material_recall = statistics['materials']['found_when_expected'] / statistics['materials']['total_expected'] * 100
        print(f"\nMaterials:")
        print(f"  Recall (found/expected): {statistics['materials']['found_when_expected']}/{statistics['materials']['total_expected']} ({material_recall:.1f}%)")
        print(f"  Extra discoveries: {statistics['materials']['found_without_source']} materials found without source data")
        print(f"  Total found: {statistics['materials']['total_found']}")
    
    print(f"\nVendors:")
    print(f"  Equipment with vendors: {statistics['vendors']['equipment_with_vendors']}/{statistics['total']} ({statistics['vendors']['equipment_with_vendors']/statistics['total']*100:.1f}%)")
    print(f"  Should have vendors: {statistics['vendors']['expected_with_vendors']}")
    print(f"  Total vendors found: {statistics['vendors']['total_found']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"ultra_simple_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'statistics': statistics,
            'results': all_results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    return statistics, all_results


if __name__ == "__main__":
    stats, results = test_all_equipment()
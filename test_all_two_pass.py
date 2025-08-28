"""
Exhaustive test of two-pass discovery on all 20 equipment.
"""

import csv
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from llm_discovery_two_pass import TwoPassLLMDiscovery
from vector_relationship_builder import FAISSVectorDB

load_dotenv()


def load_ground_truth():
    """Load ground truth data for validation."""
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
    
    material_vendors = {}
    with open('EMSTEEL_PoC_Synthetic_CSVs/Materials.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Approved_Suppliers']:
                material_vendors[row['Material_ID']] = row['Approved_Suppliers'].split(',')
    
    vendor_names = {}
    with open('EMSTEEL_PoC_Synthetic_CSVs/Vendors.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vendor_names[row['Vendor_ID']] = row['Vendor_Name']
    
    return equipment_materials, material_vendors, vendor_names


def test_all_equipment():
    """Test two-pass discovery on all equipment."""
    
    print("Loading vector database...")
    vector_db = FAISSVectorDB()
    vector_db.load(Path("faiss_vector_db"))
    
    discoverer = TwoPassLLMDiscovery(vector_db)
    
    # Load equipment list
    equipment_list = []
    with open('EMSTEEL_PoC_Synthetic_CSVs/Equipment.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            equipment_list.append({
                'id': row['Equipment_ID'],
                'name': row['Equipment_Name']
            })
    
    equipment_materials, material_vendors, vendor_names = load_ground_truth()
    
    print(f"Testing {len(equipment_list)} equipment with two-pass discovery\n")
    
    all_results = []
    statistics = {
        'total': len(equipment_list),
        'successful': 0,
        'failed': 0,
        'materials': {
            'total_found': 0,
            'correct_found': 0,
            'total_expected': 0,
            'found_without_source': 0
        },
        'vendors': {
            'total_found': 0,
            'with_proper_names': 0,
            'with_placeholders': 0,
            'equipment_with_vendors': 0
        },
        'passes': {
            'total_pass1_contexts': 0,
            'total_pass2_contexts': 0,
            'equipment_needing_pass2': 0
        }
    }
    
    for i, eq in enumerate(equipment_list, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(equipment_list)}] {eq['name']} ({eq['id']})")
        
        try:
            result = discoverer.discover_all(eq['id'], eq['name'])
            all_results.append(result)
            
            if result['success']:
                statistics['successful'] += 1
                
                # Analyze materials
                found_materials = [m['material_id'] for m in result.get('materials', [])]
                expected_materials = equipment_materials.get(eq['id'], [])
                
                statistics['materials']['total_found'] += len(found_materials)
                
                if expected_materials:
                    statistics['materials']['total_expected'] += len(expected_materials)
                    correct = len(set(found_materials) & set(expected_materials))
                    statistics['materials']['correct_found'] += correct
                    print(f"  Materials: {correct}/{len(expected_materials)} expected found, "
                          f"{len(found_materials)} total")
                else:
                    if found_materials:
                        statistics['materials']['found_without_source'] += len(found_materials)
                        print(f"  Materials: {len(found_materials)} found (no source data)")
                
                # Analyze vendors
                vendors = result.get('vendors', [])
                if vendors:
                    statistics['vendors']['equipment_with_vendors'] += 1
                    statistics['vendors']['total_found'] += len(vendors)
                    
                    # Check name quality
                    proper = 0
                    placeholder = 0
                    for vendor in vendors:
                        v_name = vendor.get('vendor_name', '')
                        v_id = vendor['vendor_id']
                        
                        if v_name in ['Company Name if known', 'Unknown', ''] or 'Company Name' in v_name:
                            placeholder += 1
                            print(f"    ❌ {v_id}: '{v_name}' (placeholder)")
                        else:
                            proper += 1
                            # Verify against ground truth
                            expected_name = vendor_names.get(v_id, 'NOT IN DB')
                            if v_name == expected_name:
                                print(f"    ✅ {v_id}: {v_name}")
                            else:
                                print(f"    ⚠️ {v_id}: {v_name} (expected: {expected_name})")
                    
                    statistics['vendors']['with_proper_names'] += proper
                    statistics['vendors']['with_placeholders'] += placeholder
                    
                    print(f"  Vendors: {len(vendors)} found, {proper} with proper names")
                
                # Pass statistics
                contexts = result.get('contexts_analyzed', {})
                pass1 = contexts.get('pass1', 0)
                pass2 = contexts.get('pass2', 0)
                statistics['passes']['total_pass1_contexts'] += pass1
                statistics['passes']['total_pass2_contexts'] += pass2
                if pass2 > 0:
                    statistics['passes']['equipment_needing_pass2'] += 1
                
                print(f"  Contexts: Pass1={pass1}, Pass2={pass2} "
                      f"(resolved {result.get('ids_resolved', {})})")
                
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
    
    # Calculate final metrics
    print(f"\n{'='*60}")
    print("FINAL STATISTICS - TWO-PASS DISCOVERY")
    print(f"{'='*60}")
    
    print(f"\nSuccess Rate: {statistics['successful']}/{statistics['total']} "
          f"({statistics['successful']/statistics['total']*100:.1f}%)")
    
    # Material metrics
    if statistics['materials']['total_expected'] > 0:
        recall = statistics['materials']['correct_found'] / statistics['materials']['total_expected']
        print(f"\nMaterials:")
        print(f"  Total found: {statistics['materials']['total_found']}")
        print(f"  Recall: {statistics['materials']['correct_found']}/{statistics['materials']['total_expected']} "
              f"({recall*100:.1f}%)")
        print(f"  Extra discoveries: {statistics['materials']['found_without_source']}")
    
    # Vendor metrics
    total_vendors = statistics['vendors']['total_found']
    if total_vendors > 0:
        name_quality = statistics['vendors']['with_proper_names'] / total_vendors
        print(f"\nVendors:")
        print(f"  Total found: {total_vendors}")
        print(f"  Equipment with vendors: {statistics['vendors']['equipment_with_vendors']}/{statistics['total']} "
              f"({statistics['vendors']['equipment_with_vendors']/statistics['total']*100:.1f}%)")
        print(f"  Name quality: {statistics['vendors']['with_proper_names']}/{total_vendors} "
              f"({name_quality*100:.1f}%) have proper names")
        print(f"  Placeholders: {statistics['vendors']['with_placeholders']}")
    
    # Pass efficiency
    print(f"\nTwo-Pass Efficiency:")
    print(f"  Equipment needing second pass: {statistics['passes']['equipment_needing_pass2']}/{statistics['total']} "
          f"({statistics['passes']['equipment_needing_pass2']/statistics['total']*100:.1f}%)")
    avg_pass1 = statistics['passes']['total_pass1_contexts'] / statistics['total']
    avg_pass2 = statistics['passes']['total_pass2_contexts'] / statistics['total']
    print(f"  Average contexts: Pass1={avg_pass1:.1f}, Pass2={avg_pass2:.1f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"two_pass_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'statistics': statistics,
            'results': all_results
        }, f, indent=2)
    
    print(f"\n✅ Full results saved to {output_file}")
    
    return statistics, all_results


if __name__ == "__main__":
    stats, results = test_all_equipment()
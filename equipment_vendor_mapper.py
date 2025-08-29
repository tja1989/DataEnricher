#!/usr/bin/env python3
"""
Two-Layer Equipment → Materials → Vendors Mapper
Uses hybrid vector database for retrieval with gap-filling
"""

import json
import csv
import numpy as np
import faiss
from openai import OpenAI
import os
import re
from typing import Dict, List, Any

# Load environment variables from .env file
def load_env():
    env_path = '.env'
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()

class EquipmentVendorMapper:
    def __init__(self):
        print("Initializing Equipment-Vendor Mapper...")
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Load FAISS indices
        self.semantic_index = faiss.read_index("vector_indices/semantic.index")
        self.structured_index = faiss.read_index("vector_indices/structured.index")
        
        # Load metadata
        with open("vector_indices/metadata.json", "r") as f:
            data = json.load(f)
            self.metadata = {int(k): v for k, v in data['metadata'].items()}
            self.stats = data['statistics']
        
        # Load PO Lines for vendor-material relationships
        self.vendor_material_map = self.load_po_lines()
        
        # Load vendor names
        self.vendor_names = self.load_vendor_names()
        
        print(f"✓ Loaded vector database with {self.stats['total_records']} records")
        print(f"✓ Loaded {len(self.vendor_material_map)} vendor-material relationships")
        print(f"✓ Loaded {len(self.vendor_names)} vendor names\n")
    
    def load_vendor_names(self):
        """Load vendor names from Vendors.csv"""
        vendor_names = {}
        vendor_file = "EMSTEEL_PoC_Synthetic_CSVs/Vendors.csv"
        
        if os.path.exists(vendor_file):
            with open(vendor_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vendor_id = row['Vendor_ID']
                    vendor_name = row['Vendor_Name']
                    vendor_names[vendor_id] = vendor_name
        
        return vendor_names
    
    def load_po_lines(self):
        """Load vendor-material relationships from PO_Lines.csv"""
        vendor_material_map = {}
        po_file = "EMSTEEL_PoC_Synthetic_CSVs/PO_Lines.csv"
        
        if os.path.exists(po_file):
            with open(po_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    material_id = row['Material_ID']
                    vendor_id = row['Vendor_ID']
                    
                    if material_id not in vendor_material_map:
                        vendor_material_map[material_id] = set()
                    vendor_material_map[material_id].add(vendor_id)
        
        # Convert sets to lists for JSON serialization
        return {k: list(v) for k, v in vendor_material_map.items()}
    
    def hybrid_search(self, query, k=10, alpha=0.5):
        """Perform hybrid search on vector database"""
        try:
            # Get embeddings
            query_semantic = self.client.embeddings.create(
                input=query,
                model="text-embedding-3-large"
            ).data[0].embedding
            
            query_structured = self.client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            ).data[0].embedding
            
            # Convert to numpy
            query_semantic = np.array([query_semantic]).astype('float32')
            query_structured = np.array([query_structured]).astype('float32')
            
            # Search both indices
            sem_distances, sem_indices = self.semantic_index.search(query_semantic, k)
            str_distances, str_indices = self.structured_index.search(query_structured, k)
            
            # Convert distances to similarities
            sem_similarities = 1 / (1 + sem_distances[0])
            str_similarities = 1 / (1 + str_distances[0])
            
            # Combine results
            results = []
            for i in range(k):
                sem_idx = int(sem_indices[0][i])
                combined_score = alpha * sem_similarities[i] + (1 - alpha) * str_similarities[i]
                
                results.append({
                    'metadata': self.metadata[sem_idx],
                    'score': float(combined_score)
                })
            
            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            return results
        
        except Exception as e:
            print(f"Error in search: {e}")
            return []
    
    def extract_ids_from_results(self, results):
        """Extract various IDs from search results"""
        extracted = {
            'equipment_ids': set(),
            'material_ids': set(),
            'vendor_ids': set()
        }
        
        for result in results:
            json_data = result['metadata'].get('json_data', {})
            
            # Extract Equipment IDs
            if 'Equipment_ID' in json_data:
                extracted['equipment_ids'].add(json_data['Equipment_ID'])
            
            # Extract Material IDs
            if 'Material_ID' in json_data:
                extracted['material_ids'].add(json_data['Material_ID'])
            
            # Extract from metadata ID field
            meta_id = result['metadata'].get('id', '')
            if meta_id.startswith('MAT-'):
                extracted['material_ids'].add(meta_id)
            elif meta_id.startswith('V-'):
                extracted['vendor_ids'].add(meta_id)
        
        return extracted
    
    def layer1_retrieval(self, equipment_name):
        """Layer 1: Initial retrieval for equipment → materials"""
        print(f"\n  Layer 1: Searching for '{equipment_name}'...")
        
        # Query for equipment and its materials
        queries = [
            (f"{equipment_name} spare parts materials", 0.5),
            (f"{equipment_name} maintenance components", 0.6),
            (equipment_name, 0.3)  # Direct equipment search
        ]
        
        all_results = []
        for query, alpha in queries:
            results = self.hybrid_search(query, k=10, alpha=alpha)
            all_results.extend(results)
        
        # Extract IDs and filter relevant results
        materials = {}
        equipment_id = None
        
        for result in all_results:
            meta = result['metadata']
            json_data = meta.get('json_data', {})
            
            # Find equipment ID
            if meta['type'] == 'Equipment' and equipment_name.lower() in str(json_data).lower():
                equipment_id = json_data.get('Equipment_ID', meta.get('id'))
            
            # Collect materials
            if meta['type'] == 'Equipment_Material':
                if equipment_id and json_data.get('Equipment_ID') == equipment_id:
                    mat_id = json_data.get('Material_ID')
                    if mat_id and mat_id not in materials:
                        materials[mat_id] = {
                            'material_id': mat_id,
                            'material_name': None,  # To be filled in Layer 2
                            'vendors': []
                        }
        
        print(f"    Found {len(materials)} materials")
        return equipment_id, materials
    
    def layer2_gap_filling(self, equipment_id, materials):
        """Layer 2: Fill gaps and get vendor information"""
        print(f"  Layer 2: Gap-filling and vendor retrieval...")
        
        # Fill material names and get vendor info
        for mat_id in list(materials.keys()):
            # Query for material details
            mat_results = self.hybrid_search(mat_id, k=5, alpha=0.1)
            
            for result in mat_results:
                json_data = result['metadata'].get('json_data', {})
                
                # Get material name/description
                if 'Material_ID' in json_data and json_data['Material_ID'] == mat_id:
                    if 'Notes' in json_data:
                        materials[mat_id]['material_name'] = json_data['Notes']
                    elif 'Material_Name' in json_data:
                        materials[mat_id]['material_name'] = json_data['Material_Name']
            
            # Get vendors from PO Lines mapping
            if mat_id in self.vendor_material_map:
                vendor_ids = self.vendor_material_map[mat_id]
                
                for vendor_id in vendor_ids:
                    # Query for vendor details
                    vendor_results = self.hybrid_search(vendor_id, k=3, alpha=0.1)
                    
                    vendor_info = {
                        'vendor_id': vendor_id,
                        'vendor_name': None
                    }
                    
                    # Try to get vendor name from results
                    for result in vendor_results:
                        if 'Vendor_Name' in str(result['metadata']):
                            # Extract vendor name if available
                            json_data = result['metadata'].get('json_data', {})
                            if 'Vendor_Name' in json_data:
                                vendor_info['vendor_name'] = json_data['Vendor_Name']
                                break
                    
                    # Fallback: Use vendor names from CSV or vendor ID
                    if not vendor_info['vendor_name']:
                        vendor_info['vendor_name'] = self.vendor_names.get(vendor_id, vendor_id)
                    
                    materials[mat_id]['vendors'].append(vendor_info)
        
        # Fallback for material names
        for mat_id, mat_info in materials.items():
            if not mat_info['material_name']:
                mat_info['material_name'] = mat_id  # Use ID as fallback
        
        print(f"    Filled gaps for {len(materials)} materials")
        return materials
    
    def process_equipment(self, equipment_name):
        """Process single equipment through both layers"""
        print(f"\nProcessing: {equipment_name}")
        
        # Layer 1: Initial retrieval
        equipment_id, materials = self.layer1_retrieval(equipment_name)
        
        if not equipment_id:
            # Try to find equipment ID through direct search
            results = self.hybrid_search(equipment_name, k=5, alpha=0.3)
            for result in results:
                if result['metadata']['type'] == 'Equipment':
                    equipment_id = result['metadata'].get('id')
                    break
        
        # If still no equipment found, use the name as ID
        if not equipment_id:
            equipment_id = equipment_name
            print(f"    Warning: Could not find equipment ID, using name as ID")
        
        # Layer 2: Gap filling
        if materials:
            materials = self.layer2_gap_filling(equipment_id, materials)
        
        # Format output
        output = {
            'equipment': equipment_name,
            'equipment_id': equipment_id,
            'materials': []
        }
        
        for mat_info in materials.values():
            material_entry = {
                'material': mat_info['material_name'],
                'material_id': mat_info['material_id'],
                'vendors': mat_info['vendors']
            }
            output['materials'].append(material_entry)
        
        return output
    
    def process_all_equipment(self, csv_file='Equipment_Master.csv'):
        """Process all equipment from master CSV file"""
        results = []
        
        print(f"\nLoading equipment from {csv_file}...")
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            equipment_list = [row['Equipment_Name'] for row in reader]
        
        print(f"Found {len(equipment_list)} equipment to process\n")
        print("="*70)
        
        for i, equipment_name in enumerate(equipment_list, 1):
            print(f"\n[{i}/{len(equipment_list)}]", end="")
            result = self.process_equipment(equipment_name)
            results.append(result)
        
        print("\n" + "="*70)
        print(f"\n✓ Processing complete!")
        
        return results


def main():
    # Initialize mapper
    mapper = EquipmentVendorMapper()
    
    # Process all equipment
    results = mapper.process_all_equipment()
    
    # Save results to JSON
    output_file = 'equipment_vendor_mapping.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Print summary statistics
    total_materials = sum(len(r['materials']) for r in results)
    total_vendors = sum(
        len(m['vendors']) 
        for r in results 
        for m in r['materials']
    )
    
    print(f"\nSummary:")
    print(f"  • Equipment processed: {len(results)}")
    print(f"  • Total materials found: {total_materials}")
    print(f"  • Total vendor relationships: {total_vendors}")
    
    # Show sample output
    print("\nSample output (first equipment with materials):")
    for result in results:
        if result['materials']:
            print(f"\nEquipment: {result['equipment']}")
            print(f"ID: {result['equipment_id']}")
            print(f"Materials: {len(result['materials'])}")
            for mat in result['materials'][:2]:  # Show first 2 materials
                print(f"  - {mat['material']} ({mat['material_id']})")
                if mat['vendors']:
                    print(f"    Vendors: {', '.join([v['vendor_name'] for v in mat['vendors']])}")
            break


if __name__ == "__main__":
    main()
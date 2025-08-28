"""
Generalized two-pass LLM discovery with complete ID-to-name resolution.
First pass: Get equipment contexts with IDs
Second pass: Resolve all IDs to their full records with names
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Any
import os
from openai import OpenAI
from dotenv import load_dotenv
from vector_relationship_builder import FAISSVectorDB

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwoPassLLMDiscovery:
    """Two-pass discovery: First get IDs, then resolve all names."""
    
    def __init__(self, vector_db: FAISSVectorDB, model: str = "gpt-4o-mini"):
        self.vector_db = vector_db
        self.model = model
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
    
    def extract_ids_needing_resolution(self, contexts: List) -> Dict[str, Set[str]]:
        """Extract all IDs that need name resolution from contexts."""
        
        ids_to_resolve = {
            'materials': set(),
            'vendors': set(),
            'equipment': set()
        }
        
        # Track what we already have names for
        materials_with_names = set()
        vendors_with_names = set()
        
        for doc, score in contexts:
            metadata = doc.metadata
            
            # Check for vendor IDs in Approved_Suppliers
            if 'Approved_Suppliers' in metadata and metadata['Approved_Suppliers']:
                vendor_ids = metadata['Approved_Suppliers'].split(',')
                ids_to_resolve['vendors'].update(vendor_ids)
            
            # Track vendor names we already have
            if 'Vendor_ID' in metadata and 'Vendor_Name' in metadata:
                vendors_with_names.add(metadata['Vendor_ID'])
            
            # Check for material IDs
            if 'Material_ID' in metadata:
                mat_id = metadata['Material_ID']
                if 'Material_Name' in metadata:
                    materials_with_names.add(mat_id)
                else:
                    ids_to_resolve['materials'].add(mat_id)
            
            # Check for cross-referenced equipment
            if 'Related_Equipment_ID' in metadata:
                ids_to_resolve['equipment'].add(metadata['Related_Equipment_ID'])
        
        # Remove IDs we already have names for
        ids_to_resolve['vendors'] = ids_to_resolve['vendors'] - vendors_with_names
        ids_to_resolve['materials'] = ids_to_resolve['materials'] - materials_with_names
        
        return ids_to_resolve
    
    def resolve_ids_to_names(self, ids_to_resolve: Dict[str, Set[str]]) -> List:
        """Second pass: Query for all IDs to get their full records with names."""
        
        additional_contexts = []
        
        # Resolve material IDs to names
        if ids_to_resolve['materials']:
            material_ids = list(ids_to_resolve['materials'])[:10]  # Limit to avoid huge queries
            logger.info(f"Resolving {len(material_ids)} material IDs: {material_ids}")
            
            material_query = f"{' '.join(material_ids)} material name description specifications"
            material_results = self.vector_db.search(material_query, k=5)
            additional_contexts.extend(material_results)
        
        # Resolve vendor IDs to names
        if ids_to_resolve['vendors']:
            vendor_ids = list(ids_to_resolve['vendors'])[:10]
            logger.info(f"Resolving {len(vendor_ids)} vendor IDs: {vendor_ids}")
            
            # Query each vendor ID individually for better results
            for vendor_id in vendor_ids:
                vendor_query = f"{vendor_id} vendor name company supplier"
                vendor_results = self.vector_db.search(vendor_query, k=2)
                additional_contexts.extend(vendor_results)
        
        # Resolve equipment IDs
        if ids_to_resolve['equipment']:
            equipment_ids = list(ids_to_resolve['equipment'])[:5]
            logger.info(f"Resolving {len(equipment_ids)} equipment IDs: {equipment_ids}")
            
            equip_query = f"{' '.join(equipment_ids)} equipment name specifications"
            equip_results = self.vector_db.search(equip_query, k=3)
            additional_contexts.extend(equip_results)
        
        return additional_contexts
    
    def discover_all(self, equipment_id: str, equipment_name: str) -> Dict[str, Any]:
        """Two-pass discovery with complete ID resolution."""
        
        logger.info(f"Two-pass discovery for {equipment_id}")
        
        # FIRST PASS: Standard equipment query
        query1 = f"{equipment_id} {equipment_name} spare parts materials vendors suppliers maintenance"
        contexts_pass1 = self.vector_db.search(query1, k=12)
        logger.info(f"First pass: Retrieved {len(contexts_pass1)} contexts")
        
        # Extract IDs that need resolution
        ids_to_resolve = self.extract_ids_needing_resolution(contexts_pass1)
        logger.info(f"IDs needing resolution - Materials: {len(ids_to_resolve['materials'])}, "
                   f"Vendors: {len(ids_to_resolve['vendors'])}, "
                   f"Equipment: {len(ids_to_resolve['equipment'])}")
        
        # SECOND PASS: Resolve IDs to get names
        contexts_pass2 = []
        if any(ids_to_resolve.values()):
            contexts_pass2 = self.resolve_ids_to_names(ids_to_resolve)
            logger.info(f"Second pass: Retrieved {len(contexts_pass2)} additional contexts")
        
        # Combine all contexts
        all_contexts = contexts_pass1 + contexts_pass2
        
        # Prepare contexts for LLM
        context_info = []
        for i, (doc, score) in enumerate(all_contexts, 1):
            metadata = doc.metadata
            relevant_metadata = {}
            
            # Include all potentially useful fields
            important_fields = [
                'Equipment_ID', 'Material_ID', 'Material_Name', 
                'Approved_Suppliers', 'Vendor_ID', 'Vendor_Name',
                'Manufacturer', 'OEM_Part_No', 'Class', 'Key_Specs'
            ]
            
            for field in important_fields:
                if field in metadata and metadata[field]:
                    relevant_metadata[field] = metadata[field]
            
            context_info.append({
                'context_num': i,
                'score': score,
                'pass': 'first' if i <= len(contexts_pass1) else 'second',
                'metadata': relevant_metadata,
                'text_snippet': doc.text[:200]
            })
        
        # Save debug info
        with open(f"debug_two_pass_{equipment_id}.json", 'w') as f:
            json.dump({
                'ids_to_resolve': {k: list(v) for k, v in ids_to_resolve.items()},
                'pass1_contexts': len(contexts_pass1),
                'pass2_contexts': len(contexts_pass2),
                'total_contexts': len(context_info)
            }, f, indent=2)
        
        # Create enhanced prompt for LLM
        system_prompt = """You are an equipment maintenance expert analyzing industrial equipment data.

CRITICAL INSTRUCTIONS:
1. Match EVERY Material_ID with its Material_Name from the contexts
2. Match EVERY Vendor_ID with its Vendor_Name from the contexts
3. If you see an ID in one context and its name in another, connect them
4. Look in BOTH first pass and second pass contexts for complete information
5. The second pass contexts specifically contain name resolutions for IDs found in first pass

For vendors:
- Approved_Suppliers field contains vendor IDs (e.g., "V-DBX-01,V-GULF-01")
- Look for matching Vendor_ID entries to find the company names
- If a Manufacturer field matches a vendor, note that connection

Return complete information with all IDs properly matched to their names."""

        user_prompt = f"""Equipment: {equipment_id} - {equipment_name}

Contexts from two-pass search:
{json.dumps(context_info, indent=2)}

IMPORTANT: 
- First pass contexts have equipment-material relationships
- Second pass contexts have ID-to-name mappings
- Cross-reference between them to get complete information

Extract and return in this exact JSON format:
{{
    "materials": [
        {{
            "material_id": "MAT-XXX",
            "material_name": "Full descriptive name from contexts",
            "source": "metadata/work_order/both_passes",
            "confidence": "high/medium/low"
        }}
    ],
    "vendors": [
        {{
            "vendor_id": "V-XXX",
            "vendor_name": "Full company name from contexts (NOT 'Company Name if known')",
            "supplies_materials": ["MAT-XXX"],
            "source": "Approved_Suppliers/Vendor_ID record"
        }}
    ],
    "summary": "Brief summary noting use of two-pass resolution"
}}

Remember: Use the second pass contexts to resolve any IDs to their proper names!"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Add metadata
            result['equipment_id'] = equipment_id
            result['equipment_name'] = equipment_name
            result['contexts_analyzed'] = {
                'pass1': len(contexts_pass1),
                'pass2': len(contexts_pass2),
                'total': len(all_contexts)
            }
            result['ids_resolved'] = {k: len(v) for k, v in ids_to_resolve.items()}
            result['success'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {equipment_id}: {e}")
            return {
                'equipment_id': equipment_id,
                'equipment_name': equipment_name,
                'materials': [],
                'vendors': [],
                'summary': f"Error: {str(e)}",
                'contexts_analyzed': {
                    'pass1': len(contexts_pass1),
                    'pass2': len(contexts_pass2),
                    'total': len(all_contexts)
                },
                'success': False
            }


def test_problematic_equipment():
    """Test with equipment that had vendor name issues."""
    
    # Equipment that previously had missing vendor names
    test_cases = [
        ("EQ-PUMP-CRN10-6-01", "Centrifugal Pump CRN 10-6"),
        ("EQ-GEAR-FLENDER-H3SH-01", "Main Drive Gearbox H3SH"),
        ("EQ-PUMP-KSB-65-160-01", "Process Pump ETA 65-160")
    ]
    
    print("Loading vector database...")
    vector_db = FAISSVectorDB()
    vector_db.load(Path("faiss_vector_db"))
    
    discoverer = TwoPassLLMDiscovery(vector_db)
    
    print(f"\n{'='*60}")
    print("TESTING TWO-PASS DISCOVERY ON PROBLEMATIC EQUIPMENT")
    print(f"{'='*60}")
    
    results = []
    
    for eq_id, eq_name in test_cases:
        print(f"\n{eq_id}:")
        result = discoverer.discover_all(eq_id, eq_name)
        results.append(result)
        
        # Analyze vendor name quality
        vendors = result.get('vendors', [])
        print(f"  Vendors found: {len(vendors)}")
        
        proper_names = 0
        for vendor in vendors:
            v_id = vendor['vendor_id']
            v_name = vendor.get('vendor_name', 'Unknown')
            
            if v_name in ['Company Name if known', 'Unknown', ''] or 'Company Name' in v_name:
                print(f"    ❌ {v_id}: '{v_name}' (STILL PLACEHOLDER)")
            else:
                proper_names += 1
                print(f"    ✅ {v_id}: {v_name}")
        
        if vendors:
            print(f"  Name quality: {proper_names}/{len(vendors)} have proper names ({proper_names/len(vendors)*100:.0f}%)")
        
        # Show resolution stats
        ids_resolved = result.get('ids_resolved', {})
        contexts = result.get('contexts_analyzed', {})
        print(f"  IDs resolved: Materials={ids_resolved.get('materials', 0)}, "
              f"Vendors={ids_resolved.get('vendors', 0)}")
        print(f"  Contexts: Pass1={contexts.get('pass1', 0)}, "
              f"Pass2={contexts.get('pass2', 0)}, Total={contexts.get('total', 0)}")
    
    # Save results
    with open("two_pass_test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY:")
    total_vendors = sum(len(r.get('vendors', [])) for r in results)
    proper_count = sum(1 for r in results for v in r.get('vendors', []) 
                      if v.get('vendor_name') not in ['Company Name if known', 'Unknown', ''] 
                      and 'Company Name' not in v.get('vendor_name', ''))
    
    print(f"Total vendors: {total_vendors}")
    print(f"With proper names: {proper_count} ({proper_count/total_vendors*100:.0f}%)")
    print("\nResults saved to two_pass_test_results.json")


if __name__ == "__main__":
    test_problematic_equipment()
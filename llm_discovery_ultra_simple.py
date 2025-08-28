"""
Ultra-simplified LLM discovery: One query, one LLM call, direct extraction.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import os
from openai import OpenAI
from dotenv import load_dotenv
from vector_relationship_builder import FAISSVectorDB

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UltraSimpleLLMDiscovery:
    """Ultra-simple approach: single query, single LLM call."""
    
    def __init__(self, vector_db: FAISSVectorDB, model: str = "gpt-4o-mini"):
        self.vector_db = vector_db
        self.model = model
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
    
    def discover_all(self, equipment_id: str, equipment_name: str) -> Dict[str, Any]:
        """Single query, single LLM call to find everything."""
        
        logger.info(f"Processing {equipment_id}")
        
        # ONE comprehensive search query
        query = f"{equipment_id} {equipment_name} spare parts materials vendors suppliers maintenance"
        contexts = self.vector_db.search(query, k=15)  # Increased from 10 to 15
        
        # Prepare contexts with metadata for LLM
        context_info = []
        for i, (doc, score) in enumerate(contexts, 1):
            # Extract key metadata fields
            metadata = doc.metadata
            relevant_metadata = {}
            
            # Fields we care about
            important_fields = [
                'Equipment_ID', 'Material_ID', 'Material_Name', 
                'Approved_Suppliers', 'Vendor_ID', 'Vendor_Name',
                'Manufacturer', 'OEM_Part_No'
            ]
            
            for field in important_fields:
                if field in metadata and metadata[field]:
                    relevant_metadata[field] = metadata[field]
            
            context_info.append({
                'context_num': i,
                'score': score,
                'metadata': relevant_metadata,
                'text_snippet': doc.text[:200]
            })
        
        # Debug: save contexts to file
        with open(f"debug_contexts_{equipment_id}.json", 'w') as f:
            json.dump(context_info, f, indent=2)
        
        # Create prompt for LLM
        system_prompt = """You are an equipment maintenance expert. 
Analyze the provided contexts to extract ALL materials/spare parts and vendors for the specified equipment.

IMPORTANT: 
- Look for Material_ID in metadata
- Look for Approved_Suppliers in metadata (contains comma-separated vendor IDs)
- Include materials mentioned in work orders or maintenance records
- Be comprehensive - include everything relevant"""

        user_prompt = f"""Equipment: {equipment_id} - {equipment_name}

Contexts with metadata:
{json.dumps(context_info, indent=2)}

Extract and return in this exact JSON format:
{{
    "materials": [
        {{
            "material_id": "MAT-XXX",
            "material_name": "Description",
            "source": "metadata/work_order/manual",
            "confidence": "high/medium/low"
        }}
    ],
    "vendors": [
        {{
            "vendor_id": "V-XXX",
            "vendor_name": "Company Name if known",
            "supplies_materials": ["MAT-XXX"],
            "source": "Approved_Suppliers field or other"
        }}
    ],
    "summary": "Brief summary of findings"
}}

REMEMBER: Check Approved_Suppliers field for vendor IDs!"""

        try:
            # Single LLM call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent JSON
                max_tokens=1500,
                response_format={"type": "json_object"}  # Force JSON response
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # Add equipment info
            result['equipment_id'] = equipment_id
            result['equipment_name'] = equipment_name
            result['contexts_analyzed'] = len(contexts)
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
                'contexts_analyzed': len(contexts),
                'success': False
            }


def test_single():
    """Test with one equipment."""
    
    # Load vector database
    vector_db = FAISSVectorDB()
    vector_db.load(Path("faiss_vector_db"))
    
    # Initialize
    discoverer = UltraSimpleLLMDiscovery(vector_db)
    
    # Test case
    equipment_id = "EQ-GEAR-FLENDER-H3SH-01"
    equipment_name = "Main Drive Gearbox H3SH"
    
    print(f"\n{'='*60}")
    print(f"Testing Ultra-Simple Discovery")
    print(f"{'='*60}")
    print(f"Equipment: {equipment_name} ({equipment_id})")
    
    # Discover
    result = discoverer.discover_all(equipment_id, equipment_name)
    
    # Display
    print(f"\nContexts analyzed: {result['contexts_analyzed']}")
    print(f"Success: {result['success']}")
    
    print(f"\nMaterials found: {len(result.get('materials', []))}")
    for mat in result.get('materials', []):
        print(f"  - {mat['material_id']}: {mat['material_name']}")
        print(f"    Confidence: {mat['confidence']}, Source: {mat['source']}")
    
    print(f"\nVendors found: {len(result.get('vendors', []))}")
    for vendor in result.get('vendors', []):
        print(f"  - {vendor['vendor_id']}: {vendor.get('vendor_name', 'Unknown')}")
        print(f"    Supplies: {', '.join(vendor.get('supplies_materials', []))}")
    
    print(f"\nSummary: {result.get('summary', '')}")
    
    # Save
    with open("ultra_simple_test.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to ultra_simple_test.json")


if __name__ == "__main__":
    test_single()
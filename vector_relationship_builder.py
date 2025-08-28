"""
FAISS vector database builder with OpenAI embeddings for equipment relationships.
"""

import json
import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time

import openai
from openai import OpenAI
from equipment_relationship_models import (
    VendorInfo, MaterialWithVendors, EquipmentRelationship, 
    RelationshipSearchResult, VectorSearchParams, CriticalityLevel
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Document with embedding and metadata."""
    id: str
    text: str
    embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]
    source_file: str
    entity_type: str  # equipment, material, vendor, work_order, etc.


class FAISSVectorDB:
    """FAISS vector database with OpenAI embeddings."""
    
    def __init__(self, embedding_model: str = "text-embedding-3-small", dimension: int = 1536):
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.documents: List[VectorDocument] = []
        self.id_to_index: Dict[str, int] = {}
        
        # Initialize OpenAI client with API key from environment
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running.")
        self.client = OpenAI(api_key=api_key)
        
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.embedding_model
            )
            embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
            
            # Normalize for cosine similarity
            for embedding in embeddings:
                embedding /= np.linalg.norm(embedding)
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """Add documents to the vector database."""
        logger.info(f"Adding {len(documents)} documents to vector database")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            texts = []
            batch_docs = []
            
            for doc in batch:
                # Create embedding text
                if 'embedding_text' in doc:
                    text = doc['embedding_text']
                else:
                    # Fallback to combining fields
                    text = json.dumps(doc.get('structured_data', doc))
                
                texts.append(text)
                
                # Create VectorDocument
                vec_doc = VectorDocument(
                    id=doc.get('metadata', {}).get('primary_key', str(doc.get('row_id', ''))),
                    text=text,
                    embedding=None,
                    metadata=doc.get('structured_data', doc),
                    source_file=doc.get('source_file', ''),
                    entity_type=doc.get('metadata', {}).get('table_type', 'unknown')
                )
                batch_docs.append(vec_doc)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Add to index
            for j, (doc, embedding) in enumerate(zip(batch_docs, embeddings)):
                doc.embedding = embedding
                self.documents.append(doc)
                self.id_to_index[doc.id] = len(self.documents) - 1
                
            # Add embeddings to FAISS
            embeddings_matrix = np.vstack(embeddings)
            self.index.add(embeddings_matrix)
            
            logger.info(f"Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
    
    def search(self, query: str, k: int = 10, filter_type: Optional[str] = None) -> List[Tuple[VectorDocument, float]]:
        """Search for similar documents."""
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, min(k * 3, len(self.documents)))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                
                # Apply filter if specified
                if filter_type and doc.entity_type != filter_type:
                    continue
                
                results.append((doc, float(dist)))
                
                if len(results) >= k:
                    break
        
        return results
    
    def search_by_id(self, entity_id: str) -> Optional[VectorDocument]:
        """Search for a document by ID."""
        if entity_id in self.id_to_index:
            return self.documents[self.id_to_index[entity_id]]
        return None
    
    def save(self, path: str):
        """Save the vector database to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save documents and metadata
        with open(path / "documents.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'id_to_index': self.id_to_index,
                'embedding_model': self.embedding_model,
                'dimension': self.dimension
            }, f)
        
        logger.info(f"Vector database saved to {path}")
    
    def load(self, path: str):
        """Load the vector database from disk."""
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load documents and metadata
        with open(path / "documents.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.id_to_index = data['id_to_index']
            self.embedding_model = data['embedding_model']
            self.dimension = data['dimension']
        
        logger.info(f"Vector database loaded from {path}")


class RelationshipBuilder:
    """Build equipment-material-vendor relationships using LLM and vector search."""
    
    def __init__(self, vector_db: FAISSVectorDB):
        self.vector_db = vector_db
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running.")
        self.client = OpenAI(api_key=api_key)
        self.relationships_cache: Dict[str, EquipmentRelationship] = {}
        
    def find_equipment_materials(self, equipment_id: str, equipment_name: str) -> List[Dict[str, Any]]:
        """Find materials related to an equipment using vector search."""
        # Search for materials related to this equipment
        queries = [
            f"spare parts materials for {equipment_name} {equipment_id}",
            f"maintenance materials {equipment_name}",
            f"{equipment_id} replacement parts",
            equipment_id  # Direct ID search
        ]
        
        all_materials = {}
        for query in queries:
            results = self.vector_db.search(query, k=20, filter_type="equipment_material")
            for doc, score in results:
                if doc.metadata.get('Equipment_ID') == equipment_id:
                    material_id = doc.metadata.get('Material_ID')
                    if material_id and material_id not in all_materials:
                        all_materials[material_id] = (doc, score)
        
        # Also search in materials directly
        material_results = self.vector_db.search(
            f"{equipment_name} spare parts materials", 
            k=20, 
            filter_type="vendor"
        )
        
        # Return unique materials
        materials = []
        for material_id, (doc, score) in all_materials.items():
            # Get full material details
            material_doc = self.vector_db.search_by_id(material_id)
            if material_doc:
                materials.append(material_doc.metadata)
            else:
                materials.append(doc.metadata)
        
        return materials
    
    def find_material_vendors(self, material_id: str, material_name: str) -> List[VendorInfo]:
        """Find vendors for a specific material."""
        vendors = []
        
        # Get material document
        material_doc = self.vector_db.search_by_id(material_id)
        if material_doc:
            approved_suppliers = material_doc.metadata.get('Approved_Suppliers', '')
            if approved_suppliers:
                vendor_ids = [v.strip() for v in approved_suppliers.split(',')]
                
                for vendor_id in vendor_ids:
                    vendor_doc = self.vector_db.search_by_id(vendor_id)
                    if vendor_doc:
                        vendor_info = VendorInfo(
                            vendor_id=vendor_id,
                            vendor_name=vendor_doc.metadata.get('Vendor_Name', ''),
                            lead_time_days=vendor_doc.metadata.get('Typical_Lead_Time_Days'),
                            contact_email=vendor_doc.metadata.get('Contact_Email'),
                            contact_phone=vendor_doc.metadata.get('Contact_Phone'),
                            is_oem_authorized=vendor_doc.metadata.get('OEM_Authorized') == 'Y',
                            region=vendor_doc.metadata.get('Region'),
                            approval_status=vendor_doc.metadata.get('Approval_Status')
                        )
                        vendors.append(vendor_info)
        
        # If no direct vendors found, search semantically
        if not vendors:
            search_results = self.vector_db.search(
                f"vendor supplier for {material_name} {material_id}",
                k=5,
                filter_type="vendor"
            )
            for doc, score in search_results:
                if score > 0.7:  # Threshold for relevance
                    vendor_info = VendorInfo(
                        vendor_id=doc.metadata.get('Vendor_ID', ''),
                        vendor_name=doc.metadata.get('Vendor_Name', ''),
                        lead_time_days=doc.metadata.get('Typical_Lead_Time_Days'),
                        contact_email=doc.metadata.get('Contact_Email'),
                        contact_phone=doc.metadata.get('Contact_Phone'),
                        is_oem_authorized=doc.metadata.get('OEM_Authorized') == 'Y',
                        region=doc.metadata.get('Region'),
                        approval_status=doc.metadata.get('Approval_Status')
                    )
                    vendors.append(vendor_info)
        
        return vendors
    
    def build_equipment_relationship(self, equipment_id: str) -> Optional[EquipmentRelationship]:
        """Build complete relationship for an equipment."""
        # Check cache
        if equipment_id in self.relationships_cache:
            return self.relationships_cache[equipment_id]
        
        # Get equipment details
        equipment_doc = self.vector_db.search_by_id(equipment_id)
        if not equipment_doc:
            logger.warning(f"Equipment {equipment_id} not found")
            return None
        
        equipment_data = equipment_doc.metadata
        
        # Find related materials
        materials = self.find_equipment_materials(
            equipment_id, 
            equipment_data.get('Equipment_Name', '')
        )
        
        # Build material with vendors list
        spare_parts = []
        for material in materials:
            material_id = material.get('Material_ID', '')
            vendors = self.find_material_vendors(
                material_id,
                material.get('Material_Name', '')
            )
            
            material_with_vendors = MaterialWithVendors(
                material_id=material_id,
                material_name=material.get('Material_Name', ''),
                material_class=material.get('Class'),
                quantity=material.get('Qty'),
                unit=material.get('Base_UOM'),
                current_stock=material.get('Current_Stock'),
                min_level=material.get('Min_Level'),
                max_level=material.get('Max_Level'),
                is_critical=material.get('Is_Critical'),
                is_alternate=material.get('Is_Alternate', False),
                vendors=vendors,
                oem_part_no=material.get('OEM_Part_No'),
                spec_sheet=material.get('Spec_Sheet_File')
            )
            spare_parts.append(material_with_vendors)
        
        # Create equipment relationship
        criticality = equipment_data.get('Criticality', 'U')
        if criticality not in ['H', 'M', 'L']:
            criticality = 'U'
        
        relationship = EquipmentRelationship(
            equipment_id=equipment_id,
            equipment_name=equipment_data.get('Equipment_Name', ''),
            manufacturer=equipment_data.get('Manufacturer', ''),
            model=equipment_data.get('Model'),
            serial_no=equipment_data.get('Serial_No'),
            criticality=CriticalityLevel(criticality),
            functional_location=equipment_data.get('FL_ID'),
            downtime_cost_per_hour=equipment_data.get('Downtime_Cost_per_Hour_AED'),
            spare_parts=spare_parts
        )
        
        # Calculate metrics
        relationship.calculate_metrics()
        
        # Assess supply chain risk using LLM
        relationship = self.assess_supply_chain_risk(relationship)
        
        # Cache the result
        self.relationships_cache[equipment_id] = relationship
        
        return relationship
    
    def assess_supply_chain_risk(self, relationship: EquipmentRelationship) -> EquipmentRelationship:
        """Use LLM to assess supply chain risk."""
        try:
            # Prepare context for LLM
            context = {
                "equipment": relationship.equipment_name,
                "criticality": relationship.criticality,
                "downtime_cost": relationship.downtime_cost_per_hour,
                "total_spare_parts": relationship.total_spare_parts,
                "total_vendors": relationship.total_vendors,
                "spare_parts_summary": [
                    {
                        "name": part.material_name,
                        "vendors_count": len(part.vendors),
                        "stock": part.current_stock,
                        "min_level": part.min_level
                    }
                    for part in relationship.spare_parts[:5]  # First 5 for context
                ]
            }
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a supply chain risk analyst. Assess the risk level and provide brief recommendations."
                    },
                    {
                        "role": "user",
                        "content": f"""Assess supply chain risk for this equipment:
                        {json.dumps(context, indent=2)}
                        
                        Provide:
                        1. Risk level (High/Medium/Low)
                        2. Key risk factors (list 2-3)
                        3. Brief recommendations (list 2-3)
                        
                        Format as JSON with keys: risk_level, risk_factors, recommendations"""
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse LLM response
            risk_assessment = json.loads(response.choices[0].message.content)
            
            relationship.supply_chain_risk = risk_assessment.get('risk_level', 'Not Assessed')
            relationship.risk_factors = risk_assessment.get('risk_factors', [])
            relationship.recommendations = risk_assessment.get('recommendations', [])
            
        except Exception as e:
            logger.error(f"Error assessing supply chain risk: {e}")
            relationship.supply_chain_risk = "Error in assessment"
        
        return relationship
    
    def build_all_relationships(self) -> Dict[str, EquipmentRelationship]:
        """Build relationships for all equipment in the database."""
        all_relationships = {}
        
        # Find all equipment
        equipment_docs = [
            doc for doc in self.vector_db.documents 
            if doc.entity_type == "equipment"
        ]
        
        logger.info(f"Building relationships for {len(equipment_docs)} equipment")
        
        for i, doc in enumerate(equipment_docs):
            equipment_id = doc.id
            relationship = self.build_equipment_relationship(equipment_id)
            if relationship:
                all_relationships[equipment_id] = relationship
            
            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i + 1}/{len(equipment_docs)} equipment")
        
        return all_relationships


def load_vector_data(vector_output_dir: Path) -> List[Dict[str, Any]]:
    """Load all vector JSON files from output directory."""
    all_documents = []
    
    for json_file in vector_output_dir.glob("*_vector.json"):
        logger.info(f"Loading {json_file.name}")
        with open(json_file, 'r') as f:
            documents = json.load(f)
            all_documents.extend(documents)
    
    logger.info(f"Loaded {len(all_documents)} total documents")
    return all_documents


def main():
    """Main execution function."""
    # Setup paths
    vector_output_dir = Path("EMSTEEL_vector_output")
    faiss_db_path = Path("faiss_vector_db")
    
    # Initialize vector database
    logger.info("Initializing FAISS vector database with OpenAI embeddings")
    vector_db = FAISSVectorDB()
    
    # Check if database already exists
    if (faiss_db_path / "index.faiss").exists():
        logger.info("Loading existing vector database")
        vector_db.load(faiss_db_path)
    else:
        # Load and index all documents
        logger.info("Building new vector database")
        documents = load_vector_data(vector_output_dir)
        vector_db.add_documents(documents)
        vector_db.save(faiss_db_path)
    
    # Initialize relationship builder
    logger.info("Initializing relationship builder")
    builder = RelationshipBuilder(vector_db)
    
    # Build relationships for all equipment
    logger.info("Building equipment relationships")
    all_relationships = builder.build_all_relationships()
    
    # Save relationships
    output_file = Path("equipment_relationships.json")
    with open(output_file, 'w') as f:
        # Convert to dict for JSON serialization
        relationships_dict = {
            eq_id: rel.model_dump() 
            for eq_id, rel in all_relationships.items()
        }
        json.dump(relationships_dict, f, indent=2, default=str)
    
    logger.info(f"Saved {len(all_relationships)} equipment relationships to {output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("EQUIPMENT RELATIONSHIP SUMMARY")
    print("="*50)
    
    total_equipment = len(all_relationships)
    total_materials = sum(len(rel.spare_parts) for rel in all_relationships.values())
    all_vendors = set()
    for rel in all_relationships.values():
        all_vendors.update(rel.unique_vendors)
    
    print(f"Total Equipment: {total_equipment}")
    print(f"Total Material Links: {total_materials}")
    print(f"Total Unique Vendors: {len(all_vendors)}")
    
    # Show high-risk equipment
    high_risk = [
        rel for rel in all_relationships.values() 
        if rel.supply_chain_risk == "High"
    ]
    if high_risk:
        print(f"\nHigh Risk Equipment ({len(high_risk)}):")
        for rel in high_risk[:3]:
            print(f"  - {rel.equipment_name} ({rel.equipment_id})")
            for factor in rel.risk_factors[:2]:
                print(f"    Risk: {factor}")
    
    return vector_db, builder, all_relationships


if __name__ == "__main__":
    main()
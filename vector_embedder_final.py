#!/usr/bin/env python3
"""
Hybrid Vector Embedder for Equipment Data (Final Version)
Uses dual embedding streams: semantic and structured
"""

import json
import os
import glob
import numpy as np
import faiss
from datetime import datetime
import time

# Set environment variable before importing sentence_transformers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from sentence_transformers import SentenceTransformer

class HybridVectorEmbedder:
    def __init__(self):
        print("="*60)
        print("Initializing Hybrid Vector Embedder")
        print("="*60)
        
        # Initialize two different models for each stream
        print("\n1. Loading semantic model (all-mpnet-base-v2)...")
        self.semantic_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')
        print("   ✓ Semantic model loaded successfully")
        
        print("\n2. Loading structured model (all-MiniLM-L6-v2)...")
        self.structured_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        print("   ✓ Structured model loaded successfully")
        
        # Get embedding dimensions
        self.semantic_dim = 768  # all-mpnet-base-v2 dimension
        self.structured_dim = 384  # all-MiniLM-L6-v2 dimension
        
        # Initialize FAISS indices
        print("\n3. Initializing FAISS indices...")
        self.semantic_index = faiss.IndexFlatL2(self.semantic_dim)
        self.structured_index = faiss.IndexFlatL2(self.structured_dim)
        print("   ✓ FAISS indices initialized")
        
        # Metadata storage
        self.metadata = {}
        
    def load_json_files(self, directory_pattern="OUTPUT_*"):
        """Dynamically load all processed JSON files from OUTPUT directories"""
        all_records = []
        
        print("\n" + "="*60)
        print("Loading JSON Files")
        print("="*60)
        
        # Find all OUTPUT directories
        output_dirs = glob.glob(directory_pattern)
        print(f"\nFound {len(output_dirs)} OUTPUT directories")
        
        for output_dir in output_dirs:
            # Find all *_processed.json files (excluding processing_summary.json)
            json_files = glob.glob(os.path.join(output_dir, "*_processed.json"))
            json_files = [f for f in json_files if "processing_summary" not in f]
            
            print(f"\nDirectory: {output_dir}")
            
            for json_file in json_files:
                file_type = os.path.basename(json_file).replace("_processed.json", "")
                
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                # Extract records from the JSON structure
                if 'records' in data:
                    print(f"  • {file_type}: {len(data['records'])} records")
                    for record in data['records']:
                        # Add source information
                        record['source_file'] = os.path.basename(json_file)
                        record['source_type'] = file_type
                        record['source_dir'] = output_dir
                        all_records.append(record)
        
        print(f"\n✓ Total records loaded: {len(all_records)}")
        return all_records
    
    def extract_id_from_record(self, record):
        """Extract the appropriate ID based on record type"""
        json_data = record.get('json_data', {})
        
        # Try different ID fields based on data type
        if 'Equipment_ID' in json_data:
            return json_data['Equipment_ID']
        elif 'FL_ID' in json_data:
            return json_data['FL_ID']
        elif 'Material_ID' in json_data:
            # For equipment materials, combine Equipment_ID and Material_ID
            eq_id = json_data.get('Equipment_ID', 'unknown')
            mat_id = json_data.get('Material_ID', 'unknown')
            return f"{eq_id}_{mat_id}"
        else:
            # Fallback to row number if no ID found
            return f"row_{record.get('row_number', 'unknown')}"
    
    def json_to_structured_text(self, json_data):
        """Convert JSON data to structured text format"""
        parts = []
        for key, value in json_data.items():
            # Convert value to string and handle None values
            if value is not None:
                parts.append(f"{key}: {value}")
        return " | ".join(parts)
    
    def process_records(self, records):
        """Process all records and create embeddings"""
        print("\n" + "="*60)
        print("Generating Embeddings")
        print("="*60)
        
        start_time = time.time()
        
        # Process each record individually to avoid batching issues
        semantic_embeddings = []
        structured_embeddings = []
        
        print(f"\nProcessing {len(records)} records...")
        
        for i, record in enumerate(records):
            if i % 5 == 0:
                print(f"  Progress: {i}/{len(records)} records processed...")
            
            # Extract semantic text
            semantic_text = record.get('semantic_statement', '')
            
            # Extract and format structured text from json_data
            json_data = record.get('json_data', {})
            structured_text = self.json_to_structured_text(json_data)
            
            # Generate embeddings (one at a time to avoid issues)
            sem_embed = self.semantic_model.encode(semantic_text, convert_to_numpy=True)
            str_embed = self.structured_model.encode(structured_text, convert_to_numpy=True)
            
            semantic_embeddings.append(sem_embed)
            structured_embeddings.append(str_embed)
            
            # Store metadata
            self.metadata[i] = {
                'id': self.extract_id_from_record(record),
                'type': record.get('source_type', 'unknown'),
                'source_file': record.get('source_file', ''),
                'source_dir': record.get('source_dir', ''),
                'row_number': record.get('row_number', -1),
                'criticality': json_data.get('Criticality', 'N/A'),
                'json_data': json_data  # Store full data for reference
            }
        
        print(f"  Progress: {len(records)}/{len(records)} records processed...")
        
        # Convert to numpy arrays
        semantic_embeddings = np.array(semantic_embeddings).astype('float32')
        structured_embeddings = np.array(structured_embeddings).astype('float32')
        
        print(f"\n✓ Generated embeddings:")
        print(f"  • Semantic: {semantic_embeddings.shape}")
        print(f"  • Structured: {structured_embeddings.shape}")
        
        # Add to FAISS indices
        print("\nAdding embeddings to FAISS indices...")
        self.semantic_index.add(semantic_embeddings)
        self.structured_index.add(structured_embeddings)
        print("✓ Embeddings added to indices")
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Processing completed in {elapsed_time:.2f} seconds")
        
        return len(records)
    
    def save_indices(self, output_dir="vector_indices"):
        """Save FAISS indices and metadata"""
        print("\n" + "="*60)
        print("Saving Indices and Metadata")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FAISS indices
        semantic_path = os.path.join(output_dir, "semantic.index")
        structured_path = os.path.join(output_dir, "structured.index")
        metadata_path = os.path.join(output_dir, "metadata.json")
        
        print(f"\nSaving to {output_dir}/")
        
        faiss.write_index(self.semantic_index, semantic_path)
        print(f"  ✓ Semantic index: {semantic_path}")
        
        faiss.write_index(self.structured_index, structured_path)
        print(f"  ✓ Structured index: {structured_path}")
        
        # Save metadata with statistics
        metadata_with_stats = {
            'metadata': self.metadata,
            'statistics': {
                'total_records': len(self.metadata),
                'semantic_dim': self.semantic_dim,
                'structured_dim': self.structured_dim,
                'semantic_model': 'all-mpnet-base-v2',
                'structured_model': 'all-MiniLM-L6-v2',
                'timestamp': datetime.now().isoformat(),
                'record_types': {}
            }
        }
        
        # Count record types
        for meta in self.metadata.values():
            record_type = meta['type']
            metadata_with_stats['statistics']['record_types'][record_type] = \
                metadata_with_stats['statistics']['record_types'].get(record_type, 0) + 1
        
        # Convert numpy int64 to regular int for JSON serialization
        for key in self.metadata:
            self.metadata[key] = {k: int(v) if isinstance(v, np.int64) else v 
                                  for k, v in self.metadata[key].items()}
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_with_stats, f, indent=2, default=str)
        print(f"  ✓ Metadata: {metadata_path}")
        
        print(f"\nRecord Statistics:")
        for record_type, count in metadata_with_stats['statistics']['record_types'].items():
            print(f"  • {record_type}: {count} records")
        print(f"  • Total: {metadata_with_stats['statistics']['total_records']} records")
    
    def hybrid_search(self, query, k=5, alpha=0.6):
        """Perform hybrid search with late fusion"""
        # Generate query embeddings
        query_semantic = self.semantic_model.encode(query, convert_to_numpy=True).reshape(1, -1)
        query_structured = self.structured_model.encode(query, convert_to_numpy=True).reshape(1, -1)
        
        # Search both indices
        sem_distances, sem_indices = self.semantic_index.search(
            query_semantic.astype('float32'), k
        )
        
        str_distances, str_indices = self.structured_index.search(
            query_structured.astype('float32'), k
        )
        
        # Combine scores (convert distances to similarities)
        sem_similarities = 1 / (1 + sem_distances[0])
        str_similarities = 1 / (1 + str_distances[0])
        
        # Late fusion
        combined_scores = alpha * sem_similarities + (1 - alpha) * str_similarities
        
        # Get results
        results = []
        for i in range(k):
            sem_idx = sem_indices[0][i]
            
            result = {
                'index': int(sem_idx),
                'metadata': self.metadata[int(sem_idx)],
                'semantic_score': float(sem_similarities[i]),
                'structured_score': float(str_similarities[i]),
                'combined_score': float(combined_scores[i])
            }
            results.append(result)
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results


def main():
    try:
        # Initialize embedder
        embedder = HybridVectorEmbedder()
        
        # Load all JSON files from OUTPUT directories
        records = embedder.load_json_files()
        
        if not records:
            print("\n❌ No records found to process!")
            return 1
        
        # Process records and create embeddings
        num_processed = embedder.process_records(records)
        
        # Save indices and metadata
        embedder.save_indices()
        
        print("\n" + "="*60)
        print("✅ EMBEDDING PROCESS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nProcessed {num_processed} records")
        print("Indices saved to: vector_indices/")
        
        # Test with sample queries
        print("\n" + "="*60)
        print("Testing Hybrid Search")
        print("="*60)
        
        test_queries = [
            ("Grundfos pump 7.5 kW", 0.3),  # More structured
            ("critical equipment in utility area", 0.7),  # More semantic
            ("EQ-PUMP-CRN10-6-01", 0.1),  # Exact ID search
        ]
        
        for query, alpha in test_queries:
            results = embedder.hybrid_search(query, k=3, alpha=alpha)
            print(f"\nQuery: '{query}'")
            print(f"Alpha: {alpha} (semantic weight)")
            print("-" * 50)
            
            for i, result in enumerate(results[:3], 1):
                print(f"\n{i}. ID: {result['metadata']['id']}")
                print(f"   Type: {result['metadata']['type']}")
                print(f"   Criticality: {result['metadata']['criticality']}")
                print(f"   Scores:")
                print(f"     • Semantic: {result['semantic_score']:.4f}")
                print(f"     • Structured: {result['structured_score']:.4f}")
                print(f"     • Combined: {result['combined_score']:.4f}")
        
        print("\n" + "="*60)
        print("✅ ALL OPERATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
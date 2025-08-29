#!/usr/bin/env python3
"""
Hybrid Vector Embedder for Equipment Data (V2)
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
        print("Initializing Hybrid Vector Embedder...")
        
        # Initialize two different models for each stream
        print("Loading semantic model (all-mpnet-base-v2)...")
        print("Note: First run will download models (~420MB each)")
        self.semantic_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')
        print("  ✓ Semantic model loaded")
        
        print("Loading structured model (all-MiniLM-L6-v2)...")
        self.structured_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        print("  ✓ Structured model loaded")
        
        # Get embedding dimensions
        self.semantic_dim = 768  # all-mpnet-base-v2 dimension
        self.structured_dim = 384  # all-MiniLM-L6-v2 dimension
        
        # Initialize FAISS indices
        self.semantic_index = faiss.IndexFlatL2(self.semantic_dim)
        self.structured_index = faiss.IndexFlatL2(self.structured_dim)
        
        # Metadata storage
        self.metadata = {}
        self.index_counter = 0
        
    def load_json_files(self, directory_pattern="OUTPUT_*"):
        """Dynamically load all processed JSON files from OUTPUT directories"""
        all_records = []
        
        # Find all OUTPUT directories
        output_dirs = glob.glob(directory_pattern)
        print(f"\nFound {len(output_dirs)} OUTPUT directories")
        
        for output_dir in output_dirs:
            # Find all *_processed.json files (excluding processing_summary.json)
            json_files = glob.glob(os.path.join(output_dir, "*_processed.json"))
            json_files = [f for f in json_files if "processing_summary" not in f]
            
            print(f"\nProcessing directory: {output_dir}")
            print(f"Found {len(json_files)} JSON files")
            
            for json_file in json_files:
                file_type = os.path.basename(json_file).replace("_processed.json", "")
                print(f"  Loading {file_type}...")
                
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                # Extract records from the JSON structure
                if 'records' in data:
                    for record in data['records']:
                        # Add source information
                        record['source_file'] = os.path.basename(json_file)
                        record['source_type'] = file_type
                        record['source_dir'] = output_dir
                        all_records.append(record)
                    
                    print(f"    Loaded {len(data['records'])} records")
        
        print(f"\nTotal records loaded: {len(all_records)}")
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
        print("\nGenerating embeddings...")
        start_time = time.time()
        
        # Prepare texts for embedding
        semantic_texts = []
        structured_texts = []
        
        for i, record in enumerate(records):
            # Extract semantic text
            semantic_text = record.get('semantic_statement', '')
            semantic_texts.append(semantic_text)
            
            # Extract and format structured text from json_data
            json_data = record.get('json_data', {})
            structured_text = self.json_to_structured_text(json_data)
            structured_texts.append(structured_text)
            
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
        
        # Generate embeddings in smaller batches to avoid memory issues
        batch_size = 5
        semantic_embeddings = []
        structured_embeddings = []
        
        print(f"  Processing {len(semantic_texts)} records in batches of {batch_size}...")
        
        for i in range(0, len(semantic_texts), batch_size):
            batch_end = min(i + batch_size, len(semantic_texts))
            print(f"    Processing batch {i//batch_size + 1}/{(len(semantic_texts) + batch_size - 1)//batch_size}...")
            
            # Process semantic batch
            sem_batch = semantic_texts[i:batch_end]
            sem_embeds = self.semantic_model.encode(sem_batch, convert_to_numpy=True)
            semantic_embeddings.extend(sem_embeds)
            
            # Process structured batch
            str_batch = structured_texts[i:batch_end]
            str_embeds = self.structured_model.encode(str_batch, convert_to_numpy=True)
            structured_embeddings.extend(str_embeds)
        
        # Convert to numpy arrays
        semantic_embeddings = np.array(semantic_embeddings).astype('float32')
        structured_embeddings = np.array(structured_embeddings).astype('float32')
        
        print(f"  Semantic embeddings shape: {semantic_embeddings.shape}")
        print(f"  Structured embeddings shape: {structured_embeddings.shape}")
        
        # Add to FAISS indices
        print("  Adding embeddings to FAISS indices...")
        self.semantic_index.add(semantic_embeddings)
        self.structured_index.add(structured_embeddings)
        
        elapsed_time = time.time() - start_time
        print(f"  Embedding generation completed in {elapsed_time:.2f} seconds")
        
        return len(records)
    
    def save_indices(self, output_dir="vector_indices"):
        """Save FAISS indices and metadata"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving indices to {output_dir}/")
        
        # Save FAISS indices
        semantic_path = os.path.join(output_dir, "semantic.index")
        structured_path = os.path.join(output_dir, "structured.index")
        metadata_path = os.path.join(output_dir, "metadata.json")
        
        faiss.write_index(self.semantic_index, semantic_path)
        print(f"  ✓ Saved semantic index: {semantic_path}")
        
        faiss.write_index(self.structured_index, structured_path)
        print(f"  ✓ Saved structured index: {structured_path}")
        
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
        print(f"  ✓ Saved metadata: {metadata_path}")
        
        print(f"\nStatistics:")
        for record_type, count in metadata_with_stats['statistics']['record_types'].items():
            print(f"  {record_type}: {count} records")
        print(f"  Total: {metadata_with_stats['statistics']['total_records']} records")
    
    def hybrid_search(self, query, k=5, alpha=0.6):
        """Perform hybrid search with late fusion"""
        print(f"\nSearching for: '{query}'")
        print(f"Using alpha={alpha} (semantic weight)")
        
        # Generate query embeddings
        query_semantic = self.semantic_model.encode([query], convert_to_numpy=True)
        query_structured = self.structured_model.encode([query], convert_to_numpy=True)
        
        # Search both indices
        sem_distances, sem_indices = self.semantic_index.search(
            query_semantic.astype('float32'), k
        )
        
        str_distances, str_indices = self.structured_index.search(
            query_structured.astype('float32'), k
        )
        
        # Combine scores (convert distances to similarities)
        # Lower distance = higher similarity
        sem_similarities = 1 / (1 + sem_distances[0])
        str_similarities = 1 / (1 + str_distances[0])
        
        # Late fusion
        combined_scores = alpha * sem_similarities + (1 - alpha) * str_similarities
        
        # Get results
        results = []
        for i in range(k):
            sem_idx = sem_indices[0][i]
            
            # For simplicity, using semantic index results with combined score
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
            print("No records found to process!")
            return
        
        # Process records and create embeddings
        num_processed = embedder.process_records(records)
        
        # Save indices and metadata
        embedder.save_indices()
        
        print("\n" + "="*60)
        print("✅ Embedding process completed successfully!")
        print(f"Processed {num_processed} records")
        print("="*60)
        
        # Test with sample queries
        print("\n" + "="*60)
        print("Testing hybrid search capability...")
        print("="*60)
        
        test_queries = [
            ("Grundfos pump 7.5 kW", 0.3),  # More structured
            ("critical equipment in utility area", 0.7),  # More semantic
            ("EQ-PUMP-CRN10-6-01", 0.1),  # Exact ID search
        ]
        
        for query, alpha in test_queries:
            results = embedder.hybrid_search(query, k=3, alpha=alpha)
            print(f"\nQuery: '{query}' (alpha={alpha})")
            print("-" * 50)
            
            for i, result in enumerate(results[:3], 1):
                print(f"{i}. ID: {result['metadata']['id']}")
                print(f"   Type: {result['metadata']['type']}")
                print(f"   Criticality: {result['metadata']['criticality']}")
                print(f"   Semantic Score: {result['semantic_score']:.4f}")
                print(f"   Structured Score: {result['structured_score']:.4f}")
                print(f"   Combined Score: {result['combined_score']:.4f}")
                print()
                
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
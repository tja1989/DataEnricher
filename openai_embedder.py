#!/usr/bin/env python3
"""
Hybrid Vector Embedder using OpenAI Embeddings
Uses dual embedding streams: semantic and structured
"""

import json
import os
import glob
import numpy as np
import faiss
from datetime import datetime
import time
from openai import OpenAI
import sys

# Load environment variables from .env file if it exists
def load_env():
    env_path = '.env'
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Load .env file
load_env()

class OpenAIHybridEmbedder:
    def __init__(self, api_key=None):
        print("="*60)
        print("Initializing OpenAI Hybrid Vector Embedder")
        print("="*60)
        
        # Initialize OpenAI client
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Will use OPENAI_API_KEY environment variable
            self.client = OpenAI()
        
        print("\n✓ OpenAI client initialized")
        
        # Model configuration for hybrid approach
        # Using different models for semantic vs structured
        self.semantic_model = "text-embedding-3-large"  # 3072 dimensions
        self.structured_model = "text-embedding-3-small"  # 1536 dimensions
        
        # Get embedding dimensions
        self.semantic_dim = 3072
        self.structured_dim = 1536
        
        print(f"\nModels configured:")
        print(f"  • Semantic: {self.semantic_model} ({self.semantic_dim} dims)")
        print(f"  • Structured: {self.structured_model} ({self.structured_dim} dims)")
        
        # Initialize FAISS indices
        self.semantic_index = faiss.IndexFlatL2(self.semantic_dim)
        self.structured_index = faiss.IndexFlatL2(self.structured_dim)
        print("\n✓ FAISS indices initialized")
        
        # Metadata storage
        self.metadata = {}
        
        # Cost tracking
        self.tokens_used = {'semantic': 0, 'structured': 0}
        
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
    
    def estimate_tokens(self, text):
        """Rough estimate of tokens (OpenAI uses ~4 chars per token)"""
        return len(text) // 4
    
    def get_embedding(self, text, model):
        """Get embedding from OpenAI API"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            raise
    
    def get_embeddings_batch(self, texts, model):
        """Get embeddings for a batch of texts (OpenAI supports up to 2048 inputs)"""
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=model
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error getting batch embeddings: {e}")
            raise
    
    def process_records(self, records):
        """Process all records and create embeddings"""
        print("\n" + "="*60)
        print("Generating Embeddings via OpenAI API")
        print("="*60)
        
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
        
        # Estimate tokens for cost tracking
        semantic_tokens = sum(self.estimate_tokens(t) for t in semantic_texts)
        structured_tokens = sum(self.estimate_tokens(t) for t in structured_texts)
        
        print(f"\nEstimated tokens:")
        print(f"  • Semantic texts: ~{semantic_tokens:,} tokens")
        print(f"  • Structured texts: ~{structured_tokens:,} tokens")
        
        semantic_cost = (semantic_tokens / 1_000_000) * 0.13  # $0.13 per 1M tokens
        structured_cost = (structured_tokens / 1_000_000) * 0.02  # $0.02 per 1M tokens
        print(f"\nEstimated cost:")
        print(f"  • Semantic embeddings: ~${semantic_cost:.4f}")
        print(f"  • Structured embeddings: ~${structured_cost:.4f}")
        print(f"  • Total: ~${semantic_cost + structured_cost:.4f}")
        
        # Process embeddings in batches (OpenAI supports up to 2048 at once)
        batch_size = 100  # Conservative batch size
        semantic_embeddings = []
        structured_embeddings = []
        
        print(f"\nProcessing {len(records)} records in batches of {batch_size}...")
        
        # Process semantic embeddings
        print("\n1. Generating semantic embeddings...")
        for i in range(0, len(semantic_texts), batch_size):
            batch_end = min(i + batch_size, len(semantic_texts))
            batch = semantic_texts[i:batch_end]
            print(f"   Batch {i//batch_size + 1}/{(len(semantic_texts) + batch_size - 1)//batch_size}")
            
            embeddings = self.get_embeddings_batch(batch, self.semantic_model)
            semantic_embeddings.extend(embeddings)
            
            # Rate limiting (be respectful of API)
            if i + batch_size < len(semantic_texts):
                time.sleep(0.5)  # Small delay between batches
        
        print("   ✓ Semantic embeddings complete")
        
        # Process structured embeddings
        print("\n2. Generating structured embeddings...")
        for i in range(0, len(structured_texts), batch_size):
            batch_end = min(i + batch_size, len(structured_texts))
            batch = structured_texts[i:batch_end]
            print(f"   Batch {i//batch_size + 1}/{(len(structured_texts) + batch_size - 1)//batch_size}")
            
            embeddings = self.get_embeddings_batch(batch, self.structured_model)
            structured_embeddings.extend(embeddings)
            
            # Rate limiting
            if i + batch_size < len(structured_texts):
                time.sleep(0.5)
        
        print("   ✓ Structured embeddings complete")
        
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
        
        self.tokens_used['semantic'] = semantic_tokens
        self.tokens_used['structured'] = structured_tokens
        
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
                'semantic_model': self.semantic_model,
                'structured_model': self.structured_model,
                'timestamp': datetime.now().isoformat(),
                'tokens_used': self.tokens_used,
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
        print(f"\nSearching for: '{query}'")
        print(f"Alpha: {alpha} (semantic weight)")
        
        # Generate query embeddings
        query_semantic = self.get_embedding(query, self.semantic_model)
        query_structured = self.get_embedding(query, self.structured_model)
        
        query_semantic = np.array([query_semantic]).astype('float32')
        query_structured = np.array([query_structured]).astype('float32')
        
        # Search both indices
        sem_distances, sem_indices = self.semantic_index.search(query_semantic, k)
        str_distances, str_indices = self.structured_index.search(query_structured, k)
        
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
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY environment variable not set!")
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        return 1
    
    try:
        # Initialize embedder
        embedder = OpenAIHybridEmbedder()
        
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
        
        # Print cost summary
        total_tokens = embedder.tokens_used['semantic'] + embedder.tokens_used['structured']
        semantic_cost = (embedder.tokens_used['semantic'] / 1_000_000) * 0.13
        structured_cost = (embedder.tokens_used['structured'] / 1_000_000) * 0.02
        total_cost = semantic_cost + structured_cost
        
        print(f"\nUsage Summary:")
        print(f"  • Total tokens used: ~{total_tokens:,}")
        print(f"  • Estimated cost: ~${total_cost:.4f}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
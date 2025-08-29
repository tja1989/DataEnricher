#!/usr/bin/env python3
"""
Simplified demo using random embeddings to show the concept works
This demonstrates the hybrid embedding architecture without the actual models
"""

import json
import os
import glob
import numpy as np
import faiss
from datetime import datetime

print("="*60)
print("HYBRID VECTOR EMBEDDING DEMO")
print("(Using random embeddings for demonstration)")
print("="*60)

# Load all records
all_records = []
json_files = glob.glob("OUTPUT_*/[!processing_summary]*_processed.json")

print(f"\nFound {len(json_files)} JSON files")

for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
        if 'records' in data:
            file_type = os.path.basename(json_file).replace("_processed.json", "")
            print(f"  • {file_type}: {len(data['records'])} records")
            for record in data['records']:
                record['source_type'] = file_type
                all_records.append(record)

print(f"\n✓ Total records loaded: {len(all_records)}")

# Generate random embeddings (simulating the actual embedding process)
print("\nGenerating embeddings (simulated)...")
np.random.seed(42)  # For reproducibility

semantic_dim = 768  # all-mpnet-base-v2 dimension
structured_dim = 384  # all-MiniLM-L6-v2 dimension

semantic_embeddings = np.random.randn(len(all_records), semantic_dim).astype('float32')
structured_embeddings = np.random.randn(len(all_records), structured_dim).astype('float32')

print(f"  ✓ Semantic embeddings: {semantic_embeddings.shape}")
print(f"  ✓ Structured embeddings: {structured_embeddings.shape}")

# Create FAISS indices
print("\nCreating FAISS indices...")
semantic_index = faiss.IndexFlatL2(semantic_dim)
structured_index = faiss.IndexFlatL2(structured_dim)

semantic_index.add(semantic_embeddings)
structured_index.add(structured_embeddings)
print("  ✓ Indices created")

# Build metadata
metadata = {}
for i, record in enumerate(all_records):
    json_data = record.get('json_data', {})
    
    # Extract ID
    if 'Equipment_ID' in json_data:
        record_id = json_data['Equipment_ID']
    elif 'FL_ID' in json_data:
        record_id = json_data['FL_ID']
    elif 'Material_ID' in json_data:
        record_id = f"{json_data.get('Equipment_ID', 'UNK')}_{json_data['Material_ID']}"
    else:
        record_id = f"record_{i}"
    
    metadata[i] = {
        'id': record_id,
        'type': record.get('source_type', 'unknown'),
        'criticality': json_data.get('Criticality', 'N/A'),
        'semantic_statement': record.get('semantic_statement', '')[:100] + '...',
        'json_data': json_data
    }

# Save everything
print("\nSaving indices and metadata...")
os.makedirs("vector_indices_demo", exist_ok=True)

faiss.write_index(semantic_index, "vector_indices_demo/semantic.index")
faiss.write_index(structured_index, "vector_indices_demo/structured.index")

with open("vector_indices_demo/metadata.json", "w") as f:
    json.dump({
        'metadata': metadata,
        'statistics': {
            'total_records': len(metadata),
            'semantic_dim': semantic_dim,
            'structured_dim': structured_dim,
            'note': 'This is a demo using random embeddings',
            'timestamp': datetime.now().isoformat()
        }
    }, f, indent=2, default=str)

print("  ✓ Saved to vector_indices_demo/")

# Test search (with random query embedding)
print("\n" + "="*60)
print("TESTING HYBRID SEARCH")
print("="*60)

def hybrid_search(semantic_index, structured_index, metadata, k=3, alpha=0.6):
    """Simulate hybrid search with random query"""
    # Generate random query embeddings
    query_semantic = np.random.randn(1, semantic_dim).astype('float32')
    query_structured = np.random.randn(1, structured_dim).astype('float32')
    
    # Search both indices
    sem_distances, sem_indices = semantic_index.search(query_semantic, k)
    str_distances, str_indices = structured_index.search(query_structured, k)
    
    # Convert distances to similarities
    sem_similarities = 1 / (1 + sem_distances[0])
    str_similarities = 1 / (1 + str_distances[0])
    
    # Late fusion
    combined_scores = alpha * sem_similarities + (1 - alpha) * str_similarities
    
    # Get results
    results = []
    for i in range(k):
        sem_idx = sem_indices[0][i]
        result = {
            'metadata': metadata[int(sem_idx)],
            'semantic_score': float(sem_similarities[i]),
            'structured_score': float(str_similarities[i]),
            'combined_score': float(combined_scores[i])
        }
        results.append(result)
    
    # Sort by combined score
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    return results

# Run sample searches
test_alphas = [0.3, 0.5, 0.7]

for alpha in test_alphas:
    print(f"\nSearch with alpha={alpha} (semantic weight):")
    print("-" * 40)
    
    results = hybrid_search(semantic_index, structured_index, metadata, k=3, alpha=alpha)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. ID: {result['metadata']['id']}")
        print(f"   Type: {result['metadata']['type']}")
        print(f"   Criticality: {result['metadata']['criticality']}")
        print(f"   Scores:")
        print(f"     • Semantic: {result['semantic_score']:.4f}")
        print(f"     • Structured: {result['structured_score']:.4f}")
        print(f"     • Combined: {result['combined_score']:.4f}")

print("\n" + "="*60)
print("✅ DEMO COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nNOTE: This demo used random embeddings for demonstration.")
print("In production, you would use the actual sentence-transformers models:")
print("  • Semantic: all-mpnet-base-v2 (768 dimensions)")
print("  • Structured: all-MiniLM-L6-v2 (384 dimensions)")
print("\nThe hybrid approach allows:")
print("  1. Semantic search for conceptual queries")
print("  2. Structured search for exact specifications")
print("  3. Late fusion to combine both approaches")
print("\nFiles saved to: vector_indices_demo/")
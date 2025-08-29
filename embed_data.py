#!/usr/bin/env python3
"""
Minimal vector embedder - processes data one by one
"""

import json
import os
import glob
import numpy as np
import faiss
from datetime import datetime
import sys

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("Starting embedding process...", flush=True)

try:
    from sentence_transformers import SentenceTransformer
    
    print("Loading models...", flush=True)
    semantic_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')
    print("✓ Semantic model loaded", flush=True)
    
    structured_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    print("✓ Structured model loaded", flush=True)
    
    # Load all records
    all_records = []
    json_files = glob.glob("OUTPUT_*/[!processing_summary]*_processed.json")
    
    print(f"\nFound {len(json_files)} JSON files", flush=True)
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            if 'records' in data:
                all_records.extend(data['records'])
                print(f"  Loaded {len(data['records'])} records from {os.path.basename(json_file)}", flush=True)
    
    print(f"\nTotal records: {len(all_records)}", flush=True)
    
    # Process embeddings
    print("\nGenerating embeddings...", flush=True)
    semantic_embeddings = []
    structured_embeddings = []
    metadata = {}
    
    for i, record in enumerate(all_records):
        if i % 10 == 0:
            print(f"  Processing record {i+1}/{len(all_records)}...", flush=True)
        
        # Get texts
        semantic_text = record.get('semantic_statement', '')
        json_data = record.get('json_data', {})
        structured_text = " | ".join([f"{k}: {v}" for k, v in json_data.items() if v is not None])
        
        # Generate embeddings - one at a time
        sem_emb = semantic_model.encode([semantic_text], convert_to_numpy=True)[0]
        str_emb = structured_model.encode([structured_text], convert_to_numpy=True)[0]
        
        semantic_embeddings.append(sem_emb)
        structured_embeddings.append(str_emb)
        
        # Store metadata
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
            'criticality': json_data.get('Criticality', 'N/A'),
            'json_data': json_data
        }
    
    print(f"  Processing record {len(all_records)}/{len(all_records)}...", flush=True)
    print("✓ All embeddings generated", flush=True)
    
    # Create FAISS indices
    print("\nCreating FAISS indices...", flush=True)
    semantic_index = faiss.IndexFlatL2(768)
    structured_index = faiss.IndexFlatL2(384)
    
    semantic_index.add(np.array(semantic_embeddings).astype('float32'))
    structured_index.add(np.array(structured_embeddings).astype('float32'))
    print("✓ Indices created", flush=True)
    
    # Save everything
    print("\nSaving indices...", flush=True)
    os.makedirs("vector_indices", exist_ok=True)
    
    faiss.write_index(semantic_index, "vector_indices/semantic.index")
    faiss.write_index(structured_index, "vector_indices/structured.index")
    
    with open("vector_indices/metadata.json", "w") as f:
        json.dump({
            'metadata': metadata,
            'statistics': {
                'total_records': len(metadata),
                'timestamp': datetime.now().isoformat()
            }
        }, f, indent=2, default=str)
    
    print("✓ Indices saved to vector_indices/", flush=True)
    print(f"\n✅ SUCCESS! Processed {len(all_records)} records", flush=True)
    
except Exception as e:
    print(f"\n❌ Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
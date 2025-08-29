#!/usr/bin/env python3
"""
Simple test to generate embeddings for one record
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
from sentence_transformers import SentenceTransformer

# Load a single record for testing
with open('OUTPUT_INPUT_CSV_20250828_204143/Equipment_processed.json', 'r') as f:
    data = json.load(f)
    
record = data['records'][0]

print("Loading models...")
semantic_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')
print("✓ Semantic model loaded")

structured_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
print("✓ Structured model loaded")

print("\nProcessing single record...")
print(f"Equipment ID: {record['json_data']['Equipment_ID']}")

# Get texts
semantic_text = record['semantic_statement']
structured_text = " | ".join([f"{k}: {v}" for k, v in record['json_data'].items()])

print("\nGenerating semantic embedding...")
sem_embed = semantic_model.encode(semantic_text, convert_to_numpy=True)
print(f"✓ Semantic embedding shape: {sem_embed.shape}")

print("\nGenerating structured embedding...")
str_embed = structured_model.encode(structured_text, convert_to_numpy=True)
print(f"✓ Structured embedding shape: {str_embed.shape}")

print("\n✅ Success! Single record processed.")
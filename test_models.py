#!/usr/bin/env python3
"""Test script to verify model loading"""

from sentence_transformers import SentenceTransformer
import sys

print("Testing model loading...")

try:
    print("\n1. Loading all-mpnet-base-v2...")
    model1 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    print("   ✓ Model loaded successfully")
    
    print("\n2. Testing encoding...")
    test_text = ["This is a test sentence"]
    embedding = model1.encode(test_text)
    print(f"   ✓ Embedding shape: {embedding.shape}")
    
    print("\n3. Loading all-MiniLM-L6-v2...")
    model2 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("   ✓ Model loaded successfully")
    
    print("\n4. Testing encoding...")
    embedding2 = model2.encode(test_text)
    print(f"   ✓ Embedding shape: {embedding2.shape}")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
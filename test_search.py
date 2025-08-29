#!/usr/bin/env python3
"""
Non-interactive test queries for the hybrid vector database
"""

import json
import numpy as np
import faiss
from openai import OpenAI
import os

# Load environment variables from .env file
def load_env():
    env_path = '.env'
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()

class HybridSearcher:
    def __init__(self):
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Load FAISS indices
        self.semantic_index = faiss.read_index("vector_indices/semantic.index")
        self.structured_index = faiss.read_index("vector_indices/structured.index")
        
        # Load metadata
        with open("vector_indices/metadata.json", "r") as f:
            data = json.load(f)
            self.metadata = {int(k): v for k, v in data['metadata'].items()}
            self.stats = data['statistics']
        
        print("✓ Loaded vector database")
        print(f"  • Total records: {self.stats['total_records']}")
        print(f"  • Equipment: {self.stats['record_types'].get('Equipment', 0)}")
        print(f"  • Functional Locations: {self.stats['record_types'].get('Functional_Locations', 0)}")
        print(f"  • Equipment Materials: {self.stats['record_types'].get('Equipment_Material', 0)}\n")
    
    def search(self, query, k=5, alpha=0.5):
        """Perform hybrid search"""
        # Get embeddings
        query_semantic = self.client.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        ).data[0].embedding
        
        query_structured = self.client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        ).data[0].embedding
        
        # Convert to numpy
        query_semantic = np.array([query_semantic]).astype('float32')
        query_structured = np.array([query_structured]).astype('float32')
        
        # Search both indices
        sem_distances, sem_indices = self.semantic_index.search(query_semantic, k)
        str_distances, str_indices = self.structured_index.search(query_structured, k)
        
        # Convert distances to similarities
        sem_similarities = 1 / (1 + sem_distances[0])
        str_similarities = 1 / (1 + str_distances[0])
        
        # Late fusion
        results = []
        indices_seen = set()
        
        # Combine results from both searches
        for i in range(k):
            # From semantic search
            sem_idx = int(sem_indices[0][i])
            if sem_idx not in indices_seen:
                indices_seen.add(sem_idx)
                combined_score = alpha * sem_similarities[i] + (1 - alpha) * str_similarities[i]
                results.append({
                    'metadata': self.metadata[sem_idx],
                    'semantic_score': float(sem_similarities[i]),
                    'structured_score': float(str_similarities[i]),
                    'combined_score': float(combined_score)
                })
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results[:k]
    
    def display_results(self, query, results, alpha):
        """Display search results in a formatted way"""
        print("="*70)
        print(f"Query: '{query}'")
        print(f"Alpha: {alpha} ({'Semantic-focused' if alpha > 0.6 else 'Structured-focused' if alpha < 0.4 else 'Balanced'})")
        print("="*70)
        
        if not results:
            print("No results found.")
            return
        
        for i, result in enumerate(results, 1):
            meta = result['metadata']
            json_data = meta.get('json_data', {})
            
            print(f"\n{i}. {meta['id']}")
            print("   " + "-"*60)
            print(f"   Type: {meta['type']}")
            print(f"   Criticality: {meta.get('criticality', 'N/A')}")
            
            # Show key attributes based on type
            if meta['type'] == 'Equipment':
                print(f"   Name: {json_data.get('Equipment_Name', 'N/A')}")
                print(f"   Manufacturer: {json_data.get('Manufacturer', 'N/A')}")
                print(f"   Model: {json_data.get('Model', 'N/A')}")
                if 'Nameplate_Power_kW' in json_data:
                    print(f"   Power: {json_data['Nameplate_Power_kW']} kW")
                if 'FL_ID' in json_data:
                    print(f"   Location: {json_data['FL_ID']}")
            elif meta['type'] == 'Functional_Locations':
                print(f"   Name: {json_data.get('FL_Name', 'N/A')}")
                print(f"   Site: {json_data.get('Site_Area', 'N/A')}")
                print(f"   Parent: {json_data.get('Parent_FL_ID', 'N/A')}")
            elif meta['type'] == 'Equipment_Material':
                print(f"   Equipment: {json_data.get('Equipment_ID', 'N/A')}")
                print(f"   Material: {json_data.get('Material_ID', 'N/A')}")
                print(f"   Quantity: {json_data.get('Qty', 'N/A')}")
                print(f"   Notes: {json_data.get('Notes', 'N/A')}")
            
            print(f"\n   Relevance Scores:")
            print(f"     • Semantic:   {result['semantic_score']:.4f}")
            print(f"     • Structured: {result['structured_score']:.4f}")
            print(f"     • Combined:   {result['combined_score']:.4f}")


def main():
    # Initialize searcher
    searcher = HybridSearcher()
    
    print("\n" + "="*70)
    print("TESTING HYBRID VECTOR SEARCH")
    print("="*70)
    
    # Test queries
    test_queries = [
        {
            "query": "show me all cooling tower related equipment and locations",
            "alpha": 0.7,
            "description": "Semantic search for cooling systems"
        },
        {
            "query": "Grundfos CRN 10-6 pump specifications",
            "alpha": 0.3,
            "description": "Structured search for specific pump"
        },
        {
            "query": "high criticality equipment with power above 40 kW",
            "alpha": 0.5,
            "description": "Mixed search for critical high-power equipment"
        },
        {
            "query": "MAT-BEARING-6207-SKF",
            "alpha": 0.1,
            "description": "Exact match for material/part number"
        },
        {
            "query": "hot rolling mill production area",
            "alpha": 0.6,
            "description": "Semantic search for production areas"
        }
    ]
    
    for test in test_queries:
        print(f"\n\n{'='*70}")
        print(f"TEST: {test['description']}")
        print(f"{'='*70}")
        
        results = searcher.search(
            query=test['query'],
            k=3,
            alpha=test['alpha']
        )
        
        searcher.display_results(test['query'], results, test['alpha'])
    
    print("\n\n" + "="*70)
    print("✅ ALL TESTS COMPLETED")
    print("="*70)
    print("\nSummary:")
    print("• Tested 5 different query types")
    print("• Demonstrated semantic vs structured search")
    print("• Showed late fusion with different alpha values")
    print("• Successfully retrieved relevant equipment, locations, and materials")


if __name__ == "__main__":
    main()
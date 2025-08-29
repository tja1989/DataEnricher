#!/usr/bin/env python3
"""
Test queries for the hybrid vector database
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
        print(f"  • Semantic model: {self.stats['semantic_model']}")
        print(f"  • Structured model: {self.stats['structured_model']}\n")
    
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
            elif meta['type'] == 'Functional_Locations':
                print(f"   Name: {json_data.get('FL_Name', 'N/A')}")
                print(f"   Site: {json_data.get('Site_Area', 'N/A')}")
            elif meta['type'] == 'Equipment_Material':
                print(f"   Equipment: {json_data.get('Equipment_ID', 'N/A')}")
                print(f"   Material: {json_data.get('Material_ID', 'N/A')}")
                print(f"   Quantity: {json_data.get('Qty', 'N/A')}")
            
            print(f"\n   Relevance Scores:")
            print(f"     • Semantic:   {result['semantic_score']:.4f}")
            print(f"     • Structured: {result['structured_score']:.4f}")
            print(f"     • Combined:   {result['combined_score']:.4f}")


def main():
    # Initialize searcher
    searcher = HybridSearcher()
    
    # Test queries with different intents
    test_queries = [
        # Equipment-specific queries
        {
            "query": "Siemens S7-1500 PLC safety systems",
            "alpha": 0.3,  # Favor structured for specific model
            "description": "Finding specific Siemens PLC equipment"
        },
        {
            "query": "high criticality pumps in utility area",
            "alpha": 0.6,  # Balanced for mixed semantic/structured
            "description": "Finding critical pumps by location"
        },
        {
            "query": "45 kW hydraulic power unit",
            "alpha": 0.4,  # Slightly favor structured for specifications
            "description": "Finding equipment by power specification"
        },
        {
            "query": "cooling tower equipment functional location",
            "alpha": 0.7,  # Favor semantic for concept understanding
            "description": "Finding cooling tower related items"
        },
        {
            "query": "ABB variable frequency drive motor control",
            "alpha": 0.5,  # Balanced search
            "description": "Finding ABB drive systems"
        },
        {
            "query": "bearing 6207 SKF maintenance materials",
            "alpha": 0.3,  # Favor structured for part numbers
            "description": "Finding equipment materials and parts"
        }
    ]
    
    print("\n" + "="*70)
    print("HYBRID VECTOR SEARCH DEMONSTRATION")
    print("="*70)
    
    for test in test_queries:
        print(f"\n\n{'='*70}")
        print(f"Test: {test['description']}")
        print(f"{'='*70}")
        
        results = searcher.search(
            query=test['query'],
            k=3,
            alpha=test['alpha']
        )
        
        searcher.display_results(test['query'], results, test['alpha'])
        
        input("\nPress Enter to continue to next query...")
    
    # Interactive mode
    print("\n\n" + "="*70)
    print("INTERACTIVE SEARCH MODE")
    print("="*70)
    print("Enter your own queries (type 'quit' to exit)")
    print("You can adjust alpha: 0.0 = pure keyword, 1.0 = pure semantic")
    
    while True:
        print("\n" + "-"*70)
        query = input("Query: ").strip()
        
        if query.lower() == 'quit':
            break
        
        if not query:
            continue
        
        alpha_str = input("Alpha (0.0-1.0, default 0.5): ").strip()
        try:
            alpha = float(alpha_str) if alpha_str else 0.5
            alpha = max(0.0, min(1.0, alpha))
        except:
            alpha = 0.5
        
        k_str = input("Number of results (default 5): ").strip()
        try:
            k = int(k_str) if k_str else 5
            k = max(1, min(10, k))
        except:
            k = 5
        
        results = searcher.search(query, k=k, alpha=alpha)
        searcher.display_results(query, results, alpha)
    
    print("\n✓ Search session completed")


if __name__ == "__main__":
    main()
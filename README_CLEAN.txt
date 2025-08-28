CLEAN WORKSPACE - LLM DISCOVERY SYSTEM
=======================================

CORE IMPLEMENTATIONS:
---------------------
1. llm_discovery_two_pass.py
   - Best implementation with 100% vendor name resolution
   - Two-pass approach: First get IDs, then resolve names
   - Handles materials, vendors, and cross-references

2. llm_discovery_ultra_simple.py  
   - Simplest implementation (140 lines)
   - Single query, single LLM call
   - 100% success rate, ~80% vendor names

3. vector_relationship_builder.py
   - FAISS vector database utilities
   - Search and embedding functions

4. csv_to_vector_db.py
   - Converts CSV files to vector database format
   - Creates JSON + semantic summaries for embedding

TEST SCRIPTS:
-------------
- test_all_two_pass.py - Test two-pass on all equipment
- test_all_ultra_simple.py - Test ultra-simple on all equipment  
- final_comparison.py - Compare all approaches

DATA:
-----
- EMSTEEL_PoC_Synthetic_CSVs/ - Source CSV files
- faiss_vector_db/ - Vector database
- EMSTEEL_vector_output/ - Vector-ready JSON files

RESULTS:
--------
- two_pass_results_*.json - Two-pass test results (100% vendor names)
- ultra_simple_results_*.json - Ultra-simple results (best simplicity)

TO TEST NEW DATA:
-----------------
1. Place CSV files in a folder
2. Run: python csv_to_vector_db.py (to create vector DB)
3. Run: python test_all_two_pass.py (for best results)
   OR: python test_all_ultra_simple.py (for simplicity)

KEY ACHIEVEMENTS:
-----------------
- 100% success rate (vs 65% with Pydantic)
- 100% vendor name resolution (two-pass)
- 2-3x more materials discovered
- 50% less code complexity
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

This is an LLM-driven equipment relationship discovery system that extracts relationships between industrial equipment, spare parts/materials, and vendors from CSV data using vector search and OpenAI's API. The system converts structured CSV data into a vector database, then uses semantic search to find relevant contexts for LLM analysis.

## Key Commands

### Environment Setup
```bash
# Install dependencies
pip3 install openai faiss-cpu numpy pandas python-dotenv

# Set OpenAI API key
export OPENAI_API_KEY="your-key-here"
# OR create .env file with: OPENAI_API_KEY=your-key-here
```

### Building Vector Database from CSV
```bash
# Convert CSV files to vector database (required before discovery)
python3 csv_to_vector_db.py

# This will:
# 1. Read all CSV files from EMSTEEL_PoC_Synthetic_CSVs/
# 2. Convert to JSON + semantic summaries
# 3. Create FAISS vector database in faiss_vector_db/
# 4. Output vector-ready files to EMSTEEL_vector_output/
```

### Running Discovery

```bash
# Two-pass approach (100% vendor name resolution, production-ready)
python3 test_all_two_pass.py

# Ultra-simple approach (fastest, 140 lines of code)
python3 test_all_ultra_simple.py

# Compare all approaches
python3 final_comparison.py
```

### Testing with New Data
```bash
# 1. Place CSV files in a new folder
# 2. Update folder path in csv_to_vector_db.py (line with 'EMSTEEL_PoC_Synthetic_CSVs')
# 3. Run: python3 csv_to_vector_db.py
# 4. Run: python3 test_all_two_pass.py
```

## Architecture

### Core Discovery Flow

The system uses a **vector search → LLM analysis** pipeline:

1. **CSV → Vector DB** (`csv_to_vector_db.py`)
   - Converts each CSV row to: `{json_data} + semantic_summary`
   - Embeds using OpenAI's text-embedding-3-small
   - Stores in FAISS for similarity search
   - Critical: Detects entity types (equipment/material/vendor) from column headers

2. **Vector Search** (`vector_relationship_builder.py`)
   - FAISSVectorDB class handles all vector operations
   - Search returns (document, score) tuples with metadata
   - Metadata contains IDs, names, and relationship fields

3. **LLM Discovery** (Two implementations)
   
   **Two-Pass Approach** (`llm_discovery_two_pass.py`):
   - Pass 1: Query equipment → Get contexts with Material_IDs and vendor IDs in Approved_Suppliers field
   - Pass 2: Query those IDs → Get vendor/material records with full names
   - Combines contexts → LLM extracts complete relationships
   - Achieves 100% ID-to-name resolution
   
   **Ultra-Simple Approach** (`llm_discovery_ultra_simple.py`):
   - Single query: `"{equipment_id} {equipment_name} spare parts materials vendors suppliers"`
   - Single LLM call with JSON response format
   - ~80% vendor name accuracy (only from coincidental matches)

### Key Data Relationships

The system discovers these relationships from CSV data:

```
Equipment (Equipment.csv)
    ↓
    ├── Materials (Equipment_Material.csv links to Materials.csv)
    │      ↓
    │      └── Vendors (Materials.csv has Approved_Suppliers field with vendor IDs)
    │
    └── Work Orders (Work_Orders.csv references materials used)
```

**Critical Issue**: Vendor names are stored separately in Vendors.csv, not with materials. The two-pass approach solves this by:
1. Finding vendor IDs in Approved_Suppliers field
2. Explicitly querying for those vendor IDs to get names

### Important Implementation Details

1. **Entity Type Detection** (`csv_to_vector_db.py:detect_table_type()`)
   - Bug: Checks "supplier" before "material", causing Materials.csv to be misclassified as vendor type
   - This prevents vendor discovery in single-pass approaches

2. **Vendor Name Resolution**
   - Materials.csv has `Approved_Suppliers` field with comma-separated vendor IDs (e.g., "V-DBX-01,V-GULF-01")
   - Vendor names are in separate Vendors.csv file
   - Two-pass approach required for 100% name resolution

3. **Context Limits**
   - Default k=10-15 contexts per query
   - Increasing k improves recall but adds cost
   - Two-pass uses k=12 for equipment, k=2-5 for ID resolution

4. **LLM Configuration**
   - Model: gpt-4o-mini (cost-effective)
   - Temperature: 0.1-0.3 for consistent output
   - Response format: `{"type": "json_object"}` for structured output

## CSV Data Schema

Expected CSV files and key columns:

- **Equipment.csv**: Equipment_ID, Equipment_Name, Criticality, Manufacturer
- **Materials.csv**: Material_ID, Material_Name, Approved_Suppliers (vendor IDs)
- **Vendors.csv**: Vendor_ID, Vendor_Name
- **Equipment_Material.csv**: Equipment_ID, Material_ID (links equipment to materials)
- **Work_Orders.csv**: Equipment_ID, Materials_Used (additional material discoveries)

## Performance Benchmarks

With 20 test equipment:
- Two-pass: 100% success, 100% vendor names, ~300 lines code
- Ultra-simple: 100% success, 80% vendor names, ~140 lines code
- API calls: 2-6 per equipment depending on approach
- Processing time: ~3-5 seconds per equipment

## Debugging

Check vector DB entity types:
```python
from vector_relationship_builder import FAISSVectorDB
vector_db = FAISSVectorDB()
vector_db.load(Path("faiss_vector_db"))
results = vector_db.search("your query", k=5)
for doc, score in results:
    print(f"Entity type: {doc.entity_type}, Score: {score}")
    print(f"Metadata: {doc.metadata}")
```

Enable debug output in two-pass:
- Creates `debug_two_pass_{equipment_id}.json` with ID resolution details
- Shows which IDs were found and what queries were made
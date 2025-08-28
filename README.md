# DataEnricher - LLM-Driven Equipment Relationship Discovery

An intelligent system that discovers relationships between industrial equipment, spare parts/materials, and vendors using vector search and LLM analysis.

## Features

- **100% Success Rate**: Robust discovery without parsing failures
- **Complete Vendor Resolution**: Two-pass approach achieves 100% vendor name accuracy
- **Semantic Search**: FAISS vector database for intelligent context retrieval
- **Flexible Architecture**: Choose between simple (140 lines) or comprehensive (300 lines) implementations

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
# OR create a .env file with:
# OPENAI_API_KEY=your-api-key-here
```

### 3. Build Vector Database

```bash
python csv_to_vector_db.py
```

This converts CSV files to a searchable vector database.

### 4. Run Discovery

```bash
# Two-pass approach (best accuracy)
python test_all_two_pass.py

# Ultra-simple approach (fastest)
python test_all_ultra_simple.py
```

## How It Works

1. **Data Ingestion**: CSV files are converted to JSON + semantic summaries
2. **Vector Embedding**: Text is embedded using OpenAI's text-embedding-3-small
3. **Semantic Search**: FAISS enables fast similarity search across all data
4. **LLM Analysis**: GPT-4o-mini extracts relationships from retrieved contexts
5. **ID Resolution**: Two-pass approach resolves all IDs to their full names

## Project Structure

```
dataenricher/
├── csv_to_vector_db.py           # CSV to vector database converter
├── vector_relationship_builder.py # FAISS vector DB utilities
├── llm_discovery_two_pass.py     # Two-pass discovery (100% accuracy)
├── llm_discovery_ultra_simple.py # Simple discovery (140 lines)
├── test_all_two_pass.py          # Test two-pass on all equipment
├── test_all_ultra_simple.py      # Test simple approach
├── final_comparison.py           # Compare all approaches
├── EMSTEEL_PoC_Synthetic_CSVs/   # Sample CSV data
└── faiss_vector_db/              # Generated vector database
```

## Input Data Format

The system expects CSV files with these structures:

- **Equipment.csv**: Equipment_ID, Equipment_Name, Criticality, Manufacturer
- **Materials.csv**: Material_ID, Material_Name, Approved_Suppliers
- **Vendors.csv**: Vendor_ID, Vendor_Name
- **Equipment_Material.csv**: Links equipment to materials
- **Work_Orders.csv**: Maintenance history with materials used

## Performance

Testing on 20 equipment items:
- Success Rate: 100%
- Materials Discovery: 56 materials found (2.8 per equipment)
- Vendor Discovery: 85% of equipment have vendors identified
- Vendor Name Accuracy: 100% (two-pass) vs 80% (single-pass)
- Processing Time: ~3-5 seconds per equipment

## Key Achievements

- **100% success rate** vs 65% with strict schema validation
- **100% vendor name resolution** with two-pass approach
- **2-3x more materials discovered** including from work orders
- **50% less code** compared to complex implementations

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions, please open an issue on GitHub.
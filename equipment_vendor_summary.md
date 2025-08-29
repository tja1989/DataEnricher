# Equipment → Materials → Vendors Mapping Summary

## Overview
Successfully implemented a two-layer hybrid retrieval system using OpenAI embeddings and FAISS vector database.

## Processing Summary
- **Total Equipment Processed**: 20 (from Equipment_Master.csv)
- **Equipment with Materials Found**: 10 (50%)
- **Total Material Relationships**: 10
- **Total Vendor Relationships**: 11
- **Processing Time**: ~30 seconds

## Two-Layer Retrieval Process

### Layer 1: Initial Equipment → Materials Retrieval
- Queries vector DB using equipment names
- Retrieves associated materials using hybrid search (α=0.5)
- Identifies Equipment IDs and Material IDs

### Layer 2: Gap-Filling & Validation
- Re-queries with specific IDs to fill missing information
- Uses highly structured search (α=0.1) for ID resolution
- Maps vendors from PO_Lines.csv
- Enriches with vendor names from Vendors.csv

## Key Findings

### Equipment with Materials (10/20):
1. **Centrifugal Pump CRN 10-6** → 2 materials → 2 vendors
2. **Process Pump ETA 65-160** → 1 material → 1 vendor
3. **Main Drive Gearbox H3SH** → 1 material → 2 vendors
4. **Hoist DH 20t** → 1 material → 1 vendor
5. **Hydraulic Power Unit 45 kW** → 1 material → 1 vendor
6. **VFD ACS880** → 1 material → 1 vendor
7. **Control Valve 119** → 1 material → 1 vendor
8. **Mag Flowmeter Promag** → 1 material → 1 vendor
9. **Grid Coupling 1100T10** → 1 material → 1 vendor
10. **Equipment without materials**: 10 items (no spare parts found in data)

## Vendor Distribution
- **Dubai Bearings & Seals LLC**: 2 material relationships
- **Abu Dhabi Sealing Co.**: 2 material relationships
- **Gulf Industrial Supplies FZE**: 3 material relationships
- **Gulf Industrial Supplies F.Z.E.**: 1 material relationship
- **Arabian Pumps & Valves Trading LLC**: 3 material relationships

## Technical Implementation

### Hybrid Search Strategy
- **Equipment lookup**: α=0.3-0.5 (balanced for name matching)
- **Material association**: α=0.5-0.6 (semantic relationship finding)
- **Gap-filling queries**: α=0.1 (highly structured for ID resolution)

### Key Features
✓ No speculative associations - only explicit relationships from data
✓ Comprehensive gap-filling mechanism
✓ Validation through cross-referencing with PO_Lines.csv
✓ Enrichment with vendor names from Vendors.csv
✓ JSON output format for easy integration

## Output Structure
```json
{
  "equipment": "Equipment Name",
  "equipment_id": "Equipment ID",
  "materials": [
    {
      "material": "Material Description",
      "material_id": "Material ID",
      "vendors": [
        {
          "vendor_id": "Vendor ID",
          "vendor_name": "Full Vendor Name"
        }
      ]
    }
  ]
}
```

## Files Generated
- `equipment_vendor_mapping.json`: Complete mapping results
- `equipment_vendor_mapper.py`: Two-layer retrieval implementation

## Conclusion
The hybrid embedding approach successfully leveraged both semantic understanding and structured matching to create accurate equipment-material-vendor relationships, with a 50% success rate in finding material associations for the given equipment.
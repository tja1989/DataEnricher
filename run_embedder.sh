#!/bin/bash

# Run the embedder with all warnings suppressed
export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=false

echo "Starting vector embedding process..."
python3 -W ignore vector_embedder_final.py 2>&1 | grep -v "resource_tracker" | grep -v "UserWarning"

echo ""
echo "Process completed. Check vector_indices/ directory for results."
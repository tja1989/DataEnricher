#!/bin/bash

# Example usage script

# 1. Set your OpenAI API key in .env file
echo "OPENAI_API_KEY=sk-your-actual-key-here" > .env

# 2. Create sample input directory with CSVs
mkdir -p sample_input
echo "Name,Age,City,Job
Alice,28,NYC,Engineer
Bob,35,LA,Designer
Carol,42,Chicago,Manager" > sample_input/employees.csv

echo "Product,Price,Stock
Laptop,999,50
Mouse,25,200
Keyboard,75,150" > sample_input/inventory.csv

# 3. Run the converter
python3 csv_to_statements.py \
  --input-dir ./sample_input \
  --output-dir ./sample_output \
  --max-words 25

# 4. View the results
echo "Results saved in ./sample_output/"
ls -la ./sample_output/
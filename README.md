# Universal CSV Processor

A comprehensive, all-in-one CSV processing solution that automatically adapts to your data. Converts any CSV file into hybrid records containing structured JSON and natural language semantic statements.

## Features

- **🚀 Auto Mode**: Automatically detects and uses the best processing approach
- **📊 Three Processing Modes**:
  - **Simple**: One row per API call (best for small files)
  - **Optimized**: Batch processing with multithreading (10-20x faster for large files)
  - **Folder**: Process entire directories of CSV files
- **🎯 Zero Configuration**: Works with ANY CSV structure without setup
- **💰 Cost Efficient**: Up to 90% reduction in API calls with batch processing
- **🔄 Smart Fallbacks**: Automatic error recovery and retry logic
- **📈 Performance Tracking**: Real-time statistics and progress monitoring

## Installation

```bash
# Install dependencies
pip install pandas openai python-dotenv

# Optional: Install tenacity for better retry logic
pip install tenacity

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
# OR create a .env file with:
# OPENAI_API_KEY=your-api-key-here
```

## Quick Start

### Auto Mode (Recommended)

The processor automatically detects the best approach:

```python
from universal_csv_processor import process_csv

# Automatically chooses the best mode
results = process_csv("data.csv")
```

### Command Line Usage

```bash
# Auto mode - let the processor decide
python universal_csv_processor.py data.csv

# Process a folder of CSVs
python universal_csv_processor.py data_folder/

# Specify mode explicitly
python universal_csv_processor.py large_file.csv --mode optimized --batch-size 20

# Analyze without processing
python universal_csv_processor.py data.csv --analyze
```

### Python API

```python
from universal_csv_processor import UniversalCSVProcessor

# Create processor
processor = UniversalCSVProcessor(mode="auto")

# Process single file
results = processor.process("data.csv", output_path="output.json")

# Process folder
summary = processor.process("csv_folder/", output_mode="separate")

# Get mode suggestion
suggestion = processor.suggest_best_mode("large_file.csv")
```

## Processing Modes

### 1. Auto Mode (Default)

Automatically selects the best mode based on:
- File size
- Row count  
- Input type (file vs directory)

```python
processor = UniversalCSVProcessor(mode="auto")
results = processor.process("any_input.csv")
```

### 2. Simple Mode

Best for small files (<10 rows or <0.5 MB):
- One API call per row
- Straightforward processing
- No batching overhead

```python
processor = UniversalCSVProcessor(mode="simple")
results = processor.process("small_file.csv")
```

### 3. Optimized Mode

Best for large files (>10 rows):
- Batch processing (multiple rows per API call)
- Multithreading for parallel processing
- 10-20x faster than simple mode
- 75-90% fewer API calls

```python
processor = UniversalCSVProcessor(mode="optimized", batch_size=10, max_workers=3)
results = processor.process("large_file.csv")
```

### 4. Folder Mode

Process entire directories:
- Handles multiple CSV files
- Parallel file processing
- Separate or consolidated output

```python
processor = UniversalCSVProcessor(mode="folder")
summary = processor.process("csv_folder/", output_mode="separate")
```

## Output Format

Each row is converted to a hybrid record:

```json
{
  "row_number": 1,
  "json_data": {
    "Product_ID": "PRD-001",
    "Product_Name": "Widget A",
    "Price": 29.99,
    "Stock": 150
  },
  "semantic_statement": "Product PRD-001, named Widget A, is priced at $29.99 with 150 units in stock.",
  "combined_text": "Product PRD-001, named Widget A... | Product_ID: PRD-001 | Product_Name: Widget A..."
}
```

## Auto Mode Decision Logic

The processor automatically selects the best mode:

```
Input Analysis
    ↓
Is it a directory?
    Yes → FOLDER MODE
    No ↓
    
Check file size and rows
    < 10 rows → SIMPLE MODE
    10-50 rows → OPTIMIZED (batch=5)
    50-200 rows → OPTIMIZED (batch=10)
    > 200 rows → OPTIMIZED (batch=20)
```

## Performance Benchmarks

| File Size | Simple Mode | Optimized Mode | Improvement |
|-----------|------------|----------------|-------------|
| 10 rows | 10 API calls, ~10s | 1 API call, ~2s | 10x fewer calls, 5x faster |
| 50 rows | 50 API calls, ~50s | 5 API calls, ~8s | 10x fewer calls, 6x faster |
| 200 rows | 200 API calls, ~200s | 10 API calls, ~15s | 20x fewer calls, 13x faster |

## Advanced Usage

### Custom Batch Sizes

```python
# Adjust batch size based on your needs
processor = UniversalCSVProcessor(
    mode="optimized",
    batch_size=20,  # More rows per API call
    max_workers=5    # More parallel threads
)
```

### Header Mapping

```python
# Map CSV headers to custom roles
mapping = {
    "Product_ID": "identifier",
    "Product_Name": "item",
    "Price": "cost"
}

processor = UniversalCSVProcessor(mode="simple")
results = processor.process("data.csv", header_mapping=mapping)
```

### Folder Processing Options

```python
# Separate files (default)
processor.process("folder/", output_mode="separate")
# Creates: file1_processed.json, file2_processed.json, etc.

# Consolidated output
processor.process("folder/", output_mode="consolidated")
# Creates: consolidated_output.json with all records
```

### Error Handling

The processor includes robust error handling:
- Automatic retry with exponential backoff
- Fallback to individual processing if batch fails
- Graceful null value handling
- Progress checkpointing for large files

## API Reference

### Main Class

```python
UniversalCSVProcessor(
    api_key: Optional[str] = None,
    mode: str = "auto",
    batch_size: int = 10,
    max_workers: int = 3,
    timeout: int = 30
)
```

### Parameters

- `api_key`: OpenAI API key (uses env var if not provided)
- `mode`: Processing mode - "auto", "simple", "optimized", "folder"
- `batch_size`: Rows per batch for optimized mode
- `max_workers`: Number of parallel threads
- `timeout`: API call timeout in seconds

### Methods

- `process(input_path, output_path=None, **kwargs)`: Main processing method
- `suggest_best_mode(csv_path)`: Get mode recommendation
- `_detect_mode(path)`: Auto-detect best mode (internal)

## Statistics and Monitoring

The processor tracks detailed statistics:

```python
processor.stats.total_rows        # Total rows in file
processor.stats.processed_rows    # Successfully processed
processor.stats.api_calls         # Number of API calls made
processor.stats.processing_time   # Total time taken
processor.stats.rows_per_second   # Processing speed
processor.stats.api_efficiency    # Rows per API call
```

## Examples

### Example 1: Process Equipment Data

```python
# Auto mode handles everything
results = process_csv("equipment.csv")

# Each equipment becomes:
{
  "json_data": {"Equipment_ID": "EQ-001", "Name": "Pump A", "Status": "Active"},
  "semantic_statement": "Equipment EQ-001, named Pump A, is currently active."
}
```

### Example 2: Process Large Dataset

```python
# Optimized mode for efficiency
processor = UniversalCSVProcessor(mode="optimized", batch_size=20)
results = processor.process("large_dataset.csv", output_path="processed.json")

print(f"Processed {processor.stats.processed_rows} rows")
print(f"Used only {processor.stats.api_calls} API calls")
print(f"Efficiency: {processor.stats.api_efficiency:.1f} rows per call")
```

### Example 3: Batch Process Multiple Files

```python
# Process all CSVs in a directory
processor = UniversalCSVProcessor(mode="folder")
summary = processor.process("data_folder/")

print(f"Processed {summary['statistics']['successful_files']} files")
print(f"Total rows: {summary['statistics']['processed_rows']}")
```

## Testing

Run the comprehensive test suite:

```bash
python test_universal.py
```

This tests:
- Auto mode detection
- All processing modes
- Null value handling
- Performance comparisons
- Error recovery

## Requirements

- Python 3.7+
- pandas
- openai
- python-dotenv
- tenacity (optional, for enhanced retry logic)

## License

MIT License - Use freely in your projects

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Support

For questions or issues, please create an issue in the repository.
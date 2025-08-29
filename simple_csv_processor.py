#!/usr/bin/env python3
"""
Simple CSV Processor - Streamlined Single-Mode Implementation

Features:
- Single processing mode with batch API calls
- Row-by-row processing for 100% accuracy
- Batch API calls to minimize costs
- Checkpoint system for resilience
- Clean and maintainable code
"""

import os
import csv
import json
import time
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleCsvProcessor:
    """
    Simple CSV processor with batch API optimization
    Processes rows sequentially but batches API calls for efficiency
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 batch_size: Optional[int] = None,
                 timeout: int = 30):
        """
        Initialize processor
        
        Args:
            api_key: OpenAI API key (uses env var if not provided)
            batch_size: Number of rows to process in single API call (auto-detected if None)
            timeout: API timeout in seconds
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. "
                "Set OPENAI_API_KEY env var or pass api_key parameter"
            )
        
        self.client = OpenAI(api_key=self.api_key, timeout=timeout)
        self.batch_size = batch_size  # Will be auto-determined if None
        self.timeout = timeout
        self.auto_batch = batch_size is None
        
        # Statistics
        self.stats = {
            'total_rows': 0,
            'processed_rows': 0,
            'failed_rows': 0,
            'api_calls': 0,
            'start_time': 0,
            'end_time': 0
        }
        
        if batch_size:
            logger.info(f"Processor initialized with fixed batch_size={batch_size}")
        else:
            logger.info("Processor initialized with automatic batch sizing")
    
    def check_api_connection(self) -> bool:
        """Test API connectivity"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False
    
    def _determine_batch_size(self, df: pd.DataFrame) -> int:
        """
        Automatically determine optimal batch size based on data characteristics
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            Optimal batch size
        """
        # Get data characteristics
        num_rows = len(df)
        num_columns = len(df.columns)
        
        # Sample first few rows to estimate complexity
        sample_size = min(5, num_rows)
        sample_rows = df.head(sample_size)
        
        # Estimate average data size per row
        total_chars = 0
        null_count = 0
        
        for _, row in sample_rows.iterrows():
            row_dict = row.to_dict()
            # Count characters in non-null values
            for value in row_dict.values():
                if pd.notna(value):
                    total_chars += len(str(value))
                else:
                    null_count += 1
        
        avg_chars_per_row = total_chars / sample_size if sample_size > 0 else 100
        data_density = 1 - (null_count / (sample_size * num_columns)) if num_columns > 0 else 0.5
        
        # Calculate batch size based on complexity
        # Approximate token usage: chars/4 + overhead
        estimated_tokens_per_row = (avg_chars_per_row / 4) + 50
        
        # Determine batch size with constraints
        # Target ~2000 tokens per API call for efficiency
        # Max tokens per call is typically 4000-8000
        target_tokens_per_call = 2000
        
        if estimated_tokens_per_row < 50:
            # Very simple data: larger batches
            batch_size = 25
        elif estimated_tokens_per_row < 100:
            # Simple data
            batch_size = 20
        elif estimated_tokens_per_row < 200:
            # Medium complexity
            batch_size = 15
        elif estimated_tokens_per_row < 400:
            # Complex data
            batch_size = 10
        else:
            # Very complex data: smaller batches
            batch_size = 5
        
        # Adjust based on total rows
        if num_rows < 10:
            # Small file: process all at once
            batch_size = min(batch_size, num_rows)
        elif num_rows < 50:
            # Medium file: slightly smaller batches for better progress feedback
            batch_size = min(batch_size, 10)
        
        # Apply data density factor
        if data_density < 0.3:
            # Sparse data: can handle larger batches
            batch_size = int(batch_size * 1.5)
        elif data_density > 0.8:
            # Dense data: reduce batch size
            batch_size = int(batch_size * 0.8)
        
        # Ensure reasonable bounds
        batch_size = max(1, min(batch_size, 30))
        
        logger.info(
            f"Auto-detected batch size: {batch_size} "
            f"(rows={num_rows}, cols={num_columns}, "
            f"avg_chars={avg_chars_per_row:.0f}, density={data_density:.2f})"
        )
        
        return batch_size
    
    def process(self, 
                input_path: str,
                output_path: Optional[str] = None,
                checkpoint_file: str = 'checkpoint.json') -> List[Dict]:
        """
        Process CSV file with checkpoint support
        
        Args:
            input_path: Path to CSV file
            output_path: Optional output file path  
            checkpoint_file: Checkpoint file for resume capability
        
        Returns:
            List of processed records
        """
        # Validate API connection
        print("🔄 Checking API connection...")
        if not self.check_api_connection():
            raise ConnectionError(
                "Cannot establish API connection. Please check:\n"
                "1. Your internet connection\n"
                "2. OpenAI API key is valid\n"
                "3. OpenAI service status"
            )
        print("✅ API connection verified\n")
        
        # Load checkpoint if exists
        start_row, processed_results = self._load_checkpoint(checkpoint_file)
        
        # Read CSV
        df = pd.read_csv(input_path)
        self.stats['total_rows'] = len(df)
        self.stats['start_time'] = time.time()
        
        # Auto-determine batch size if needed
        if self.auto_batch:
            self.batch_size = self._determine_batch_size(df)
        
        print(f"📊 Processing CSV: {input_path}")
        print(f"   Total rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Batch size: {self.batch_size} {'(auto-detected)' if self.auto_batch else '(fixed)'}")
        
        # Resume from checkpoint if needed
        if start_row > 0:
            print(f"📂 Resuming from row {start_row + 1}")
            print(f"   Already processed: {len(processed_results)} rows\n")
            df = df.iloc[start_row:]
        else:
            print()
        
        # Process in batches
        results = self._process_with_batches(
            df, 
            processed_results, 
            start_row,
            checkpoint_file
        )
        
        self.stats['end_time'] = time.time()
        self.stats['processed_rows'] = len(results)
        
        # Save results
        if output_path:
            self._save_results(results, output_path)
            print(f"\n✅ Results saved to: {output_path}")
        
        # Clean up checkpoint
        if Path(checkpoint_file).exists():
            os.remove(checkpoint_file)
            logger.info("Checkpoint file removed (processing complete)")
        
        # Print statistics
        self._print_statistics()
        
        return results
    
    def _process_with_batches(self, 
                             df: pd.DataFrame, 
                             processed_results: List[Dict],
                             offset: int,
                             checkpoint_file: str) -> List[Dict]:
        """
        Process dataframe rows in batches
        
        Args:
            df: DataFrame to process
            processed_results: Already processed results (from checkpoint)
            offset: Row offset for numbering (currently unused but kept for compatibility)
            checkpoint_file: Checkpoint file path
        
        Returns:
            Complete list of processed records
        """
        _ = offset  # Mark as intentionally unused
        results = processed_results.copy()
        batch_buffer = []
        
        for idx, row in df.iterrows():
            actual_row_num = idx + 1  # 1-based numbering
            
            # Convert row to dictionary
            row_data = row.to_dict()
            
            # Clean null values
            row_data = {
                k: (v if pd.notna(v) else None) 
                for k, v in row_data.items()
            }
            
            # Add to batch buffer
            batch_buffer.append((actual_row_num, row_data))
            
            # Process batch when full or last row
            is_last_row = (idx == df.index[-1])
            if len(batch_buffer) >= self.batch_size or is_last_row:
                try:
                    # Process batch via API
                    batch_results = self._process_batch(batch_buffer)
                    
                    # Verify all rows got responses
                    if len(batch_results) != len(batch_buffer):
                        raise ValueError(
                            f"API returned {len(batch_results)} results "
                            f"for {len(batch_buffer)} rows"
                        )
                    
                    # Add to results
                    results.extend(batch_results)
                    self.stats['processed_rows'] += len(batch_results)
                    
                    # Progress update
                    total_processed = len(results)
                    progress = (total_processed / self.stats['total_rows']) * 100
                    print(
                        f"   Processed: {total_processed}/{self.stats['total_rows']} "
                        f"rows ({progress:.1f}%)"
                    )
                    
                    # Save checkpoint
                    self._save_checkpoint(
                        checkpoint_file, 
                        actual_row_num, 
                        results
                    )
                    
                    # Clear buffer
                    batch_buffer = []
                    
                except Exception as e:
                    # Log error and suspend
                    logger.error(f"Batch processing failed at row {actual_row_num}: {e}")
                    
                    # Save state
                    failed_row = batch_buffer[0][0] if batch_buffer else actual_row_num
                    self._save_checkpoint(
                        checkpoint_file, 
                        failed_row - 1, 
                        results
                    )
                    
                    print(f"\n❌ Processing suspended at row {failed_row}")
                    print(f"   Error: {e}")
                    print(f"   Checkpoint saved: {checkpoint_file}")
                    print(f"   Successfully processed: {len(results)} rows")
                    print("\n   To resume: Run the same command again")
                    
                    raise
        
        return results
    
    def _process_batch(self, batch: List[tuple]) -> List[Dict]:
        """
        Process a batch of rows with single API call
        
        Args:
            batch: List of (row_number, row_data) tuples
        
        Returns:
            List of processed records
        """
        # Create prompt
        prompt = self._create_batch_prompt(batch)
        
        # Call API
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a data analyst. Convert each data row into a natural language description."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200 * len(batch),
            response_format={"type": "json_object"}
        )
        
        self.stats['api_calls'] += 1
        
        # Parse response
        api_response = json.loads(response.choices[0].message.content)
        
        # Build results
        results = []
        for i, (curr_row_num, row_data) in enumerate(batch):
            # Get semantic description from API response
            semantic = api_response.get(f"row_{i+1}", {}).get("description", "")
            
            if not semantic:
                raise ValueError(f"No description returned for row {curr_row_num}")
            
            # Create combined text for embeddings
            structured = " | ".join([
                f"{k}: {v}" for k, v in row_data.items() 
                if v is not None
            ])
            combined = f"{semantic} | {structured}"
            
            results.append({
                "row_number": curr_row_num,
                "json_data": row_data,
                "semantic_statement": semantic,
                "combined_text": combined
            })
        
        return results
    
    def _create_batch_prompt(self, batch: List[tuple]) -> str:
        """Create prompt for batch processing"""
        prompt = """Process each data row and create a natural language description.

Data rows:
"""
        
        for i, (row_num, row_data) in enumerate(batch, 1):
            prompt += f"\n--- Row {i} ---\n"
            prompt += json.dumps(row_data, indent=2)
        
        prompt += """

Return JSON with this exact format:
{
  "row_1": {"description": "Natural language description including all information..."},
  "row_2": {"description": "Natural language description including all information..."},
  ...
}

Rules:
- Each description must include ALL fields and their values
- Make descriptions natural and readable
- Be concise but complete
"""
        
        return prompt
    
    def _save_checkpoint(self, checkpoint_file: str, last_row: int, results: List[Dict]):
        """Save processing checkpoint"""
        checkpoint = {
            'last_successful_row': last_row,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_processed': len(results),
            'results': results
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.debug(f"Checkpoint saved: {last_row} rows processed")
    
    def _load_checkpoint(self, checkpoint_file: str) -> tuple:
        """Load checkpoint if exists"""
        if Path(checkpoint_file).exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                return checkpoint['last_successful_row'], checkpoint['results']
        return 0, []
    
    def _save_results(self, results: List[Dict], output_path: str):
        """Save processing results"""
        processing_time = self.stats['end_time'] - self.stats['start_time']
        
        output = {
            "metadata": {
                "total_records": len(results),
                "processing_time": round(processing_time, 2),
                "api_calls": self.stats['api_calls'],
                "batch_size": self.batch_size,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "records": results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    def _print_statistics(self):
        """Print processing statistics"""
        processing_time = self.stats['end_time'] - self.stats['start_time']
        success_rate = (
            (self.stats['processed_rows'] / self.stats['total_rows'] * 100)
            if self.stats['total_rows'] > 0 else 0
        )
        rows_per_call = (
            self.stats['processed_rows'] / self.stats['api_calls']
            if self.stats['api_calls'] > 0 else 0
        )
        
        print("\n" + "="*50)
        print("Processing Statistics:")
        print(f"  Total rows: {self.stats['total_rows']}")
        print(f"  Processed: {self.stats['processed_rows']}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  API calls: {self.stats['api_calls']}")
        print(f"  Rows per API call: {rows_per_call:.1f}")
        print(f"  Processing time: {processing_time:.1f} seconds")
        print("="*50)
    
    def process_folder(self, 
                      folder_path: str,
                      output_dir: Optional[str] = None) -> Dict:
        """
        Process all CSV files in a folder
        
        Args:
            folder_path: Path to folder containing CSV files
            output_dir: Optional output directory for processed files
            
        Returns:
            Summary dictionary with processing results
        """
        from datetime import datetime
        
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        if not folder.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")
        
        # Find all CSV files
        csv_files = list(folder.glob("*.csv"))
        
        if not csv_files:
            print(f"⚠️ No CSV files found in {folder_path}")
            return {"error": "No CSV files found", "folder": str(folder)}
        
        # Set output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            # Create timestamped output folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"OUTPUT_{folder.name}_{timestamp}")
            output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"📁 Processing folder: {folder_path}")
        print(f"📊 Found {len(csv_files)} CSV files")
        print(f"📂 Output directory: {output_path}")
        print(f"{'='*60}\n")
        
        # Track overall statistics
        overall_start = time.time()
        file_results = []
        total_rows_all = 0
        total_processed_all = 0
        total_api_calls_all = 0
        
        # Process each CSV file
        for idx, csv_file in enumerate(csv_files, 1):
            print(f"\n[{idx}/{len(csv_files)}] Processing {csv_file.name}")
            print("-" * 40)
            
            # Create file-specific checkpoint
            checkpoint_file = f"checkpoint_{csv_file.stem}.json"
            
            # Create output filename
            output_file = output_path / f"{csv_file.stem}_processed.json"
            
            try:
                # Reset stats for this file
                self.stats = {
                    'total_rows': 0,
                    'processed_rows': 0,
                    'failed_rows': 0,
                    'api_calls': 0,
                    'start_time': 0,
                    'end_time': 0
                }
                
                # Process the file
                results = self.process(
                    str(csv_file),
                    str(output_file),
                    checkpoint_file
                )
                
                # Collect statistics
                file_result = {
                    'file_name': csv_file.name,
                    'total_rows': self.stats['total_rows'],
                    'processed_rows': self.stats['processed_rows'],
                    'api_calls': self.stats['api_calls'],
                    'processing_time': self.stats['end_time'] - self.stats['start_time'],
                    'output_file': str(output_file),
                    'success': True,
                    'error': None
                }
                
                total_rows_all += self.stats['total_rows']
                total_processed_all += self.stats['processed_rows']
                total_api_calls_all += self.stats['api_calls']
                
            except Exception as e:
                print(f"❌ Error processing {csv_file.name}: {e}")
                file_result = {
                    'file_name': csv_file.name,
                    'total_rows': 0,
                    'processed_rows': 0,
                    'api_calls': 0,
                    'processing_time': 0,
                    'output_file': None,
                    'success': False,
                    'error': str(e)
                }
            
            file_results.append(file_result)
        
        # Calculate overall statistics
        overall_time = time.time() - overall_start
        successful_files = [f for f in file_results if f['success']]
        failed_files = [f for f in file_results if not f['success']]
        
        # Generate summary
        summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'folder': str(folder),
            'output_directory': str(output_path),
            'statistics': {
                'total_files': len(csv_files),
                'successful_files': len(successful_files),
                'failed_files': len(failed_files),
                'total_rows': total_rows_all,
                'processed_rows': total_processed_all,
                'total_api_calls': total_api_calls_all,
                'total_time': round(overall_time, 2),
                'api_efficiency': round(total_processed_all / total_api_calls_all, 1) if total_api_calls_all > 0 else 0
            },
            'files': file_results
        }
        
        # Save summary to file
        summary_file = output_path / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("FOLDER PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Successful: {len(successful_files)}/{len(csv_files)} files")
        print(f"📊 Total rows processed: {total_processed_all}/{total_rows_all}")
        print(f"⚡ Total API calls: {total_api_calls_all}")
        print(f"💰 API efficiency: {summary['statistics']['api_efficiency']} rows/call")
        print(f"⏱️ Total time: {overall_time:.1f} seconds")
        print(f"\n📁 Results saved to: {output_path}")
        print(f"📋 Summary saved to: {summary_file}")
        
        return summary


def main():
    """Command line interface"""
    import argparse
    from datetime import datetime
    
    # Default input folder
    DEFAULT_INPUT = "INPUT_CSV"
    
    parser = argparse.ArgumentParser(
        description="Simple CSV Processor - Process CSV files with batch optimization",
        epilog=f"Default: Processes all CSVs in '{DEFAULT_INPUT}/' folder if no input specified"
    )
    parser.add_argument(
        "input",
        nargs='?',  # Makes input optional
        default=DEFAULT_INPUT,
        help=f"Path to CSV file or folder to process (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file/directory path (auto-generated if not specified)"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=None,
        help="Number of rows per API call (default: auto-detect based on data)"
    )
    parser.add_argument(
        "-c", "--checkpoint",
        default="checkpoint.json",
        help="Checkpoint file for resume capability (default: checkpoint.json)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="API timeout in seconds (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Check if input exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Path not found: {args.input}")
        exit(1)
    
    try:
        # Create processor
        processor = SimpleCsvProcessor(
            batch_size=args.batch_size,
            timeout=args.timeout
        )
        
        # Determine if input is a file or folder
        if input_path.is_dir():
            # Process folder
            print(f"📁 Detected folder input: {args.input}")
            summary = processor.process_folder(
                args.input,
                output_dir=args.output
            )
            
        elif input_path.is_file() and args.input.endswith('.csv'):
            # Process single file
            print(f"📄 Detected CSV file input: {args.input}")
            
            # Generate output path if not provided
            if not args.output:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                input_name = input_path.stem
                args.output = f"{input_name}_processed_{timestamp}.json"
            
            results = processor.process(
                args.input,
                args.output,
                args.checkpoint
            )
            
            print(f"\n✅ Successfully processed {len(results)} records")
            
        else:
            print(f"❌ Error: Input must be a CSV file or a folder containing CSV files")
            exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Processing interrupted by user")
        if input_path.is_file():
            print(f"   Checkpoint saved to: {args.checkpoint}")
        else:
            print("   Individual checkpoint files saved for each CSV")
        print("   Run the same command to resume")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
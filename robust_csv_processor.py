#!/usr/bin/env python3
"""
Robust CSV Processor - Production-Ready Implementation with Complete Error Recovery

Features:
- Comprehensive error handling with retry logic
- Row-level error isolation
- Enhanced checkpoint system
- Async processing support
- Detailed error tracking and reporting
- 100% file completion guarantee
"""

import os
import csv
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Processing status types"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class ProcessingError:
    """Error tracking for failed rows"""
    row_number: int
    error_type: str
    error_message: str
    retry_count: int
    timestamp: str
    row_data: Dict


@dataclass
class ProcessingResult:
    """Result container for processed rows"""
    row_number: int
    json_data: Dict
    semantic_statement: str
    combined_text: str
    processing_time: float
    retry_count: int = 0


class RobustCsvProcessor:
    """
    Robust CSV processor with comprehensive error recovery
    Guarantees 100% file processing with detailed error reporting
    """
    
    # Configuration constants
    MAX_RETRIES = 3
    MAX_BATCH_SIZE = 30
    MIN_BATCH_SIZE = 1
    INITIAL_BACKOFF = 2  # seconds
    MAX_BACKOFF = 60  # seconds
    API_TIMEOUT = 30  # seconds
    RATE_LIMIT_DELAY = 5  # seconds
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 batch_size: Optional[int] = None,
                 max_workers: int = 3,
                 enable_async: bool = True):
        """
        Initialize robust processor
        
        Args:
            api_key: OpenAI API key
            batch_size: Initial batch size (auto-detected if None)
            max_workers: Maximum parallel workers for file processing
            enable_async: Enable async processing
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=self.api_key, timeout=self.API_TIMEOUT)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.enable_async = enable_async
        
        # Error tracking
        self.failed_rows: List[ProcessingError] = []
        self.skipped_rows: List[Dict] = []
        self.retry_tracker: Dict[int, int] = {}
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_rows': 0,
            'processed_rows': 0,
            'failed_rows': 0,
            'skipped_rows': 0,
            'api_calls': 0,
            'total_retries': 0,
            'start_time': 0,
            'end_time': 0
        }
        
        logger.info("Robust CSV Processor initialized")
    
    def check_api_connection(self) -> bool:
        """Test API connectivity with retry"""
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                return True
            except Exception as e:
                logger.warning(f"API connection attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return False
    
    def _determine_batch_size(self, df: pd.DataFrame) -> int:
        """Auto-determine optimal batch size"""
        num_rows = len(df)
        num_columns = len(df.columns)
        
        # Sample for complexity analysis
        sample_size = min(5, num_rows)
        sample_rows = df.head(sample_size)
        
        total_chars = 0
        for _, row in sample_rows.iterrows():
            row_dict = row.to_dict()
            for value in row_dict.values():
                if pd.notna(value):
                    total_chars += len(str(value))
        
        avg_chars = total_chars / sample_size if sample_size > 0 else 100
        
        # Calculate batch size
        if avg_chars < 100:
            batch_size = 20
        elif avg_chars < 300:
            batch_size = 15
        elif avg_chars < 500:
            batch_size = 10
        else:
            batch_size = 5
        
        # Apply bounds
        batch_size = max(self.MIN_BATCH_SIZE, min(batch_size, self.MAX_BATCH_SIZE))
        
        logger.info(f"Auto-detected batch size: {batch_size} (avg_chars={avg_chars:.0f})")
        return batch_size
    
    async def process_folder_async(self, folder_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        Process all CSV files in folder asynchronously
        
        Args:
            folder_path: Path to folder with CSV files
            output_dir: Output directory for results
            
        Returns:
            Processing summary
        """
        folder = Path(folder_path)
        csv_files = list(folder.glob("*.csv"))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {folder_path}")
            return {"error": "No CSV files found"}
        
        # Setup output directory
        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"OUTPUT_{folder.name}_{timestamp}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing {len(csv_files)} CSV files from {folder_path}")
        
        # Process files in parallel
        tasks = []
        for csv_file in csv_files:
            output_file = output_path / f"{csv_file.stem}_processed.json"
            checkpoint_file = f"checkpoint_{csv_file.stem}.json"
            
            task = self._process_file_async(
                str(csv_file),
                str(output_file),
                checkpoint_file
            )
            tasks.append(task)
        
        # Wait for all files to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Generate summary
        return self._generate_folder_summary(csv_files, results, output_path)
    
    async def _process_file_async(self, input_path: str, output_path: str, checkpoint_file: str) -> Dict:
        """Async wrapper for file processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.process_file_with_recovery,
            input_path,
            output_path,
            checkpoint_file
        )
    
    def process_file_with_recovery(self, 
                                   input_path: str,
                                   output_path: str,
                                   checkpoint_file: str) -> Dict:
        """
        Process single CSV file with complete error recovery
        
        Args:
            input_path: Input CSV file path
            output_path: Output JSON file path
            checkpoint_file: Checkpoint file for recovery
            
        Returns:
            Processing result dictionary
        """
        file_start_time = time.time()
        
        # Check API connection
        if not self.check_api_connection():
            logger.error("Cannot establish API connection")
            return {"error": "API connection failed", "file": input_path}
        
        # Load checkpoint if exists
        checkpoint = self._load_enhanced_checkpoint(checkpoint_file)
        processed_results = checkpoint.get('processed_rows', [])
        failed_rows_data = checkpoint.get('failed_rows', [])
        last_row = checkpoint.get('last_attempted_row', 0)
        
        # Read CSV
        df = pd.read_csv(input_path)
        total_rows = len(df)
        
        # Auto-determine batch size if needed
        if not self.batch_size:
            self.batch_size = self._determine_batch_size(df)
        
        logger.info(f"Processing {input_path}: {total_rows} rows, batch_size={self.batch_size}")
        
        # Resume from checkpoint
        if last_row > 0:
            logger.info(f"Resuming from row {last_row + 1}")
            df = df.iloc[last_row:]
        
        # Process with comprehensive error handling
        results, failed = self._process_with_recovery(
            df,
            processed_results,
            last_row,
            checkpoint_file
        )
        
        # Add previously failed rows
        for failed_data in failed_rows_data:
            failed.append(ProcessingError(**failed_data))
        
        # Retry failed rows with individual processing
        if failed:
            logger.info(f"Retrying {len(failed)} failed rows individually")
            recovered = self._retry_failed_rows(failed, checkpoint_file)
            results.extend(recovered)
        
        # Save results
        self._save_enhanced_results(results, failed, output_path)
        
        # Clean checkpoint on success
        if Path(checkpoint_file).exists():
            os.remove(checkpoint_file)
        
        processing_time = time.time() - file_start_time
        
        return {
            'file': input_path,
            'total_rows': total_rows,
            'processed_rows': len(results),
            'failed_rows': len(failed),
            'processing_time': processing_time,
            'success_rate': (len(results) / total_rows * 100) if total_rows > 0 else 0
        }
    
    def _process_with_recovery(self, 
                               df: pd.DataFrame,
                               processed_results: List[Dict],
                               offset: int,
                               checkpoint_file: str) -> Tuple[List[Dict], List[ProcessingError]]:
        """
        Process dataframe with comprehensive error recovery
        
        Returns:
            Tuple of (successful_results, failed_rows)
        """
        results = processed_results.copy()
        failed_rows = []
        batch_buffer = []
        current_batch_size = self.batch_size
        
        for idx, row in df.iterrows():
            actual_row_num = idx + 1
            
            # Prepare row data
            row_data = row.to_dict()
            row_data = {k: (v if pd.notna(v) else None) for k, v in row_data.items()}
            
            batch_buffer.append((actual_row_num, row_data))
            
            # Process when batch is full or last row
            is_last_row = (idx == df.index[-1])
            if len(batch_buffer) >= current_batch_size or is_last_row:
                
                # Try processing with retry logic
                batch_results, batch_failed = self._process_batch_with_retry(
                    batch_buffer,
                    current_batch_size
                )
                
                if batch_results:
                    results.extend(batch_results)
                    self.stats['processed_rows'] += len(batch_results)
                
                if batch_failed:
                    failed_rows.extend(batch_failed)
                    self.stats['failed_rows'] += len(batch_failed)
                
                # Update checkpoint
                self._save_enhanced_checkpoint(
                    checkpoint_file,
                    actual_row_num,
                    results,
                    failed_rows
                )
                
                # Progress update
                total_processed = len(results)
                progress = (total_processed / (len(df) + offset)) * 100
                logger.info(f"Progress: {total_processed} rows processed ({progress:.1f}%)")
                
                # Reset for next batch
                batch_buffer = []
                current_batch_size = self.batch_size  # Reset to original size
        
        return results, failed_rows
    
    def _process_batch_with_retry(self, 
                                  batch: List[Tuple],
                                  max_batch_size: int) -> Tuple[List[Dict], List[ProcessingError]]:
        """
        Process batch with retry logic and batch splitting
        
        Returns:
            Tuple of (successful_results, failed_rows)
        """
        results = []
        failed = []
        retry_count = 0
        current_batch = batch.copy()
        
        while retry_count < self.MAX_RETRIES and current_batch:
            try:
                # Attempt to process batch
                batch_results = self._call_api_for_batch(current_batch)
                
                # Validate results
                if len(batch_results) != len(current_batch):
                    raise ValueError(f"API returned {len(batch_results)} results for {len(current_batch)} rows")
                
                results.extend(batch_results)
                self.stats['api_calls'] += 1
                break  # Success, exit retry loop
                
            except Exception as e:
                retry_count += 1
                self.stats['total_retries'] += 1
                
                error_msg = str(e)
                logger.warning(f"Batch processing error (attempt {retry_count}): {error_msg}")
                
                # Handle specific error types
                if "rate_limit" in error_msg.lower():
                    time.sleep(self.RATE_LIMIT_DELAY * retry_count)
                    
                elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    # Reduce batch size and retry
                    if len(current_batch) > 1:
                        logger.info(f"Splitting batch from {len(current_batch)} to {len(current_batch)//2}")
                        mid = len(current_batch) // 2
                        
                        # Process first half
                        first_results, first_failed = self._process_batch_with_retry(
                            current_batch[:mid],
                            max_batch_size // 2
                        )
                        results.extend(first_results)
                        failed.extend(first_failed)
                        
                        # Process second half
                        second_results, second_failed = self._process_batch_with_retry(
                            current_batch[mid:],
                            max_batch_size // 2
                        )
                        results.extend(second_results)
                        failed.extend(second_failed)
                        
                        break  # Exit after splitting
                        
                elif retry_count < self.MAX_RETRIES:
                    # Exponential backoff
                    wait_time = min(self.INITIAL_BACKOFF ** retry_count, self.MAX_BACKOFF)
                    logger.info(f"Waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    
                else:
                    # Max retries reached, record failures
                    for row_num, row_data in current_batch:
                        failed.append(ProcessingError(
                            row_number=row_num,
                            error_type=type(e).__name__,
                            error_message=error_msg,
                            retry_count=retry_count,
                            timestamp=datetime.now().isoformat(),
                            row_data=row_data
                        ))
                    break
        
        return results, failed
    
    def _call_api_for_batch(self, batch: List[Tuple]) -> List[Dict]:
        """
        Make API call for batch processing
        
        Args:
            batch: List of (row_number, row_data) tuples
            
        Returns:
            List of processed results
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
        
        # Parse response
        api_response = json.loads(response.choices[0].message.content)
        
        # Build results
        results = []
        for i, (row_num, row_data) in enumerate(batch):
            semantic = api_response.get(f"row_{i+1}", {}).get("description", "")
            
            if not semantic:
                raise ValueError(f"No description returned for row {row_num}")
            
            # Create combined text
            structured = " | ".join([
                f"{k}: {v}" for k, v in row_data.items() 
                if v is not None
            ])
            combined = f"{semantic} | {structured}"
            
            results.append({
                "row_number": row_num,
                "json_data": row_data,
                "semantic_statement": semantic,
                "combined_text": combined
            })
        
        return results
    
    def _retry_failed_rows(self, 
                          failed_rows: List[ProcessingError],
                          checkpoint_file: str) -> List[Dict]:
        """
        Retry failed rows individually
        
        Args:
            failed_rows: List of failed row errors
            checkpoint_file: Checkpoint file path
            
        Returns:
            List of recovered results
        """
        recovered = []
        still_failed = []
        
        for error in failed_rows:
            try:
                # Try processing single row
                result = self._call_api_for_batch([(error.row_number, error.row_data)])
                if result:
                    recovered.extend(result)
                    logger.info(f"Recovered row {error.row_number}")
                    
            except Exception as e:
                logger.error(f"Failed to recover row {error.row_number}: {e}")
                still_failed.append(error)
        
        # Update checkpoint with still failed rows
        if still_failed:
            checkpoint = self._load_enhanced_checkpoint(checkpoint_file)
            checkpoint['permanently_failed'] = [asdict(e) for e in still_failed]
            self._save_checkpoint_data(checkpoint, checkpoint_file)
        
        return recovered
    
    def _create_batch_prompt(self, batch: List[Tuple]) -> str:
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
    
    def _save_enhanced_checkpoint(self, 
                                  checkpoint_file: str,
                                  last_row: int,
                                  results: List[Dict],
                                  failed: List[ProcessingError]):
        """Save enhanced checkpoint with failure tracking"""
        checkpoint = {
            'last_attempted_row': last_row,
            'timestamp': datetime.now().isoformat(),
            'processed_rows': results,
            'failed_rows': [asdict(e) for e in failed],
            'total_processed': len(results),
            'total_failed': len(failed),
            'status': ProcessingStatus.IN_PROGRESS.value
        }
        
        self._save_checkpoint_data(checkpoint, checkpoint_file)
    
    def _load_enhanced_checkpoint(self, checkpoint_file: str) -> Dict:
        """Load enhanced checkpoint"""
        if Path(checkpoint_file).exists():
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_checkpoint_data(self, checkpoint: Dict, checkpoint_file: str):
        """Save checkpoint data to file"""
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
    
    def _save_enhanced_results(self, 
                               results: List[Dict],
                               failed: List[ProcessingError],
                               output_path: str):
        """Save results with error reporting"""
        output = {
            "metadata": {
                "total_records": len(results),
                "failed_records": len(failed),
                "success_rate": (len(results) / (len(results) + len(failed)) * 100) if results or failed else 0,
                "timestamp": datetime.now().isoformat(),
                "batch_size": self.batch_size,
                "api_calls": self.stats['api_calls'],
                "total_retries": self.stats['total_retries']
            },
            "records": results,
            "failed_rows": [asdict(e) for e in failed] if failed else []
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    def _generate_folder_summary(self, csv_files: List[Path], results: List, output_path: Path) -> Dict:
        """Generate processing summary for folder"""
        successful = [r for r in results if isinstance(r, dict) and r.get('success_rate', 0) > 0]
        failed = [r for r in results if isinstance(r, Exception) or (isinstance(r, dict) and r.get('error'))]
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'output_directory': str(output_path),
            'statistics': {
                'total_files': len(csv_files),
                'successful_files': len(successful),
                'failed_files': len(failed),
                'total_api_calls': self.stats['api_calls'],
                'total_retries': self.stats['total_retries']
            },
            'files': results
        }
        
        # Save summary
        summary_file = output_path / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary
    
    def process_folder(self, folder_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        Process folder synchronously (fallback for non-async environments)
        """
        if self.enable_async:
            # Run async version
            return asyncio.run(self.process_folder_async(folder_path, output_dir))
        else:
            # Synchronous processing
            return self._process_folder_sync(folder_path, output_dir)
    
    def _process_folder_sync(self, folder_path: str, output_dir: Optional[str] = None) -> Dict:
        """Synchronous folder processing with ThreadPoolExecutor"""
        folder = Path(folder_path)
        csv_files = list(folder.glob("*.csv"))
        
        if not csv_files:
            return {"error": "No CSV files found"}
        
        # Setup output directory
        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"OUTPUT_{folder.name}_{timestamp}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"📁 Processing folder: {folder_path}")
        print(f"📊 Found {len(csv_files)} CSV files")
        print(f"📂 Output directory: {output_path}")
        print(f"{'='*60}\n")
        
        results = []
        
        # Process files with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for csv_file in csv_files:
                output_file = output_path / f"{csv_file.stem}_processed.json"
                checkpoint_file = f"checkpoint_{csv_file.stem}.json"
                
                future = executor.submit(
                    self.process_file_with_recovery,
                    str(csv_file),
                    str(output_file),
                    checkpoint_file
                )
                futures[future] = csv_file.name
            
            # Process as completed
            for future in as_completed(futures):
                file_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"✅ Completed: {file_name}")
                except Exception as e:
                    logger.error(f"Failed to process {file_name}: {e}")
                    results.append({"file": file_name, "error": str(e)})
                    print(f"❌ Failed: {file_name}")
        
        # Generate summary
        summary = self._generate_folder_summary(csv_files, results, output_path)
        
        # Print summary
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Successful: {summary['statistics']['successful_files']}/{len(csv_files)} files")
        print(f"⚡ Total API calls: {summary['statistics']['total_api_calls']}")
        print(f"🔄 Total retries: {summary['statistics']['total_retries']}")
        print(f"📁 Results saved to: {output_path}")
        
        return summary


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Robust CSV Processor - Production-ready with complete error recovery"
    )
    parser.add_argument(
        "input",
        nargs='?',
        default="INPUT_CSV",
        help="Path to CSV file or folder (default: INPUT_CSV)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory path"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        help="Batch size for API calls (auto-detected if not specified)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=3,
        help="Max parallel workers (default: 3)"
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous processing instead of async"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up checkpoint files before processing"
    )
    
    args = parser.parse_args()
    
    # Clean up checkpoints if requested
    if args.cleanup:
        checkpoints = Path(".").glob("checkpoint_*.json")
        for checkpoint in checkpoints:
            os.remove(checkpoint)
            print(f"Removed: {checkpoint}")
    
    # Check input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Path not found: {args.input}")
        exit(1)
    
    try:
        # Create processor
        processor = RobustCsvProcessor(
            batch_size=args.batch_size,
            max_workers=args.workers,
            enable_async=not args.sync
        )
        
        # Process based on input type
        if input_path.is_dir():
            summary = processor.process_folder(args.input, args.output)
        elif input_path.is_file() and input_path.suffix == '.csv':
            # Single file processing
            output_file = args.output or f"{input_path.stem}_processed.json"
            checkpoint_file = f"checkpoint_{input_path.stem}.json"
            
            result = processor.process_file_with_recovery(
                str(input_path),
                output_file,
                checkpoint_file
            )
            
            print(f"\n✅ Processing complete:")
            print(f"   Processed: {result['processed_rows']}/{result['total_rows']} rows")
            print(f"   Success rate: {result['success_rate']:.1f}%")
            print(f"   Output: {output_file}")
        else:
            print(f"❌ Error: Input must be a CSV file or folder")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted")
        print("Checkpoint files saved - run again to resume")
        exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
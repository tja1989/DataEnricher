#!/usr/bin/env python3
"""
Universal CSV Processor - All-in-One Solution

Combines simple, optimized, and multi-CSV processing in a single file.
Automatically selects the best processing mode based on input.

Features:
- Simple mode: One row per API call (small files)
- Optimized mode: Batch processing with threading (large files)
- Folder mode: Process entire directories (multiple CSVs)
- Auto mode: Automatically selects best approach
"""

import os
import csv
import json
import glob
import time
import logging
import hashlib
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from openai import OpenAI
from dotenv import load_dotenv

# Try to import tenacity for retry logic, fall back to simple retry if not available
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    from openai import APIError, APITimeoutError, RateLimitError
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False
    print("Warning: tenacity not installed. Using simple retry logic.")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class ProcessingSuspendedException(Exception):
    """Raised when processing is suspended due to API issues"""
    pass

class APIConnectionError(Exception):
    """Raised when API connection cannot be established"""
    pass

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ProcessingStats:
    """Track processing statistics"""
    total_rows: int = 0
    processed_rows: int = 0
    failed_rows: int = 0
    api_calls: int = 0
    start_time: float = 0
    end_time: float = 0
    mode: str = "simple"
    
    @property
    def success_rate(self) -> float:
        if self.total_rows == 0:
            return 0
        return (self.processed_rows / self.total_rows) * 100
    
    @property
    def processing_time(self) -> float:
        return self.end_time - self.start_time if self.end_time else time.time() - self.start_time
    
    @property
    def rows_per_second(self) -> float:
        if self.processing_time == 0:
            return 0
        return self.processed_rows / self.processing_time
    
    @property
    def api_efficiency(self) -> float:
        if self.api_calls == 0:
            return 0
        return self.processed_rows / self.api_calls


@dataclass
class FileProcessingResult:
    """Result for a single CSV file"""
    file_name: str
    file_path: str
    total_rows: int
    processed_rows: int
    processing_time: float
    api_calls: int
    mode_used: str
    success: bool
    error_message: Optional[str] = None
    output_file: Optional[str] = None


# ============================================================================
# MAIN UNIVERSAL PROCESSOR
# ============================================================================

class UniversalCSVProcessor:
    """
    Universal CSV processor with automatic mode selection.
    Combines simple, optimized, and folder processing in one class.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 mode: str = "auto",
                 batch_size: int = 10,
                 max_workers: int = 3,
                 timeout: int = 30):
        """
        Initialize universal processor
        
        Args:
            api_key: OpenAI API key (uses env var if not provided)
            mode: Processing mode - "auto", "simple", "optimized", "folder"
            batch_size: Rows per batch (for optimized mode)
            max_workers: Thread workers (for optimized mode)
            timeout: API timeout in seconds
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter")
        
        self.client = OpenAI(api_key=self.api_key, timeout=timeout)
        self.mode = mode
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.timeout = timeout
        self.stats = ProcessingStats()
        self.results_lock = threading.Lock()
        
        logger.info(f"UniversalCSVProcessor initialized in '{mode}' mode")
    
    def check_api_connection(self) -> bool:
        """Test API connectivity before processing"""
        try:
            # Simple test call
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False
    
    def wait_for_api_connection(self, max_wait_seconds=300, check_interval=10):
        """Wait for API to become available"""
        start_time = time.time()
        attempt = 1
        
        while time.time() - start_time < max_wait_seconds:
            print(f"Checking API connection (attempt {attempt})...")
            
            if self.check_api_connection():
                print("✅ API connection established!")
                return True
            
            print(f"❌ API unavailable. Retrying in {check_interval} seconds...")
            time.sleep(check_interval)
            attempt += 1
        
        return False
    
    def process(self, 
                input_path: str,
                output_path: Optional[str] = None,
                header_mapping: Optional[Dict[str, str]] = None,
                **kwargs) -> Any:
        """
        Main processing method - automatically selects best approach
        
        Args:
            input_path: Path to CSV file or directory
            output_path: Optional output path
            header_mapping: Optional header mapping for simple mode
            **kwargs: Additional arguments for specific modes
        
        Returns:
            Processing results (format depends on mode)
        """
        # Check API connection first
        print("Checking API connection...")
        if not self.check_api_connection():
            print("⚠️ API unavailable. Attempting to reconnect...")
            if not self.wait_for_api_connection():
                raise APIConnectionError(
                    "Cannot establish API connection. Please check:\n"
                    "1. Your internet connection\n"
                    "2. OpenAI API key is valid\n"
                    "3. OpenAI service status"
                )
        
        print("✅ API connection verified")
        
        path = Path(input_path)
        
        # Determine processing mode
        if self.mode == "auto":
            detected_mode = self._detect_mode(path)
            logger.info(f"Auto-detected mode: {detected_mode}")
            self.mode = detected_mode
        
        # Route to appropriate processor
        if self.mode == "folder":
            return self._process_folder(path, output_path, **kwargs)
        elif self.mode == "optimized":
            return self._process_optimized(path, output_path, **kwargs)
        else:  # simple
            return self._process_simple(path, output_path, header_mapping, **kwargs)
    
    def _detect_mode(self, path: Path) -> str:
        """
        Automatically detect the best processing mode
        
        Args:
            path: Input path
        
        Returns:
            Detected mode: "simple", "optimized", or "folder"
        """
        # If directory, use folder mode
        if path.is_dir():
            return "folder"
        
        # If not a CSV file, default to simple
        if not path.suffix.lower() == '.csv':
            return "simple"
        
        # Check file size and row count
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
            
            # Quick row count (read first column only)
            row_count = len(pd.read_csv(path, usecols=[0]))
            
            # Decision logic
            if row_count <= 10 or size_mb < 0.5:
                logger.info(f"Small file detected ({row_count} rows, {size_mb:.1f} MB) -> simple mode")
                return "simple"
            else:
                logger.info(f"Large file detected ({row_count} rows, {size_mb:.1f} MB) -> optimized mode")
                # Adjust batch size based on file size
                if row_count > 100:
                    self.batch_size = min(20, row_count // 10)
                return "optimized"
        except Exception as e:
            logger.warning(f"Mode detection failed: {e}, defaulting to simple mode")
            return "simple"
    
    # ========================================================================
    # SIMPLE MODE PROCESSING
    # ========================================================================
    
    def _process_simple(self, 
                       path: Path,
                       output_path: Optional[str] = None,
                       header_mapping: Optional[Dict[str, str]] = None,
                       **kwargs) -> List[Dict]:
        """
        Simple processing - one row per API call
        
        Args:
            path: CSV file path
            output_path: Optional output file path
            header_mapping: Optional header to role mapping
        
        Returns:
            List of processed records
        """
        logger.info(f"Processing CSV in SIMPLE mode: {path}")
        self.stats = ProcessingStats(start_time=time.time(), mode="simple")
        
        # Read CSV
        df = pd.read_csv(path)
        self.stats.total_rows = len(df)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        results = []
        
        # Process each row individually
        for idx, row in df.iterrows():
            try:
                # Convert row to JSON
                json_obj = self._row_to_json(row, header_mapping)
                
                # Generate semantic statement
                semantic = self._generate_semantic_simple(json_obj)
                
                # Create combined text
                combined = self._create_combined_text(json_obj, semantic)
                
                record = {
                    "row_number": idx + 1,
                    "json_data": json_obj,
                    "semantic_statement": semantic,
                    "combined_text": combined
                }
                
                results.append(record)
                self.stats.processed_rows += 1
                
                # Progress logging
                if (idx + 1) % 10 == 0:
                    logger.info(f"Progress: {idx + 1}/{len(df)} rows")
                    
            except Exception as e:
                logger.error(f"Error processing row {idx + 1}: {e}")
                self.stats.failed_rows += 1
                results.append({
                    "row_number": idx + 1,
                    "json_data": row.to_dict(),
                    "semantic_statement": f"Error: {str(e)}",
                    "combined_text": str(row.to_dict())
                })
        
        self.stats.end_time = time.time()
        
        # Save results if output path provided
        if output_path:
            self._save_results(results, output_path)
        
        self._log_statistics()
        
        return results
    
    # ========================================================================
    # OPTIMIZED MODE PROCESSING
    # ========================================================================
    
    def _process_optimized(self,
                          path: Path,
                          output_path: Optional[str] = None,
                          show_progress: bool = True,
                          **kwargs) -> List[Dict]:
        """
        Optimized processing - batch processing with threading
        
        Args:
            path: CSV file path
            output_path: Optional output file path
            show_progress: Show progress updates
        
        Returns:
            List of processed records
        """
        logger.info(f"Processing CSV in OPTIMIZED mode: {path}")
        logger.info(f"Batch size: {self.batch_size}, Workers: {self.max_workers}")
        self.stats = ProcessingStats(start_time=time.time(), mode="optimized")
        
        # Read CSV
        df = pd.read_csv(path)
        self.stats.total_rows = len(df)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Convert to list of dictionaries
        rows = df.to_dict('records')
        
        # Create batches
        batches = self._create_batches(rows)
        logger.info(f"Created {len(batches)} batches")
        
        # Process batches in parallel
        results = self._process_batches_parallel(batches, show_progress)
        
        # Sort by row number
        results.sort(key=lambda x: x['row_number'])
        
        self.stats.end_time = time.time()
        
        # Save results
        if output_path:
            self._save_results(results, output_path)
        
        self._log_statistics()
        
        return results
    
    def _create_batches(self, rows: List[Dict]) -> List[List[Tuple[int, Dict]]]:
        """Create batches of rows with row numbers"""
        batches = []
        for i in range(0, len(rows), self.batch_size):
            batch = [(j+1, row) for j, row in enumerate(rows[i:i+self.batch_size], start=i)]
            batches.append(batch)
        return batches
    
    def _process_batches_parallel(self, 
                                 batches: List[List[Tuple[int, Dict]]], 
                                 show_progress: bool) -> List[Dict]:
        """Process batches in parallel using thread pool"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for batch_idx, batch in enumerate(batches):
                future = executor.submit(self._process_batch_with_retry, batch)
                futures[future] = batch_idx
            
            completed = 0
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    batch_results = future.result()
                    
                    with self.results_lock:
                        results.extend(batch_results)
                        completed += len(batch_results)
                        
                        if show_progress:
                            progress = (completed / self.stats.total_rows) * 100
                            logger.info(f"Progress: {completed}/{self.stats.total_rows} rows ({progress:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")
                    self.stats.failed_rows += len(batches[batch_idx])
        
        self.stats.processed_rows = len(results)
        return results
    
    def _process_batch_with_retry(self, batch: List[Tuple[int, Dict]]) -> List[Dict]:
        """Process batch with retry logic - fail if cannot process"""
        import random
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                return self._process_batch(batch)
            except Exception as e:
                if attempt == max_retries - 1:
                    # Don't fallback - raise the error
                    error_msg = f"Batch processing failed after {max_retries} attempts: {e}"
                    logger.error(error_msg)
                    raise ProcessingSuspendedException(error_msg)
                
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
    
    def _process_batch(self, batch: List[Tuple[int, Dict]]) -> List[Dict]:
        """Process a batch of rows in single API call"""
        # Create batch prompt
        prompt = self._create_batch_prompt(batch)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data analyst. Process each row independently."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200 * len(batch),
                response_format={"type": "json_object"}
            )
            
            self.stats.api_calls += 1
            
            # Parse response
            batch_response = json.loads(response.choices[0].message.content)
            
            # Build results
            results = []
            for i, (row_num, row_data) in enumerate(batch):
                semantic = batch_response.get(f"row_{i+1}", {}).get("semantic", "")
                
                if not semantic:
                    # Don't use fallback - this is a real error
                    error_msg = f"API returned empty response for row {row_num}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                results.append({
                    "row_number": row_num,
                    "json_data": row_data,
                    "semantic_statement": semantic,
                    "combined_text": self._create_combined_text(row_data, semantic)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise  # Propagate error up - don't hide it
    
    
    # ========================================================================
    # FOLDER MODE PROCESSING
    # ========================================================================
    
    def _process_folder(self,
                       path: Path,
                       output_dir: Optional[str] = None,
                       pattern: str = "*.csv",
                       output_mode: str = "separate",
                       **kwargs) -> Dict:
        """
        Process all CSV files in a folder
        
        Args:
            path: Directory path
            output_dir: Output directory
            pattern: File pattern to match
            output_mode: "separate" or "consolidated"
        
        Returns:
            Processing summary
        """
        logger.info(f"Processing folder in FOLDER mode: {path}")
        start_time = time.time()
        
        # Find CSV files
        csv_files = list(path.glob(pattern))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {path}")
            return {"error": "No CSV files found", "folder": str(path)}
        
        logger.info(f"Found {len(csv_files)} CSV files")
        
        # Set output directory
        if not output_dir:
            output_dir = path
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process files
        if output_mode == "consolidated":
            results = self._process_folder_consolidated(csv_files, output_dir)
        else:
            results = self._process_folder_separate(csv_files, output_dir)
        
        # Generate summary
        summary = self._generate_folder_summary(results, time.time() - start_time)
        
        # Save summary
        summary_file = Path(output_dir) / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to {summary_file}")
        
        return summary
    
    def _process_folder_separate(self, csv_files: List[Path], output_dir: Path) -> List[FileProcessingResult]:
        """Process each CSV to separate output files"""
        results = []
        
        for i, csv_file in enumerate(csv_files, 1):
            logger.info(f"[{i}/{len(csv_files)}] Processing {csv_file.name}")
            
            try:
                # Auto-detect mode for each file
                file_mode = self._detect_mode(csv_file)
                
                # Create new processor instance for thread safety
                processor = UniversalCSVProcessor(
                    api_key=self.api_key,
                    mode=file_mode,
                    batch_size=self.batch_size,
                    max_workers=self.max_workers
                )
                
                # Process file
                output_file = output_dir / f"{csv_file.stem}_processed.json"
                records = processor.process(str(csv_file), str(output_file))
                
                # Create result
                result = FileProcessingResult(
                    file_name=csv_file.name,
                    file_path=str(csv_file),
                    total_rows=processor.stats.total_rows,
                    processed_rows=processor.stats.processed_rows,
                    processing_time=processor.stats.processing_time,
                    api_calls=processor.stats.api_calls,
                    mode_used=file_mode,
                    success=True,
                    output_file=str(output_file)
                )
                results.append(result)
                
                logger.info(f"  ✓ {csv_file.name}: {processor.stats.processed_rows} rows in {processor.stats.processing_time:.1f}s")
                
            except Exception as e:
                logger.error(f"  ✗ {csv_file.name}: {e}")
                results.append(FileProcessingResult(
                    file_name=csv_file.name,
                    file_path=str(csv_file),
                    total_rows=0,
                    processed_rows=0,
                    processing_time=0,
                    api_calls=0,
                    mode_used="error",
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    def _process_folder_consolidated(self, csv_files: List[Path], output_dir: Path) -> List[FileProcessingResult]:
        """Process all CSVs to single consolidated output"""
        all_records = []
        results = []
        
        for i, csv_file in enumerate(csv_files, 1):
            logger.info(f"[{i}/{len(csv_files)}] Processing {csv_file.name}")
            
            try:
                # Process file
                file_mode = self._detect_mode(csv_file)
                processor = UniversalCSVProcessor(
                    api_key=self.api_key,
                    mode=file_mode,
                    batch_size=self.batch_size
                )
                
                records = processor.process(str(csv_file))
                
                # Add source file info
                for record in records:
                    record['source_file'] = csv_file.name
                
                all_records.extend(records)
                
                results.append(FileProcessingResult(
                    file_name=csv_file.name,
                    file_path=str(csv_file),
                    total_rows=processor.stats.total_rows,
                    processed_rows=processor.stats.processed_rows,
                    processing_time=processor.stats.processing_time,
                    api_calls=processor.stats.api_calls,
                    mode_used=file_mode,
                    success=True
                ))
                
            except Exception as e:
                logger.error(f"Failed to process {csv_file.name}: {e}")
                results.append(FileProcessingResult(
                    file_name=csv_file.name,
                    file_path=str(csv_file),
                    total_rows=0,
                    processed_rows=0,
                    processing_time=0,
                    api_calls=0,
                    mode_used="error",
                    success=False,
                    error_message=str(e)
                ))
        
        # Save consolidated output
        output_file = output_dir / "consolidated_output.json"
        with open(output_file, 'w') as f:
            json.dump({
                "metadata": {
                    "total_files": len(csv_files),
                    "total_records": len(all_records)
                },
                "records": all_records
            }, f, indent=2)
        
        logger.info(f"Consolidated output saved to {output_file}")
        
        return results
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _row_to_json(self, row: pd.Series, mapping: Optional[Dict] = None) -> Dict:
        """Convert pandas row to JSON with optional mapping"""
        json_obj = {}
        
        for header, value in row.items():
            # Handle null values
            if pd.isna(value):
                value = None
            elif isinstance(value, float) and value.is_integer():
                value = int(value)
            
            # Apply mapping if provided
            if mapping and header in mapping:
                key = mapping[header]
            else:
                key = header
            
            json_obj[key] = value
        
        return json_obj
    
    def _generate_semantic_simple(self, json_obj: Dict) -> str:
        """Generate semantic statement for single row"""
        prompt = f"""Convert this data into a natural language sentence that includes all information:

{json.dumps(json_obj, indent=2)}

Rules:
- Include ALL values with their context
- Make it sound natural
- Keep it concise but complete

Generate the sentence:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            self.stats.api_calls += 1
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise  # Don't fallback - raise the error for proper handling
    
    def _create_batch_prompt(self, batch: List[Tuple[int, Dict]]) -> str:
        """Create prompt for batch processing"""
        prompt = """Process each row INDEPENDENTLY and create natural language descriptions.

Rows to process:
"""
        
        for i, (row_num, row_data) in enumerate(batch, 1):
            prompt += f"\n--- Row {i} ---\n"
            prompt += json.dumps(row_data, indent=2)
        
        prompt += """

Return JSON with format:
{
  "row_1": {"semantic": "Natural language description..."},
  "row_2": {"semantic": "Natural language description..."},
  ...
}

Each description must include ALL fields and values."""
        
        return prompt
    
    
    def _create_combined_text(self, json_obj: Dict, semantic: str) -> str:
        """Create combined text for vector embedding"""
        structured = " | ".join([f"{k}: {v}" for k, v in json_obj.items() if v is not None])
        return f"{semantic} | {structured}"
    
    def _save_results(self, results: List[Dict], output_path: str):
        """Save processing results to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "total_records": len(results),
                    "processor": "UniversalCSVProcessor",
                    "mode": self.stats.mode,
                    "processing_time": self.stats.processing_time,
                    "api_calls": self.stats.api_calls
                },
                "records": results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
    
    def _generate_folder_summary(self, results: List[FileProcessingResult], total_time: float) -> Dict:
        """Generate summary for folder processing"""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        total_rows = sum(r.total_rows for r in results)
        processed_rows = sum(r.processed_rows for r in results)
        total_api_calls = sum(r.api_calls for r in results)
        
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "statistics": {
                "total_files": len(results),
                "successful_files": len(successful),
                "failed_files": len(failed),
                "total_rows": total_rows,
                "processed_rows": processed_rows,
                "success_rate": (processed_rows / total_rows * 100) if total_rows > 0 else 0,
                "total_api_calls": total_api_calls,
                "total_time": total_time,
                "api_efficiency": processed_rows / total_api_calls if total_api_calls > 0 else 0
            },
            "files": [asdict(r) for r in results]
        }
    
    def _log_statistics(self):
        """Log processing statistics"""
        logger.info("="*50)
        logger.info(f"Processing Statistics ({self.stats.mode} mode):")
        logger.info(f"  Total rows: {self.stats.total_rows}")
        logger.info(f"  Processed: {self.stats.processed_rows}")
        logger.info(f"  Failed: {self.stats.failed_rows}")
        logger.info(f"  Success rate: {self.stats.success_rate:.1f}%")
        logger.info(f"  API calls: {self.stats.api_calls}")
        logger.info(f"  Time: {self.stats.processing_time:.1f} seconds")
        logger.info(f"  Rows/second: {self.stats.rows_per_second:.1f}")
        logger.info(f"  API efficiency: {self.stats.api_efficiency:.1f} rows/call")
        logger.info("="*50)
    
    def _save_checkpoint(self, checkpoint_file: str, last_row: int, results: List[Dict]):
        """Save processing state for resume"""
        checkpoint = {
            'last_successful_row': last_row,
            'timestamp': time.time(),
            'total_processed': len(results),
            'results': results
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"Checkpoint saved: {last_row} rows processed")
    
    def process_with_suspension(self, filepath: str, output_path: Optional[str] = None, 
                               checkpoint_file: str = 'processing_state.json') -> List[Dict]:
        """Process with automatic suspension on API failure"""
        
        # Load checkpoint if exists
        start_row = 0
        processed_results = []
        
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_row = checkpoint['last_successful_row']
                processed_results = checkpoint['results']
                print(f"📂 Resuming from row {start_row + 1}")
                print(f"   Already processed: {len(processed_results)} rows")
        
        # Check API before starting
        if not self.check_api_connection():
            print("❌ API is not available. Waiting for connection...")
            if not self.wait_for_api_connection():
                raise ConnectionError("API connection could not be established. Process suspended.")
        
        try:
            # Process with checkpointing
            df = pd.read_csv(filepath)
            total_rows = len(df)
            
            # Skip already processed rows
            if start_row > 0:
                df = df.iloc[start_row:]
            
            print(f"📊 Processing {len(df)} remaining rows (out of {total_rows} total)")
            
            for idx, row in df.iterrows():
                actual_row_num = idx + 1  # 1-based row numbering
                
                try:
                    # Process single row
                    json_obj = self._row_to_json(row)
                    semantic = self._generate_semantic_simple(json_obj)  # Will raise on failure
                    
                    result = {
                        "row_number": actual_row_num,
                        "json_data": json_obj,
                        "semantic_statement": semantic,
                        "combined_text": self._create_combined_text(json_obj, semantic)
                    }
                    
                    processed_results.append(result)
                    
                    # Progress update
                    if actual_row_num % 10 == 0:
                        print(f"   Progress: {actual_row_num}/{total_rows} rows processed")
                    
                    # Save checkpoint every 100 rows
                    if actual_row_num % 100 == 0:
                        self._save_checkpoint(checkpoint_file, actual_row_num, processed_results)
                        
                except Exception as e:
                    # Save state and suspend
                    print(f"\n❌ Processing failed at row {actual_row_num}: {e}")
                    self._save_checkpoint(checkpoint_file, actual_row_num - 1, processed_results)
                    
                    print("\n🔄 Process suspended. State saved.")
                    print(f"   Checkpoint file: {checkpoint_file}")
                    print(f"   Successfully processed: {len(processed_results)} rows")
                    print(f"   To resume, run the same command again.")
                    
                    raise ProcessingSuspendedException(
                        f"Processing suspended at row {actual_row_num}. "
                        f"Checkpoint saved to {checkpoint_file}"
                    )
            
            # Success - save results
            if output_path:
                self._save_results(processed_results, output_path)
            
            # Clean up checkpoint
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                print(f"✅ Checkpoint file removed (processing complete)")
                
            print(f"\n✅ Processing completed successfully! Total rows: {len(processed_results)}")
            return processed_results
            
        except Exception as e:
            if not isinstance(e, ProcessingSuspendedException):
                logger.error(f"Unexpected error: {e}")
                # Save checkpoint even for unexpected errors
                if processed_results:
                    self._save_checkpoint(checkpoint_file, len(processed_results), processed_results)
            raise
    
    def suggest_best_mode(self, csv_path: str) -> str:
        """
        Suggest the best processing mode for a given CSV file
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            Suggested mode with explanation
        """
        path = Path(csv_path)
        
        if path.is_dir():
            return "folder - Input is a directory"
        
        if not path.exists():
            return "Error - File does not exist"
        
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
            df_preview = pd.read_csv(path, nrows=5)
            total_rows = len(pd.read_csv(path, usecols=[0]))
            
            print(f"\nFile Analysis: {path.name}")
            print(f"  Size: {size_mb:.2f} MB")
            print(f"  Rows: {total_rows}")
            print(f"  Columns: {len(df_preview.columns)}")
            print()
            
            if total_rows <= 10:
                return "simple - Small file (<10 rows)"
            elif total_rows <= 50:
                return "optimized (batch_size=5) - Medium file"
            elif total_rows <= 200:
                return "optimized (batch_size=10) - Large file"
            else:
                return "optimized (batch_size=20) - Very large file"
                
        except Exception as e:
            return f"Error analyzing file: {e}"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def process_csv(file_path: str, 
                mode: str = "auto",
                output_path: Optional[str] = None,
                **kwargs) -> Any:
    """
    Convenience function for quick CSV processing
    
    Args:
        file_path: Path to CSV file or directory
        mode: Processing mode ("auto", "simple", "optimized", "folder")
        output_path: Optional output path
        **kwargs: Additional arguments
    
    Returns:
        Processing results
    """
    processor = UniversalCSVProcessor(mode=mode)
    return processor.process(file_path, output_path, **kwargs)


def analyze_csv(csv_path: str) -> str:
    """
    Analyze a CSV file and suggest best processing mode
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Mode suggestion with explanation
    """
    processor = UniversalCSVProcessor()
    return processor.suggest_best_mode(csv_path)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    # Default paths
    DEFAULT_INPUT_DIR = "INPUT_CSV"
    DEFAULT_OUTPUT_DIR = f"OUTPUT_PROCESSED_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    parser = argparse.ArgumentParser(
        description="Universal CSV Processor - Process CSV files with automatic optimization",
        epilog=f"Default: Processes all CSVs in '{DEFAULT_INPUT_DIR}/' folder if no input specified"
    )
    parser.add_argument(
        "input", 
        nargs='?',  # Makes input optional
        default=DEFAULT_INPUT_DIR,
        help=f"CSV file or directory to process (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument("-o", "--output", help="Output file/directory path (auto-generated if not specified)")
    parser.add_argument(
        "-m", "--mode", 
        choices=["auto", "simple", "optimized", "folder"],
        default="auto",
        help="Processing mode (default: auto)"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=10,
        help="Batch size for optimized mode (default: 10)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=3,
        help="Number of worker threads (default: 3)"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze file and suggest best mode without processing"
    )
    parser.add_argument(
        "--output-mode",
        choices=["separate", "consolidated"],
        default="separate",
        help="Output mode for folder processing"
    )
    
    args = parser.parse_args()
    
    try:
        # If analyze mode, just analyze and exit
        if args.analyze:
            suggestion = analyze_csv(args.input)
            print(f"Suggested mode: {suggestion}")
            exit(0)
        
        # Create processor
        processor = UniversalCSVProcessor(
            mode=args.mode,
            batch_size=args.batch_size,
            max_workers=args.workers
        )
        
        # Process based on input type
        path = Path(args.input)
        
        if path.is_dir():
            # Auto-generate output directory if not specified
            output_path = args.output or DEFAULT_OUTPUT_DIR
            
            print(f"\n{'='*60}")
            print(f"📁 Input folder: {args.input}")
            print(f"📂 Output folder: {output_path}")
            print(f"⚙️  Mode: {args.mode}")
            print(f"{'='*60}\n")
            
            # Folder processing
            results = processor.process(
                args.input,
                output_path=output_path,
                output_mode=args.output_mode
            )
            
            print("\n" + "="*60)
            print("PROCESSING COMPLETE")
            print("="*60)
            stats = results.get('statistics', {})
            print(f"✅ Files processed: {stats.get('successful_files', 0)}/{stats.get('total_files', 0)}")
            print(f"📊 Rows processed: {stats.get('processed_rows', 0)}")
            print(f"⚡ API calls: {stats.get('total_api_calls', 0)}")
            print(f"⏱️ Total time: {stats.get('total_time', 0):.1f}s")
            
        else:
            # Single file processing
            output_path = args.output or f"{path.stem}_processed.json"
            results = processor.process(args.input, output_path)
            
            print(f"\n✅ Successfully processed {len(results)} records")
            print(f"📁 Results saved to: {output_path}")
            
            # Show sample
            if results:
                print("\n📋 Sample record:")
                print(json.dumps(results[0], indent=2)[:500] + "...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
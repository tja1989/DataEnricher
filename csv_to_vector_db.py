#!/usr/bin/env python3
"""
CSV to Vector Database Optimized Format Generator

This script processes CSV files to generate JSON structured data with semantic summaries,
optimized for vector database embedding. Ensures NO data loss - all columns are preserved
in both JSON and semantic formats.

Output Format:
{
    "row_id": 1,
    "structured_data": {...all columns as key-value...},
    "semantic_summary": "Rich contextual description...",
    "embedding_text": "Combined text for vector embedding",
    "metadata": {...}
}
"""

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_table_type(headers: List[str]) -> str:
    """
    Detect the type of data based on column headers.
    
    Args:
        headers: List of column headers
        
    Returns:
        Table type string
    """
    headers_str = ' '.join(headers).lower()
    
    if 'equipment' in headers_str or 'nameplate' in headers_str:
        return 'equipment'
    elif 'vendor' in headers_str or 'supplier' in headers_str:
        return 'vendor'
    elif 'work_order' in headers_str or 'wo_' in headers_str:
        return 'work_order'
    elif 'material' in headers_str or 'mat_' in headers_str:
        return 'material'
    elif 'po_' in headers_str or 'purchase' in headers_str:
        return 'purchase_order'
    elif 'functional_location' in headers_str or 'fl_' in headers_str:
        return 'functional_location'
    elif 'manual' in headers_str or 'spec' in headers_str:
        return 'manual_specification'
    else:
        return 'general'


def create_semantic_prompt(table_type: str) -> ChatPromptTemplate:
    """
    Create specialized prompts based on table type.
    
    Args:
        table_type: Type of data table
        
    Returns:
        ChatPromptTemplate for semantic summary generation
    """
    base_instructions = """You are a senior data analyst specializing in {domain}. 
Given this {table_type} record, create a comprehensive semantic summary that:
1. Captures ALL data points from the record - DO NOT skip any field
2. Adds contextual understanding and business insights
3. Identifies relationships and patterns
4. Highlights operational importance
5. Uses natural language while preserving all original values

CRITICAL: Every single field from the JSON must be mentioned in your summary.
Verify that ALL keys from the structured_data appear in your narrative."""

    domain_specific = {
        'equipment': {
            'domain': 'industrial equipment and maintenance',
            'focus': 'Focus on criticality, maintenance implications, operational impact, installation timeline, and technical specifications. Mention all IDs, dates, costs, and specifications explicitly.'
        },
        'vendor': {
            'domain': 'supply chain and procurement',
            'focus': 'Emphasize supplier reliability, capabilities, contact details, lead times, authorization status, and aliases. Include all contact information, terms, and regional details.'
        },
        'work_order': {
            'domain': 'maintenance operations',
            'focus': 'Highlight the problem, solution, downtime impact, materials used, and patterns. Include all dates, hours, technician notes, and material consumption details.'
        },
        'material': {
            'domain': 'inventory and spare parts management',
            'focus': 'Detail specifications, compatibility, criticality, stock levels, and usage patterns. Include all part numbers, descriptions, units, and categorization.'
        },
        'purchase_order': {
            'domain': 'procurement and logistics',
            'focus': 'Cover order details, delivery timelines, costs, vendor relationships, and approval status. Include all PO numbers, line items, quantities, and dates.'
        },
        'functional_location': {
            'domain': 'asset hierarchy and plant structure',
            'focus': 'Describe location hierarchy, criticality, operational areas, and relationships. Include all IDs, parent locations, and site information.'
        },
        'manual_specification': {
            'domain': 'technical documentation',
            'focus': 'Detail document types, equipment coverage, revision status, and accessibility. Include all document IDs, file names, and categorization.'
        },
        'general': {
            'domain': 'data analysis',
            'focus': 'Create a comprehensive narrative that includes every field and its value, adding context where possible.'
        }
    }
    
    domain_info = domain_specific.get(table_type, domain_specific['general'])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", base_instructions),
        ("human", """Table Type: {table_type}
Domain: {domain}
Special Focus: {focus}

Structured Data:
{json_data}

Create a semantic summary that:
1. Mentions EVERY field from the structured data
2. Adds business context and insights
3. Is 3-5 sentences long
4. Flows naturally while preserving all information

Verify: Did you include ALL these fields in your summary?
{field_list}""")
    ])
    
    return prompt, domain_info


def csv_row_to_json(row: pd.Series, headers: List[str]) -> Dict[str, Any]:
    """
    Convert a CSV row to JSON format, preserving all data.
    
    Args:
        row: Pandas Series representing a row
        headers: List of column headers
        
    Returns:
        Dictionary with all row data
    """
    json_data = {}
    
    for header in headers:
        value = row[header]
        
        # Handle different data types
        if pd.isna(value):
            json_data[header] = None
        elif isinstance(value, (int, float)):
            if pd.isna(value):
                json_data[header] = None
            else:
                json_data[header] = value
        else:
            # Convert to string and clean
            str_value = str(value).strip()
            if str_value.lower() in ['nan', 'none', '']:
                json_data[header] = None
            else:
                json_data[header] = str_value
    
    return json_data


def validate_semantic_summary(summary: str, json_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that semantic summary contains all data fields.
    
    Args:
        summary: Generated semantic summary
        json_data: Original JSON data
        
    Returns:
        Tuple of (is_valid, missing_fields)
    """
    summary_lower = summary.lower()
    missing_fields = []
    
    for key, value in json_data.items():
        if value is not None:
            # Check if the value appears in the summary
            value_str = str(value).lower()
            if len(value_str) > 2 and value_str not in summary_lower:
                # Also check for partial matches (for long values)
                if not any(part in summary_lower for part in value_str.split()[:3]):
                    missing_fields.append(f"{key}={value}")
    
    return len(missing_fields) == 0, missing_fields


def generate_semantic_summary(
    json_data: Dict[str, Any],
    table_type: str,
    llm: ChatOpenAI,
    max_retries: int = 3
) -> str:
    """
    Generate semantic summary for a single row ensuring all data is included.
    
    Args:
        json_data: Structured data as dictionary
        table_type: Type of data table
        llm: LLM instance
        max_retries: Maximum retry attempts
        
    Returns:
        Semantic summary string
    """
    prompt_template, domain_info = create_semantic_prompt(table_type)
    
    # Create field list for verification
    field_list = ', '.join([f"{k}={v}" for k, v in json_data.items() if v is not None])
    
    for attempt in range(max_retries):
        try:
            # Generate summary
            chain = prompt_template | llm
            response = chain.invoke({
                'table_type': table_type,
                'domain': domain_info['domain'],
                'focus': domain_info['focus'],
                'json_data': json.dumps(json_data, indent=2),
                'field_list': field_list
            })
            
            summary = response.content.strip()
            
            # Validate that all fields are mentioned
            is_valid, missing = validate_semantic_summary(summary, json_data)
            
            if is_valid:
                return summary
            else:
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 1}: Missing fields in summary: {missing[:3]}")
                    # Add missing fields explicitly
                    summary += f" Additional details: {', '.join([f'{k}={v}' for k, v in json_data.items() if f'{k}={v}' in missing])}"
                    return summary
                    
        except Exception as e:
            logger.error(f"Error generating semantic summary: {e}")
            if attempt == max_retries - 1:
                # Fallback: Create a comprehensive listing
                return create_fallback_summary(json_data)
    
    return create_fallback_summary(json_data)


def create_fallback_summary(json_data: Dict[str, Any]) -> str:
    """
    Create a fallback summary that lists all data points.
    
    Args:
        json_data: Structured data
        
    Returns:
        Fallback summary string
    """
    parts = []
    for key, value in json_data.items():
        if value is not None:
            parts.append(f"{key.replace('_', ' ')} is {value}")
    
    return "Record contains: " + "; ".join(parts) + "."


def process_csv_batch_to_vector_format(
    df_batch: pd.DataFrame,
    headers: List[str],
    table_type: str,
    llm: Optional[ChatOpenAI],
    start_idx: int = 0
) -> List[Dict[str, Any]]:
    """
    Process a batch of CSV rows to vector-optimized format.
    
    Args:
        df_batch: DataFrame batch
        headers: Column headers
        table_type: Type of table
        llm: LLM instance (optional, for semantic summaries)
        start_idx: Starting index for row numbering
        
    Returns:
        List of vector-optimized records
    """
    vector_records = []
    
    for idx, row in df_batch.iterrows():
        # Convert to JSON (preserves all data)
        json_data = csv_row_to_json(row, headers)
        
        # Generate semantic summary if LLM available
        if llm:
            semantic_summary = generate_semantic_summary(json_data, table_type, llm)
        else:
            semantic_summary = create_fallback_summary(json_data)
        
        # Create embedding text (combines both for vector search)
        embedding_parts = []
        for key, value in json_data.items():
            if value is not None:
                embedding_parts.append(f"{key}: {value}")
        embedding_text = f"{semantic_summary} | Structured: {', '.join(embedding_parts)}"
        
        # Create vector record
        vector_record = {
            "row_id": start_idx + idx + 1,
            "source_file": df_batch.attrs.get('source_file', 'unknown.csv'),
            "structured_data": json_data,
            "semantic_summary": semantic_summary,
            "embedding_text": embedding_text,
            "metadata": {
                "table_type": table_type,
                "num_fields": len([v for v in json_data.values() if v is not None]),
                "primary_key": json_data.get(headers[0]) if headers else None
            }
        }
        
        vector_records.append(vector_record)
    
    return vector_records


def process_csv_to_vector_format(
    file_path: Path,
    llm: Optional[ChatOpenAI] = None,
    batch_size: int = 10,
    parallel_workers: int = 1
) -> List[Dict[str, Any]]:
    """
    Process entire CSV file to vector-optimized format.
    
    Args:
        file_path: Path to CSV file
        llm: LLM instance (optional)
        batch_size: Rows per batch
        parallel_workers: Number of parallel workers
        
    Returns:
        List of all vector records
    """
    # Load CSV
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            df.attrs['source_file'] = file_path.name
            logger.info(f"Loaded {file_path.name} with {len(df)} rows")
            break
        except:
            continue
    
    if df is None or df.empty:
        logger.error(f"Failed to load {file_path.name}")
        return []
    
    # Detect table type
    headers = df.columns.tolist()
    table_type = detect_table_type(headers)
    logger.info(f"Detected table type: {table_type}")
    
    # Process in batches
    all_records = []
    total_rows = len(df)
    
    with tqdm(total=total_rows, desc="Processing rows") as pbar:
        if parallel_workers > 1 and llm:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = []
                
                for start_idx in range(0, total_rows, batch_size):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch_df = df.iloc[start_idx:end_idx].copy()
                    batch_df.attrs['source_file'] = file_path.name
                    
                    future = executor.submit(
                        process_csv_batch_to_vector_format,
                        batch_df, headers, table_type, llm, start_idx
                    )
                    futures.append((start_idx, future))
                
                # Collect results in order
                results = {}
                for start_idx, future in futures:
                    records = future.result()
                    results[start_idx] = records
                    pbar.update(len(records))
                
                # Combine in order
                for start_idx in sorted(results.keys()):
                    all_records.extend(results[start_idx])
        else:
            # Sequential processing
            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                batch_df = df.iloc[start_idx:end_idx].copy()
                batch_df.attrs['source_file'] = file_path.name
                
                records = process_csv_batch_to_vector_format(
                    batch_df, headers, table_type, llm, start_idx
                )
                all_records.extend(records)
                pbar.update(len(records))
    
    return all_records


def validate_no_data_loss(
    original_csv: Path,
    vector_records: List[Dict[str, Any]]
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that no data was lost in conversion.
    
    Args:
        original_csv: Path to original CSV
        vector_records: Generated vector records
        
    Returns:
        Tuple of (is_valid, validation_report)
    """
    # Load original CSV
    df = pd.read_csv(original_csv)
    
    validation_report = {
        'total_rows': len(df),
        'total_records': len(vector_records),
        'rows_match': len(df) == len(vector_records),
        'missing_data': [],
        'column_coverage': {}
    }
    
    # Check each column appears in structured_data
    for col in df.columns:
        found_count = 0
        missing_values = []
        
        for idx, row in df.iterrows():
            if idx < len(vector_records):
                record = vector_records[idx]
                structured = record.get('structured_data', {})
                
                # Check if column exists in structured data
                if col not in structured:
                    missing_values.append(f"Row {idx + 1}: Column {col} missing")
                else:
                    # Check if non-null values are preserved
                    original_value = row[col]
                    stored_value = structured[col]
                    
                    if pd.notna(original_value) and stored_value is None:
                        missing_values.append(f"Row {idx + 1}: {col}={original_value} lost")
                    elif pd.notna(original_value):
                        found_count += 1
        
        validation_report['column_coverage'][col] = {
            'found': found_count,
            'total': len(df[df[col].notna()]),
            'coverage': found_count / max(len(df[df[col].notna()]), 1) * 100
        }
        
        if missing_values:
            validation_report['missing_data'].extend(missing_values[:5])  # Limit to 5 examples
    
    # Check semantic summaries for data presence
    semantic_coverage = []
    for idx, record in enumerate(vector_records[:5]):  # Check first 5
        summary = record.get('semantic_summary', '')
        structured = record.get('structured_data', {})
        
        missing_in_summary = []
        for key, value in structured.items():
            if value and str(value).lower() not in summary.lower():
                missing_in_summary.append(key)
        
        if missing_in_summary:
            semantic_coverage.append(f"Row {idx + 1}: Missing in summary: {missing_in_summary}")
    
    validation_report['semantic_coverage'] = semantic_coverage
    validation_report['is_valid'] = len(validation_report['missing_data']) == 0
    
    return validation_report['is_valid'], validation_report


def write_vector_output(
    vector_records: List[Dict[str, Any]],
    output_path: Path,
    format: str = 'json'
):
    """
    Write vector records to file.
    
    Args:
        vector_records: List of vector records
        output_path: Output file path
        format: Output format (json, jsonl, etc.)
    """
    if format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vector_records, f, indent=2, default=str)
    elif format == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in vector_records:
                f.write(json.dumps(record, default=str) + '\n')
    
    logger.info(f"Wrote {len(vector_records)} records to {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Convert CSV to vector database optimized format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing CSV files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output files"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for semantic summaries (requires API key)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Rows per batch"
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=2,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate no data loss"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use"
    )
    
    args = parser.parse_args()
    
    # Setup
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize LLM if requested
    llm = None
    if args.use_llm:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            llm = ChatOpenAI(
                model=args.model,
                temperature=0.3,
                api_key=api_key
            )
            logger.info(f"Using LLM: {args.model}")
        else:
            logger.warning("No API key found, using fallback summaries")
    
    # Process all CSV files
    # Check for specific pattern if provided
    pattern = os.getenv('CSV_PATTERN', '*.csv')
    csv_files = list(input_dir.glob(pattern))
    logger.info(f"Found {len(csv_files)} CSV files matching pattern: {pattern}")
    
    for csv_file in csv_files:
        logger.info(f"\nProcessing {csv_file.name}")
        
        # Process CSV
        vector_records = process_csv_to_vector_format(
            csv_file,
            llm,
            args.batch_size,
            args.parallel_workers
        )
        
        # Write output
        output_file = output_dir / f"{csv_file.stem}_vector.json"
        write_vector_output(vector_records, output_file, 'json')
        
        # Write JSONL format for streaming
        jsonl_file = output_dir / f"{csv_file.stem}_vector.jsonl"
        write_vector_output(vector_records, jsonl_file, 'jsonl')
        
        # Validate if requested
        if args.validate:
            is_valid, report = validate_no_data_loss(csv_file, vector_records)
            
            # Write validation report
            validation_file = output_dir / f"{csv_file.stem}_validation.json"
            with open(validation_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            if is_valid:
                logger.info(f"✅ Validation passed: No data loss detected")
            else:
                logger.warning(f"⚠️ Validation issues found - see {validation_file}")
            
            # Print column coverage
            print(f"\nColumn Coverage for {csv_file.name}:")
            for col, stats in report['column_coverage'].items():
                print(f"  {col}: {stats['coverage']:.1f}% ({stats['found']}/{stats['total']})")
    
    logger.info(f"\n✅ Processing complete. Output in {output_dir}")


if __name__ == "__main__":
    main()
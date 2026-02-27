"""
scripts/06_db_push.py
=====================
Step 6 - Push DATA_*.parquet files from OUTPUT_DIR to Supabase.

- Filters rows with ANY null values
- Shows summary and asks for confirmation
- Batch-inserts 1000 rows at a time

Run standalone: python scripts/06_db_push.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import cfg

#!/usr/bin/env python3
"""
Script to push parquet files from Output folder to Supabase
- Filters out rows with ANY null values
- Shows summary before pushing
- Requires confirmation to proceed
- Handles type conversions for database schema
"""

import os
import pandas as pd
from pathlib import Path
from supabase import create_client, Client
from typing import Dict, List, Tuple
import json

# Supabase credentials
# Credentials loaded from cfg (set via .env)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Mapping of file patterns to table names
TABLE_MAPPING = {
    "DATA_NDVI": "data_ndvi",
    "DATA_NDWI": "data_ndwi",
    "DATA_SMART_GROWTH": "data_sg",
    "DATA_WEED": "data_maleza"
}

class ParquetPusher:
    def __init__(self, output_folder: str = "Output"):
        self.output_folder = Path(output_folder)
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.file_stats = {}
        
    def get_table_name(self, filename: str) -> str:
        """Determine which table a file should go to based on filename"""
        for pattern, table_name in TABLE_MAPPING.items():
            if pattern in filename:
                return table_name
        return None
    
    def clean_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Remove rows with ANY null/NaN values
        Returns: (cleaned_df, num_rows_dropped)
        """
        original_count = len(df)
        
        # Drop rows with any null/NaN values
        df_clean = df.dropna(how='any')
        
        rows_dropped = original_count - len(df_clean)
        
        return df_clean, rows_dropped
    
    def convert_types_for_db(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """
        Convert DataFrame types to match database schema
        
        Schema reference:
        - data_ndvi: zafra (double), edad (double), company_id (bigint), ingenio_id (bigint)
        - data_ndwi: zafra (bigint), edad (bigint), company_id (bigint), ingenio_id (bigint)
        - data_sg: zafra (text), edad (bigint), company_id (bigint), ingenio_id (bigint)
        - data_maleza: zafra (text), edad (bigint), company_id (bigint), ingenio_id (bigint)
        """
        df = df.copy()
        
        # Define columns that should be integers (bigint) for each table
        integer_columns_map = {
            'data_ndvi': ['company_id', 'ingenio_id'],  # zafra and edad are double precision
            'data_ndwi': ['zafra', 'edad', 'company_id', 'ingenio_id'],  # all bigint
            'data_sg': ['edad', 'company_id', 'ingenio_id'],  # zafra is text
            'data_maleza': ['edad', 'company_id', 'ingenio_id']  # zafra is text
        }
        
        # Convert integer columns
        if table_name in integer_columns_map:
            for col in integer_columns_map[table_name]:
                if col in df.columns:
                    # Convert to Int64 (nullable integer type)
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        
        # Convert text columns (zafra in data_sg and data_maleza)
        text_columns_map = {
            'data_sg': ['zafra'],
            'data_maleza': ['zafra']
        }
        
        if table_name in text_columns_map:
            for col in text_columns_map[table_name]:
                if col in df.columns:
                    # Convert to string
                    df[col] = df[col].astype(str)
                    # Replace 'nan' string with None
                    df[col] = df[col].replace('nan', None)
        
        return df
    
    def analyze_files(self) -> Dict:
        """
        Analyze all parquet files and return statistics
        Returns dict with file stats before cleaning
        """
        parquet_files = list(self.output_folder.glob("*DATA_*.parquet"))
        
        if not parquet_files:
            print(f"❌ No parquet files found in {self.output_folder}")
            return {}
        
        print(f"\n📊 Analyzing {len(parquet_files)} parquet files...\n")
        
        stats = {}
        
        for file_path in sorted(parquet_files):
            table_name = self.get_table_name(file_path.name)
            
            if not table_name:
                print(f"⚠️  Skipping {file_path.name} - no table mapping found")
                continue
            
            # Read parquet
            df = pd.read_parquet(file_path)
            
            # Clean dataframe
            df_clean, rows_dropped = self.clean_dataframe(df)
            
            # Store stats
            if table_name not in stats:
                stats[table_name] = {
                    'files': [],
                    'total_rows_original': 0,
                    'total_rows_clean': 0,
                    'total_rows_dropped': 0
                }
            
            file_info = {
                'filename': file_path.name,
                'rows_original': len(df),
                'rows_clean': len(df_clean),
                'rows_dropped': rows_dropped,
                'df_clean': df_clean
            }
            
            stats[table_name]['files'].append(file_info)
            stats[table_name]['total_rows_original'] += len(df)
            stats[table_name]['total_rows_clean'] += len(df_clean)
            stats[table_name]['total_rows_dropped'] += rows_dropped
        
        self.file_stats = stats
        return stats
    
    def print_summary(self, stats: Dict):
        """Print summary of what will be inserted"""
        print("=" * 80)
        print("📋 SUMMARY OF DATA TO BE PUSHED")
        print("=" * 80)
        
        total_files = 0
        total_rows_original = 0
        total_rows_clean = 0
        total_rows_dropped = 0
        
        for table_name, table_stats in stats.items():
            print(f"\n🗂️  TABLE: {table_name}")
            print(f"   Files to process: {len(table_stats['files'])}")
            print(f"   Total rows (original): {table_stats['total_rows_original']:,}")
            print(f"   Total rows (clean): {table_stats['total_rows_clean']:,}")
            print(f"   Total rows (dropped): {table_stats['total_rows_dropped']:,}")
            
            if table_stats['total_rows_dropped'] > 0:
                drop_percentage = (table_stats['total_rows_dropped'] / table_stats['total_rows_original']) * 100
                print(f"   ⚠️  Dropping {drop_percentage:.2f}% of rows due to nulls")
            
            # Show per-file breakdown if there are dropped rows
            if table_stats['total_rows_dropped'] > 0:
                print(f"\n   Per-file breakdown:")
                for file_info in table_stats['files']:
                    if file_info['rows_dropped'] > 0:
                        print(f"      • {file_info['filename']}: "
                              f"{file_info['rows_dropped']} rows dropped "
                              f"({file_info['rows_clean']} will be inserted)")
            
            total_files += len(table_stats['files'])
            total_rows_original += table_stats['total_rows_original']
            total_rows_clean += table_stats['total_rows_clean']
            total_rows_dropped += table_stats['total_rows_dropped']
        
        print("\n" + "=" * 80)
        print(f"📊 OVERALL SUMMARY")
        print(f"   Total files: {total_files}")
        print(f"   Total rows (original): {total_rows_original:,}")
        print(f"   Total rows (clean): {total_rows_clean:,}")
        print(f"   Total rows (dropped): {total_rows_dropped:,}")
        
        if total_rows_dropped > 0:
            drop_percentage = (total_rows_dropped / total_rows_original) * 100
            print(f"   ⚠️  {drop_percentage:.2f}% of all rows will be skipped due to nulls")
        
        print("=" * 80)
    
    def push_to_supabase(self):
        """Push cleaned data to Supabase"""
        print("\n🚀 Starting push to Supabase...\n")
        
        for table_name, table_stats in self.file_stats.items():
            print(f"\n📤 Processing table: {table_name}")
            
            for file_info in table_stats['files']:
                df_clean = file_info['df_clean']
                
                if len(df_clean) == 0:
                    print(f"   ⏭️  Skipping {file_info['filename']} - no rows to insert after filtering")
                    continue
                
                print(f"   → Inserting {len(df_clean):,} rows from {file_info['filename']}...")
                
                try:
                    # Convert types to match database schema
                    df_clean = self.convert_types_for_db(df_clean, table_name)
                    
                    # Convert DataFrame to list of dicts
                    records = df_clean.to_dict('records')
                    
                    # Clean records for JSON serialization
                    cleaned_records = []
                    for record in records:
                        cleaned_record = {}
                        for k, v in record.items():
                            # Handle pandas Timestamp objects
                            if isinstance(v, pd.Timestamp):
                                cleaned_record[k] = v.isoformat()
                            # Handle pandas NA/NaT
                            elif pd.isna(v):
                                cleaned_record[k] = None
                            # Handle numpy/pandas integer types (convert to Python int)
                            elif hasattr(v, 'item'):
                                try:
                                    cleaned_record[k] = v.item()
                                except (ValueError, OverflowError):
                                    cleaned_record[k] = None
                            else:
                                cleaned_record[k] = v
                        cleaned_records.append(cleaned_record)
                    
                    # Insert in batches of 1000 to avoid timeouts
                    batch_size = 1000
                    total_inserted = 0
                    
                    for i in range(0, len(cleaned_records), batch_size):
                        batch = cleaned_records[i:i+batch_size]
                        response = self.supabase.table(table_name).insert(batch).execute()
                        total_inserted += len(batch)
                        
                        # Show progress for large files
                        if len(cleaned_records) > batch_size:
                            print(f"      → Batch {i//batch_size + 1}: {total_inserted:,}/{len(cleaned_records):,} rows")
                    
                    print(f"   ✅ Successfully inserted {len(cleaned_records):,} rows")
                    
                except Exception as e:
                    print(f"   ❌ Error inserting {file_info['filename']}: {str(e)}")
                    
                    # Debug: print first record and its types
                    if cleaned_records:
                        print(f"   🔍 Debug - First record sample:")
                        sample = cleaned_records[0]
                        for key, value in list(sample.items())[:5]:  # Show first 5 columns
                            print(f"      {key}: {value} (type: {type(value).__name__})")
                    
                    continue
        
        print("\n✅ Push to Supabase completed!")
    
    def run(self):
        """Main execution flow"""
        # Analyze files
        stats = self.analyze_files()
        
        if not stats:
            return
        
        # Print summary
        self.print_summary(stats)
        
        # Ask for confirmation
        print("\n❓ Do you want to proceed with pushing this data to Supabase?")
        confirmation = input("   Type 'Y' or 'yes' to continue: ").strip().lower()
        
        if confirmation in ['y', 'yes']:
            self.push_to_supabase()
        else:
            print("\n❌ Push cancelled by user")

def main():
    # Check if Output folder exists
    output_folder = cfg.OUTPUT_DIR
    if not output_folder.exists():
        print(f"❌ Error: {output_folder} folder not found!")
        return
    
    print("🔧 Parquet to Supabase Pusher")
    print(f"📁 Working directory: {output_folder.absolute()}")
    
    pusher = ParquetPusher(output_folder=OUTPUT_DIR_ENV)
    pusher.run()


def run() -> None:
    """Push parquet files from cfg.OUTPUT_DIR to Supabase."""
    output_folder = cfg.OUTPUT_DIR
    if not output_folder.exists():
        print(f"❌ Error: {output_folder} not found!")
        return
    print("🔧 Parquet to Supabase Pusher")
    print(f"📁 Working directory: {output_folder.absolute()}")
    pusher = ParquetPusher(output_folder=str(output_folder))
    pusher.run()


if __name__ == "__main__":
    print(cfg.summary())
    run()

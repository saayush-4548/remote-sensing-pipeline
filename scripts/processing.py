"""
scripts/05_processing.py
========================
Step 5 - Excel sync + NDVI / NDWI / Smart Growth / Weed processing.

Run standalone: python scripts/05_processing.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import cfg

# Map cfg → variable names used by original notebook
INGENIO_ENV    = cfg.ingenio_display
FECHAS_ENV     = cfg.FECHAS
INPUT_DIR_ENV  = str(cfg.INPUT_DIR)
OUTPUT_DIR_ENV = str(cfg.OUTPUT_DIR)

# -*- coding: utf-8 -*-
"""
excel_to_supabase_sync.py - INCREMENTAL UPDATE WITH ZAFRA LOGIC
Implements progressive sync with fecha_fin/fecha_inicio logic and ciclo validation
"""

import pandas as pd
import os
import logging
from datetime import datetime
from supabase import create_client, Client
import warnings
warnings.filterwarnings('ignore')

# =======================================================
# LOGGING CONFIGURATION
# =======================================================

def setup_logging(log_file: str = None):
    """Setup file logging (detailed) and console logging (minimal)"""
    
    logger = logging.getLogger('excel_sync')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    # File handler - DETAILED logging
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


# Initialize logger with timestamp
log_filename = f'sync_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
logger = setup_logging(log_file=log_filename)


# =======================================================
# CONFIGURATION
# =======================================================

ALL_COLUMNS = [
    'id_parcela', 'zafra', 'temporada_activa', 'company', 'ingenio',
    'area_calculada', 'area_cosechada', 'area_estimada', 'area_potencial',
    'area_real', 'azucar_estimada', 'azucar_potencial', 'azucar_real',
    'tch_cosechado', 'tch_estimado', 'tch_potencial', 'tch_real',
    'tah_estimado', 'tah_real', 'ton_cosechadas', 'ton_estimada',
    'ton_potencial', 'ton_real', 'ciclo', 'variedad', 'textura_suelo',
    'tipo_riego', 'tipo_corte', 'tipo_cosecha_estimado',
    'tipo_cosecha_real', 'distancia_surco', 'fecha_inicio', 'INICIO',
    'fecha_fin_estimada', 'fecha_fin', 'division_01', 'division_02',
    'division_03', 'division_04', 'division_05', 'division_06',
    'division_07', 'division_08', 'division_09', 'division_10',
    'division_11', 'division_12', 'division_13', 'division_14',
    'division_15', 'division_16', 'division_17', 'division_18',
    'division_19', 'division_20', 'division_21', 'division_22',
    'division_23', 'division_24', 'division_25', 'division_26',
    'division_27', 
    # ✅ Already added by you:
    'geometry_polygon', 'ingenio_id', 'company_id',
    # ✅ Add these for completeness:
    'geometry_centroid',         # Inherit centroid too
    'division_28', 'division_29', 'division_30', 'division_31',  # Extra divisions
    'division_32', 'division_33', 'division_34', 'division_35',
    'kg_t_core_zp',             # Metric field
    'zafra_st'                  # Status field
]

TABLE_NAME = 'parcelas_ingenios_reprocess'


def get_supabase_client():
    """Initialize Supabase client"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables required")
    logger.info(f"Supabase client initialized: {url[:30]}...")
    return create_client(url, key)


def load_excel_data(excel_path: str) -> dict:
    """Load Excel data from both 2026 and 2027 sheets"""
    
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel not found: {excel_path}")
    
    data = {}
    for sheet in ['2026', '2027']:
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet)
            df.columns = df.columns.str.strip().str.lower()
            df['zafra'] = int(sheet)
            logger.info(f"Sheet '{sheet}': {len(df)} records, columns: {list(df.columns)}")
            data[sheet] = df
        except Exception as e:
            logger.warning(f"Sheet '{sheet}' error: {e}")
            data[sheet] = pd.DataFrame()
    
    return data


def get_past_zafra_record(supabase: Client, id_parcela: str, ingenio: str, current_zafra: int) -> dict:
    """Get most recent record from past zafras for a given parcel"""
    
    try:
        response = supabase.table(TABLE_NAME) \
            .select('*') \
            .eq('id_parcela', id_parcela) \
            .eq('ingenio', ingenio) \
            .lt('zafra', current_zafra) \
            .order('zafra', desc=True) \
            .limit(1) \
            .execute()
        
        if response.data and len(response.data) > 0:
            record = response.data[0]
            logger.debug(f"Past record found: {id_parcela}, zafra={record.get('zafra')}, ciclo={record.get('ciclo')}")
            return record
        return None
        
    except Exception as e:
        logger.error(f"Error fetching past record for {id_parcela}: {e}")
        return None


def validate_ciclo(past_ciclo, new_ciclo, id_parcela: str, past_zafra: int = None, new_zafra: int = None) -> tuple:
    """
    Validate ciclo logic: must be +1 from last zafra or 0
    
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    # Log validation details to file
    logger.info(f"CICLO VALIDATION: {id_parcela}")
    logger.info(f"  Zafra: {past_zafra} → {new_zafra}")
    logger.info(f"  Ciclo: {past_ciclo} → {new_ciclo}")
    
    if pd.isna(new_ciclo):
        logger.info(f"  Result: SKIP (no new ciclo)")
        return True, "No ciclo provided"
    
    try:
        new_ciclo_int = int(float(new_ciclo))
    except (ValueError, TypeError):
        logger.warning(f"  Result: INVALID (cannot convert: {new_ciclo})")
        return False, f"Invalid ciclo value: {new_ciclo}"
    
    if past_ciclo is None or pd.isna(past_ciclo):
        logger.info(f"  Result: ACCEPT (new parcel, ciclo={new_ciclo_int})")
        return True, f"New parcel, ciclo={new_ciclo_int}"
    
    try:
        past_ciclo_int = int(float(past_ciclo))
    except (ValueError, TypeError):
        logger.warning(f"  Result: ACCEPT (invalid past ciclo: {past_ciclo})")
        return True, f"Could not validate (past_ciclo invalid)"
    
    if new_ciclo_int == past_ciclo_int + 1:
        logger.info(f"  Result: VALID (increment {past_ciclo_int} → {new_ciclo_int})")
        return True, f"Ciclo incremented: {past_ciclo_int} → {new_ciclo_int}"
    elif new_ciclo_int == 0:
        logger.info(f"  Result: VALID (reset {past_ciclo_int} → 0)")
        return True, f"Ciclo reset: {past_ciclo_int} → 0"
    else:
        logger.warning(f"  Result: INVALID ({past_ciclo_int} → {new_ciclo_int}, expected {past_ciclo_int + 1} or 0)")
        return False, f"Invalid ciclo transition: {past_ciclo_int} → {new_ciclo_int}"


def prepare_record_data(row: pd.Series, excel_cols: list) -> dict:
    """Convert pandas row to dict with proper type handling"""
    data = {}
    
    # Columns that must be numeric (int or float) — skip if not convertible
    NUMERIC_COLS = {
        'area_calculada', 'area_cosechada', 'area_estimada', 'area_potencial',
        'area_real', 'azucar_estimada', 'azucar_potencial', 'azucar_real',
        'tch_cosechado', 'tch_estimado', 'tch_potencial', 'tch_real',
        'tah_estimado', 'tah_real', 'ton_cosechadas', 'ton_estimada',
        'ton_potencial', 'ton_real', 'ciclo', 'distancia_surco', 'kg_t_core_zp',
    }
    # Division columns are also numeric
    NUMERIC_COLS.update({f'division_{str(i).zfill(2)}' for i in range(1, 36)})
    
    DATE_COLS = {'fecha_inicio', 'fecha_fin', 'fecha_fin_estimada', 'INICIO'}
    
    STRING_COLS = {
        'id_parcela', 'zafra', 'company', 'ingenio', 'variedad',
        'textura_suelo', 'tipo_riego', 'tipo_corte', 'tipo_cosecha_estimado',
        'tipo_cosecha_real', 'geometry_polygon', 'geometry_centroid',
        'ingenio_id', 'company_id', 'zafra_st',
    }
    
    # Garbage strings that should be treated as null
    GARBAGE = {'', '-', '—', 'N/A', 'n/a', 'NA', 'None', 'nan', '#N/A', '#VALUE!', '#REF!'}
    
    for col in excel_cols:
        if col not in row.index:
            continue
        value = row[col]
        
        # --- Null check ---
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (ValueError, TypeError):
            pass
        
        # --- String garbage filter ---
        if isinstance(value, str):
            value = value.strip()
            if value in GARBAGE:
                logger.debug(f"  Skipping garbage in {col}: {repr(value)}")
                continue
        
        # --- Route by column type ---
        if col == 'zafra':
            try:
                data[col] = str(int(float(value)))
            except (ValueError, TypeError):
                logger.warning(f"  Skipping zafra: cannot convert {repr(value)}")
            continue
        
        if col == 'temporada_activa':
            if isinstance(value, bool):
                data[col] = value
            elif isinstance(value, str):
                data[col] = value.lower() in ('true', '1', 'yes')
            else:
                data[col] = bool(value)
            continue
        
        if col in NUMERIC_COLS:
            try:
                num = float(value)
                data[col] = int(num) if num == int(num) else num
            except (ValueError, TypeError):
                logger.warning(f"  Skipping non-numeric in {col}: {repr(value)}")
            continue
        
        if col in DATE_COLS:
            if isinstance(value, pd.Timestamp):
                data[col] = value.strftime('%Y-%m-%d')
            elif isinstance(value, str):
                try:
                    pd.to_datetime(value)
                    data[col] = value
                except (ValueError, TypeError):
                    logger.warning(f"  Skipping invalid date in {col}: {repr(value)}")
            continue
        
        if col in STRING_COLS:
            data[col] = str(value)
            continue
        
        # --- Fallback ---
        if isinstance(value, pd.Timestamp):
            data[col] = value.strftime('%Y-%m-%d')
        elif isinstance(value, (int, float)):
            data[col] = int(value) if float(value).is_integer() else float(value)
        else:
            data[col] = str(value)
    
    return data


def log_record_update(id_parcela: str, update_data: dict, action: str, zafra: int):
    """Log record update details to file only"""
    logger.info(f"{'─'*60}")
    logger.info(f"{action}: {id_parcela} (zafra {zafra})")
    logger.info(f"  Columns ({len(update_data)}): {sorted(update_data.keys())}")
    for col, val in update_data.items():
        logger.debug(f"    {col}: {val}")


def sync_excel_to_supabase(
    excel_path: str,
    ingenio: str,
    processing_date: str,
    dry_run: bool = False
) -> dict:
    """
    Main synchronization logic with incremental updates
    
    Process:
    1. Load 2026 & 2027 sheets
    2. Step 1: Close finished parcels from 2026 (fecha_fin <= processing_date)
    3. Step 2: Process new/continuing parcels from 2027 (fecha_inicio <= processing_date)
    """
    
    # Console output - Header
    print(f"\n{'='*70}")
    print(f"  SYNC: {ingenio} | Date: {processing_date} | Mode: {'DRY RUN' if dry_run else 'PRODUCTION'}")
    print(f"{'='*70}")
    
    # File logging - Detailed header
    logger.info("=" * 80)
    logger.info(f"SYNC START: {excel_path}")
    logger.info(f"  Ingenio: {ingenio}")
    logger.info(f"  Processing Date: {processing_date}")
    logger.info(f"  Mode: {'DRY RUN' if dry_run else 'PRODUCTION'}")
    logger.info(f"  Table: {TABLE_NAME}")
    logger.info("=" * 80)
    
    proc_date = pd.to_datetime(processing_date)
    
    # Load Excel
    excel_data = load_excel_data(excel_path)
    df_2026 = excel_data.get('2026', pd.DataFrame())
    df_2027 = excel_data.get('2027', pd.DataFrame())
    
    # Filter by ingenio
    if 'ingenio' in df_2026.columns:
        ingenio_lower = ingenio.lower().replace('_', ' ')
        df_2026 = df_2026[df_2026['ingenio'].str.lower().str.replace('_', ' ') == ingenio_lower]
    if 'ingenio' in df_2027.columns:
        ingenio_lower = ingenio.lower().replace('_', ' ')
        df_2027 = df_2027[df_2027['ingenio'].str.lower().str.replace('_', ' ') == ingenio_lower]
    
    print(f"  📊 Loaded: 2026={len(df_2026)} | 2027={len(df_2027)} records")
    logger.info(f"After ingenio filter: 2026={len(df_2026)}, 2027={len(df_2027)}")
    
    stats = {
        'closed_2026': 0,
        'inserted_2027': 0,
        'updated_2027': 0,
        'ciclo_errors': 0,
        'errors': 0,
        'skipped': 0,
        'columns_updated_2026': set(),
        'columns_updated_2027': set()
    }
    
    ciclo_errors_log = []
    update_log_2026 = []
    update_log_2027 = []
    
    # Connect to Supabase
    supabase = None if dry_run else get_supabase_client()
    
    # ================================================================
    # STEP 1: Process 2026 - Close finished parcels
    # ================================================================
    print(f"\n  📌 Step 1: Closing 2026 parcels (fecha_fin <= {processing_date})...")
    logger.info("=" * 80)
    logger.info("STEP 1: Close 2026 parcels where fecha_fin <= processing_date")
    logger.info("=" * 80)
    
    if not df_2026.empty and 'fecha_fin' in df_2026.columns:
        df_2026['fecha_fin'] = pd.to_datetime(df_2026['fecha_fin'], errors='coerce')
        df_to_close = df_2026[
            (df_2026['fecha_fin'].notna()) & 
            (df_2026['fecha_fin'] <= proc_date)
        ].copy()
        
        logger.info(f"Parcels to close: {len(df_to_close)}")
        
        if not dry_run:
            for idx, row in df_to_close.iterrows():
                id_parcela = str(row['id_parcela'])
                
                try:
                    excel_cols = [col for col in df_2026.columns if col in ALL_COLUMNS]
                    update_data = prepare_record_data(row, excel_cols)
                    update_data['temporada_activa'] = False
                    
                    stats['columns_updated_2026'].update(update_data.keys())
                    log_record_update(id_parcela, update_data, "CLOSE", 2026)
                    
                    update_log_2026.append({
                        'id_parcela': id_parcela,
                        'fecha_fin': str(row['fecha_fin']),
                        'columns_updated': list(update_data.keys()),
                        'column_count': len(update_data)
                    })
                    
                    response = supabase.table(TABLE_NAME) \
                        .update(update_data) \
                        .eq('id_parcela', id_parcela) \
                        .eq('ingenio', ingenio) \
                        .eq('zafra', '2026') \
                        .execute()
                    
                    if response.data and len(response.data) > 0:
                        stats['closed_2026'] += 1
                    else:
                        stats['skipped'] += 1
                        
                except Exception as e:
                    stats['errors'] += 1
                    logger.error(f"Error closing {id_parcela}: {e}")
                    print(f"  ❌ Error closing {id_parcela}: {str(e)[:50]}")
        else:
            stats['closed_2026'] = len(df_to_close)  # Simulated count
        
        print(f"     ✓ Closed: {stats['closed_2026']} parcels ({len(stats['columns_updated_2026'])} columns updated)")
    else:
        print(f"     ⚠ No fecha_fin column or empty sheet")
        logger.warning("No fecha_fin column or empty 2026 sheet")
    
    # ================================================================
    # STEP 2: Process 2027 - New/continuing parcels
    # ================================================================
    print(f"\n  📌 Step 2: Processing 2027 parcels (fecha_inicio <= {processing_date})...")
    logger.info("=" * 80)
    logger.info("STEP 2: Process 2027 parcels where fecha_inicio <= processing_date")
    logger.info("  - Validate ciclo for existing parcels")
    logger.info("  - Compare with most recent past zafra in DB")
    logger.info("=" * 80)
    
    if not df_2027.empty and 'fecha_inicio' in df_2027.columns:
        df_2027['fecha_inicio'] = pd.to_datetime(df_2027['fecha_inicio'], errors='coerce')
        df_to_process = df_2027[
            (df_2027['fecha_inicio'].notna()) & 
            (df_2027['fecha_inicio'] <= proc_date)
        ].copy()
        
        logger.info(f"Parcels to process: {len(df_to_process)}")
        
        if not dry_run:
            for idx, row in df_to_process.iterrows():
                id_parcela = str(row['id_parcela'])
                
                try:
                    past_record = get_past_zafra_record(supabase, id_parcela, ingenio, 2027)
                    
                    if past_record:
                        # EXISTING PARCEL - Validate ciclo
                        past_zafra = past_record.get('zafra')
                        past_ciclo = past_record.get('ciclo')
                        new_ciclo = row['ciclo'] if 'ciclo' in row.index and pd.notna(row['ciclo']) else None
                        
                        is_valid, message = validate_ciclo(
                            past_ciclo, new_ciclo, id_parcela,
                            past_zafra=past_zafra, new_zafra=2027
                        )
                        
                        if not is_valid:
                            stats['ciclo_errors'] += 1
                            ciclo_errors_log.append({
                                'id_parcela': id_parcela,
                                'past_zafra': past_zafra,
                                'new_zafra': 2027,
                                'past_ciclo': past_ciclo,
                                'new_ciclo': new_ciclo,
                                'message': message
                            })
                            print(f"  ❌ Ciclo error: {id_parcela} ({past_ciclo}→{new_ciclo})")
                            continue
                        
                        # Build record: inherit from past + overwrite with Excel
                        new_record = {k: v for k, v in past_record.items() 
                                     if k in ALL_COLUMNS and k not in ['id', 'created_at', 'updated_at']}
                        
                        excel_cols = [col for col in df_2027.columns if col in ALL_COLUMNS]
                        excel_data = prepare_record_data(row, excel_cols)
                        new_record.update(excel_data)
                        
                        new_record['zafra'] = '2027'
                        new_record['temporada_activa'] = True
                        new_record['ingenio'] = ingenio
                        new_record['id_parcela'] = id_parcela
                        
                        stats['columns_updated_2027'].update(new_record.keys())
                        
                        # Check if 2027 record exists
                        existing = supabase.table(TABLE_NAME) \
                            .select('id') \
                            .eq('id_parcela', id_parcela) \
                            .eq('ingenio', ingenio) \
                            .eq('zafra', '2027') \
                            .execute()
                        
                        if existing.data and len(existing.data) > 0:
                            log_record_update(id_parcela, new_record, "UPDATE", 2027)
                            supabase.table(TABLE_NAME) \
                                .update(new_record) \
                                .eq('id_parcela', id_parcela) \
                                .eq('ingenio', ingenio) \
                                .eq('zafra', '2027') \
                                .execute()
                            stats['updated_2027'] += 1
                            
                            update_log_2027.append({
                                'id_parcela': id_parcela,
                                'action': 'UPDATE',
                                'inherited_from_zafra': past_zafra,
                                'ciclo_transition': f"{past_ciclo}→{new_ciclo}",
                                'column_count': len(new_record)
                            })
                        else:
                            log_record_update(id_parcela, new_record, "INSERT", 2027)
                            supabase.table(TABLE_NAME).insert(new_record).execute()
                            stats['inserted_2027'] += 1
                            
                            update_log_2027.append({
                                'id_parcela': id_parcela,
                                'action': 'INSERT',
                                'inherited_from_zafra': past_zafra,
                                'ciclo_transition': f"{past_ciclo}→{new_ciclo}",
                                'column_count': len(new_record)
                            })
                    
                    else:
                        # NEW PARCEL - No past record
                        excel_cols = [col for col in df_2027.columns if col in ALL_COLUMNS]
                        new_record = prepare_record_data(row, excel_cols)
                        new_record['id_parcela'] = id_parcela
                        new_record['ingenio'] = ingenio
                        new_record['zafra'] = '2027'
                        new_record['temporada_activa'] = True
                        
                        stats['columns_updated_2027'].update(new_record.keys())
                        log_record_update(id_parcela, new_record, "INSERT (NEW)", 2027)
                        
                        supabase.table(TABLE_NAME).insert(new_record).execute()
                        stats['inserted_2027'] += 1
                        
                        update_log_2027.append({
                            'id_parcela': id_parcela,
                            'action': 'INSERT_NEW',
                            'inherited_from_zafra': None,
                            'ciclo_transition': f"N/A→{new_record.get('ciclo', 'N/A')}",
                            'column_count': len(new_record)
                        })
                        
                except Exception as e:
                    stats['errors'] += 1
                    logger.error(f"Error processing {id_parcela}: {e}")
                    print(f"  ❌ Error: {id_parcela}: {str(e)[:50]}")
        else:
            stats['inserted_2027'] = len(df_to_process)  # Simulated
        
        print(f"     ✓ Inserted: {stats['inserted_2027']} | Updated: {stats['updated_2027']} ({len(stats['columns_updated_2027'])} columns)")
    else:
        print(f"     ⚠ No fecha_inicio column or empty sheet")
        logger.warning("No fecha_inicio column or empty 2027 sheet")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'─'*70}")
    print(f"  📊 SUMMARY")
    print(f"{'─'*70}")
    print(f"     2026 Closed:    {stats['closed_2026']}")
    print(f"     2027 Inserted:  {stats['inserted_2027']}")
    print(f"     2027 Updated:   {stats['updated_2027']}")
    
    if stats['ciclo_errors'] > 0:
        print(f"     ⚠ Ciclo Errors: {stats['ciclo_errors']}")
    if stats['errors'] > 0:
        print(f"     ❌ Errors:       {stats['errors']}")
    
    print(f"{'─'*70}")
    print(f"  📄 Detailed log: {log_filename}")
    print(f"{'='*70}\n")
    
    # Log summary to file
    logger.info("=" * 80)
    logger.info("SYNC COMPLETE")
    logger.info(f"  Closed (2026): {stats['closed_2026']}")
    logger.info(f"  Inserted (2027): {stats['inserted_2027']}")
    logger.info(f"  Updated (2027): {stats['updated_2027']}")
    logger.info(f"  Ciclo Errors: {stats['ciclo_errors']}")
    logger.info(f"  Other Errors: {stats['errors']}")
    logger.info(f"  Columns updated in 2026: {sorted(stats['columns_updated_2026'])}")
    logger.info(f"  Columns updated in 2027: {sorted(stats['columns_updated_2027'])}")
    logger.info("=" * 80)
    
    # Save CSV logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if update_log_2026:
        log_path = f'update_log_2026_{timestamp}.csv'
        pd.DataFrame(update_log_2026).to_csv(log_path, index=False)
        logger.info(f"2026 update log saved: {log_path}")
    
    if update_log_2027:
        log_path = f'update_log_2027_{timestamp}.csv'
        pd.DataFrame(update_log_2027).to_csv(log_path, index=False)
        logger.info(f"2027 update log saved: {log_path}")
    
    if ciclo_errors_log:
        log_path = f'ciclo_errors_{timestamp}.csv'
        pd.DataFrame(ciclo_errors_log).to_csv(log_path, index=False)
        logger.info(f"Ciclo errors saved: {log_path}")
        print(f"  ⚠ Ciclo errors saved: {log_path}")
    
    # Convert sets for JSON
    stats['columns_updated_2026'] = list(stats['columns_updated_2026'])
    stats['columns_updated_2027'] = list(stats['columns_updated_2027'])
    
    return stats


def procesar_con_sync(
    excel_path: str,
    ingenio: str,
    fechas: list,
    productos: list,
    input_dir: str,
    output_dir: str,
    bd_insert: bool = False,
    dry_run_sync: bool = False
):
    """
    Process multiple dates with incremental sync
    """
   # from main_processing import procesar_todos_productos
    
    if isinstance(fechas, str):
        fechas = [fechas]
    
    print(f"\n{'#'*70}")
    print(f"  MULTI-DATE PROCESSING: {ingenio}")
    print(f"  Dates: {fechas}")
    print(f"{'#'*70}")
    
    logger.info("=" * 80)
    logger.info(f"MULTI-DATE PROCESSING: {ingenio}")
    logger.info(f"  Dates: {fechas}")
    logger.info(f"  Products: {productos}")
    logger.info(f"  Excel: {excel_path}")
    logger.info("=" * 80)
    
    all_results = {}
    all_sync_stats = {}
    
    for idx, fecha in enumerate(fechas, 1):
        print(f"\n{'─'*70}")
        print(f"  [{idx}/{len(fechas)}] Processing: {fecha}")
        print(f"{'─'*70}")
        
        # STEP 1: Sync
        try:
            sync_stats = sync_excel_to_supabase(
                excel_path=excel_path,
                ingenio=ingenio,
                processing_date=fecha,
                dry_run=dry_run_sync
            )
            all_sync_stats[fecha] = sync_stats
        except Exception as e:
            print(f"  ❌ Sync error: {e}")
            logger.error(f"Sync error for {fecha}: {e}")
            all_sync_stats[fecha] = {'error': str(e)}
        
        # STEP 2: Processing
        try:
            resultados = procesar_todos_productos(
                ingenio=ingenio,
                fecha=fecha,
                productos=productos,
                input_dir=input_dir,
                output_dir=output_dir,
                zafras=None,
                bd_insert=bd_insert,
                id_field='id_parcela'
            )
            all_results[fecha] = resultados
            
            # Simple summary
            success = sum(1 for df in resultados.values() if df is not None)
            print(f"  ✓ Processed: {success}/{len(productos)} products")
            
        except Exception as e:
            print(f"  ❌ Processing error: {e}")
            logger.error(f"Processing error for {fecha}: {e}")
            all_results[fecha] = None
    
    # Final Summary
    print(f"\n{'#'*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'#'*70}")
    
    for fecha in fechas:
        sync = all_sync_stats.get(fecha, {})
        results = all_results.get(fecha)
        
        if 'error' in sync:
            print(f"  {fecha}: ❌ Sync Error")
        elif results is None:
            print(f"  {fecha}: ❌ Processing Error")
        else:
            closed = sync.get('closed_2026', 0)
            inserted = sync.get('inserted_2027', 0)
            updated = sync.get('updated_2027', 0)
            products_ok = sum(1 for df in results.values() if df is not None)
            print(f"  {fecha}: ✓ Sync({closed}/{inserted}/{updated}) | Products({products_ok}/{len(productos)})")
    
    print(f"{'#'*70}")
    print(f"  📄 Log file: {log_filename}")
    print(f"{'#'*70}\n")
    
    return all_results

# -*- coding: utf-8 -*-
"""
Procesamiento de productos finales usando bibliotecas open source
Procesa: NDVI, NDWI, Smart Growth y Weed
Reemplazo completo de ArcGIS ModelBuilder con geopandas, rasterio, numpy, pandas
"""

import geopandas as gpd
from pathlib import Path
import gc
# from supabase.lib.client_options import ClientOptions 
from supabase import create_client, Client
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes, rasterize
import numpy as np
import pandas as pd
from shapely.geometry import shape, mapping, box 
from datetime import datetime
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from shapely import wkb, wkt
import json

# Supabase (solo importar si BD_INSERT=True)
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# # =======================================================
# # CARGAR CURVAS POTENCIALES
# # =======================================================
# print("📊 Cargando curvas potenciales...")

from pathlib import Path

# Ruta a las curvas potenciales
BASE_DIR = Path.cwd()  # or Path(__file__).resolve().parent

CURVAS_DIR = Path(OUTPUT_DIR_ENV)
if not CURVAS_DIR.is_absolute():
    CURVAS_DIR = BASE_DIR / CURVAS_DIR


# Diccionario para almacenar curvas por ingenio
CURVAS_POTENCIALES = {}

# Función para obtener valor potencial de la curva
def obtener_potencial_curva(ingenio, edad_dias, indice='NDRE'):
    """
    Obtiene el valor potencial para una edad dada desde la curva cargada.
    
    Args:
        ingenio: Nombre del ingenio ('CAC', 'SANTA_CLARA', etc.)
        edad_dias: Edad en días
        indice: 'NDRE' o 'NDWI'
    
    Returns:
        Valor potencial para esa edad, o None si no hay curva disponible
    """
    if ingenio not in CURVAS_POTENCIALES:
        return None
    
    df_curva = CURVAS_POTENCIALES[ingenio]
    col = f'{indice}_potencial'
    
    if col not in df_curva.columns:
        return None
    
    # Si la edad está en la tabla, retornar directamente
    match = df_curva[df_curva['edad_dias'] == edad_dias]
    if len(match) > 0:
        return match.iloc[0][col]
    
    # Si no está, interpolar
    edad_min = df_curva['edad_dias'].min()
    edad_max = df_curva['edad_dias'].max()
    
    if edad_dias < edad_min:
        return df_curva.iloc[0][col]
    elif edad_dias > edad_max:
        return df_curva.iloc[-1][col]
    else:
        # Interpolación lineal
        df_sorted = df_curva.sort_values('edad_dias')
        return np.interp(edad_dias, df_sorted['edad_dias'], df_sorted[col])

# =======================================================
# DICCIONARIO MAESTRO DE INGENIOS
# =======================================================
INGENIOS_META = {
    "Pantaleon": { # Use the name you filter by in the CSV
        "EMPRESA": "Grupo Pantaleon", #"Pantaleon", # Use the code from the file name convention # GP
        "PAIS": "GT",      # Based on file prefix / bounds context
        "AOI_BOUNDS": {"west": -91.530544, "south": 13.947144, "east": -90.590395, "north": 14.445327}, # Your provided bounds
        "CRS": "EPSG:32615",  # **CRITICAL: Must match the raster CRS you confirmed!** #/ EPSG:4326
        "TILE_SIZE_KM": 20 # Dummy value, as you suspected
    },
    "Monte_Rosa": {
        "EMPRESA": "Grupo Pantaleon",
        "PAIS": "NI",
        "AOI_BOUNDS": {
            "west": -87.502239,
            "south": 12.279927,
            "east": -86.79237,
            "north": 12.948203
        },
        "CRS": "EPSG:32616",
        "TILE_SIZE_KM": 20
    },

    "Amajac": {
        "EMPRESA": "Grupo Pantaleon",
        "PAIS": "MX06",
        "AOI_BOUNDS": {
            "west": -98.543185,
            "south": 21.662828,
            "east": -98.021094,
            "north": 22.280718
        },
        "CRS": "EPSG:32614",
        "TILE_SIZE_KM": 20
    },

    "EMSA": {
        "EMPRESA": "Grupo Pantaleon",
        "PAIS": "MX07",
        "AOI_BOUNDS": {
            "west": -99.353038,
            "south": 22.437112,
            "east": -98.49084,
            "north": 23.31672
        },
        "CRS": "EPSG:32614",
        "TILE_SIZE_KM": 20
    },

    "IPSA": {
        "EMPRESA": "Grupo Pantaleon",
        "PAIS": "MX02",
        "AOI_BOUNDS": {
            "west": -98.623859,
            "south": 21.65155,
            "east": -97.987145,
            "north": 22.508884
        },
        "CRS": "EPSG:32614",
        "TILE_SIZE_KM": 20
    }
}

print("✅ Diccionario de ingenios cargado")
print(f"   Total ingenios configurados: {len(INGENIOS_META)}")
for ing, meta in INGENIOS_META.items():
    bounds = meta['AOI_BOUNDS']
    area_km2 = (bounds['east'] - bounds['west']) * 111 * (bounds['north'] - bounds['south']) * 111
    print(f"   - {ing:15} ({meta['PAIS']}/{meta['EMPRESA']}): ~{area_km2:.0f} km²")

# =======================================================
# FORMULAS POLINOMICAS (FALLBACK)
# =======================================================
# Se usan solo si no existe el archivo parquet de curva.
# x = edad en dias
FORMULAS_POTENCIALES = {
    "EMSA": {
        "NDRE": lambda x: (
            0.000000000000098 * (x**5) - 
            0.000000000051161 * (x**4) - 
            0.000000012525467 * (x**3) + 
            0.000005959336438 * (x**2) + 
            0.000739203312554 * x + 
            0.199493195410753
        ),
        "NDWI": lambda x: (
            -0.000000000000268 * (x**5) + 
            0.000000000365123 * (x**4) - 
            0.000000187095158 * (x**3) + 
            0.000037974035589 * (x**2) - 
            0.001211159730837 * x - 
            0.103648224595588
        )
    },
    "Amajac": {
        "NDRE": lambda x: (
            -0.000000000000656 * (x**5) + 
            0.000000000626723 * (x**4) - 
            0.000000228534347 * (x**3) + 
            0.000033607870585 * (x**2) - 
            0.000346568129761 * x + 
            0.221016483453894
        ),
        "NDWI": lambda x: (
            -0.000000000001057 * (x**5) + 
            0.000000001000451 * (x**4) - 
            0.000000355156276 * (x**3) + 
            0.000051721075749 * (x**2) - 
            0.001023270894495 * x - 
            0.087364430989297
        )
    },
    "IPSA": {
        "NDRE": lambda x: (
            -0.000000000000676 * (x**5) + 
            0.000000000533030 * (x**4) - 
            0.000000155353342 * (x**3) + 
            0.000017696957070 * (x**2) + 
            0.000556844849307 * x + 
            0.214562788170895
        ),
        "NDWI": lambda x: (
            -0.000000000001388 * (x**5) + 
            0.000000001173416 * (x**4) - 
            0.000000344287333 * (x**3) + 
            0.000035529951513 * (x**2) + 
            0.000857207455698 * x - 
            0.087356195373687
        )
    },
    "Pantaleon": {
        "NDRE": lambda x: (
            -0.000000000000047 * (x**5) + 
            0.000000000019120 * (x**4) + 
            0.000000005022705 * (x**3) - 
            0.000010922540182 * (x**2) + 
            0.003816187604734 * x + 
            0.149990990306719
        ),
        "NDWI": lambda x: (
            -0.000000000001043 * (x**5) + 
            0.000000000994730 * (x**4) - 
            0.000000335383715 * (x**3) + 
            0.000037700434707 * (x**2) + 
            0.001739266187149 * x - 
            0.134608730808750
        )
    },
    "Monte Rosa": {
        "NDRE": lambda x: (
            0.000000000000096 * (x**5) + 
            0.000000000055232 * (x**4) - 
            0.000000086619735 * (x**3) + 
            0.000018148797811 * (x**2) + 
            0.001270882911438 * x + 
            0.178959414402129
        ),
        "NDWI": lambda x: (
            -0.000000000001572 * (x**5) + 
            0.000000001636878 * (x**4) - 
            0.000000621366573 * (x**3) + 
            0.000092411080706 * (x**2) - 
            0.001803549089450 * x - 
            0.113900219568478
        )
    }
}


# =======================================================
# CONFIGURACIONES DE PRODUCTOS
# =======================================================
PRODUCTOS_CONFIG = {
    "NDVI": {
        "tipo": "simple",  # Procesamiento estándar
        "input_suffix": "NDRE",
        "potencial_formula": lambda edad, ingenio: (
            # 1. Try loading parquet file
            obtener_potencial_curva(ingenio, edad, 'NDRE') 
            # 2. If it is None, search in the formula dictionary
            if obtener_potencial_curva(ingenio, edad, 'NDRE') is not None
            else (
                FORMULAS_POTENCIALES[ingenio]['NDRE'](edad) 
                if ingenio in FORMULAS_POTENCIALES 
                else None # Si no hay ni archivo ni formula, error controlado
            )
        ),
        
        # "potencial_formula": lambda edad, ingenio='CAC': (
        #     # Intentar usar curva cargada, si no existe usar fórmula hardcodeada
        #     obtener_potencial_curva(ingenio, edad, 'NDRE') or (
        #         0.000000000000028 * (edad**6) - 
        #         0.000000000031353 * (edad**5) + 
        #         0.000000013102096 * (edad**4) - 
        #         0.000002457541752 * (edad**3) + 
        #         0.000175180254613 * (edad**2) + 
        #         0.002445789888924 * edad + 
        #         0.188780255063534
        #     )
        # ),

        "reclass_ranges": [
            (-1, 0.1, 1), (0.1, 0.2, 2), (0.2, 0.3, 3), (0.3, 0.4, 4),
            (0.40, 0.5, 5), (0.5, 0.55, 6), (0.55, 0.6, 7), (0.60, 0.625, 8),
            (0.625, 0.65, 9), (0.65, 0.675, 10), (0.675, 0.70, 11), (0.70, 0.725, 12),
            (0.725, 0.75, 13), (0.75, 1.0, 14)
        ],
        "reclass_potencial": [
            (-9000, 0.6, 1), (0.6, 0.9, 2), (0.9, 1.1, 3),
            (1.1, 1.3, 4), (1.3, 9000, 5)
        ],
        "cosechado_classes": [1, 2, 3]
    },
    "NDWI": {
        "tipo": "simple",  # Procesamiento estándar
        "input_suffix": "NDWI",
        "potencial_formula": lambda edad, ingenio: (
            # 1. Try loading parquet file
            obtener_potencial_curva(ingenio, edad, 'NDWI') 
            # 2. If it is None, search in the formula dictionary
            if obtener_potencial_curva(ingenio, edad, 'NDWI') is not None
            else (
                FORMULAS_POTENCIALES[ingenio]['NDWI'](edad) 
                if ingenio in FORMULAS_POTENCIALES 
                else None
            )
        ),

        # "potencial_formula": lambda edad, ingenio='CAC': (
        #     # Intentar usar curva cargada, si no existe usar fórmula hardcodeada
        #     obtener_potencial_curva(ingenio, edad, 'NDWI') or (
        #         0.0000046831 * (edad**2) - 0.0023419769 * edad - 0.2889615388
        #     )
        # ),
        "reclass_ranges": [
            (-1, -0.15, 1), (-0.15, -0.05, 2), (-0.05, 0.05, 3), (0.05, 0.15, 4),
            (0.15, 0.20, 5), (0.20, 0.25, 6), (0.25, 0.30, 7), (0.30, 0.325, 8),
            (0.325, 0.35, 9), (0.35, 0.375, 10), (0.375, 0.40, 11), (0.40, 0.425, 12),
            (0.425, 0.45, 13), (0.45, 1, 14)
        ],
        "reclass_potencial": [
            (-9000, 0.6, 1), (0.6, 0.9, 2), (0.9, 1.1, 3),
            (1.1, 1.3, 4), (1.3, 9000, 5)
        ],
        "cosechado_classes": []
    },
    "SMART_GROWTH": {
        "tipo": "combinado",  # Combina potenciales NDVI y NDWI
        "input_suffix": None,
        "requires": ["NDVI", "NDWI"],
        "potencial_formula": None,
        "reclass_ranges": None,
        "reclass_potencial": None,
        "cosechado_classes": []
    },
    "WEED": {
        "tipo": "complejo",  # Procesamiento especial para detección de malezas
        "input_suffix": "NDRE",
        "potencial_formula": None,
        "reclass_ranges": [
            (-10, 0, 2),  # ≤ 0: NO maleza (NDVI ≤ umbral = normal)
            (0, 10, 1)    # > 0: SÍ maleza (NDVI > umbral = exceso de vigor)
        ],
        "reclass_potencial": None,
        "cosechado_classes": [],
        "buffer_distance": 0,  # Sin buffer
        "maleza_threshold_formula": lambda mean, std: mean + (1.5 * std),  # Umbral ALTO para detectar exceso
        "maleza_min": 0,
        "maleza_max": 1,
        "maleza_critica_percent": 15
    }
}

# =======================================================
# FUNCIONES AUXILIARES
# =======================================================

def get_supabase_client():
    """Obtiene cliente de Supabase desde variables de entorno"""
    if not SUPABASE_AVAILABLE:
        raise ImportError("Librería 'supabase' no instalada. Ejecute: pip install supabase")
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        raise ValueError("Variables de entorno SUPABASE_URL y SUPABASE_KEY requeridas")
    
    # REMOVE ALL ClientOptions code - just return simple client
    return create_client(url, key)

from pathlib import Path

# Curve paths resolved dynamically from cfg.OUTPUT_DIR at runtime
def _build_curvas_paths():
    out = cfg.OUTPUT_DIR
    display = cfg.ingenio_display   # e.g. "Monte Rosa"
    key = cfg.INGENIO               # e.g. "Monte_Rosa"
    return {
        display: out / f"{display}_curvas" / f"curva_global_{display}.parquet",
        key:     out / f"{key}_curvas"     / f"curva_global_{key}.parquet",
    }

CURVAS_PARQUET_PATHS = {}   # populated lazily in cargar_curva_dinamica


def cargar_curva_dinamica(ingenio):
    """
    Carga la curva potencial. Primero busca el parquet en OUTPUT_DIR,
    si falla usa las fórmulas polinómicas de respaldo.
    """
    global CURVAS_PARQUET_PATHS
    if not CURVAS_PARQUET_PATHS:
        CURVAS_PARQUET_PATHS = _build_curvas_paths()

    print(f"📊 Buscando curvas potenciales para: {ingenio}...")

    ruta_completa = CURVAS_PARQUET_PATHS.get(ingenio)

    if ruta_completa is not None:
        try:
            if ruta_completa.exists():
                df_curva = pd.read_parquet(ruta_completa)
                CURVAS_POTENCIALES[ingenio] = df_curva
                print(f"   ✅ Curva Parquet cargada exitosamente: {len(df_curva)} edades")
                return True
            else:
                print(f"   ⚠️  Archivo no encontrado: {ruta_completa}")
        except Exception as e:
            print(f"   ❌ Error leyendo archivo parquet: {str(e)}")
    else:
        print(f"   ⚠️  No hay ruta hardcodeada para {ingenio}")

    # 🔁 Fallback a fórmulas
    print(f"   ⚠️  No se cargó archivo de curva. Verificando fórmulas de respaldo...")

    if ingenio in FORMULAS_POTENCIALES:
        print(f"   ✅ Fórmulas polinómicas encontradas para {ingenio}.")
        print(f"      -> Se usarán ecuaciones hardcodeadas (R² ~ 0.90+)")
        return False
    else:
        print(f"   ❌ CRITICO: No hay ni archivo Parquet ni Fórmula para {ingenio}!")
        print(f"      -> El cálculo de potencial fallará (NaN).")
        return False



def obtener_zafras_activas(ingenio):
    """
    Obtiene lista de zafras activas para un ingenio (solo informativo)
    Filtra por temporada_activa = True
    """
    supabase = get_supabase_client()
    
    print(f"        → Consultando zafras activas para {ingenio}...")
    
    try:
        response = supabase.table('parcelas_ingenios_reprocess') \
            .select('zafra') \
            .eq('ingenio', ingenio) \
            .eq('temporada_activa', True) \
            .execute()
        
        if not response.data:
            raise ValueError(f"No se encontraron parcelas activas para ingenio='{ingenio}'")
        
        # Obtener zafras únicas
        zafras = sorted(list(set([row['zafra'] for row in response.data])))
        
        print(f"        ✓ Zafras activas encontradas: {zafras}")
        
        return zafras
        
    except Exception as e:
        print(f"        ❌ Error consultando zafras activas: {str(e)}")
        raise

def cargar_parcelas_desde_supabase(ingenio):
    """
    Carga parcelas desde Supabase OPTIMIZADO
    Solo carga columnas necesarias para reducir transferencia de datos
    """
    supabase = get_supabase_client()
    
    print(f"        → Consultando parcelas desde Supabase...")
    print(f"           Filtros: ingenio='{ingenio}', temporada_activa=True")
    
    # ============================================
    # CRÍTICO: Solo seleccionar columnas necesarias
    # Esto reduce ENORMEMENTE el tiempo de query
    # ============================================
    columns_needed = 'id_parcela,zafra,fecha_inicio,area_calculada,ingenio,company,company_id,ingenio_id,geometry_polygon'
    # columns_needed = 'id_parcela,zafra,fecha_inicio,area_calculada,ingenio,company,geometry_polygon'
    
    try:
        import time
        start = time.time()
        
        response = supabase.table('parcelas_ingenios_reprocess') \
            .select(columns_needed) \
            .eq('ingenio', ingenio) \
            .eq('temporada_activa', True) \
            .execute()
        
        elapsed = time.time() - start
        print(f"        ✓ Query completada en {elapsed:.2f} segundos")
        
        if not response.data:
            raise ValueError(f"No se encontraron parcelas activas para ingenio='{ingenio}'")
        
        df = pd.DataFrame(response.data)
        print(f"        ✓ {len(df)} parcelas encontradas")

        # --- SANITY CHECK ---
        if len(df) != df['id_parcela'].nunique():
            print(f"        ⚠ WARNING: Detectados {len(df) - df['id_parcela'].nunique()} IDs duplicados. Revise la BD.")
        # --------------------
        
        # =========================================================
        # Parse geometry (tu código existente)
        # =========================================================
        def parse_geometry(val):
            """
            Optimistic parsing: Assumes Supabase ALWAYS returns a Dict (GeoJSON).
            """
            if val is None or pd.isna(val):
                return None
            try:
                if isinstance(val, dict):
                    return shape(val)
                # -------------------------------------------------
                # If it's not a dict, return None (Strict Mode)
                return None
            
            except Exception as e:
                print(f"        ⚠ Error parseando geometría: {e}")
                return None

        df['geometry'] = df['geometry_polygon'].apply(parse_geometry)
        df_valid = df[df['geometry'].notna()].copy()
        
        if len(df_valid) < len(df):
            print(f"        ⚠ {len(df) - len(df_valid)} parcelas sin geometría válida (excluidas)")
        
        if len(df_valid) == 0:
            raise ValueError("Error Crítico: Ninguna geometría se pudo parsear. Verifique formato en BD.")

        # Detectar CRS
        sample_geom = df_valid.iloc[0].geometry
        if sample_geom.geom_type == 'Polygon':
            sample_coords = sample_geom.exterior.coords[0]
        else:
            sample_coords = list(sample_geom.geoms[0].exterior.coords)[0]
        
        x, y = sample_coords[0], sample_coords[1]
        
        if abs(x) < 180 and abs(y) < 90:
            print(f"        → Coordenadas detectadas: WGS84 (lon={x:.4f}, lat={y:.4f})")
            gdf = gpd.GeoDataFrame(df_valid, geometry='geometry', crs='EPSG:4326')
        else:
            if 100000 <= abs(x) <= 900000:
                ingenio_utm_zones = {
                    'CAC': 'EPSG:32619',
                    'Pantaleon': 'EPSG:32615',
                    'SANTA_CLARA': 'EPSG:32615',
                    'Monte Rosa' : 'EPSG:32616',
                    'Amajac' : 'EPSG:32614',
                    'EMSA' : 'EPSG:32614',
                    'IPSA' : 'EPSG:32614'
                }
                utm_crs = ingenio_utm_zones.get(ingenio, 'EPSG:32615')
                print(f"        → Coordenadas detectadas: UTM (x={x:.0f}, y={y:.0f})")
                print(f"        → CRS detectado: {utm_crs}")
                gdf = gpd.GeoDataFrame(df_valid, geometry='geometry', crs=utm_crs)
                
                print(f"        → Reproyectando {utm_crs} → EPSG:4326...")
                gdf = gdf.to_crs('EPSG:4326')
            else:
                print(f"        ⚠ Coordenadas no reconocidas (x={x:.2f}, y={y:.2f}), asumiendo WGS84")
                gdf = gpd.GeoDataFrame(df_valid, geometry='geometry', crs='EPSG:4326')
        
        sample_geom_final = gdf.iloc[0].geometry
        if sample_geom_final.geom_type == 'Polygon':
            sample_coords_final = sample_geom_final.exterior.coords[0]
        else:
            sample_coords_final = list(sample_geom_final.geoms[0].exterior.coords)[0]
        print(f"        ✓ Coordenadas finales WGS84: (lon={sample_coords_final[0]:.6f}, lat={sample_coords_final[1]:.6f})")
        
        return gdf
        
    except Exception as e:
        print(f"        ❌ Error cargando parcelas desde Supabase: {str(e)}")
        raise



def insertar_a_supabase(df, producto, ingenio, fecha):
    """
    Inserta datos del DataFrame a Supabase con UPSERT
    
    Tablas y campos por producto:
    - NDVI: public.data_ndvi
    - NDWI: public.data_ndwi  
    - SMART_GROWTH: public.data_smart_growth
    - WEED: public.data_maleza
    """
    supabase = get_supabase_client()
    
    # Mapeo de productos a tablas
    tabla_map = {
        "NDVI": "data_ndvi",
        "NDWI": "data_ndwi",
        "SMART_GROWTH": "data_sg",
        "WEED": "data_maleza"
    }
    
    if producto not in tabla_map:
        print(f"        ⚠ Producto {producto} no tiene tabla configurada")
        return
    
    tabla = tabla_map[producto]
    
    # Convertir DataFrame a lista de dicts
    df_copy = df.copy()
    
    # Convertir fecha_img a string ISO
    if 'fecha_img' in df_copy.columns:
        df_copy['fecha_img'] = df_copy['fecha_img'].dt.strftime('%Y-%m-%d')
    
    # Convertir NaN a None para JSON
    df_copy = df_copy.where(pd.notna(df_copy), None)
    
    records = df_copy.to_dict('records')
    
    print(f"        → Insertando {len(records)} registros a {tabla}...")
    
    try:
        # Inserción simple sin ON CONFLICT
        # Las tablas no tienen constraints únicos definidos
        response = supabase.table(tabla).insert(
            records
        ).execute()
        
        print(f"        ✓ {len(records)} registros insertados en {tabla}")
        
    except Exception as e:
        error_msg = str(e)
        print(f"        ❌ Error insertando a {tabla}: {error_msg}")
        
        # Si el error es por registros duplicados, intentar con delete + insert
        if 'duplicate' in error_msg.lower() or 'unique' in error_msg.lower():
            print(f"        → Intentando eliminar registros existentes y reinsertar...")
            try:
                # Eliminar registros existentes para esta fecha e ingenio
                supabase.table(tabla).delete().eq('fecha_img', fecha).execute()
                # Reintentar inserción
                response = supabase.table(tabla).insert(records).execute()
                print(f"        ✓ {len(records)} registros reinsertados en {tabla}")
            except Exception as e2:
                print(f"        ❌ Error en reintento: {str(e2)}")
                raise
        else:
            raise

def get_ingenio_meta(ingenio_name):
    """Obtiene metadata del ingenio desde el diccionario"""
    if ingenio_name not in INGENIOS_META:
        raise ValueError(f"Ingenio '{ingenio_name}' no encontrado. Opciones: {list(INGENIOS_META.keys())}")
    return INGENIOS_META[ingenio_name]

def generar_nombres_archivos(pais, empresa, ingenio, fecha_str, producto="NDVI"):
    """Genera nombres de archivos según convención: PAIS_EMPRESA_PRODUCTO_INGENIO_FECHA"""
    config = PRODUCTOS_CONFIG[producto]
    
    if config['tipo'] == 'combinado':
        # Smart Growth no tiene archivo de entrada directo
        return {
            'input_raster': None,
            'extracted': f"{pais}_{empresa}_{producto}_{ingenio}_{fecha_str}.tif",
            'potencial': None,
            'data_output': f"{pais}_{empresa}_DATA_{producto}_{ingenio}_{fecha_str}.parquet"
        }
    
    input_suffix = config['input_suffix']
    base_input = f"{pais}_{empresa}_{input_suffix}_{ingenio}_{fecha_str}"
    base_output = f"{pais}_{empresa}_{producto}_{ingenio}_{fecha_str}"
    
    return {
        'input_raster': f"{base_input}_cloudfill.tif",
        'extracted': f"{base_output}_extracted.tif",
        'potencial': f"{pais}_{empresa}_POTENCIAL_{producto}_{ingenio}_{fecha_str}.tif" if config['potencial_formula'] else None,
        'data_output': f"{pais}_{empresa}_DATA_{producto}_{ingenio}_{fecha_str}.parquet"
    }

# Calculate age based on the new logic by Sergio (where the max age should be capped to 450 days)
def calculate_days(fecha_img, fecha_inicio):
    """
    Calcula días desde fecha de inicio de zafra.
    Directive: CLIPPED to a maximum of 450 days.
    """
    raw_days = (fecha_img - fecha_inicio).days
    
    # Logic: If days > 450, force it to 450. Otherwise keep it as is.
    if raw_days > 450:
        return 450
    else:
        return raw_days

# def calculate_days(fecha_img, fecha_inicio):
#     """Calcula días desde fecha de inicio de zafra"""
#     return (fecha_img - fecha_inicio).days

def calcular_etapa_fenologica(edad):
    """Clasifica la edad en etapa fenológica"""
    if edad is None or pd.isna(edad):
        return "SF"
    elif edad <= 60:
        return "Iniciacion"
    elif edad <= 120:
        return "Macollamiento"
    elif edad <= 210:
        return "Elongacion I"
    elif edad <= 300:
        return "Elongacion II"
    else:
        return "Maduracion"

def zonal_statistics(raster_path, zones_gdf, zone_field, stats=['mean', 'std', 'min', 'max', 'median']):
    """Calcula estadísticas zonales"""
    results = []
    
    with rasterio.open(raster_path) as src:
        if zones_gdf.crs != src.crs:
            zones_gdf = zones_gdf.to_crs(src.crs)
        
        for idx, row in zones_gdf.iterrows():
            zone_id = row[zone_field]
            geom = [mapping(row.geometry)]
            
            try:
                out_image, out_transform = mask(src, geom, crop=True, all_touched=False)
                data = out_image[0]
                valid_data = data[data != src.nodata] if src.nodata is not None else data.flatten()
                valid_data = valid_data[~np.isnan(valid_data)]
                
                if len(valid_data) > 0:
                    stat_dict = {zone_field: zone_id}
                    if 'mean' in stats:
                        stat_dict['MEAN'] = float(np.mean(valid_data))
                    if 'std' in stats:
                        stat_dict['STD'] = float(np.std(valid_data))
                    if 'min' in stats:
                        stat_dict['MIN'] = float(np.min(valid_data))
                    if 'max' in stats:
                        stat_dict['MAX'] = float(np.max(valid_data))
                    if 'median' in stats:
                        stat_dict['MEDIAN'] = float(np.median(valid_data))
                    results.append(stat_dict)
            except Exception as e:
                print(f"      ⚠ Error procesando zona {zone_id}: {e}")
                continue
    
    return pd.DataFrame(results)

def buffer_geometry(gdf, distance):
    """Crea buffer de geometrías en metros"""
    original_crs = gdf.crs
    
    if gdf.crs.is_geographic:
        gdf_projected = gdf.to_crs(epsg=32615)
        gdf_buffered = gdf_projected.copy()
        gdf_buffered['geometry'] = gdf_projected.geometry.buffer(distance)
        gdf_buffered = gdf_buffered.to_crs(original_crs)
    else:
        gdf_buffered = gdf.copy()
        gdf_buffered['geometry'] = gdf.geometry.buffer(distance)
    
    return gdf_buffered

def extract_by_mask(raster_path, mask_gdf, output_path):
    """Extrae un raster usando una máscara de polígonos"""
    with rasterio.open(raster_path) as src:
        if mask_gdf.crs != src.crs:
            mask_gdf = mask_gdf.to_crs(src.crs)
        
        geoms = [mapping(geom) for geom in mask_gdf.geometry]
        out_image, out_transform = mask(src, geoms, crop=True)
        
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

def reclassify_raster(raster_path, reclass_dict, output_path, nodata_value=-9999):
    """Reclasifica un raster basado en rangos"""
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        out_data = np.full_like(data, nodata_value, dtype=np.float32)
        
        for min_val, max_val, new_val in reclass_dict:
            mask_range = (data >= min_val) & (data < max_val)
            out_data[mask_range] = new_val
        
        out_meta = src.meta.copy()
        out_meta.update({'dtype': 'float32', 'nodata': nodata_value})
        
        with rasterio.open(output_path, 'w', **out_meta) as dest:
            dest.write(out_data, 1)

def raster_to_polygons(raster_path, simplify_tolerance=0):
    """Convierte un raster a polígonos"""
    with rasterio.open(raster_path) as src:
        image = src.read(1)
        results = []
        for geom, value in shapes(image, transform=src.transform):
            if value != src.nodata:
                results.append({'geometry': shape(geom), 'gridcode': int(value)})
        
        gdf = gpd.GeoDataFrame(results, crs=src.crs)
        if simplify_tolerance > 0:
            gdf['geometry'] = gdf.geometry.simplify(simplify_tolerance)
        return gdf

def polygon_to_raster(gdf, value_field, output_path, cell_size=None, bounds=None, reference_raster=None, nodata_value=-9999, all_touched=True):
    """Convierte polígonos a raster - usa reference_raster para coincidir dimensiones exactas
       UPDATED: Ensures correct NoData value (-9999) is written to file metadata and background.
    """
    if reference_raster is not None:
        # Usar las dimensiones exactas del raster de referencia
        with rasterio.open(reference_raster) as ref:
            height, width = ref.shape
            transform = ref.transform
            crs = ref.crs
            bounds = ref.bounds
    else:
        # Modo legacy - calcular desde cell_size
        if bounds is None:
            bounds = gdf.total_bounds
        
        minx, miny, maxx, maxy = bounds
        width = int((maxx - minx) / cell_size)
        height = int((maxy - miny) / cell_size)
        transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
        crs = gdf.crs
    
    # Asegurar que gdf está en el mismo CRS
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    
    shapes_with_values = ((mapping(geom), value) for geom, value in 
                          zip(gdf.geometry, gdf[value_field]))
    
    raster = rasterize(shapes_with_values, out_shape=(height, width),
                      transform=transform, fill=nodata_value, dtype=np.float32, all_touched=all_touched) # earlier it was fill=0
    
    with rasterio.open(output_path, 'w', driver='GTiff', height=height,
                      width=width, count=1, dtype=np.float32, crs=crs,
                      transform=transform, nodata=nodata_value, BIGTIFF='YES') as dst:
        dst.write(raster, 1)

def raster_calculator(raster1_path, raster2_path, output_path, operation='divide'):
    """
    Operaciones entre rasters - asegura que ambos tengan las mismas dimensiones.
    FIX: Maneja correctamente los valores NoData del header Y los hardcoded (-9999, -32768).
    """
    from rasterio.warp import reproject, Resampling
    
    with rasterio.open(raster1_path) as src1:
        data1 = src1.read(1).astype(np.float32)
        nodata1 = src1.nodata if src1.nodata is not None else -9999
        meta = src1.meta.copy()
        
        with rasterio.open(raster2_path) as src2:
            nodata2 = src2.nodata if src2.nodata is not None else -9999
            
            # Si las dimensiones no coinciden, reproyectar src2 a src1
            if src1.shape != src2.shape or src1.transform != src2.transform:
                data2 = np.empty(src1.shape, dtype=np.float32)
                reproject(
                    source=rasterio.band(src2, 1),
                    destination=data2,
                    src_transform=src2.transform,
                    src_crs=src2.crs,
                    dst_transform=src1.transform,
                    dst_crs=src1.crs,
                    resampling=Resampling.bilinear
                )
            else:
                data2 = src2.read(1).astype(np.float32)
            
            # --- ROBUST VALIDITY CHECK ---
            # 1. Check against the file's defined NoData
            mask1 = (data1 != nodata1)
            mask2 = (data2 != nodata2)
            
            # 2. ALSO Check against common hardcoded garbage values (Float & Int standards)
            # This protects us if the header is missing but the pixels are -32768
            garbage_values = [-9999, -32768]
            for bad_val in garbage_values:
                mask1 &= (data1 != bad_val)
                mask2 &= (data2 != bad_val)
            
            # 3. Combine: Pixel valid ONLY if both inputs are valid
            valid_mask = mask1 & mask2
            # -----------------------------
            
            # Inicializar resultado con NoData
            result = np.full_like(data1, nodata1)
            
            # Ejecutar operación solo en pixeles válidos
            if operation == 'divide':
                with np.errstate(divide='ignore', invalid='ignore'):
                    denom = data2[valid_mask]
                    safe_mask = denom != 0
                    
                    vals = np.zeros_like(denom)
                    vals[safe_mask] = data1[valid_mask][safe_mask] / denom[safe_mask]
                    result[valid_mask] = vals
                    
            elif operation == 'multiply':
                result[valid_mask] = data1[valid_mask] * data2[valid_mask]
            elif operation == 'add':
                result[valid_mask] = data1[valid_mask] + data2[valid_mask]
            elif operation == 'subtract':
                result[valid_mask] = data1[valid_mask] - data2[valid_mask]
            
            # Asegurar que el output tenga el NoData correcto definido
            meta.update(nodata=nodata1)
            
            with rasterio.open(output_path, 'w', **meta) as dest:
                dest.write(result.astype(np.float32), 1)



def clip_raster_with_polygons(raster_path, polygons_gdf, output_path, all_touched=True):
    """Recorta un raster usando polígonos (mask) - solo mantiene áreas dentro de las parcelas"""
    from rasterio.mask import mask as rasterio_mask
    
    # Eliminar archivo de salida si existe
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except PermissionError:
            pass  # Intentar de todos modos
    
    # Asegurar que los polígonos están en el mismo CRS que el raster
    with rasterio.open(raster_path) as src:
        if polygons_gdf.crs != src.crs:
            polygons_gdf = polygons_gdf.to_crs(src.crs)
        
        # Convertir geometrías a formato GeoJSON
        geoms = [mapping(geom) for geom in polygons_gdf.geometry]
        
        # Hacer el clip (crop=True recorta al extent de los polígonos)
        out_image, out_transform = rasterio_mask(src, geoms, crop=True, all_touched=all_touched)
        
        # Actualizar metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        # Guardar raster recortado
        with rasterio.open(output_path, 'w', **out_meta) as dest:
            dest.write(out_image)

# =======================================================
# FUNCIONES ESPECÍFICAS POR TIPO DE PRODUCTO
# =======================================================

def procesar_smart_growth(ingenio, fecha, parcelas_gdf, output_dir, id_field, meta, archivos):
    """
    Procesa Smart Growth combinando potenciales de NDVI y NDWI
    Formula: (potencial_ndvi * 12) + (potencial_ndwi * 8)
    FIX 1: Maneja correctamente los valores NoData (-9999)
    FIX 2: Busca archivos usando guiones bajos (Monte_Rosa)
    """
    from rasterio.warp import reproject, Resampling
    
    print(f"\n📐 Procesando Smart Growth (combina NDVI y NDWI)...")
    
    # Buscar archivos de potencial NDVI y NDWI
    pais = meta['PAIS']
    empresa = meta['EMPRESA']
    
    # --- FILENAME LOGIC ---
    # Los archivos de salida (Potencial) se generaron con guiones bajos en el paso anterior.
    # Por eso forzamos '_' aquí para poder encontrarlos.
    ingenio_safe = ingenio.replace(' ', '_') 
    
    fecha_dt = datetime.strptime(fecha, "%Y-%m-%d")
    fecha_str = fecha_dt.strftime("%Y_%m_%d")
    
    # Calcular fecha_img y edad
    parcelas_gdf['fecha_inicio_converted'] = pd.to_datetime(parcelas_gdf['fecha_inicio'], format='%Y-%m-%d', errors='coerce')
    parcelas_gdf['fecha_img'] = fecha_dt
    parcelas_gdf['edad'] = parcelas_gdf.apply(
        lambda row: calculate_days(row['fecha_img'], row['fecha_inicio_converted'])
        if pd.notna(row['fecha_inicio_converted']) else None, axis=1
    )
    
    # Usamos ingenio_safe (Monte_Rosa) para encontrar los archivos
    potencial_ndvi_path = os.path.join(output_dir, f"{pais}_{empresa}_POTENCIAL_NDVI_{ingenio_safe}_{fecha_str}.tif")
    potencial_ndwi_path = os.path.join(output_dir, f"{pais}_{empresa}_POTENCIAL_NDWI_{ingenio_safe}_{fecha_str}.tif")
    
    if not os.path.exists(potencial_ndvi_path):
        raise FileNotFoundError(f"Potencial NDVI no encontrado: {potencial_ndvi_path}")
    if not os.path.exists(potencial_ndwi_path):
        raise FileNotFoundError(f"Potencial NDWI no encontrado: {potencial_ndwi_path}")
    
    sg_raster_path = os.path.join(output_dir, archivos['extracted'])
    
    with rasterio.open(potencial_ndvi_path) as src_ndvi:
        ndvi_data = src_ndvi.read(1).astype(np.float32)
        ndvi_nodata = src_ndvi.nodata if src_ndvi.nodata is not None else -9999
        meta_out = src_ndvi.meta.copy()
        
        with rasterio.open(potencial_ndwi_path) as src_ndwi:
            # Si las dimensiones no coinciden, reproyectar NDWI a NDVI
            if src_ndvi.shape != src_ndwi.shape or src_ndvi.transform != src_ndwi.transform:
                ndwi_data = np.empty(src_ndvi.shape, dtype=np.float32)
                reproject(
                    source=rasterio.band(src_ndwi, 1),
                    destination=ndwi_data,
                    src_transform=src_ndwi.transform,
                    src_crs=src_ndwi.crs,
                    dst_transform=src_ndvi.transform,
                    dst_crs=src_ndvi.crs,
                    resampling=Resampling.bilinear
                )
            else:
                ndwi_data = src_ndwi.read(1).astype(np.float32)
            
            ndwi_nodata = src_ndwi.nodata if src_ndwi.nodata is not None else -9999

            # --- MATH FIX START (-9999 Issue) ---
            # 1. Crear mascara de validez: Un pixel es válido SOLO si ambos inputs son válidos
            valid_mask = (ndvi_data != ndvi_nodata) & (ndwi_data != ndwi_nodata)
            
            # 2. Inicializar el canvas completo con el valor NoData
            sg_data = np.full_like(ndvi_data, ndvi_nodata)
            
            # 3. Calcular la fórmula SOLO en los pixeles válidos
            # Esto evita que -9999 + -9999 se convierta en -199980
            sg_data[valid_mask] = (ndvi_data[valid_mask] * 12) + (ndwi_data[valid_mask] * 8)
            
            # 4. Asegurar que la metadata de salida defina el NoData correcto
            meta_out.update(nodata=ndvi_nodata)
            # --- MATH FIX END ---
            
            # Guardar raster temporal completo
            sg_temp_path = sg_raster_path.replace('.tif', '_temp.tif')
            with rasterio.open(sg_temp_path, 'w', **meta_out) as dest:
                dest.write(sg_data, 1)
    
    print(f"        ✓ Smart Growth calculado correctamente (NoData preservado)")
    
    # Recortar a parcelas
    print(f"        ✓ Recortando Smart Growth a parcelas...")
    clip_raster_with_polygons(sg_temp_path, parcelas_gdf, sg_raster_path)
    
    # Eliminar temporal
    if os.path.exists(sg_temp_path):
        os.remove(sg_temp_path)
    
    # Estadísticas zonales
    stats = zonal_statistics(sg_raster_path, parcelas_gdf, id_field)
    
    if not stats.empty:
        stats.columns = [id_field, 'sg_mean', 'sg_stdv', 'sg_min', 'sg_max', 'sg_median']
        parcelas_gdf = parcelas_gdf.merge(stats, on=id_field, how='left')
    else:
        # Fallback por si acaso
        for col in ['sg_mean', 'sg_stdv', 'sg_min', 'sg_max', 'sg_median']:
            parcelas_gdf[col] = 0
            
    return parcelas_gdf



def procesar_weed(ingenio, fecha, parcelas_gdf, ndvi_raster_path, output_dir, id_field, meta, archivos, config):
    """
    Procesa detección de malezas (Weed)
    Calcula umbral, filtra, buffers, resta rasters, calcula % de maleza
    Genera raster de temporalidad (overlapping de últimas 5 detecciones)
    """
    print(f"\n🌱 Procesando detección de malezas (Weed)...")
    
    pais = meta['PAIS']
    empresa = meta['EMPRESA']
    fecha_dt = datetime.strptime(fecha, "%Y-%m-%d")
    
    # Archivo histórico acumulativo (GeoJSON)
    historico_path = os.path.join(output_dir, f"{pais}_{empresa}_WEED_HISTORICO_{ingenio}.geojson")
    
    # Calcular fecha_img y edad
    parcelas_gdf['fecha_inicio_converted'] = pd.to_datetime(parcelas_gdf['fecha_inicio'], format='%Y-%m-%d', errors='coerce')
    parcelas_gdf['fecha_img'] = fecha_dt
    parcelas_gdf['edad'] = parcelas_gdf.apply(
        lambda row: calculate_days(row['fecha_img'], row['fecha_inicio_converted'])
        if pd.notna(row['fecha_inicio_converted']) else None, axis=1
    )

    # ==============================================================================
    # 🛑 LOGIC CHANGE: RESTRICT TO AGE <= 90 DAYS (ADDED ON 12-02-2026)
    # ==============================================================================
    print(f"   [FILTER] Applying agronomic filter: Age <= 90 days...")
    initial_count = len(parcelas_gdf)
    
    # Keep only young cane
    parcelas_gdf = parcelas_gdf[parcelas_gdf['edad'] <= 90].copy()
    
    filtered_count = len(parcelas_gdf)
    dropped_count = initial_count - filtered_count
    print(f"        ✓ Retained: {filtered_count} parcels (Dropped {dropped_count} parcels > 90 days)")

    # HANDLE EDGE CASE: If no parcels are <= 90 days
    if filtered_count == 0:
        print(f"        ⚠ STOPPING WEED: No parcels meet the age criteria (<= 90).")
        print(f"        → Generating BLANK (NoData) rasters for consistency...")

        # ---------------------------------------------------------
        # GENERATE GHOST RASTERS (So file system doesn't break)
        # ---------------------------------------------------------
        # Define paths exactly as they are defined later in the code
        # 1. Main Weed Raster
        weed_out = os.path.join(output_dir, archivos['extracted'])
        # 2. Diff Raster
        diff_out = os.path.join(output_dir, f"{pais}_{empresa}_WEED_DIFF_{ingenio}_{fecha.replace('-', '_')}.tif")
        # 3. Threshold Raster
        thresh_out = os.path.join(output_dir, f"{pais}_{empresa}_WEED_THRESHOLD_{ingenio}_{fecha.replace('-', '_')}.tif")
        # 4. Temporalidad Raster
        temp_out = os.path.join(output_dir, archivos['extracted'].replace('_extracted.tif', '_temporalidad.tif'))

        # Create blank TIFs using input raster as template
        with rasterio.open(ndvi_raster_path) as src:
            meta = src.meta.copy()
            # Ensure nodata is set
            nodata_val = src.nodata if src.nodata is not None else -9999
            meta.update(nodata=nodata_val)
            
            # Create array full of NoData
            empty_img = np.full((src.count, src.height, src.width), nodata_val, dtype=meta['dtype'])
            
            # Write the 4 files
            for fpath in [weed_out, diff_out, thresh_out, temp_out]:
                with rasterio.open(fpath, 'w', **meta) as dst:
                    dst.write(empty_img)
                print(f"        ✓ Created blank file: {os.path.basename(fpath)}")

        # ---------------------------------------------------------
        # RETURN EMPTY DATAFRAME
        # ---------------------------------------------------------
        empty_cols = [id_field, 'zafra', 'fecha_img', 'edad', 'area_maleza', 'area_parcela', 
                      'percent_maleza', 'etapa_f', 'status_maleza', 'ingenio', 'company', 
                      'ingenio_id', 'company_id']
        
        # Ensure cols exist in empty df
        empty_df = pd.DataFrame(columns=empty_cols)
        if id_field != 'id_parcela':
            empty_df.rename(columns={id_field: 'id_parcela'}, inplace=True)
            
        return empty_df
    # ==============================================================================
    
    # 1. Calcular umbral de maleza
    print(f"   [1/9] Calculando umbral de maleza (MEAN + 1.5*STD)...")
    stats = zonal_statistics(ndvi_raster_path, parcelas_gdf, id_field)
    stats.columns = [id_field, 'MEAN', 'STD', 'MIN', 'MAX', 'MEDIAN']
    stats['MALEZA'] = stats.apply(lambda row: config['maleza_threshold_formula'](row['MEAN'], row['STD']), axis=1)
    
    parcelas_gdf = parcelas_gdf.merge(stats[[id_field, 'MALEZA']], on=id_field, how='left')
    
    # 2. Filtrar parcelas con maleza válida
    print(f"   [2/9] Filtrando parcelas con maleza (0 < MALEZA < 1)...")
    parcelas_maleza = parcelas_gdf[
        (parcelas_gdf['MALEZA'].notna()) &
        (parcelas_gdf['MALEZA'] > config['maleza_min']) &
        (parcelas_gdf['MALEZA'] < config['maleza_max'])
    ].copy()
    
    if len(parcelas_maleza) == 0:
        print(f"        ⚠ No se encontraron parcelas con maleza")
        parcelas_gdf['area_maleza'] = 0
        parcelas_gdf['area_parcela'] = parcelas_gdf.get('area_calculada', 0)
        parcelas_gdf['percent_maleza'] = 0
        parcelas_gdf['status_maleza'] = "Sin Maleza"
        parcelas_gdf['etapa_f'] = parcelas_gdf['edad'].apply(calcular_etapa_fenologica)
        return parcelas_gdf[['id_parcela' if 'id_parcela' in parcelas_gdf.columns else id_field, 'zafra', 'fecha_img', 'edad', 'area_maleza', 'area_parcela', 'percent_maleza', 'etapa_f', 'status_maleza']]
    
    print(f"        ✓ {len(parcelas_maleza)} parcelas con maleza detectada")
    
    # 3. Rasterizar umbral de maleza (SIN buffer)
    print(f"   [3/9] Rasterizando umbral de maleza por parcela...")
    maleza_raster_path = os.path.join(output_dir, 'maleza_raster_temp.tif')
    # Usar parcelas SIN buffer - cada parcela con su propio umbral (MEAN + 1.4*STD)
    polygon_to_raster(parcelas_maleza, 'MALEZA', maleza_raster_path, reference_raster=ndvi_raster_path, all_touched=False)
    
    # 4. Restar NDVI - Umbral: negativo = maleza (bajo vigor)
    print(f"   [4/9] Calculando NDVI - Umbral (negativo = maleza)...")
    ndvi_minus_maleza_path = os.path.join(output_dir, 'ndvi_minus_maleza_temp.tif')
    raster_calculator(ndvi_raster_path, maleza_raster_path, ndvi_minus_maleza_path, 'subtract')
    
    # 5. Reclasificar: >0 = maleza (1), ≤0 = no maleza (2)
    print(f"   [5/9] Reclasificando (1=Maleza >0, 2=No maleza ≤0)...")
    reclass_path = os.path.join(output_dir, 'weed_reclass_temp.tif')
    reclassify_raster(ndvi_minus_maleza_path, config['reclass_ranges'], reclass_path)
    
    # 6. Convertir a polígonos: SOLO gridcode=1 (maleza)
    print(f"   [6/9] Calculando áreas y % de maleza...")
    polygons = raster_to_polygons(reclass_path)

    # === CRITICAL CRS FIX ===
    # We align the raster polygons (UTM) to the parcel CRS (WGS84).
    if polygons.crs != parcelas_gdf.crs:
        polygons = polygons.to_crs(parcelas_gdf.crs)
    # ========================

    polygons_maleza = polygons[polygons['gridcode'] == 1].copy()  # Solo maleza (gridcode=1)
    
    if len(polygons_maleza) == 0:
        print(f"        ⚠ No se detectaron áreas con maleza después de reclasificación")
        parcelas_gdf['area_maleza'] = 0
        parcelas_gdf['percent_maleza'] = 0
        parcelas_gdf['status_maleza'] = "Sin Maleza"
        parcelas_gdf['etapa_f'] = parcelas_gdf['edad'].apply(calcular_etapa_fenologica)
    else:
        clipped = gpd.overlay(polygons_maleza, parcelas_gdf[[id_field, 'geometry', 'area_calculada']], how='intersection')
        single = clipped.explode(index_parts=False).reset_index(drop=True)
        
        crs_utm = meta['CRS']
        single_proj = single.to_crs(crs_utm)
        single_proj['area_ha'] = single_proj.geometry.area / 10000
        
        maleza_por_parcela = single_proj.groupby(id_field).agg({'area_ha': 'sum'}).reset_index()
        maleza_por_parcela.columns = [id_field, 'area_maleza']
        
        parcelas_gdf = parcelas_gdf.merge(maleza_por_parcela, on=id_field, how='left')
        parcelas_gdf['area_maleza'] = parcelas_gdf['area_maleza'].fillna(0)
        parcelas_gdf['area_parcela'] = parcelas_gdf.get('area_calculada', 0)
        parcelas_gdf['percent_maleza'] = np.where(
            parcelas_gdf['area_parcela'] > 0,
            (parcelas_gdf['area_maleza'] / parcelas_gdf['area_parcela'] * 100).round(2),
            0
        )
        parcelas_gdf['status_maleza'] = np.where(
            parcelas_gdf['percent_maleza'] >= config['maleza_critica_percent'],
            "Critica", "Evaluar"
        )
        parcelas_gdf['etapa_f'] = parcelas_gdf['edad'].apply(calcular_etapa_fenologica)
    
    # 7. Recortar y guardar raster final de malezas
    print(f"   [7/9] Recortando y guardando raster TIFF de malezas...")
    weed_temp_path = os.path.join(output_dir, 'weed_temp_full.tif')
    weed_output_path = os.path.join(output_dir, archivos['extracted'])
    
    # Primero copiar a temporal
    import shutil
    shutil.copy2(reclass_path, weed_temp_path)
    
    # Recortar con los polígonos de las parcelas
    clip_raster_with_polygons(weed_temp_path, parcelas_gdf, weed_output_path, all_touched=False)
    
    # Eliminar temporal
    if os.path.exists(weed_temp_path):
        os.remove(weed_temp_path)
    
    print(f"        ✓ Raster WEED guardado: {archivos['extracted']}")
    print(f"        ✓ Valores: 1=Maleza (NDVI>umbral, exceso vigor), 2=No maleza (NDVI≤umbral)")
    print(f"        ✓ Recortado al extent de las parcelas")
    
    # 7b. Guardar raster de diferencia (NDVI - Umbral) sin reclasificar
    print(f"   [7b/9] Guardando raster de diferencia (NDVI - Umbral)...")
    diff_output_name = f"{pais}_{empresa}_WEED_DIFF_{ingenio}_{fecha.replace('-', '_')}.tif"
    diff_output_path = os.path.join(output_dir, diff_output_name)
    diff_temp_path = os.path.join(output_dir, 'weed_diff_temp.tif')
    
    # Copiar y recortar raster de diferencia
    shutil.copy2(ndvi_minus_maleza_path, diff_temp_path)
    clip_raster_with_polygons(diff_temp_path, parcelas_gdf, diff_output_path, all_touched=False)
    
    if os.path.exists(diff_temp_path):
        os.remove(diff_temp_path)
    
    print(f"        ✓ Raster DIFF guardado: {diff_output_name}")
    print(f"        ✓ Valores continuos: >0 = NDVI excede umbral, ≤0 = NDVI bajo/normal")
    
    # 7c. Guardar raster de umbrales por parcela
    print(f"   [7c/9] Guardando raster de umbrales por parcela...")
    threshold_output_name = f"{pais}_{empresa}_WEED_THRESHOLD_{ingenio}_{fecha.replace('-', '_')}.tif"
    threshold_output_path = os.path.join(output_dir, threshold_output_name)
    threshold_temp_path = os.path.join(output_dir, 'weed_threshold_temp.tif')
    
    # Copiar y recortar raster de umbrales
    shutil.copy2(maleza_raster_path, threshold_temp_path)
    clip_raster_with_polygons(threshold_temp_path, parcelas_gdf, threshold_output_path, all_touched=False)
    
    if os.path.exists(threshold_temp_path):
        os.remove(threshold_temp_path)
    
    print(f"        ✓ Raster THRESHOLD guardado: {threshold_output_name}")
    print(f"        ✓ Cada parcela tiene su umbral (MEAN + 1.5*STD)")
    
    # 8. Generar raster de temporalidad (overlapping)
    print(f"   [8/9] Generando raster de temporalidad (overlapping)...")
    
    # Guardar detección actual en histórico
    deteccion_actual = clipped.copy() if len(clipped) > 0 else gpd.GeoDataFrame()
    if len(deteccion_actual) > 0:
        deteccion_actual['fecha_deteccion'] = fecha_dt
        # zafra ya viene en los datos, no asignar
        
        # Cargar o crear histórico
        if os.path.exists(historico_path):
            historico = gpd.read_file(historico_path)
            
            # ELIMINAR fecha actual si ya existe (evitar duplicados)
            historico = historico[historico['fecha_deteccion'] != fecha_dt]
            
            # Mantener solo últimas 5 fechas únicas
            fechas_unicas = sorted(historico['fecha_deteccion'].unique(), reverse=True)
            if len(fechas_unicas) >= 5:
                fechas_mantener = fechas_unicas[:4]  # Últimas 4 + la actual = 5
                historico = historico[historico['fecha_deteccion'].isin(fechas_mantener)]
            
            # Append nueva detección
            historico = pd.concat([historico, deteccion_actual], ignore_index=True)
        else:
            historico = deteccion_actual
        
        # Guardar histórico actualizado
        historico.to_file(historico_path, driver='GeoJSON')
        
        # Filtrar solo última fecha para clip
        ultima_fecha = historico['fecha_deteccion'].max()
        deteccion_ultima = historico[historico['fecha_deteccion'] == ultima_fecha].copy()
        
        # Clip histórico con última detección
        if len(deteccion_ultima) > 0:
            historico_clipped = gpd.overlay(historico, deteccion_ultima[['geometry']], how='intersection')
            
            # Count overlapping (groupby geometría)
            from shapely.ops import unary_union
            overlapping = historico_clipped.copy()
            overlapping['geometry_wkt'] = overlapping.geometry.apply(lambda g: g.wkt)
            overlapping_count = overlapping.groupby('geometry_wkt').size().reset_index(name='COUNT_')
            overlapping_count['geometry'] = overlapping_count['geometry_wkt'].apply(lambda wkt: gpd.GeoSeries.from_wkt([wkt])[0])
            overlapping_gdf = gpd.GeoDataFrame(overlapping_count, geometry='geometry', crs=historico.crs)
            
            # Limitar a máximo 5
            overlapping_gdf['COUNT_'] = overlapping_gdf['COUNT_'].clip(upper=5)
            
            # Generar raster de temporalidad (temporal completo)
            temporalidad_temp_path = os.path.join(output_dir, 'temporalidad_temp_full.tif')
            temporalidad_raster_path = os.path.join(output_dir, archivos['extracted'].replace('_extracted.tif', '_temporalidad.tif'))
            polygon_to_raster(overlapping_gdf, 'COUNT_', temporalidad_temp_path, reference_raster=ndvi_raster_path, all_touched=False)
            
            # Recortar raster de temporalidad con parcelas
            clip_raster_with_polygons(temporalidad_temp_path, parcelas_gdf, temporalidad_raster_path, all_touched=False)
            
            # Eliminar temporal
            if os.path.exists(temporalidad_temp_path):
                os.remove(temporalidad_temp_path)
            
            print(f"        ✓ Raster temporalidad guardado: {os.path.basename(temporalidad_raster_path)}")
            print(f"        ✓ Rango temporalidad: 1-{int(overlapping_gdf['COUNT_'].max())} detecciones")
            print(f"        ✓ Lógica: 1ª imagen=1, 2ª imagen=1-2, 3ª=1-2-3, hasta máx 5")
            print(f"        ✓ Recortado al extent de las parcelas")
        else:
            print(f"        ⚠ No hay detecciones en última fecha para overlapping")
    else:
        print(f"        ⚠ No hay detecciones para guardar en histórico")
    
    # 9. Limpiar archivos temporales
    print(f"   [9/9] Limpiando archivos temporales...")
    for temp_file in [maleza_raster_path, ndvi_minus_maleza_path, reclass_path, 
                      os.path.join(output_dir, 'weed_temp_full.tif'),
                      os.path.join(output_dir, 'temporalidad_temp_full.tif')]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print(f"\n        ✓ Parcelas con maleza detectada: {len(parcelas_gdf[parcelas_gdf['area_maleza'] > 0])}")
    print(f"        ✓ Parcelas críticas (≥15%): {len(parcelas_gdf[parcelas_gdf.get('status_maleza', '') == 'Critica'])}")
    
    # Renombrar id_field a 'id_parcela' para salida estandarizada
    # ADDED: ingenio_id, company_id
    cols_weed = [id_field, 'zafra', 'fecha_img', 'edad', 'area_maleza', 'area_parcela', 'percent_maleza', 'etapa_f', 'status_maleza', 'ingenio_id', 'company_id', 'ingenio', 'company']
    # Ensure columns exist before selecting (safety check)
    cols_weed = [c for c in cols_weed if c in parcelas_gdf.columns]
    
    result_gdf = parcelas_gdf[cols_weed].copy()
    result_gdf.rename(columns={id_field: 'id_parcela'}, inplace=True)
    return result_gdf
    
    # # Renombrar id_field a 'id_parcela' para salida estandarizada
    # result_gdf = parcelas_gdf[[id_field, 'zafra', 'fecha_img', 'edad', 'area_maleza', 'area_parcela', 'percent_maleza', 'etapa_f', 'status_maleza']].copy()
    # result_gdf.rename(columns={id_field: 'id_parcela'}, inplace=True)
    # return result_gdf

# =======================================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO
# =======================================================

def procesar_producto(ingenio, fecha, producto, input_dir, output_dir, zafras=None, bd_insert=False, id_field='Clave_area', parcelas_geojson_path=None):
    """
    Función principal que procesa un producto (NDVI, NDWI, Smart Growth o Weed)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if producto not in PRODUCTOS_CONFIG:
        raise ValueError(f"Producto '{producto}' no válido. Opciones: {list(PRODUCTOS_CONFIG.keys())}")
    
    config = PRODUCTOS_CONFIG[producto]
    
    # Normalizar nombre de ingenio
    ingenio_key = ingenio.replace(' ', '_')
    meta = get_ingenio_meta(ingenio_key)
    pais, empresa, crs_utm = meta['PAIS'], meta['EMPRESA'], meta['CRS']
    
    fecha_dt = datetime.strptime(fecha, "%Y-%m-%d")
    fecha_str = fecha_dt.strftime("%Y_%m_%d")
    archivos = generar_nombres_archivos(pais, empresa, ingenio_key, fecha_str, producto)
    
    # Auto-detectar zafras
    if zafras is None and not parcelas_geojson_path:
        zafras = obtener_zafras_activas(ingenio)
    elif zafras is None:
        zafras = [2025]
    
    if not isinstance(zafras, list):
        zafras = [zafras]
    
    print(f"\n{'='*70}")
    print(f"  Procesando {producto} - {ingenio} ({pais}/{empresa})")
    print(f"  Fecha: {fecha} | Zafras: {zafras}")
    print(f"{'='*70}\n")
    
    # [1/9] Cargar parcelas
    print("\n[1/9] Cargando parcelas...")
    if parcelas_geojson_path:
        print(f"        → Fuente: GeoJSON local ({parcelas_geojson_path})")
        parcelas_gdf = gpd.read_file(parcelas_geojson_path)
        if 'zafra' not in parcelas_gdf.columns:
            parcelas_gdf['zafra'] = zafras[0]
        print(f"        ✓ {len(parcelas_gdf)} parcelas cargadas")
    else:
        print(f"        → Fuente: Supabase (public.parcelas_ingenios_reprocess)")
        parcelas_gdf = cargar_parcelas_desde_supabase(ingenio)

        # ==============================================================================
        # LOGIC: SAVE IDS WITH MISSING DATES TO JSON
        # ==============================================================================
        # Detect missing dates BEFORE filtering
        missing_dates_mask = parcelas_gdf['fecha_inicio'].isna()
        missing_count = missing_dates_mask.sum()
        
        if missing_count > 0:
            print(f"        ⚠ Found {missing_count} parcels with missing 'fecha_inicio'. saving log...")
            
            # Extract IDs
            missing_ids = parcelas_gdf.loc[missing_dates_mask, id_field].tolist()
            
            # Define JSON filename
            json_filename = f"{pais}_{empresa}_MISSING_DATES_{ingenio}_{fecha_str}.json"
            json_path = os.path.join(output_dir, json_filename)
            
            # Save to JSON
            log_data = {
                "ingenio": ingenio,
                "date": fecha,
                "total_missing": int(missing_count),
                "parcel_ids": missing_ids
            }
            
            with open(json_path, 'w') as f:
                json.dump(log_data, f, indent=4)
                
            print(f"        📝 Log saved: {json_filename}")
        # ==============================================================================
        
        # ==============================================================================
        # <--- CORRECCIÓN 1: FILTRO FECHA_INICIO (Eliminar Nulos)
        # ==============================================================================
        total_encontradas = len(parcelas_gdf)
        print(f"        ✓ {total_encontradas} parcelas cargadas desde BD")
        
        # Filtrar donde fecha_inicio NO es NaT/None/NaN
        parcelas_gdf = parcelas_gdf[parcelas_gdf['fecha_inicio'].notna()].copy()
        total_validas = len(parcelas_gdf)
        eliminadas = total_encontradas - total_validas
        
        print(f"        ✓ Filtrado por fecha_inicio: {total_validas} válidas")
        if eliminadas > 0:
            print(f"        ⚠ ATENCIÓN: Se eliminaron {eliminadas} parcelas por tener 'fecha_inicio' NULA")
        
        if total_validas == 0:
            print("        ❌ Error: No quedan parcelas válidas después del filtro de fechas.")
            return None
        # ==============================================================================

        zafras_cargadas = sorted(parcelas_gdf['zafra'].unique())
        print(f"        ✓ Distribución por zafra: {dict(parcelas_gdf['zafra'].value_counts())}")
    
    # PROCESAMIENTO SEGÚN TIPO
    if config['tipo'] == 'combinado':
        # SMART GROWTH
        print("\n[2/9] Smart Growth requiere NDVI y NDWI procesados previamente...")
        resultado = procesar_smart_growth(ingenio, fecha, parcelas_gdf, output_dir, id_field, meta, archivos)
        columns_to_save = [id_field, 'zafra', 'fecha_img', 'edad', 'sg_mean', 'sg_stdv', 'sg_min', 'sg_max', 'sg_median', 'ingenio_id', 'company_id','ingenio', 'company']
        
    elif config['tipo'] == 'complejo':
        # WEED
        input_raster_path = os.path.join(input_dir, archivos['input_raster'])
        if not os.path.exists(input_raster_path):
            raise FileNotFoundError(f"Archivo no encontrado: {input_raster_path}")
        
        print(f"📂 Input: {archivos['input_raster']}")
        
        # Procesar fechas y edad
        print("\n[2/9] Procesando fechas y edad...")
        parcelas_gdf['fecha_inicio_converted'] = pd.to_datetime(parcelas_gdf['fecha_inicio'], format='%Y-%m-%d', errors='coerce')
        parcelas_gdf['fecha_img'] = fecha_dt
        parcelas_gdf['edad'] = parcelas_gdf.apply(
            lambda row: calculate_days(row['fecha_img'], row['fecha_inicio_converted'])
            if pd.notna(row['fecha_inicio_converted']) else None, axis=1
        )
        
        resultado = procesar_weed(ingenio, fecha, parcelas_gdf, input_raster_path, output_dir, id_field, meta, archivos, config)
        columns_to_save = [id_field, 'zafra', 'fecha_img', 'edad', 'area_maleza', 'area_parcela', 'percent_maleza', 'etapa_f', 'status_maleza', 'ingenio_id', 'company_id','ingenio', 'company']
        
    else:
        # SIMPLE (NDVI, NDWI)
        input_raster_path = os.path.join(input_dir, archivos['input_raster'])
        if not os.path.exists(input_raster_path):
            raise FileNotFoundError(f"Archivo no encontrado: {input_raster_path}")
        
        print(f"📂 Input: {archivos['input_raster']}")
        
        # Filtrar parcelas que intersectan con el raster
        print("\n[2/9] Filtrando parcelas dentro del área del raster...")
        with rasterio.open(input_raster_path) as src:
            raster_crs = src.crs
            raster_bounds = src.bounds
            raster_bbox_wgs84 = gpd.GeoSeries([box(*raster_bounds)], crs=raster_crs).to_crs("EPSG:4326").iloc[0]
            
            parcelas_in_raster = parcelas_gdf[parcelas_gdf.intersects(raster_bbox_wgs84)].copy()
            
            if len(parcelas_in_raster) == 0:
                print("        ⚠ Still zero intersection after reprojection – check raster/path/parcels")
                return None
            
            print(f"        ✓ {len(parcelas_in_raster)} parcelas intersectan el raster (de {len(parcelas_gdf)} totales)")
            parcelas_gdf = parcelas_in_raster
        
        # Procesar fechas, edad y potencial
        print("\n[3/9] Procesando fechas y edad...")
        parcelas_gdf['fecha_inicio_converted'] = pd.to_datetime(parcelas_gdf['fecha_inicio'], format='%Y-%m-%d', errors='coerce')
        parcelas_gdf['fecha_img'] = fecha_dt
        parcelas_gdf['edad'] = parcelas_gdf.apply(
            lambda row: calculate_days(row['fecha_img'], row['fecha_inicio_converted'])
            if pd.notna(row['fecha_inicio_converted']) else None, axis=1
        )
        
        print("\n[3/9] Calculando potencial...")
        print(f"        → Usando curva potencial para {ingenio}")
        parcelas_gdf['potencial'] = parcelas_gdf['edad'].apply(
            lambda x: config['potencial_formula'](x, ingenio) if pd.notna(x) else None
        )
        
        # Estadisticas zonales
        print(f"\n[4/9] Calculando estadísticas zonales de {producto}...")
        stats = zonal_statistics(input_raster_path, parcelas_gdf, id_field)
        
        if len(stats) == 0:
            print(f"        ⚠ No se pudieron calcular estadísticas.")
            return None
        
        prefix = producto.lower()
        stats.columns = [id_field, f'{prefix}_mean', f'{prefix}_stdv', f'{prefix}_min', f'{prefix}_max', f'{prefix}_median']
        parcelas_gdf = parcelas_gdf.merge(stats, on=id_field, how='left')
        print(f"        ✓ Estadísticas calculadas para {len(stats)} parcelas")
        
        # Buffer y extraer
        print(f"\n[5/9] Creando buffer y extrayendo {producto}...")
        parcelas_buffer = buffer_geometry(parcelas_gdf, 20)
        extracted_temp_path = os.path.join(output_dir, f'{producto.lower()}_extracted_temp.tif')
        extract_by_mask(input_raster_path, parcelas_buffer, extracted_temp_path)
        
        # Recortar a parcelas (sin buffer)
        extracted_path = os.path.join(output_dir, archivos['extracted'])
        clip_raster_with_polygons(extracted_temp_path, parcelas_gdf, extracted_path)
        if os.path.exists(extracted_temp_path):
            os.remove(extracted_temp_path)
        print(f"        ✓ Extraído y recortado: {archivos['extracted']}")
        
        # Reclasificar
        print(f"\n[6/9] Reclasificando {producto}...")
        reclass_path = os.path.join(output_dir, f'{producto.lower()}_reclass_temp.tif')

        # FIX: Use 'extracted_path' (The raster already clipped to the parcels from Step 5)
        extracted_path = os.path.join(output_dir, archivos['extracted']) 
        reclassify_raster(extracted_path, config['reclass_ranges'], reclass_path)
        # reclassify_raster(input_raster_path, config['reclass_ranges'], reclass_path)
        
        # Convertir a polígonos y calcular áreas
        print(f"\n[7/9] Procesando distribución de {producto} por clase...")
        polygons = raster_to_polygons(reclass_path)
        
        # ==============================================================================
        # <--- CORRECCIÓN 2: CRS MATCH (Evitar Fallo Silencioso de Columnas)
        # ==============================================================================
        if polygons.crs != parcelas_gdf.crs:
            # Reproyectar polígonos del raster (UTM) al CRS de parcelas (WGS84)
            polygons = polygons.to_crs(parcelas_gdf.crs)
        # ==============================================================================

        clipped = gpd.overlay(polygons, parcelas_gdf[[id_field, 'geometry']], how='intersection')
        
        if len(clipped) == 0:
            print(f"        ⚠ WARNING: No se generaron intersecciones para la distribución.")
        else:
            single = clipped.explode(index_parts=False).reset_index(drop=True)
            single_proj = single.to_crs(crs_utm)
            single_proj['area_ha'] = single_proj.geometry.area / 10000
            
            dissolved = single_proj.groupby([id_field, 'gridcode']).agg({'area_ha': 'sum'}).reset_index()
            pivot = dissolved.pivot(index=id_field, columns='gridcode', values='area_ha').reset_index()
            pivot.columns = [id_field] + [f'{prefix}_{int(col)}' for col in pivot.columns[1:]]
            pivot = pivot.fillna(0)
            parcelas_gdf = parcelas_gdf.merge(pivot, on=id_field, how='left')
            print(f"        ✓ Distribución por clase calculada")
        
        # Procesar potencial
        print(f"\n[8/9] Procesando potencial {producto}...")
        potencial_raster_path = os.path.join(output_dir, f'potencial_{producto.lower()}_temp.tif')
        
        polygon_to_raster(parcelas_gdf[parcelas_gdf['potencial'].notna()], 'potencial',
                         potencial_raster_path, reference_raster=extracted_path) # reference_raster=input_raster_path
        
        ratio_path = os.path.join(output_dir, f'ratio_{producto.lower()}_temp.tif')
        raster_calculator(extracted_path, potencial_raster_path, ratio_path, 'divide')

        # raster_calculator(input_raster_path, potencial_raster_path, ratio_path, 'divide')
        
        potencial_reclass_temp = os.path.join(output_dir, f'potencial_{producto.lower()}_reclass_temp.tif')
        reclassify_raster(ratio_path, config['reclass_potencial'], potencial_reclass_temp)
        
        # Recortar potencial a parcelas
        potencial_final_path = os.path.join(output_dir, archivos['potencial'])
        clip_raster_with_polygons(potencial_reclass_temp, parcelas_gdf, potencial_final_path)
        if os.path.exists(potencial_reclass_temp):
            os.remove(potencial_reclass_temp)
        
        pot_polygons = raster_to_polygons(potencial_final_path)
        
        # ==============================================================================
        # <--- CORRECCIÓN 3: CRS MATCH (Evitar Fallo Silencioso de Potencial)
        # ==============================================================================
        if pot_polygons.crs != parcelas_gdf.crs:
            pot_polygons = pot_polygons.to_crs(parcelas_gdf.crs)
        # ==============================================================================

        pot_clipped = gpd.overlay(pot_polygons, parcelas_gdf[[id_field, 'geometry']], how='intersection')
        
        if len(pot_clipped) > 0:
            pot_single = pot_clipped.explode(index_parts=False).reset_index(drop=True)
            pot_single_proj = pot_single.to_crs(crs_utm)
            pot_single_proj['area_ha'] = pot_single_proj.geometry.area / 10000
            
            pot_dissolved = pot_single_proj.groupby([id_field, 'gridcode']).agg({'area_ha': 'sum'}).reset_index()
            pot_pivot = pot_dissolved.pivot(index=id_field, columns='gridcode', values='area_ha').reset_index()
            pot_pivot.columns = [id_field] + [f'{prefix}_pot_{int(col)}' for col in pot_pivot.columns[1:]]
            pot_pivot = pot_pivot.fillna(0)
            parcelas_gdf = parcelas_gdf.merge(pot_pivot, on=id_field, how='left')
        
        print(f"        ✓ Potencial {producto}: {archivos['potencial']}")
        
        # Limpiar temporales
        for temp_file in [potencial_raster_path, ratio_path, reclass_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # Calcular cosechado
        if config['cosechado_classes']:
            cosechado_cols = [f'{prefix}_{c}' for c in config['cosechado_classes'] if f'{prefix}_{c}' in parcelas_gdf.columns]
            if cosechado_cols:
                parcelas_gdf['cosechado'] = parcelas_gdf[cosechado_cols].sum(axis=1, min_count=1)
            else:
                parcelas_gdf['cosechado'] = 0
            
            if 'area_calculada' in parcelas_gdf.columns:
                parcelas_gdf['cultivo_en_pie'] = parcelas_gdf['area_calculada'] - parcelas_gdf['cosechado']
            else:
                parcelas_gdf['cultivo_en_pie'] = 0
        
        resultado = parcelas_gdf
        
        # Seleccionar columnas
        columns_to_save = [id_field, 'zafra', 'fecha_img', 'edad','ingenio', 'company', 'ingenio_id', 'company_id']
        image_cols = [col for col in parcelas_gdf.columns if any([
            col.startswith(f'{prefix}_'),
            col in [f'{prefix}_mean', f'{prefix}_stdv', f'{prefix}_min', f'{prefix}_max', f'{prefix}_median', 'potencial']
        ])]
        columns_to_save.extend(image_cols)
    
    # Guardar resultado final
    print(f"\n[9/9] Guardando resultado final...")
    output_parquet = os.path.join(output_dir, archivos['data_output'])
    
    columns_to_save = [col for col in columns_to_save if col in resultado.columns]
    df_output = resultado[columns_to_save].copy()
    
    if id_field != 'id_parcela':
        df_output = df_output.rename(columns={id_field: 'id_parcela'})
    
    df_output.to_parquet(output_parquet, index=False)
    
    print(f"        ✓ Datos guardados: {archivos['data_output']}")
    
    if bd_insert:
        print(f"\n[SUPABASE] Insertando datos a base de datos...")
        try:
            insertar_a_supabase(df_output, producto, ingenio, fecha)
        except Exception as e:
            print(f"        ⚠ Error en inserción a Supabase: {str(e)}")
            print(f"        → Los datos están guardados en {archivos['data_output']}")
    
    print(f"\n{'='*70}")
    print(f"  ✅ {producto} COMPLETADO")
    print(f"{'='*70}")
    print(f"\n📊 Resumen:")
    print(f"   - Parcelas procesadas: {len(df_output)}")
    if 'zafra' in df_output.columns:
        print(f"   - Distribución por zafra: {dict(df_output['zafra'].value_counts())}")
    print(f"   - Columnas generadas: {len(df_output.columns)}")
    
    return df_output


# =======================================================
# FUNCIÓN PARA PROCESAR MÚLTIPLES PRODUCTOS
# =======================================================

def procesar_todos_productos(ingenio, fecha, productos, input_dir, output_dir, zafras=None, bd_insert=False, id_field='id_parcela', parcelas_geojson_path=None):
    """
    Procesa múltiples productos para un ingenio y fecha
    Consolida todas las zafras activas en un solo archivo de salida por producto
    
    Args:
        zafras: Lista de zafras a procesar o None para auto-detectar desde BD (temporada_activa=True)
    """
    # -----------------------------------------------------------
    # SETUP: DEFINIR VARIABLES DE METADATA PARA NOMBRES DE ARCHIVO
    # -----------------------------------------------------------
    # Necesario para reconstruir el nombre del archivo en la auditoría
    ingenio_key = ingenio.replace(' ', '_')
    try:
        meta = get_ingenio_meta(ingenio_key)
        pais = meta['PAIS']
        empresa = meta['EMPRESA']
        fecha_dt = datetime.strptime(fecha, "%Y-%m-%d")
        fecha_str = fecha_dt.strftime("%Y_%m_%d")
    except Exception as e:
        print(f"⚠ Warning: No se pudo cargar metadata para generar nombres de archivo: {e}")
        pais, empresa, fecha_str = "UNK", "UNK", fecha.replace('-', '_')
    # -----------------------------------------------------------

    resultados = {}
    
    # Auto-detectar zafras activas si no se especifican
    if zafras is None and not parcelas_geojson_path:
        print(f"\n{'~'*80}")
        print(f"  DETECCIÓN AUTOMÁTICA DE ZAFRAS ACTIVAS")
        print(f"{'~'*80}")
        zafras = obtener_zafras_activas(ingenio)
        print(f"\n  → Se procesarán {len(zafras)} zafra(s): {zafras}")
        print(f"  → Salida: 1 archivo por producto (consolidado)\n")
    elif zafras is None:
        # Modo GeoJSON local: usar zafra por defecto
        zafras = [2025]
        print(f"\n  ⚠ Modo GeoJSON local: usando zafra por defecto {zafras[0]}\n")
    
    # Asegurar que zafras sea lista
    if not isinstance(zafras, list):
        zafras = [zafras]

    # === INSERT THIS BLOCK HERE ===
    # Cargar curva específica para este ingenio antes de procesar
    cargar_curva_dinamica(ingenio)
    # ==============================
    
    print(f"\n{'#'*80}")
    print(f"# PROCESAMIENTO DE PRODUCTOS FINALES")
    print(f"# Ingenio: {ingenio}")
    print(f"# Fecha: {fecha}")
    print(f"# Zafras activas: {', '.join(map(str, zafras))}")
    print(f"# Productos: {', '.join(productos)}")
    print(f"# Fuente parcelas: {'Supabase (auto)' if not parcelas_geojson_path else 'GeoJSON local'}")
    print(f"# Insertar a BD: {'SÍ' if bd_insert else 'NO'}")
    print(f"{'#'*80}\n")
    
    # Procesar cada producto (sin iterar por zafra - se procesan todas juntas)
    for i, producto in enumerate(productos, 1):
        print(f"\n{'~'*80}")
        print(f"  [{i}/{len(productos)}] Iniciando {producto}...")
        print(f"{'~'*80}")
        
        try:
            resultado = procesar_producto(
                ingenio=ingenio, fecha=fecha, producto=producto,
                input_dir=input_dir, output_dir=output_dir, 
                zafras=zafras, bd_insert=bd_insert, id_field=id_field,
                parcelas_geojson_path=parcelas_geojson_path
            )

            # --- AUDITORÍA DE ARCHIVOS GENERADOS ---
            if resultado is not None:
                # Reconstruir el nombre exacto del archivo parquet
                # Nota: Usamos ingenio_key (con guiones bajos) para coincidir con la convención
                filename = f"{pais}_{empresa}_DATA_{producto}_{ingenio_key}_{fecha_str}.parquet"
                output_parquet = os.path.join(output_dir, filename)
                
                # Ejecutar auditoría
                audit_parquet_file(output_parquet, producto)
            # ---------------------------------------
            
            resultados[producto] = resultado
            
            if resultado is not None:
                total_parcelas = len(resultado)
                print(f"\n✅ {producto} completado - TOTAL: {total_parcelas} parcelas\n")
            else:
                print(f"\n⚠ {producto} - Sin resultados\n")

            # =======================================================
            #  GARBAGE COLLECTION ADDITION (Newly added for efficiency)
            # =======================================================
            # Delete the large result variable from memory (we stored it in the 'resultados' dict anyway)
            del resultado 
            
            # Force Python to clean up RAM immediately
            gc.collect()
            print(f"      🧹 Memory cleaned after {producto}")
            # =======================================================
                
        except Exception as e:
            print(f"\n❌ Error procesando {producto}: {str(e)}\n")
            import traceback
            traceback.print_exc()
            resultados[producto] = None
            
            # Also clean up if it crashes, so the next product has a fresh start
            gc.collect()
    
    print(f"\n{'#'*80}")
    print(f"# RESUMEN FINAL - CONSOLIDADO")
    print(f"{'#'*80}")
    print(f"# Zafras procesadas: {', '.join(map(str, zafras))}")
    print(f"# Salida: 1 archivo por producto (todas las zafras combinadas)")
    print(f"{'#'*80}")
    for producto, resultado in resultados.items():
        if resultado is not None:
            zafras_en_resultado = resultado['zafra'].unique() if 'zafra' in resultado.columns else []
            print(f"  ✅ {producto:20} - {len(resultado)} parcelas (zafras: {list(zafras_en_resultado)})")
        else:
            print(f"  ❌ {producto:20} - Error en procesamiento")
    print(f"{'#'*80}\n")
    
    return resultados

def audit_parquet_file(file_path, expected_product):
    """
    Audits the generated Parquet file.
    UPDATED: Smart detection of product type to avoid false alarms on SG/Weed.
    """
    if not os.path.exists(file_path):
        print(f"❌ CRITICAL: Output file not found: {file_path}")
        return

    df = pd.read_parquet(file_path)
    total_rows = len(df)
    
    print(f"\n🕵️ STARTING AUDIT: {os.path.basename(file_path)}")
    print(f"   Total Rows: {total_rows}")

    # ----------------------------------------------------
    # CHECK 1: NULL VALUES (Data Quality - Applies to ALL)
    # ----------------------------------------------------
    if 'edad' in df.columns:
        missing_age = df['edad'].isna().sum()
        if missing_age > 0:
            print(f"   ⚠️ WARNING: {missing_age} parcels ({missing_age/total_rows:.1%}) have NaN 'edad'.")
        else:
            print(f"   ✅ PASS: All parcels have valid Age.")

    # ----------------------------------------------------
    # CHECK 2: PRODUCT-SPECIFIC COLUMNS
    # ----------------------------------------------------
    product_key = expected_product.upper()
    
    # CASE A: SIMPLE PRODUCTS (NDVI, NDWI) -> Check for Distribution Classes (_1, _2...)
    if product_key in ["NDVI", "NDWI"]:
        prefix = product_key.lower()
        # We expect at least the first few classes to exist (e.g., ndvi_1, ndvi_2)
        expected_cols = [f'{prefix}_{i}' for i in range(1, 3)] 
        missing_cols = [c for c in expected_cols if c not in df.columns]
        
        if missing_cols:
            print(f"   ❌ FAIL: Distribution columns (Histogram) are MISSING!")
            print(f"      Expected {expected_cols}, but not found.")
        else:
            print(f"   ✅ PASS: Distribution columns present (Histogram data ok).")
            
        # Check Mean
        mean_col = f'{prefix}_mean'
        if mean_col in df.columns and df[mean_col].isna().sum() > 0:
             print(f"   ❌ FAIL: Some parcels have NaN '{mean_col}'.")
        else:
             print(f"   ✅ PASS: Mean values are valid.")

    # CASE B: SMART GROWTH -> Check for Statistics (sg_mean)
    elif product_key == "SMART_GROWTH":
        if 'sg_mean' not in df.columns:
            print(f"   ❌ FAIL: 'sg_mean' column is MISSING!")
        else:
            # Check if values are within expected range (20-100)
            # Ignoring 0s which might be potential issues, looking for real data
            valid_sg = df[df['sg_mean'] > 0]['sg_mean']
            if len(valid_sg) > 0:
                print(f"   ✅ PASS: Smart Growth stats present. Range: {valid_sg.min():.1f} - {valid_sg.max():.1f}")
            else:
                print(f"   ⚠️ WARNING: Smart Growth columns exist but values seem empty/zero.")

    # CASE C: WEED -> Check for Weed Specifics (percent_maleza)
    elif product_key == "WEED":
        required = ['area_maleza', 'percent_maleza', 'status_maleza']
        missing = [c for c in required if c not in df.columns]
        
        if missing:
            print(f"   ❌ FAIL: Missing Weed columns: {missing}")
        else:
            critica = df[df['status_maleza'] == 'Critica'].shape[0]
            print(f"   ✅ PASS: Weed data present. Found {critica} critical parcels.")

    print(f"👮 AUDIT COMPLETE\n")

# =======================================================
# USO
# =======================================================
# ============================================
# SUPABASE CONFIGURATION
# ============================================

# ============================================
# MAIN PARAMETERS
# ============================================
def run() -> dict:
    """Execute Excel sync + product processing for all dates in FECHAS."""
    # Inline the original __main__ block
    INGENIO    = INGENIO_ENV
    FECHAS     = FECHAS_ENV
    EXCEL_PATH = cfg.EXCEL_PATH
    PRODUCTOS  = cfg.PRODUCTOS
    INPUT_DIR  = INPUT_DIR_ENV
    OUTPUT_DIR = OUTPUT_DIR_ENV
    ID_FIELD   = "id_parcela"
    PARCELAS_GEOJSON = None
    ZAFRAS     = None
    BD_INSERT  = cfg.BD_INSERT
    DRY_RUN_SYNC = cfg.DRY_RUN_SYNC

    print("\n" + "="*80)
    print("  STARTING FULL PIPELINE WITH INCREMENTAL SYNC")
    print("="*80)

    all_results = procesar_con_sync(
        excel_path=EXCEL_PATH,
        ingenio=INGENIO,
        fechas=FECHAS,
        productos=PRODUCTOS,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        bd_insert=BD_INSERT,
        dry_run_sync=DRY_RUN_SYNC,
    )

    print("\n🎉 Processing completed!")
    return all_results


if __name__ == "__main__":
    print(cfg.summary())
    run()

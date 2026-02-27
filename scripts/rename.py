"""
scripts/04_rename.py
====================
Step 4 - Rename and move NDRE/NDWI cloudfill TIFs to INPUT_DIR.

Naming convention:
    PAIS_EMPRESA_INDEX_INGENIO_YYYY_MM_DD_cloudfill.tif

Run standalone: python scripts/04_rename.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import cfg
import os
import shutil
from datetime import datetime

import os
import shutil
from pathlib import Path
from datetime import datetime

def rename_and_move_results(
    inference_folder="amajac-auto",
    output_folder="inputs-amajac-auto",
    prefix="MX06_Grupo Pantaleon",
    location="Amajac"
):
    """
    Rename and move result files from cloud_free_output folder to output folder.
    
    NEW STRUCTURE:
    inference_folder/
    ├── pairs/
    └── cloud_free_output/  <- Files are here now
        ├── cloud_free_2026-02-19.tif
        ├── ndvi_PROD_20260219.tif
        ├── ndre_PROD_20260219.tif
        └── ndwi_PROD_20260219.tif
    
    Example transformation:
    ndvi_PROD_20260219.tif -> MX06_Grupo Pantaleon_NDVI_Amajac_2026_02_19_cloudfill.tif
    """
    
    # Paths
    inference_path = Path(inference_folder)
    cloud_free_path = inference_path / "cloud_free_output"
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    print(f"Processing files from: {cloud_free_path}")
    print(f"Output folder: {output_path}\n")
    
    if not cloud_free_path.exists():
        print(f"❌ Cloud-free output folder not found: {cloud_free_path}")
        print("   Run Cell 2 (cloud removal) first!")
        return
    
    moved_count = 0
    skipped_count = 0
    
    # Process all .tif files in cloud_free_output folder
    for file in sorted(cloud_free_path.glob("*.tif")):
        filename = file.stem  # filename without extension
        
        # Skip the main cloud_free file (we only want vegetation indices)
        if filename.startswith("cloud_free"):
            print(f"⊙ Skipping cloud-free base file: {file.name}")
            continue
        
        # Extract index type and date from filename
        # Format: ndvi_PROD_20260219.tif or ndre_PROD_20260219.tif
        index_type = None
        date_str = None
        
        for idx in ["ndvi", "ndwi", "ndre", "evi", "savi"]:
            if filename.lower().startswith(idx):
                index_type = idx.upper()
                # Extract date: ndvi_PROD_20260219 -> 20260219
                parts = filename.split("_")
                if len(parts) >= 3 and parts[1].lower() == "prod":
                    date_str = parts[2]  # 20260219
                break
        
        if not index_type or not date_str:
            print(f"⊘ Skipping unknown file format: {file.name}")
            continue
        
        # Convert date: 20260219 -> 2026_02_19
        try:
            if len(date_str) == 8:  # YYYYMMDD
                date_formatted = f"{date_str[0:4]}_{date_str[4:6]}_{date_str[6:8]}"
            else:
                print(f"⊘ Invalid date format in: {file.name}")
                continue
        except:
            print(f"⊘ Could not parse date from: {file.name}")
            continue
        
        # Create new filename
        new_filename = f"{prefix}_{index_type}_{location}_{date_formatted}_cloudfill.tif"
        new_filepath = output_path / new_filename
        
        # Check if file already exists
        if new_filepath.exists():
            print(f"⊙ Already exists, skipping: {new_filename}")
            skipped_count += 1
            continue
        
        # Copy file (use shutil.move() to move instead of copy)
        shutil.copy2(file, new_filepath)
        print(f"✓ {file.name} -> {new_filename}")
        moved_count += 1
    
    print(f"\n✅ Processing complete!")
    print(f"   Files moved: {moved_count}")
    print(f"   Files skipped (already exist): {skipped_count}")
    print(f"   Output folder: {output_path}")



def run() -> None:
    """Rename and move cloudfill TIFs using config from .env."""
    rename_and_move_results(**cfg.NAME_CONFIG)


if __name__ == "__main__":
    print(cfg.summary())
    run()

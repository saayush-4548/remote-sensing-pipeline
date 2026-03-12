"""
setup_dependencies.py

Downloads all dependency files from S3 and places them in the exact
folder structure the notebook expects.

Usage:
    python setup_dependencies.py [--work-dir /path/to/workdir]

Result structure (relative to work_dir):
    ./GT01_BoundingBox_margin001.geojson          (AOI geojsons in root)
    ./EM01_BoundingBox_margin001.geojson
    ./Plantillas-new.xlsx                          (Excel in root)
    ./Output-gt/
    │   ├── GT_Grupo Pantaleon_WEED_HISTORICO_Pantaleon.geojson
    │   └── Pantaleon_curvas/
    │       └── curva_global_Pantaleon.parquet
    ./Output-emsa/
    │   ├── MX07_Grupo Pantaleon_WEED_HISTORICO_EMSA.geojson
    │   └── EMSA_curvas/
    │       └── curva_global_EMSA.parquet
    ... (same pattern for all mills)
"""

import boto3
import os
import argparse
from pathlib import Path

# ── S3 Config ───────────────────────────────────────────────────────────
S3_BUCKET = "carrier-pdfs"
S3_PREFIX = "stomasense-dependencies"

# ── Mill definitions ────────────────────────────────────────────────────
# Maps each mill to its output folder, curvas subfolder, and weed historico filename
MILLS = {
    "Pantaleon": {
        "output_dir": "Output-gt",
        "curvas_folder": "Pantaleon_curvas",
        "curva_file": "curva_global_Pantaleon.parquet",
        "weed_file": "GT_Grupo Pantaleon_WEED_HISTORICO_Pantaleon.geojson",
    },
    "EMSA": {
        "output_dir": "Output-emsa",
        "curvas_folder": "EMSA_curvas",
        "curva_file": "curva_global_EMSA.parquet",
        "weed_file": "MX07_Grupo Pantaleon_WEED_HISTORICO_EMSA.geojson",
    },
    "Monte Rosa": {
        "output_dir": "Output-monterosa",
        "curvas_folder": "Monte Rosa_curvas",
        "curva_file": "curva_global_Monte Rosa.parquet",
        "weed_file": "NI_Grupo Pantaleon_WEED_HISTORICO_Monte Rosa.geojson",
    },
    "IPSA": {
        "output_dir": "Output-ipsa",
        "curvas_folder": "IPSA_curvas",  
        "curva_file": "curva_global_IPSA.parquet",
        "weed_file": "MX02_Grupo Pantaleon_WEED_HISTORICO_IPSA.geojson",
    },
    "Amajac": {
        "output_dir": "Output-amajac",
        "curvas_folder": "Amajac_curvas",
        "curva_file": "curva_global_Amajac.parquet",
        "weed_file": "MX06_Grupo Pantaleon_WEED_HISTORICO_Amajac.geojson",
    },
}

# ── AOI GeoJSONs (go in root) ──────────────────────────────────────────
AOI_GEOJSONS = [
    "GT01_BoundingBox_margin001.geojson",
    "EM01_BoundingBox_margin001.geojson",
    "MR01_BoundingBox_margin001.geojson",
    "IP01_BoundingBox_margin001.geojson",
    "AM01_BoundingBox_margin001.geojson",
]

# ── Excel (goes in root) ───────────────────────────────────────────────
EXCEL_FILE = "Plantillas-new.xlsx"


def download_from_s3(s3_client, s3_key, local_path):
    """Download a file from S3, creating parent dirs as needed."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ⬇️  s3://{S3_BUCKET}/{s3_key} → {local_path}")
    s3_client.download_file(S3_BUCKET, s3_key, str(local_path))


def main():
    parser = argparse.ArgumentParser(description="Download dependencies from S3")
    parser.add_argument("--work-dir", default=".", help="Working directory (default: current)")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    os.chdir(work_dir)
    print(f"📂 Working directory: {work_dir}\n")

    s3 = boto3.client("s3")

    # ── 1. AOI GeoJSONs → root ──────────────────────────────────────────
    print("📍 Downloading AOI GeoJSONs...")
    for geojson in AOI_GEOJSONS:
        s3_key = f"{S3_PREFIX}/{geojson}"
        download_from_s3(s3, s3_key, geojson)

    # ── 2. Excel → root ─────────────────────────────────────────────────
    print("\n📊 Downloading Plantillas Excel...")
    download_from_s3(s3, f"{S3_PREFIX}/{EXCEL_FILE}", EXCEL_FILE)

    # ── 3. Per-mill: curva parquets + weed historico ─────────────────────
    print("\n🏭 Downloading per-mill files...")
    for mill_name, mill in MILLS.items():
        print(f"\n  [{mill_name}]")
        output_dir = mill["output_dir"]

        # Curva parquet → Output-{x}/{Mill}_curvas/curva_global_{Mill}.parquet
        curva_local = os.path.join(output_dir, mill["curvas_folder"], mill["curva_file"])
        curva_s3 = f"{S3_PREFIX}/{mill['curva_file']}"
        download_from_s3(s3, curva_s3, curva_local)

        # Weed historico → Output-{x}/{weed_file}
        weed_local = os.path.join(output_dir, mill["weed_file"])
        weed_s3 = f"{S3_PREFIX}/{mill['weed_file']}"
        download_from_s3(s3, weed_s3, weed_local)

    print(f"\n{'='*60}")
    print("✅ All dependencies downloaded and placed correctly!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
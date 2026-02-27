#!/usr/bin/env python3
"""
setup_dependencies.py
=====================
Downloads all static dependency files from S3 and places them in the
exact folder structure the pipeline expects.

Run ONCE before running the pipeline (or whenever static files are missing):
    python setup_dependencies.py [--work-dir /path/to/workdir] [--mill EMSA]

What it downloads
-----------------
Root (work_dir/):
    GT01_BoundingBox_margin001.geojson
    EM01_BoundingBox_margin001.geojson
    MR01_BoundingBox_margin001.geojson
    IP01_BoundingBox_margin001.geojson
    AM01_BoundingBox_margin001.geojson
    Plantillas-new.xlsx

Per mill (e.g. for EMSA):
    Output-emsa/
    ├── MX07_Grupo Pantaleon_WEED_HISTORICO_EMSA.geojson
    └── EMSA_curvas/
        └── curva_global_EMSA.parquet

S3 credentials are read from .env (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
or from the standard AWS environment / IAM role — whichever is present.
"""

import os
import sys
import argparse
from pathlib import Path

# ── Bootstrap config (reads .env) ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.settings import cfg, INGENIOS_META


# ── S3 coordinates (come from .env / defaults) ───────────────────────────────
S3_BUCKET = os.getenv("S3_BUCKET", "carrier-pdfs")
S3_PREFIX = os.getenv("S3_PREFIX", "stomasense-dependencies")


# ── Static file manifest ──────────────────────────────────────────────────────
# Derived from INGENIOS_META so there's a single source of truth.

AOI_GEOJSONS = [meta["AOI_GEOJSON"] for meta in INGENIOS_META.values()]
EXCEL_FILE = "Plantillas-new.xlsx"

# Build per-mill file list from INGENIOS_META + naming conventions
def _mill_files():
    """
    Returns dict: mill_key → {output_dir, curvas_folder, curva_file, weed_file}
    Derived entirely from INGENIOS_META so names stay in sync automatically.
    """
    result = {}
    for mill_key, meta in INGENIOS_META.items():
        display  = meta["DISPLAY"]        # e.g. "Monte Rosa"
        pais     = meta["PAIS"]           # e.g. "NI"
        empresa  = meta["EMPRESA"]        # e.g. "Grupo Pantaleon"

        # Output dir convention: Output-{lowercase key without _}
        # Pantaleon→Output-gt, Monte_Rosa→Output-monterosa etc.
        _output_map = {
            "Pantaleon":  "Output-gt",
            "Monte_Rosa": "Output-monterosa",
            "Amajac":     "Output-amajac",
            "EMSA":       "Output-emsa",
            "IPSA":       "Output-ipsa",
        }
        output_dir = _output_map[mill_key]

        curvas_folder = f"{display}_curvas"
        curva_file    = f"curva_global_{display}.parquet"
        weed_file     = f"{pais}_{empresa}_WEED_HISTORICO_{display}.geojson"

        result[mill_key] = {
            "display":       display,
            "output_dir":    output_dir,
            "curvas_folder": curvas_folder,
            "curva_file":    curva_file,
            "weed_file":     weed_file,
        }
    return result

MILLS = _mill_files()


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _s3_client():
    try:
        import boto3
    except ImportError:
        raise SystemExit(
            "❌ boto3 is not installed. Run: pip install boto3"
        )

    session_kwargs = {}
    key    = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    secret = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1").strip()

    if key and secret:
        session_kwargs = dict(
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            region_name=region,
        )
        print(f"🔑 Using AWS credentials from .env (region: {region})")
    else:
        print("🔑 Using default AWS credential chain (IAM role / env / ~/.aws)")

    import boto3
    return boto3.client("s3", **session_kwargs)


def _download(s3, s3_key: str, local_path: Path, skip_existing: bool = True):
    """Download one file from S3. Skips if already present (unless forced)."""
    if skip_existing and local_path.exists():
        print(f"  ⏭  Already exists, skipping: {local_path}")
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)
    full_key = f"{S3_PREFIX}/{s3_key}"
    print(f"  ⬇️  s3://{S3_BUCKET}/{full_key}")
    print(f"       → {local_path}")
    try:
        s3.download_file(S3_BUCKET, full_key, str(local_path))
    except Exception as exc:
        print(f"  ❌  FAILED: {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(work_dir: Path, mills_to_fetch: list = None, force: bool = False):
    """
    Download all static dependencies.

    Parameters
    ----------
    work_dir      : Root directory where files are placed.
    mills_to_fetch: List of mill keys to fetch per-mill files for.
                    Defaults to ALL mills.
    force         : Re-download even if file already exists.
    """
    os.chdir(work_dir)
    print(f"\n📂 Working directory: {work_dir}")
    print(f"📦 S3 bucket        : s3://{S3_BUCKET}/{S3_PREFIX}/\n")

    s3 = _s3_client()
    skip = not force

    # ── AOI GeoJSONs (all mills, go in root) ─────────────────────────────
    print("📍 AOI GeoJSONs...")
    for geojson in AOI_GEOJSONS:
        _download(s3, geojson, work_dir / geojson, skip)

    # ── Excel template (root) ─────────────────────────────────────────────
    print(f"\n📊 Excel template...")
    _download(s3, EXCEL_FILE, work_dir / EXCEL_FILE, skip)

    # ── Per-mill files ────────────────────────────────────────────────────
    targets = mills_to_fetch or list(MILLS.keys())

    print(f"\n🏭 Per-mill files ({len(targets)} mills)...")
    for mill_key in targets:
        if mill_key not in MILLS:
            print(f"  ⚠  Unknown mill '{mill_key}' — skipping")
            continue

        m = MILLS[mill_key]
        print(f"\n  [{mill_key}]  →  {m['output_dir']}/")

        # Curva parquet: Output-{x}/{Mill}_curvas/curva_global_{Mill}.parquet
        curva_local = work_dir / m["output_dir"] / m["curvas_folder"] / m["curva_file"]
        _download(s3, m["curva_file"], curva_local, skip)

        # Weed historico: Output-{x}/{weed_file}
        weed_local = work_dir / m["output_dir"] / m["weed_file"]
        _download(s3, m["weed_file"], weed_local, skip)

    print(f"\n{'='*60}")
    print("✅ Dependencies ready!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download static pipeline dependencies from S3"
    )
    parser.add_argument(
        "--work-dir", default=".",
        help="Working directory where files are placed (default: current dir)"
    )
    parser.add_argument(
        "--mill", default=None,
        help=(
            "Fetch only this mill's per-mill files. "
            "One of: Pantaleon, Monte_Rosa, Amajac, EMSA, IPSA. "
            "Omit to fetch ALL mills."
        )
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download files even if they already exist"
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    mills = [args.mill] if args.mill else None

    # Validate --mill value early
    if args.mill and args.mill not in MILLS:
        parser.error(
            f"Unknown mill '{args.mill}'. "
            f"Choose from: {list(MILLS.keys())}"
        )

    run(work_dir=work_dir, mills_to_fetch=mills, force=args.force)


if __name__ == "__main__":
    main()

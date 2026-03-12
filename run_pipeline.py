#!/usr/bin/env python3
"""
run_pipeline.py
===============
Single entry point for the satellite pipeline.

Change your mill / dates in .env, then run:
    python run_pipeline.py

Steps
------
1. Download   — Sentinel-1/2 imagery into pairs/
2. Cloud Rem  — Cloud removal, S1-S2 fusion, NDRE/NDWI
3. Fix NoData — Set nodata=-32768 on index rasters
4. Rename     — Move cloudfill TIFs to INPUT_DIR
5. Processing — Excel sync + NDVI/NDWI/Smart Growth/Weed
6. DB Push    — Push parquet files to Supabase

Any step can be skipped by setting SKIP_<STEP>=true in .env.
"""

import sys
import traceback
from pathlib import Path

# ── Bootstrap config ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.settings import cfg

print(cfg.summary())
print()


def step(name: str, fn, *args, skip: bool = False, **kwargs):
    if skip:
        print(f"⏭  SKIP: {name}")
        return None
    print(f"\n{'='*70}")
    print(f"▶  {name}")
    print(f"{'='*70}")
    try:
        result = fn(*args, **kwargs)
        print(f"✅ {name} complete")
        return result
    except Exception as exc:
        print(f"\n❌ {name} FAILED: {exc}")
        traceback.print_exc()
        raise SystemExit(1)


# ── Import scripts lazily so import errors are visible per step ───────────────
def main():
    # Step 1 – Download
    from scripts import download
    download_result = step(
        "Step 1: Download S1/S2",
        download.run,
        skip=cfg.SKIP_DOWNLOAD,
    )

    # Step 2 – Cloud removal + vegetation indices
    from scripts import cloud_removal
    effective_download_result = download_result or {}
    cloud_result = step(
        "Step 2: Cloud removal + vegetation indices",
        cloud_removal.run,
        effective_download_result,
        skip=cfg.SKIP_CLOUD_REMOVAL,
    )

    # Step 3 – Fix NoData
    from scripts import fix_nodata
    step(
        "Step 3: Fix NoData on index rasters",
        fix_nodata.run,
        cloud_result or {},
        skip=cfg.SKIP_CLOUD_REMOVAL,
    )

    # Step 4 – Rename & move
    from scripts import rename
    step(
        "Step 4: Rename & move cloudfill TIFs",
        rename.run,
        skip=cfg.SKIP_RENAME,
    )

    # Step 5 – Processing
    from scripts import processing
    step(
        "Step 5: Processing (NDVI / NDWI / Smart Growth / Weed)",
        processing.run,
        skip=cfg.SKIP_PROCESSING,
    )

    # # Step 6 – DB push
    # from scripts import db_push
    # step(
    #     "Step 6: Push parquet → Supabase",
    #     db_push.run,
    #     skip=cfg.SKIP_DB_PUSH,
    # )

    print(f"\n{'='*70}")
    print("🎉  PIPELINE COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

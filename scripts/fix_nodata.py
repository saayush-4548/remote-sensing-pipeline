"""
scripts/fix_nodata.py
========================
Step 3 - Fix NoData value on NDRE and NDWI rasters to match cloud_free (-32768).

Run standalone: python scripts/fix_nodata.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import cfg
import rasterio
import numpy as np


def run(cloud_removal_result: dict) -> None:
    """Set NoData=-32768 on NDRE/NDWI rasters."""

    # ── These are the same regardless of which path we take ──────────────────
    cloud_free_dir = cfg.CLOUD_FREE_DIR
    NODATA_VALUE = -32768.0

    # ── Resolve inference date ────────────────────────────────────────────────
    if not cloud_removal_result:
        print("⏭  fix_nodata: no cloud_removal_result, inferring from disk...")
        tifs = sorted(cloud_free_dir.glob("ndre_PROD_*.tif"))
        if not tifs:
            print("   No index rasters found, skipping.")
            return
        # ndre_PROD_20260219.tif  →  "20260219"
        compact = tifs[-1].stem.split("_")[-1]
        INFERENCE_DATE = f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}"
        print(f"   Inferred inference date: {INFERENCE_DATE}")
    else:
        INFERENCE_DATE = cloud_removal_result["INFERENCE_DATE"]

    # ── Build file list ───────────────────────────────────────────────────────
    inference_compact = INFERENCE_DATE.replace('-', '')
    ndre_path = cloud_free_dir / f"ndre_PROD_{inference_compact}.tif"
    ndwi_path = cloud_free_dir / f"ndwi_PROD_{inference_compact}.tif"

    files_to_fix = []
    if ndre_path.exists():
        files_to_fix.append(("NDRE", ndre_path))
    if ndwi_path.exists():
        files_to_fix.append(("NDWI", ndwi_path))

    if not files_to_fix:
        print(f"   ⚠️  No NDRE/NDWI rasters found for {INFERENCE_DATE}, skipping.")
        return

    print(f"🔧 Setting NoData={NODATA_VALUE} for index rasters")
    print(f"   Inference date: {INFERENCE_DATE}")
    print(f"   Output dir:     {cloud_free_dir}")

    # ── Fix each file ─────────────────────────────────────────────────────────
    for name, path in files_to_fix:
        with rasterio.open(path) as src:
            old_nodata = src.nodata
            profile = src.profile.copy()
            data = src.read()

        print(f"   {name}: {path.name}")
        print(f"      Before: nodata={old_nodata}")

        nodata_mask = np.isnan(data) | (data == 0)
        data[nodata_mask] = NODATA_VALUE
        profile.update(nodata=NODATA_VALUE)

        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(data)

        with rasterio.open(path) as src:
            print(f"      After:  nodata={src.nodata}")
        print(f"      ✅ Done")

    print(f"\n✅ All index rasters updated with nodata={NODATA_VALUE}")


if __name__ == "__main__":
    raise SystemExit("Run via run_pipeline.py")
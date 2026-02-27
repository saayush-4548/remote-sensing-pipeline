"""
scripts/03_fix_nodata.py
========================
Step 3 - Fix NoData value on NDRE and NDWI rasters to match cloud_free (-32768).

Run standalone: python scripts/03_fix_nodata.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import cfg
import rasterio
import numpy as np



def run(cloud_removal_result: dict) -> None:
    """Set NoData=-32768 on NDRE/NDWI rasters."""
    INFERENCE_DATE = cloud_removal_result["INFERENCE_DATE"]
    cloud_free_dir = cfg.CLOUD_FREE_DIR

    NODATA_VALUE = -32768.0
    inference_compact = INFERENCE_DATE.replace('-', '')

    ndre_path = cloud_free_dir / f"ndre_PROD_{inference_compact}.tif"
    ndwi_path = cloud_free_dir / f"ndwi_PROD_{inference_compact}.tif"

    files_to_fix = []
    if ndre_path.exists():
        files_to_fix.append(("NDRE", ndre_path))
    if ndwi_path.exists():
        files_to_fix.append(("NDWI", ndwi_path))

    print(f"🔧 Setting NoData={NODATA_VALUE} for index rasters")
    print(f"   Inference date: {INFERENCE_DATE}")
    print(f"   Output dir: {cloud_free_dir}")

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

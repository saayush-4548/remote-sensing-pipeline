"""
scripts/02_cloud_removal.py
===========================
Step 2 - Cloud removal, S1-S2 fusion, vegetation indices (NDRE, NDWI).

Run standalone: python scripts/02_cloud_removal.py
(Requires download_result dict from step 1, or re-scans pairs/ from disk)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import cfg

"""
Cloud Removal Pipeline - Compatible with Pairs Folder Download Structure
Uses INFERENCE_DATE, PREVIOUS_DATES, DOWNLOAD_PAIRS, TARGET_FOLDERS from Cell 1
Downloads go directly into pairs/ folder - no separate S1/S2 dirs
Includes confidence metrics based on prev image count, cloud %, nodata %
Never stops - always produces output + confidence values
"""
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
import warnings
from scipy import ndimage
from sklearn.ensemble import RandomForestRegressor
import gc
import re

warnings.filterwarnings('ignore')

# Check for GPU availability
GPU_AVAILABLE = False
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    GPU_AVAILABLE = True
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    gpu_mem = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / (1024**3)
    print(f"✅ GPU available via CuPy. Device: {gpu_name} ({gpu_mem:.1f} GB)")
    mempool = cp.get_default_memory_pool()
    mempool.set_limit(fraction=0.7)
except ImportError:
    print("ℹ️ CuPy not available. Using CPU only.")

print(f"NumPy version: {np.__version__}")
print(f"GPU acceleration (CuPy): {'Enabled' if GPU_AVAILABLE else 'Disabled'}")


# ============================================================================
# CONFIDENCE TRACKING
# ============================================================================
@dataclass
class PipelineConfidence:
    """
    Tracks confidence metrics for the cloud removal pipeline output.
    Confidence is based on:
      - Number of previous images available (within 30 days)
      - Cloud cover % of inference image
      - NoData % of inference image
      - Fill method distribution (temporal vs fusion vs spatial)
    """
    # Input metrics
    inference_date: str = ""
    num_previous_images: int = 0
    max_possible_previous: int = 5
    inference_cloud_cover_pct: float = 0.0
    inference_nodata_pct: float = 0.0
    inference_is_broken: bool = False

    # Per-previous-image info
    previous_image_dates: list = field(default_factory=list)
    previous_image_cloud_pcts: list = field(default_factory=list)
    previous_image_days_gap: list = field(default_factory=list)

    # Fill method breakdown
    total_pixels: int = 0
    clear_pixels: int = 0
    cloud_pixels: int = 0
    temporal_filled: int = 0
    fusion_filled: int = 0
    spatial_filled: int = 0
    unfilled_pixels: int = 0

    # Derived confidence scores (0-100)
    confidence_prev_images: float = 0.0    # Based on count of previous images
    confidence_cloud_cover: float = 0.0    # Based on inference cloud %
    confidence_nodata: float = 0.0         # Based on inference nodata %
    confidence_fill_quality: float = 0.0   # Based on fill method distribution
    confidence_overall: float = 0.0        # Weighted combination

    # Confidence level label
    confidence_level: str = "UNKNOWN"

    def calculate(self):
        """Calculate all confidence scores from collected metrics."""

        # 1. Previous images confidence (0-100)
        #    5 images = 100, 4=90, 3=75, 2=55, 1=30, 0=5
        prev_score_map = {5: 100, 4: 90, 3: 75, 2: 55, 1: 30, 0: 5}
        n = min(self.num_previous_images, 5)
        self.confidence_prev_images = prev_score_map.get(n, 5)

        # 2. Cloud cover confidence (0-100)
        #    0% cloud = 100 confidence, 100% cloud = 5 confidence
        if self.inference_cloud_cover_pct <= 0:
            self.confidence_cloud_cover = 100.0
        elif self.inference_cloud_cover_pct >= 100:
            self.confidence_cloud_cover = 5.0
        else:
            # Non-linear: low cloud = high confidence, degrades faster at high cloud
            self.confidence_cloud_cover = max(
                5.0,
                100.0 - (self.inference_cloud_cover_pct ** 1.3) * 0.8
            )

        # 3. NoData confidence (0-100)
        #    0% nodata = 100, >50% nodata = very low
        if self.inference_nodata_pct <= 0:
            self.confidence_nodata = 100.0
        elif self.inference_nodata_pct >= 50:
            self.confidence_nodata = 5.0
        else:
            self.confidence_nodata = max(
                5.0,
                100.0 - self.inference_nodata_pct * 2.0
            )

        # 4. Fill quality confidence (0-100)
        #    Based on HOW pixels were filled
        #    Temporal = best, Fusion = good, Spatial = fair, Unfilled = bad
        if self.cloud_pixels > 0:
            temporal_ratio = self.temporal_filled / self.cloud_pixels
            fusion_ratio = self.fusion_filled / self.cloud_pixels
            spatial_ratio = self.spatial_filled / self.cloud_pixels
            unfilled_ratio = self.unfilled_pixels / self.cloud_pixels

            # Weighted quality score
            self.confidence_fill_quality = (
                temporal_ratio * 100.0 +    # Temporal = full confidence
                fusion_ratio * 75.0 +       # Fusion = 75% confidence
                spatial_ratio * 40.0 +      # Spatial = 40% confidence
                unfilled_ratio * 0.0        # Unfilled = 0% confidence
            )
            self.confidence_fill_quality = max(5.0, min(100.0, self.confidence_fill_quality))
        else:
            # No clouds = perfect
            self.confidence_fill_quality = 100.0

        # 5. Overall confidence (weighted combination)
        #    Previous images: 30% weight (most important for quality)
        #    Cloud cover:     25% weight
        #    Fill quality:    30% weight
        #    NoData:          15% weight
        self.confidence_overall = (
            self.confidence_prev_images * 0.30 +
            self.confidence_cloud_cover * 0.25 +
            self.confidence_fill_quality * 0.30 +
            self.confidence_nodata * 0.15
        )
        self.confidence_overall = max(0.0, min(100.0, self.confidence_overall))

        # 6. Confidence level label
        if self.confidence_overall >= 85:
            self.confidence_level = "HIGH"
        elif self.confidence_overall >= 65:
            self.confidence_level = "MEDIUM"
        elif self.confidence_overall >= 40:
            self.confidence_level = "LOW"
        elif self.confidence_overall >= 20:
            self.confidence_level = "VERY LOW"
        else:
            self.confidence_level = "MINIMAL"

    def to_dict(self) -> dict:
        return {
            'inference_date': self.inference_date,
            'num_previous_images': self.num_previous_images,
            'max_possible_previous': self.max_possible_previous,
            'inference_cloud_cover_pct': round(self.inference_cloud_cover_pct, 2),
            'inference_nodata_pct': round(self.inference_nodata_pct, 2),
            'inference_is_broken': self.inference_is_broken,
            'previous_image_dates': self.previous_image_dates,
            'previous_image_cloud_pcts': [round(c, 2) for c in self.previous_image_cloud_pcts],
            'previous_image_days_gap': self.previous_image_days_gap,
            'total_pixels': self.total_pixels,
            'clear_pixels': self.clear_pixels,
            'cloud_pixels': self.cloud_pixels,
            'temporal_filled': self.temporal_filled,
            'fusion_filled': self.fusion_filled,
            'spatial_filled': self.spatial_filled,
            'unfilled_pixels': self.unfilled_pixels,
            'confidence_prev_images': round(self.confidence_prev_images, 1),
            'confidence_cloud_cover': round(self.confidence_cloud_cover, 1),
            'confidence_nodata': round(self.confidence_nodata, 1),
            'confidence_fill_quality': round(self.confidence_fill_quality, 1),
            'confidence_overall': round(self.confidence_overall, 1),
            'confidence_level': self.confidence_level,
        }

    def print_report(self):
        """Print a formatted confidence report."""
        level_emoji = {
            "HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🟠",
            "VERY LOW": "🔴", "MINIMAL": "⚫", "UNKNOWN": "⚪"
        }
        emoji = level_emoji.get(self.confidence_level, "⚪")

        print(f"\n{'='*70}")
        print(f"📊 CONFIDENCE REPORT")
        print(f"{'='*70}")
        print(f"")
        print(f"  Inference Date     : {self.inference_date}")
        print(f"  Broken Image       : {'YES ⚠️' if self.inference_is_broken else 'NO ✅'}")
        print(f"")
        print(f"  ┌─────────────────────────────────────────────────────┐")
        print(f"  │  {emoji} OVERALL CONFIDENCE: {self.confidence_overall:.1f}/100 "
              f"({self.confidence_level}){' '*(22 - len(self.confidence_level))}│")
        print(f"  └─────────────────────────────────────────────────────┘")
        print(f"")
        print(f"  Component Scores:")
        print(f"  {'─'*55}")
        print(f"  {'Component':<25} {'Score':>8} {'Weight':>8} {'Contribution':>14}")
        print(f"  {'─'*55}")

        components = [
            ("Prev Images", self.confidence_prev_images, 0.30),
            ("Cloud Cover", self.confidence_cloud_cover, 0.25),
            ("Fill Quality", self.confidence_fill_quality, 0.30),
            ("NoData", self.confidence_nodata, 0.15),
        ]
        for name, score, weight in components:
            contrib = score * weight
            bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
            print(f"  {name:<25} {score:>7.1f} {weight:>7.0%} {contrib:>13.1f}")

        print(f"  {'─'*55}")
        print(f"  {'OVERALL':<25} {self.confidence_overall:>7.1f} {'100%':>8}")

        print(f"\n  Input Metrics:")
        print(f"    Previous images      : {self.num_previous_images} / "
              f"{self.max_possible_previous}")
        if self.previous_image_dates:
            for i, (d, c, g) in enumerate(zip(
                    self.previous_image_dates,
                    self.previous_image_cloud_pcts,
                    self.previous_image_days_gap)):
                c_str = f"{c:.1f}%" if c is not None else "N/A"
                print(f"      prev{i+1}: {d} (cloud: {c_str}, gap: {g}d)")
        print(f"    Inference cloud%     : {self.inference_cloud_cover_pct:.2f}%")
        print(f"    Inference nodata%    : {self.inference_nodata_pct:.2f}%")

        print(f"\n  Fill Breakdown:")
        print(f"    Total pixels         : {self.total_pixels:,}")
        print(f"    Clear (untouched)    : {self.clear_pixels:,} "
              f"({self.clear_pixels/max(self.total_pixels,1)*100:.1f}%)")
        print(f"    Cloud pixels         : {self.cloud_pixels:,} "
              f"({self.cloud_pixels/max(self.total_pixels,1)*100:.1f}%)")

        if self.cloud_pixels > 0:
            print(f"    ├── Temporal fill    : {self.temporal_filled:,} "
                  f"({self.temporal_filled/self.cloud_pixels*100:.1f}%)")
            print(f"    ├── S1-S2 fusion     : {self.fusion_filled:,} "
                  f"({self.fusion_filled/self.cloud_pixels*100:.1f}%)")
            print(f"    ├── Spatial fill     : {self.spatial_filled:,} "
                  f"({self.spatial_filled/self.cloud_pixels*100:.1f}%)")
            print(f"    └── Unfilled         : {self.unfilled_pixels:,} "
                  f"({self.unfilled_pixels/self.cloud_pixels*100:.1f}%)")

        print(f"\n{'='*70}")


# ============================================================================
# AUTO-DETECT PATHS FROM DOWNLOAD CELL VARIABLES
# ============================================================================
def build_paths_from_pairs(
    inference_date: str,
    previous_dates: list,
    download_pairs: list,
    target_folders: dict,
    pairs_dir: Path
) -> Tuple[dict, dict, str, list]:
    """
    Build S2_FILE_PATHS and S1_FILE_PATHS from the pairs folder structure.
    Handles multiple naming conventions from download code:
      S2: S2_YYYYMMDD.tif, S2_YYYY-MM-DD.tif, openEO_YYYY-MM-DDZ.tif
      S1: s1_YYYYMMDD.tif, S1_YYYYMMDD.tif, S1_YYYY-MM-DD.tif, openEO_YYYY-MM-DDZ.tif
    """
    print(f"\n{'='*70}")
    print(f"📂 BUILDING PATHS FROM PAIRS FOLDER")
    print(f"{'='*70}")

    s2_paths = {}
    s1_paths = {}

    for pair in download_pairs:
        s2_date = pair['s2_date']
        s1_date = pair.get('s1_date')
        date_compact = s2_date.replace('-', '')
        s1_date_compact = s1_date.replace('-', '') if s1_date else None

        folder = target_folders.get(s2_date)
        if folder is None:
            for candidate in pairs_dir.iterdir():
                if candidate.is_dir() and s2_date in candidate.name:
                    folder = candidate
                    break

        if folder is None or not folder.exists():
            print(f"   ⚠️  No folder for {s2_date}")
            continue

        # ---- Find S2 file (multiple naming conventions) ----
        s2_file = None
        s2_search_patterns = [
            f"S2_{date_compact}.tif",           # S2_20260222.tif
            f"S2_{s2_date}.tif",                 # S2_2026-02-22.tif
            f"openEO_{s2_date}Z.tif",            # openEO_2026-02-22Z.tif
        ]
        # Try exact matches first
        for pattern in s2_search_patterns:
            candidate = folder / pattern
            if candidate.exists() and candidate.stat().st_size >= 1000:
                s2_file = candidate
                break

        # Fallback: glob for any S2/openEO tif
        if s2_file is None:
            all_s2 = (list(folder.glob("S2_*.tif")) +
                      list(folder.glob("openEO_*.tif")))
            valid_s2 = [f for f in all_s2 if f.stat().st_size >= 1000]
            if valid_s2:
                s2_file = max(valid_s2, key=lambda f: f.stat().st_size)

        if s2_file:
            s2_paths[s2_date] = str(s2_file)
        else:
            print(f"   ⚠️  No S2 in {folder.name}/")

        # ---- Find S1 file (multiple naming conventions) ----
        if s1_date:
            s1_file = None
            s1_search_patterns = [
                f"s1_{s1_date_compact}.tif",             # s1_20260222.tif (lowercase)
                f"s1_{s1_date_compact}_filled.tif",      # s1_20260222_filled.tif
                f"S1_{s1_date_compact}.tif",             # S1_20260222.tif
                f"S1_{s1_date}.tif",                     # S1_2026-02-22.tif
                f"S1_{s1_date}_filled.tif",              # S1_2026-02-22_filled.tif
                f"openEO_{s1_date}Z.tif",                # openEO_2026-02-22Z.tif
            ]

            # Try exact matches first (prefer filled versions)
            filled_file = None
            regular_file = None
            for pattern in s1_search_patterns:
                candidate = folder / pattern
                if candidate.exists() and candidate.stat().st_size >= 1000:
                    if '_filled' in pattern:
                        filled_file = candidate
                    elif regular_file is None:
                        regular_file = candidate

            # Prefer filled over regular
            s1_file = filled_file or regular_file

            # Fallback: glob for any S1/s1/openEO tif
            if s1_file is None:
                all_s1 = (list(folder.glob("S1_*.tif")) +
                          list(folder.glob("s1_*.tif")) +
                          list(folder.glob("openEO_*.tif")))
                # Exclude files already matched as S2
                if s2_file:
                    all_s1 = [f for f in all_s1 if f != s2_file]
                # Prefer filled
                filled_s1 = [f for f in all_s1
                             if '_filled' in f.name and f.stat().st_size >= 1000]
                regular_s1 = [f for f in all_s1
                              if '_filled' not in f.name and f.stat().st_size >= 1000]
                if filled_s1:
                    s1_file = max(filled_s1, key=lambda f: f.stat().st_size)
                elif regular_s1:
                    s1_file = max(regular_s1, key=lambda f: f.stat().st_size)

            if s1_file:
                # Extract actual S1 date from filename
                actual_s1_date = s1_date
                m = re.match(r'(?:S1_|s1_)(\d{4}-?\d{2}-?\d{2})', s1_file.name)
                if m:
                    d = m.group(1).replace('-', '')
                    actual_s1_date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
                s1_paths[s2_date] = {
                    's1_date': actual_s1_date, 's1_file': str(s1_file)}
            else:
                print(f"   ⚠️  No valid S1 in {folder.name}/")
        else:
            print(f"   ⚠️  No S1 date for {s2_date}")

    sorted_dates = sorted(s2_paths.keys(), reverse=True)

    print(f"\n📊 Path Resolution:")
    print(f"   {'Date':<14} {'Role':<12} {'S2 File':<40} {'S1 File'}")
    print(f"   {'-'*100}")

    for date in sorted_dates:
        role = "INFERENCE" if date == inference_date else "PREVIOUS"
        s2_name = Path(s2_paths[date]).name
        s1_info = s1_paths.get(date)
        s1_name = Path(s1_info['s1_file']).name if s1_info else "MISSING"
        filled_tag = " [filled]" if s1_info and "filled" in s1_name else ""
        print(f"   {date:<14} {role:<12} {s2_name:<40} {s1_name}{filled_tag}")

    print(f"\n   S2: {len(s2_paths)} | S1: {len(s1_paths)} | Target: {inference_date}")
    return s2_paths, s1_paths, inference_date, sorted_dates


def detect_band_names(s2_filepath: str) -> Tuple[list, list, int]:
    """Detect band names from downloaded S2 file."""
    with rasterio.open(s2_filepath) as src:
        n_bands = src.count
        descriptions = [src.descriptions[i] if src.descriptions[i] else f"band_{i+1}"
                        for i in range(n_bands)]

    print(f"\n🔍 Detecting bands: {Path(s2_filepath).name}")
    print(f"   Bands: {n_bands}, Descriptions: {descriptions}")

    FULL_15 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
               'B08', 'B8A', 'B09', 'B11', 'B12', 'WVP', 'AOT', 'SCL']

    CONFIGS = {
        15: FULL_15,
        13: ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A',
             'B09', 'B11', 'B12', 'AOT', 'SCL'],
        5:  ['B02', 'B03', 'B04', 'B08', 'SCL'],
        6:  ['B02', 'B03', 'B04', 'B08', 'B11', 'SCL'],
    }

    known = {'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
             'B08', 'B8A', 'B09', 'B11', 'B12', 'WVP', 'AOT', 'SCL'}

    detected = []
    for desc in descriptions:
        upper = desc.upper().strip()
        if upper in known:
            detected.append(upper)
        else:
            matched = False
            for kb in known:
                if kb in upper:
                    detected.append(kb)
                    matched = True
                    break
            if not matched:
                detected.append(desc)

    if len(set(detected) & known) >= n_bands * 0.5:
        band_names = detected
    elif n_bands in CONFIGS:
        band_names = CONFIGS[n_bands]
    else:
        band_names = FULL_15[:n_bands] if n_bands <= 15 else [
            f"band_{i+1}" for i in range(n_bands)]

    scl_idx = None
    for i, bn in enumerate(band_names):
        if bn.upper() == 'SCL':
            scl_idx = i
            break
    if scl_idx is None:
        scl_idx = n_bands - 1

    print(f"   Band names: {band_names}")
    print(f"   SCL index: {scl_idx}")
    return band_names, ['VV', 'VH'], scl_idx



def run(download_result: dict) -> dict:
    """
    Execute cloud removal pipeline.

    Parameters
    ----------
    download_result : dict returned by 01_download.run()

    Returns
    -------
    dict with CONFIDENCE_OVERALL, CONFIDENCE_LEVEL, FILL_* variables etc.
    """

    # Unpack download result
    INFERENCE_DATE        = download_result["INFERENCE_DATE"]
    INFERENCE_IS_BROKEN   = download_result["INFERENCE_IS_BROKEN"]
    PREVIOUS_DATES        = download_result["PREVIOUS_DATES"]
    DOWNLOAD_PAIRS        = download_result["DOWNLOAD_PAIRS"]
    INFERENCE_WINDOW      = download_result["INFERENCE_WINDOW"]
    TARGET_FOLDERS        = download_result["TARGET_FOLDERS"]
    PAIRS_DIR             = download_result["PAIRS_DIR"]
    CLOUD_PCT_PER_DATE    = download_result["CLOUD_PCT_PER_DATE"]
    NODATA_PCT_S2_PER_DATE= download_result["NODATA_PCT_S2_PER_DATE"]
    S2_FILE_PATHS         = download_result["S2_FILE_PATHS"]
    S1_FILE_PATHS         = download_result["S1_FILE_PATHS"]
    
    # ============================================================================
    # BUILD PATHS FROM CELL 1
    # ============================================================================
    print(f"\n{'='*70}")
    print(f"🔗 CONNECTING TO DOWNLOAD CELL OUTPUTS")
    print(f"{'='*70}")
    
    required_vars = ['INFERENCE_DATE', 'PREVIOUS_DATES', 'DOWNLOAD_PAIRS',
                     'TARGET_FOLDERS', 'PAIRS_DIR']
    # NEW (works)
    missing_vars = []
    for v in required_vars:
        try:
            eval(v)
        except NameError:
            missing_vars.append(v)
    
    if missing_vars:
        print(f"❌ Missing: {missing_vars}")
        raise RuntimeError(f"Run Cell 1 first! Missing: {missing_vars}")
    
    print(f"✅ Cell 1 variables found:")
    print(f"   INFERENCE_DATE  : {INFERENCE_DATE}")
    print(f"   PREVIOUS_DATES  : {PREVIOUS_DATES}")
    print(f"   PAIRS_DIR       : {PAIRS_DIR}")
    print(f"   DOWNLOAD_PAIRS  : {len(DOWNLOAD_PAIRS)} pairs")
    
    S2_FILE_PATHS, S1_FILE_PATHS, TARGET_DATE, SORTED_DATES = build_paths_from_pairs(
        INFERENCE_DATE, PREVIOUS_DATES, DOWNLOAD_PAIRS, TARGET_FOLDERS, PAIRS_DIR)
    
    first_s2_path = S2_FILE_PATHS[SORTED_DATES[0]]
    BAND_NAMES, S1_BAND_NAMES, SCL_BAND_INDEX = detect_band_names(first_s2_path)
    
    SPECTRAL_BAND_NAMES_FULL = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06',
                                 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12')
    AUX_BAND_NAMES = ('WVP', 'AOT', 'SCL')
    
    SPECTRAL_BAND_INDICES = []
    SPECTRAL_BANDS_AVAILABLE = []
    for bn in SPECTRAL_BAND_NAMES_FULL:
        if bn in BAND_NAMES:
            SPECTRAL_BAND_INDICES.append(BAND_NAMES.index(bn))
            SPECTRAL_BANDS_AVAILABLE.append(bn)
    
    AUX_BAND_INDICES = []
    for bn in AUX_BAND_NAMES:
        if bn in BAND_NAMES:
            AUX_BAND_INDICES.append(BAND_NAMES.index(bn))
    
    print(f"\n📊 Band Mapping:")
    print(f"   Spectral: {len(SPECTRAL_BAND_INDICES)} → {SPECTRAL_BANDS_AVAILABLE}")
    print(f"   Aux: {AUX_BAND_INDICES}")
    print(f"   SCL: {SCL_BAND_INDEX}")
    
    # Cell 1 metadata
    try:
        if 'CLOUD_PCT_PER_DATE' in dir():
            print(f"\n📊 Cell 1 Metadata:")
            for d in SORTED_DATES:
                cl = CLOUD_PCT_PER_DATE.get(d)
                nd_s2 = NODATA_PCT_S2_PER_DATE.get(d)
                role = "INF" if d == INFERENCE_DATE else "PRV"
                cl_s = f"{cl:.1f}%" if cl is not None else "N/A"
                nd_s = f"{nd_s2:.2f}%" if nd_s2 is not None else "N/A"
                print(f"   {d} [{role}] Cloud:{cl_s} NoData:{nd_s}")
    except:
        pass
    
    
    # ============================================================================
    # Configuration
    # ============================================================================
    @dataclass
    class CloudRemovalConfig:
        """Configuration for cloud removal pipeline"""
        CLOUD_SCL_CLASSES: Tuple[int, ...] = (8, 9, 10, 3)
        SPECTRAL_BANDS: Tuple[str, ...] = tuple(SPECTRAL_BANDS_AVAILABLE)
        SPECTRAL_BAND_INDICES: Tuple[int, ...] = tuple(SPECTRAL_BAND_INDICES)
        AUX_BAND_INDICES: Tuple[int, ...] = tuple(AUX_BAND_INDICES)
        SCL_BAND_INDEX: int = SCL_BAND_INDEX
        MAX_PREVIOUS_IMAGES: int = 5
        CHUNK_SIZE: int = 1024
        BUFFER_PIXELS: int = 5
        MAX_CHANGE_RATE_PER_DAY: float = 500.0
        MIN_CLEAR_OBSERVATIONS: int = 2
        TRAINING_SAMPLE_FRACTION: float = 0.5
        TRAINING_MAX_SAMPLES: int = 200_000
        MAX_INTERPOLATION_DISTANCE: int = 50
        N_JOBS: int = -1
    
    config = CloudRemovalConfig()
    
    print(f"\n✅ Configuration loaded")
    print(f" • Cloud SCL classes: {config.CLOUD_SCL_CLASSES}")
    print(f" • Spectral bands: {len(config.SPECTRAL_BANDS)} → {config.SPECTRAL_BANDS}")
    print(f" • SCL index: {config.SCL_BAND_INDEX}")
    print(f" • Chunk size: {config.CHUNK_SIZE}x{config.CHUNK_SIZE}")
    
    
    # ============================================================================
    # Output Directory
    # ============================================================================
    OUTPUT_DIR = str(PAIRS_DIR.parent / "cloud_free_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f'cloud_free_{TARGET_DATE}.tif')
    
    print(f"\n📁 Output: {OUTPUT_PATH}")
    
    
    # ============================================================================
    # Utility Functions
    # ============================================================================
    def parse_date(date_str: str) -> datetime:
        return datetime.strptime(date_str, '%Y-%m-%d')
    
    def days_between(date1: str, date2: str) -> int:
        return abs((parse_date(date1) - parse_date(date2)).days)
    
    def get_previous_dates(target_date: str, all_dates: List[str],
                           max_count: int = 5) -> List[str]:
        target_dt = parse_date(target_date)
        previous = []
        for date in sorted(all_dates, reverse=True):
            if parse_date(date) < target_dt:
                previous.append(date)
                if len(previous) >= max_count:
                    break
        return previous
    
    def validate_files_exist() -> bool:
        missing = []
        for date, path in S2_FILE_PATHS.items():
            if not os.path.exists(path):
                missing.append(f"S2 {date}: {path}")
        for date, info in S1_FILE_PATHS.items():
            if not os.path.exists(info['s1_file']):
                missing.append(f"S1 {date}: {info['s1_file']}")
        if missing:
            print("❌ Missing files:")
            for m in missing:
                print(f"   • {m}")
            return False
        print("✅ All files validated")
        return True
    
    def get_image_info(filepath: str) -> dict:
        with rasterio.open(filepath) as src:
            return {
                'shape': (src.count, src.height, src.width),
                'dtype': src.dtypes[0], 'crs': src.crs,
                'transform': src.transform, 'bounds': src.bounds}
    
    validate_files_exist()
    
    print(f"\n📊 Image Info:")
    sample_s2 = get_image_info(S2_FILE_PATHS[SORTED_DATES[0]])
    print(f" S2 shape: {sample_s2['shape']}, dtype: {sample_s2['dtype']}")
    if S1_FILE_PATHS:
        sample_s1 = get_image_info(list(S1_FILE_PATHS.values())[0]['s1_file'])
        print(f" S1 shape: {sample_s1['shape']}")
    
    
    # ============================================================================
    # Cloud Mask Generation
    # ============================================================================
    def create_cloud_mask(scl_band: np.ndarray, buffer_size: int = 5,
                          cloud_classes: tuple = (3, 8, 9, 10)) -> Tuple[np.ndarray, np.ndarray]:
        if GPU_AVAILABLE:
            scl_gpu = cp.asarray(scl_band)
            cm_gpu = cp.zeros_like(scl_gpu, dtype=cp.bool_)
            for cls in cloud_classes:
                cm_gpu |= (scl_gpu == cls)
            cloud_mask = cp.asnumpy(cm_gpu)
            del scl_gpu, cm_gpu
            cp.get_default_memory_pool().free_all_blocks()
        else:
            cloud_mask = np.isin(scl_band, cloud_classes)
    
        if buffer_size > 0:
            struct = ndimage.generate_binary_structure(2, 1)
            cloud_mask_buffered = ndimage.binary_dilation(
                cloud_mask, structure=struct, iterations=buffer_size)
        else:
            cloud_mask_buffered = cloud_mask.copy()
    
        return cloud_mask, cloud_mask_buffered
    
    def analyze_cloud_coverage(scl_band: np.ndarray,
                               cloud_classes: tuple = (3, 8, 9, 10)) -> dict:
        total = scl_band.size
        cloud_mask = np.isin(scl_band, cloud_classes)
        cloud_px = np.sum(cloud_mask)
        return {
            'total_pixels': total, 'cloud_pixels': int(cloud_px),
            'clear_pixels': int(total - cloud_px),
            'cloud_percentage': (cloud_px / total) * 100}
    
    print(f"\n🔍 Analyzing cloud coverage...")
    cloud_stats = {}
    for date in tqdm(SORTED_DATES, desc="Analyzing clouds"):
        with rasterio.open(S2_FILE_PATHS[date]) as src:
            scl = src.read(config.SCL_BAND_INDEX + 1)
            cloud_stats[date] = analyze_cloud_coverage(scl, config.CLOUD_SCL_CLASSES)
        del scl
        gc.collect()
    
    print(f"\n📊 Cloud Coverage:")
    print(f"{'Date':<15} {'Cloud %':>10} {'Pixels':>15} {'Status':<15}")
    print("-" * 60)
    for date in SORTED_DATES:
        s = cloud_stats[date]
        status = "🔴 High" if s['cloud_percentage'] > 30 else \
                 "🟡 Medium" if s['cloud_percentage'] > 10 else "🟢 Low"
        role = " ← INFERENCE" if date == TARGET_DATE else ""
        print(f"{date:<15} {s['cloud_percentage']:>9.2f}% {s['cloud_pixels']:>15,} "
              f"{status}{role}")
    
    
    # ============================================================================
    # Image Loader
    # ============================================================================
    class ImageLoader:
        def __init__(self, s2_paths: dict, s1_paths: dict):
            self.s2_paths = s2_paths
            self.s1_paths = s1_paths
            self._metadata = None
    
        def get_metadata(self) -> dict:
            if self._metadata is None:
                first_path = list(self.s2_paths.values())[0]
                with rasterio.open(first_path) as src:
                    self._metadata = {
                        'height': src.height, 'width': src.width,
                        'count': src.count, 'dtype': src.dtypes[0],
                        'crs': src.crs, 'transform': src.transform,
                        'profile': src.profile.copy()}
            return self._metadata
    
        def load_s2_full(self, date: str) -> np.ndarray:
            with rasterio.open(self.s2_paths[date]) as src:
                return src.read()
    
        def load_s1_full(self, date: str) -> np.ndarray:
            if date not in self.s1_paths:
                print(f"   ⚠️  No S1 for {date}, returning zeros")
                meta = self.get_metadata()
                return np.zeros((2, meta['height'], meta['width']), dtype=np.float32)
            with rasterio.open(self.s1_paths[date]['s1_file']) as src:
                return src.read()
    
        def load_s2_band(self, date: str, band_idx: int) -> np.ndarray:
            with rasterio.open(self.s2_paths[date]) as src:
                return src.read(band_idx + 1)
    
        def get_chunk_windows(self, chunk_size: int = 512) -> List[Window]:
            meta = self.get_metadata()
            h, w = meta['height'], meta['width']
            windows = []
            for r in range(0, h, chunk_size):
                for c in range(0, w, chunk_size):
                    windows.append(Window(c, r, min(chunk_size, w - c),
                                          min(chunk_size, h - r)))
            return windows
    
    loader = ImageLoader(S2_FILE_PATHS, S1_FILE_PATHS)
    metadata = loader.get_metadata()
    
    print(f"\n✅ Loader: {metadata['height']}x{metadata['width']}, "
          f"{metadata['count']} bands, {metadata['dtype']}")
    chunks = loader.get_chunk_windows(config.CHUNK_SIZE)
    print(f"   Chunks: {len(chunks)}")
    
    
    # ============================================================================
    # Temporal Rate of Change
    # ============================================================================
    def calculate_temporal_rate_of_change_fast(
        target_data, previous_data, target_cloud_mask, previous_cloud_masks,
        days_from_target, spectral_indices, max_rate=500.0
    ) -> Tuple[np.ndarray, np.ndarray]:
    
        n_bands, height, width = target_data.shape
        n_prev = previous_data.shape[0]
    
        filled_data = target_data.astype(np.float32).copy()
        fill_success = ~target_cloud_mask.copy()
    
        # Handle 0 previous images
        if n_prev == 0:
            print("   ⚠️  No previous images for temporal interpolation")
            return filled_data, fill_success
    
        clear_indicators = (~previous_cloud_masks).astype(np.float32)
        inv_days = 1.0 / np.maximum(days_from_target, 1.0)
        temporal_weights = clear_indicators * inv_days[:, np.newaxis, np.newaxis]
        weight_sum = np.sum(temporal_weights, axis=0)
        n_clear = np.sum(clear_indicators, axis=0)
        has_clear = n_clear >= 1
        cloudy_with_clear = target_cloud_mask & has_clear
    
        print(f"   Cloudy pixels with clear reference: {np.sum(cloudy_with_clear):,}")
    
        if GPU_AVAILABLE:
            print("   Using GPU...")
            days_gpu = cp.asarray(days_from_target)
            tw_gpu = cp.asarray(temporal_weights)
            ws_gpu = cp.asarray(weight_sum)
            nc_gpu = cp.asarray(n_clear)
            cm_gpu = cp.asarray(target_cloud_mask)
    
            for bi in tqdm(spectral_indices, desc="Bands (GPU)"):
                pbv = cp.asarray(previous_data[:, bi, :, :].astype(np.float32))
    
                single = (nc_gpu == 1) & cm_gpu
                if cp.any(single):
                    ws_val = cp.sum(pbv * tw_gpu, axis=0)
                    ws_d = cp.maximum(ws_gpu, 1e-10)
                    fb = cp.asarray(filled_data[bi])
                    fb[single] = (ws_val / ws_d)[single]
                    filled_data[bi] = cp.asnumpy(fb)
                    fill_success[cp.asnumpy(single)] = True
                    del fb, ws_val, ws_d
    
                multi = (nc_gpu >= 2) & cm_gpu
                if cp.any(multi):
                    mr_gpu, mc_gpu_idx = cp.where(multi)
                    mr = cp.asnumpy(mr_gpu)
                    mc = cp.asnumpy(mc_gpu_idx)
    
                    csz = 100000
                    for cs in range(0, len(mr), csz):
                        ce = min(cs + csz, len(mr))
                        cr, cc = mr[cs:ce], mc[cs:ce]
    
                        cv = pbv[:, cr, cc]
                        ctw = tw_gpu[:, cr, cc]
                        x = days_gpu[:, cp.newaxis]
                        w = ctw
    
                        sw = cp.sum(w, axis=0)
                        swx = cp.sum(w * x, axis=0)
                        swy = cp.sum(w * cv, axis=0)
                        swxx = cp.sum(w * x * x, axis=0)
                        swxy = cp.sum(w * x * cv, axis=0)
    
                        den = sw * swxx - swx * swx
                        valid = cp.abs(den) > 1e-10
    
                        slope = cp.zeros(len(cr), dtype=cp.float32)
                        intercept = cp.zeros(len(cr), dtype=cp.float32)
    
                        slope[valid] = (sw[valid] * swxy[valid] -
                                        swx[valid] * swy[valid]) / den[valid]
                        slope = cp.clip(slope, -max_rate, max_rate)
                        intercept[valid] = (swy[valid] -
                                            slope[valid] * swx[valid]) / sw[valid]
                        intercept[~valid] = swy[~valid] / cp.maximum(
                            sw[~valid], 1e-10)
    
                        filled_data[bi, cr, cc] = cp.asnumpy(
                            cp.maximum(intercept, 0))
    
                        del cv, ctw, w, x, sw, swx, swy, swxx, swxy
                        del den, valid, slope, intercept
    
                    fill_success[cp.asnumpy(multi)] = True
                    del mr_gpu, mc_gpu_idx
    
                del pbv
                cp.get_default_memory_pool().free_all_blocks()
    
            del days_gpu, tw_gpu, ws_gpu, nc_gpu, cm_gpu
            cp.get_default_memory_pool().free_all_blocks()
    
        else:
            for bi in tqdm(spectral_indices, desc="Bands (CPU)"):
                pbv = previous_data[:, bi, :, :].astype(np.float32)
    
                single = (n_clear == 1) & target_cloud_mask
                if np.any(single):
                    ws_val = np.sum(pbv * temporal_weights, axis=0)
                    ws_d = np.maximum(weight_sum, 1e-10)
                    filled_data[bi, single] = (ws_val / ws_d)[single]
                    fill_success[single] = True
                    del ws_val, ws_d
    
                multi = (n_clear >= 2) & target_cloud_mask
                if np.any(multi):
                    mr, mc = np.where(multi)
                    csz = 100000
                    for cs in range(0, len(mr), csz):
                        ce = min(cs + csz, len(mr))
                        cr, cc = mr[cs:ce], mc[cs:ce]
    
                        cv = pbv[:, cr, cc]
                        ctw = temporal_weights[:, cr, cc]
                        x = days_from_target[:, np.newaxis]
                        w = ctw
    
                        sw = np.sum(w, axis=0)
                        swx = np.sum(w * x, axis=0)
                        swy = np.sum(w * cv, axis=0)
                        swxx = np.sum(w * x * x, axis=0)
                        swxy = np.sum(w * x * cv, axis=0)
    
                        den = sw * swxx - swx * swx
                        valid = np.abs(den) > 1e-10
    
                        slope = np.zeros(len(cr), dtype=np.float32)
                        intercept = np.zeros(len(cr), dtype=np.float32)
    
                        slope[valid] = (sw[valid] * swxy[valid] -
                                        swx[valid] * swy[valid]) / den[valid]
                        slope = np.clip(slope, -max_rate, max_rate)
                        intercept[valid] = (swy[valid] -
                                            slope[valid] * swx[valid]) / sw[valid]
                        intercept[~valid] = swy[~valid] / np.maximum(
                            sw[~valid], 1e-10)
    
                        filled_data[bi, cr, cc] = np.maximum(intercept, 0)
    
                        del cv, ctw, w, x, sw, swx, swy, swxx, swxy
    
                    fill_success[multi] = True
    
                del pbv
                gc.collect()
    
        return filled_data, fill_success
    
    print("✅ Temporal functions ready")
    
    
    # ============================================================================
    # S1-S2 Fusion Model
    # ============================================================================
    class S1S2FusionModel:
        def __init__(self, n_estimators=100, sample_fraction=0.5,
                     max_samples=200_000):
            self.n_estimators = n_estimators
            self.sample_fraction = sample_fraction
            self.max_samples = max_samples
            self.models = {}
            self.is_trained = False
            self.band_stats = {}
    
        def _get_s1_valid_mask(self, s1_data):
            vv, vh = s1_data[0], s1_data[1]
            return (np.isfinite(vv) & np.isfinite(vh) &
                    (vv != 0) & (vh != 0) &
                    (np.abs(vv) < 1e6) & (np.abs(vh) < 1e6))
    
        def _prepare_features(self, s1_data, rows, cols):
            n = len(rows)
            vv_f = s1_data[0].astype(np.float32)
            vh_f = s1_data[1].astype(np.float32)
            vv, vh = vv_f[rows, cols], vh_f[rows, cols]
            vv_s = np.maximum(np.abs(vv), 1e-10)
            vh_s = np.maximum(np.abs(vh), 1e-10)
    
            vvm3 = ndimage.uniform_filter(vv_f, size=3)
            vhm3 = ndimage.uniform_filter(vh_f, size=3)
            vvm7 = ndimage.uniform_filter(vv_f, size=7)
            vhm7 = ndimage.uniform_filter(vh_f, size=7)
            vv_var = np.maximum(
                ndimage.uniform_filter(vv_f**2, 5) -
                ndimage.uniform_filter(vv_f, 5)**2, 0)
            vh_var = np.maximum(
                ndimage.uniform_filter(vh_f**2, 5) -
                ndimage.uniform_filter(vh_f, 5)**2, 0)
    
            X = np.zeros((n, 14), dtype=np.float32)
            X[:, 0] = vv;  X[:, 1] = vh
            X[:, 2] = vv / vh_s;  X[:, 3] = vh / vv_s
            X[:, 4] = vv * vh
            X[:, 5] = np.log10(vv_s);  X[:, 6] = np.log10(vh_s)
            X[:, 7] = vvm3[rows, cols]; X[:, 8] = vhm3[rows, cols]
            X[:, 9] = vvm7[rows, cols]; X[:, 10] = vhm7[rows, cols]
            X[:, 11] = np.sqrt(vv_var[rows, cols])
            X[:, 12] = np.sqrt(vh_var[rows, cols])
            X[:, 13] = (vv - vh) / (vv_s + vh_s)
    
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            del vv_f, vh_f, vvm3, vhm3, vvm7, vhm7, vv_var, vh_var
            gc.collect()
            return X
    
        def train_multi_date(self, s2_list, s1_list, cm_list, spectral_idx):
            if not s2_list:
                print("   ⚠️  No training data for S1-S2 fusion")
                self.is_trained = False
                return
    
            all_rows, all_cols = [], []
            all_s1_feat = []
            all_s2_vals = {bi: [] for bi in spectral_idx}
            date_idx = []
    
            for i, (s2, s1, cm) in enumerate(zip(s2_list, s1_list, cm_list)):
                s1v = self._get_s1_valid_mask(s1)
                trainable = (~cm) & s1v
                cr, cc = np.where(trainable)
                na = len(cr)
    
                if na < 500:
                    print(f"   Date {i}: skip ({na} trainable)")
                    continue
    
                budget = self.max_samples // max(len(s2_list), 1)
                nt = min(int(na * self.sample_fraction), budget)
                nt = max(nt, 500)
    
                idx = np.random.choice(na, nt, replace=False)
                sr, sc = cr[idx], cc[idx]
    
                all_rows.append(sr)
                all_cols.append(sc)
                all_s1_feat.append((s1, sr, sc))
                date_idx.append(i)
    
                for bi in spectral_idx:
                    all_s2_vals[bi].append(s2[bi, sr, sc].astype(np.float32))
    
                print(f"   Date {i}: {nt:,} / {na:,}")
    
            if not all_rows:
                print("   ⚠️  No trainable pairs")
                self.is_trained = False
                return
    
            X_parts = [self._prepare_features(s1, r, c)
                        for s1, r, c in all_s1_feat]
            X = np.vstack(X_parts)
            print(f"   Total samples: {X.shape[0]:,}")
            del X_parts, all_s1_feat
            gc.collect()
    
            for bi in tqdm(spectral_idx, desc="Training fusion"):
                y = np.nan_to_num(np.concatenate(all_s2_vals[bi]), nan=0.0)
                self.band_stats[bi] = {
                    'mean': float(np.mean(y)), 'std': float(np.std(y)),
                    'p1': float(np.percentile(y, 1)),
                    'p99': float(np.percentile(y, 99))}
    
                model = RandomForestRegressor(
                    n_estimators=self.n_estimators, max_depth=15,
                    min_samples_leaf=5, max_features='sqrt',
                    n_jobs=-1, random_state=42)
                model.fit(X, y)
                self.models[bi] = model
    
            self.is_trained = True
            print(f"   ✅ Trained on {len(date_idx)} dates, {X.shape[0]:,} samples")
    
        def predict(self, s1_data, mask, spectral_idx):
            if not self.is_trained:
                return {}
    
            s1v = self._get_s1_valid_mask(s1_data)
            predictable = mask & s1v
            pr, pc = np.where(predictable)
            n = len(pr)
    
            if n == 0:
                print(f"   ⚠️  No valid S1 in mask")
                return {}
    
            print(f"   Pixels to predict: {n:,}")
            predictions = {}
            csz = 500_000
    
            for bi in tqdm(spectral_idx, desc="Predicting", leave=False):
                if bi not in self.models:
                    continue
                all_p = np.zeros(n, dtype=np.float32)
                for cs in range(0, n, csz):
                    ce = min(cs + csz, n)
                    X_c = self._prepare_features(s1_data, pr[cs:ce], pc[cs:ce])
                    p_c = self.models[bi].predict(X_c)
                    p1 = self.band_stats[bi]['p1']
                    p99 = self.band_stats[bi]['p99']
                    margin = (p99 - p1) * 0.1
                    all_p[cs:ce] = np.clip(
                        p_c, max(0, p1 - margin), p99 + margin).astype(np.float32)
                    del X_c, p_c
                    gc.collect()
                predictions[bi] = (pr.copy(), pc.copy(), all_p)
    
            return predictions
    
    print("✅ S1-S2 Fusion ready")
    
    
    # ============================================================================
    # Spatial Interpolation
    # ============================================================================
    def spatial_interpolate_band_fast(data, mask, max_distance=50):
        result = data.copy().astype(np.float32)
        success = np.zeros_like(mask, dtype=bool)
        if not np.any(mask):
            success[:] = True
            return result, success
        if not np.any(~mask):
            return result, success
        dist, indices = ndimage.distance_transform_edt(
            mask, return_distances=True, return_indices=True)
        fillable = mask & (dist <= max_distance)
        if np.any(fillable):
            result[fillable] = data[indices[0][fillable], indices[1][fillable]]
            success[fillable] = True
        success[~mask] = True
        return result, success
    
    def apply_spatial_interpolation(data, mask, spectral_idx, max_distance=50):
        result = data.copy()
        overall = np.zeros(mask.shape, dtype=bool)
        for bi in tqdm(spectral_idx, desc="Spatial interpolation"):
            br, bs = spatial_interpolate_band_fast(data[bi], mask, max_distance)
            result[bi] = br
            overall |= bs
        return result, overall
    
    print("✅ Spatial interpolation ready")
    
    
    # ============================================================================
    # Edge Blending
    # ============================================================================
    def create_blend_weights(cloud_mask, cloud_mask_buffered, buffer_size=5):
        if buffer_size <= 0:
            return cloud_mask.astype(np.float32)
        weights = np.zeros_like(cloud_mask, dtype=np.float32)
        weights[cloud_mask] = 1.0
        bz = cloud_mask_buffered & ~cloud_mask
        if np.any(bz):
            d = ndimage.distance_transform_edt(~cloud_mask)
            bw = np.maximum(1.0 - d / buffer_size, 0.0)
            weights[bz] = bw[bz]
            del d, bw
        return weights
    
    def blend_images(original, filled, weights, spectral_idx):
        result = original.copy().astype(np.float32)
        if GPU_AVAILABLE:
            w_gpu = cp.asarray(weights)
            ow = 1.0 - w_gpu
            for bi in spectral_idx:
                o = cp.asarray(original[bi].astype(np.float32))
                f = cp.asarray(filled[bi].astype(np.float32))
                result[bi] = cp.asnumpy(o * ow + f * w_gpu)
                del o, f
            del w_gpu, ow
            cp.get_default_memory_pool().free_all_blocks()
        else:
            bm = weights > 0
            if np.any(bm):
                for bi in spectral_idx:
                    result[bi][bm] = (
                        original[bi].astype(np.float32)[bm] * (1 - weights[bm]) +
                        filled[bi].astype(np.float32)[bm] * weights[bm])
        return result
    
    print("✅ Blending ready")
    
    
    # ============================================================================
    # Vegetation Indices (NDRE and NDWI only)
    # ============================================================================
    def generate_vegetation_indices(cloud_free_path, output_dir, target_date, band_names):
        print(f"\n{'='*70}")
        print(f"🌿 GENERATING VEGETATION INDICES (NDRE, NDWI)")
        print(f"{'='*70}")
    
        CLEAN = Path(cloud_free_path)
        OUT = Path(output_dir)
        bu = [b.upper() for b in band_names]
    
        def find_band(name):
            try:
                return bu.index(name.upper())
            except ValueError:
                return None
    
        B05 = find_band('B05')
        B08 = find_band('B08')
        B11 = find_band('B11')
    
        avail = []
        if B05 is not None and B08 is not None:
            avail.append('NDRE')
        if B08 is not None and B11 is not None:
            avail.append('NDWI')
    
        print(f"   Available: {avail}")
    
        if not CLEAN.exists():
            print(f"   ❌ File not found: {CLEAN}")
            return
    
        dc = target_date.replace('-', '')
    
        with rasterio.open(CLEAN) as src:
            profile = src.profile.copy()
            height = src.height
            width = src.width
            dtype = src.dtypes[0]
            nir = src.read(B08 + 1).astype(np.float32) if B08 is not None else None
            re1 = src.read(B05 + 1).astype(np.float32) if B05 is not None else None
            swir = src.read(B11 + 1).astype(np.float32) if B11 is not None else None
    
        print(f"   Image size: {width} x {height}")
        print(f"   Bands: {src.count if False else 'read'}")
        print(f"   Dtype: {dtype}")
    
        ip = profile.copy()
        ip.update(dtype='float32', count=1, nodata=None, compress='lzw')
    
        # ---- Compute NDRE = (NIR - Red Edge 1) / (NIR + Red Edge 1) ----
        if 'NDRE' in avail:
            print(f"\n🔬 Computing NDRE...")
    
            ndre = (nir - re1) / (nir + re1 + 1e-6)
            ndre = np.clip(ndre, -1.0, 1.0).astype(np.float32)
    
            print(f"   NDRE range: [{ndre.min():.4f}, {ndre.max():.4f}]")
            print(f"   NDRE mean:  {ndre.mean():.4f}")
    
            # Save NDRE
            ndre_output_path = OUT / f"ndre_PROD_{dc}.tif"
    
            print(f"\n💾 Saving NDRE...")
    
            ndre_profile = profile.copy()
            ndre_profile.update(
                dtype='float32',
                count=1,
                nodata=None,
                compress='lzw'
            )
    
            with rasterio.open(ndre_output_path, 'w', **ndre_profile) as dst:
                dst.write(ndre, 1)
                dst.set_band_description(1, "NDRE = (B08_NIR - B05_RedEdge1) / (B08_NIR + B05_RedEdge1)")
    
            ndre_size_mb = ndre_output_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ Saved: {ndre_output_path}")
            print(f"   File size: {ndre_size_mb:.2f} MB")
    
            del ndre
            gc.collect()
    
        # ---- Compute NDWI (Gao) = (NIR - SWIR1) / (NIR + SWIR1) ----
        if 'NDWI' in avail:
            print(f"\n🔬 Computing NDWI...")
    
            ndwi = (nir - swir) / (nir + swir + 1e-6)
            ndwi = np.clip(ndwi, -1.0, 1.0).astype(np.float32)
    
            print(f"   NDWI range: [{ndwi.min():.4f}, {ndwi.max():.4f}]")
            print(f"   NDWI mean:  {ndwi.mean():.4f}")
    
            # Save NDWI
            ndwi_output_path = OUT / f"ndwi_PROD_{dc}.tif"
    
            print(f"\n💾 Saving NDWI...")
    
            ndwi_profile = profile.copy()
            ndwi_profile.update(
                dtype='float32',
                count=1,
                nodata=None,
                compress='lzw'
            )
    
            with rasterio.open(ndwi_output_path, 'w', **ndwi_profile) as dst:
                dst.write(ndwi, 1)
                dst.set_band_description(1, "NDWI = (B08_NIR - B11_SWIR1) / (B08_NIR + B11_SWIR1)")
    
            ndwi_size_mb = ndwi_output_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ Saved: {ndwi_output_path}")
            print(f"   File size: {ndwi_size_mb:.2f} MB")
    
            del ndwi
            gc.collect()
    
        # Summary
        print(f"\n{'='*70}")
        print(f"📊 VEGETATION INDICES SUMMARY")
        print(f"{'='*70}")
        print(f"  Input:  {CLEAN.name}")
        print(f"  Date:   {target_date}")
        print(f"")
        print(f"  Output files:")
        if 'NDRE' in avail:
            print(f"    NDRE: {OUT / f'ndre_PROD_{dc}.tif'}")
            print(f"           Size:  {(OUT / f'ndre_PROD_{dc}.tif').stat().st_size / (1024*1024):.2f} MB")
            print(f"")
        if 'NDWI' in avail:
            print(f"    NDWI: {OUT / f'ndwi_PROD_{dc}.tif'}")
            print(f"           Size:  {(OUT / f'ndwi_PROD_{dc}.tif').stat().st_size / (1024*1024):.2f} MB")
        print(f"{'='*70}")
        print(f"✅ Vegetation indices generation complete!")
    
        # Free memory
        del nir, re1, swir
        gc.collect()
    
    print("✅ Vegetation indices generation function ready (NDRE, NDWI)")
    
    
    # ============================================================================
    # Main Pipeline
    # ============================================================================
    class CloudRemovalPipeline:
        def __init__(self, config, loader, s2_paths, s1_paths, band_names):
            self.config = config
            self.loader = loader
            self.s2_paths = s2_paths
            self.s1_paths = s1_paths
            self.band_names = band_names
            self.fusion_model = S1S2FusionModel(
                n_estimators=100,
                sample_fraction=config.TRAINING_SAMPLE_FRACTION,
                max_samples=config.TRAINING_MAX_SAMPLES)
            self.stats = {}
            self.confidence = None  # Will be set after processing
    
        def process_date(self, target_date, output_path=None):
            print(f"\n{'='*70}")
            print(f"🚀 PROCESSING: {target_date}")
            print(f"{'='*70}")
    
            # Initialize confidence tracker
            conf = PipelineConfidence()
            conf.inference_date = target_date
            conf.inference_is_broken = INFERENCE_IS_BROKEN if 'INFERENCE_IS_BROKEN' in dir() else False
    
            # Step 1: Load target
            print("\n📥 Step 1: Loading target image...")
            target_data = self.loader.load_s2_full(target_date)
            target_s1 = self.loader.load_s1_full(target_date)
            original_dtype = target_data.dtype
            print(f"   Shape: {target_data.shape}, dtype: {original_dtype}")
    
            # Step 2: Cloud mask
            print("\n☁️ Step 2: Creating cloud mask...")
            scl_band = target_data[self.config.SCL_BAND_INDEX]
            cloud_mask, cloud_mask_buffered = create_cloud_mask(
                scl_band, buffer_size=self.config.BUFFER_PIXELS,
                cloud_classes=self.config.CLOUD_SCL_CLASSES)
    
            initial_cloud = int(np.sum(cloud_mask))
            total_px = cloud_mask.size
            cloud_pct = (initial_cloud / total_px) * 100
            buffered_px = int(np.sum(cloud_mask_buffered))
    
            print(f"   • Cloud: {initial_cloud:,} ({cloud_pct:.2f}%)")
            print(f"   • Buffered: {buffered_px:,}")
    
            # Populate confidence input metrics
            conf.total_pixels = total_px
            conf.cloud_pixels = initial_cloud
            conf.clear_pixels = total_px - initial_cloud
            conf.inference_cloud_cover_pct = cloud_pct
    
            # Get nodata from Cell 1 if available
            try:
                conf.inference_nodata_pct = NODATA_PCT_S2_PER_DATE.get(
                    target_date, 0.0) or 0.0
            except:
                conf.inference_nodata_pct = 0.0
    
            # No clouds case
            if initial_cloud == 0:
                print("✅ No clouds! Returning original.")
                if output_path:
                    self._save_result(target_data, target_date, output_path)
    
                conf.temporal_filled = 0
                conf.fusion_filled = 0
                conf.spatial_filled = 0
                conf.unfilled_pixels = 0
    
                # Count previous images (even though not used)
                prev_dates = get_previous_dates(
                    target_date, list(self.s2_paths.keys()),
                    self.config.MAX_PREVIOUS_IMAGES)
                conf.num_previous_images = len(prev_dates)
                conf.previous_image_dates = prev_dates
                conf.previous_image_cloud_pcts = [
                    cloud_stats.get(d, {}).get('cloud_percentage', 0)
                    for d in prev_dates]
                conf.previous_image_days_gap = [
                    days_between(target_date, d) for d in prev_dates]
    
                conf.calculate()
                self.confidence = conf
                self.stats[target_date] = {
                    'initial_cloud_pixels': 0,
                    'initial_cloud_percentage': 0,
                    'temporal_filled': 0, 'fusion_filled': 0,
                    'spatial_filled': 0, 'unfilled': 0,
                    'success_rate': 100.0}
                return target_data
    
            # Step 3: Load previous
            print("\n📥 Step 3: Loading previous images...")
            previous_dates = get_previous_dates(
                target_date, list(self.s2_paths.keys()),
                self.config.MAX_PREVIOUS_IMAGES)
    
            if not previous_dates:
                print("   ⚠️  No previous dates! Using all other dates...")
                all_other = [d for d in self.s2_paths.keys() if d != target_date]
                previous_dates = sorted(all_other, reverse=True)[
                    :self.config.MAX_PREVIOUS_IMAGES]
    
            # Populate confidence with previous image info
            conf.num_previous_images = len(previous_dates)
            conf.previous_image_dates = list(previous_dates)
            conf.previous_image_cloud_pcts = [
                cloud_stats.get(d, {}).get('cloud_percentage', 0)
                for d in previous_dates]
            conf.previous_image_days_gap = [
                days_between(target_date, d) for d in previous_dates]
    
            print(f"   • Previous dates ({len(previous_dates)}): {previous_dates}")
            if not previous_dates:
                print("   ⚠️  ZERO previous images available!")
    
            prev_data_list, prev_masks_list = [], []
            prev_s1_list, prev_s2_fusion, prev_cm_fusion = [], [], []
            days_list = []
    
            for pd in tqdm(previous_dates, desc="Loading previous"):
                pi = self.loader.load_s2_full(pd)
                ps1 = self.loader.load_s1_full(pd)
                pscl = pi[self.config.SCL_BAND_INDEX]
                pcm, _ = create_cloud_mask(
                    pscl, buffer_size=0,
                    cloud_classes=self.config.CLOUD_SCL_CLASSES)
    
                prev_data_list.append(pi)
                prev_masks_list.append(pcm)
                prev_s1_list.append(ps1)
                prev_s2_fusion.append(pi)
                prev_cm_fusion.append(pcm)
                days_list.append(days_between(target_date, pd))
    
                del pscl
                gc.collect()
    
            # Safe array creation for 0 previous images
            if prev_data_list:
                prev_arr = np.array(prev_data_list)
                prev_masks_arr = np.array(prev_masks_list)
            else:
                # Create empty arrays with correct shape
                n_bands = target_data.shape[0]
                h, w = target_data.shape[1], target_data.shape[2]
                prev_arr = np.empty((0, n_bands, h, w), dtype=target_data.dtype)
                prev_masks_arr = np.empty((0, h, w), dtype=bool)
    
            days_arr = np.array(days_list, dtype=np.float32) if days_list else np.array([], dtype=np.float32)
    
            del prev_data_list, prev_masks_list
            gc.collect()
    
            print(f"   • Days from target: {days_list}")
    
            # Step 4: Temporal fill
            print("\n⏱️ Step 4: Temporal rate-of-change interpolation...")
            spectral_idx = list(self.config.SPECTRAL_BAND_INDICES)
    
            filled_data, fill_success = calculate_temporal_rate_of_change_fast(
                target_data.astype(np.float32),
                prev_arr.astype(np.float32),
                cloud_mask, prev_masks_arr,
                days_arr, spectral_idx,
                self.config.MAX_CHANGE_RATE_PER_DAY)
    
            temporal_filled = int(np.sum(cloud_mask & fill_success))
            remaining_mask = cloud_mask & ~fill_success
            remaining = int(np.sum(remaining_mask))
    
            print(f"   ✅ Temporal: {temporal_filled:,}")
            print(f"   • Remaining: {remaining:,}")
    
            del prev_arr, prev_masks_arr
            gc.collect()
            if GPU_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
    
            # Step 5: S1-S2 Fusion
            fusion_filled = 0
            if remaining > 0 and prev_s2_fusion:
                print("\n🛰️ Step 5: S1-S2 Fusion...")
    
                self.fusion_model.train_multi_date(
                    prev_s2_fusion, prev_s1_list,
                    prev_cm_fusion, spectral_idx)
    
                if self.fusion_model.is_trained:
                    predictions = self.fusion_model.predict(
                        target_s1, remaining_mask, spectral_idx)
    
                    if predictions:
                        for bi, (rows, cols, vals) in predictions.items():
                            filled_data[bi, rows, cols] = vals
    
                        first_bi = list(predictions.keys())[0]
                        pr, pc, _ = predictions[first_bi]
                        fusion_filled = len(pr)
                        fill_success[pr, pc] = True
                        remaining_mask = cloud_mask & ~fill_success
                        remaining = int(np.sum(remaining_mask))
                        print(f"   ✅ Fusion: {fusion_filled:,}")
    
                    del predictions
                    gc.collect()
            elif remaining > 0:
                print("\n🛰️ Step 5: S1-S2 Fusion → SKIPPED (no training data)")
    
            del prev_s1_list, prev_s2_fusion, prev_cm_fusion, target_s1
            gc.collect()
    
            # Step 6: Spatial interpolation
            spatial_filled = 0
            if remaining > 0:
                print(f"\n🔲 Step 6: Spatial interpolation ({remaining:,} remaining)...")
    
                filled_spatial, spatial_success = apply_spatial_interpolation(
                    filled_data, remaining_mask, spectral_idx,
                    self.config.MAX_INTERPOLATION_DISTANCE)
                filled_data = filled_spatial
                spatial_filled = int(np.sum(remaining_mask & spatial_success))
                fill_success |= spatial_success
    
                del filled_spatial, spatial_success
                gc.collect()
                print(f"   ✅ Spatial: {spatial_filled:,}")
    
            # Step 7: Blend
            print("\n🎨 Step 7: Edge blending...")
            blend_w = create_blend_weights(
                cloud_mask, cloud_mask_buffered, self.config.BUFFER_PIXELS)
    
            result = blend_images(
                target_data, filled_data, blend_w, spectral_idx)
    
            del filled_data, blend_w
            gc.collect()
    
            # Ensure clear pixels untouched
            clear_mask = ~cloud_mask_buffered
            for bi in spectral_idx:
                result[bi][clear_mask] = target_data[bi][clear_mask].astype(np.float32)
    
            # Clip & cast
            if np.issubdtype(original_dtype, np.integer):
                max_val = np.iinfo(original_dtype).max
            else:
                max_val = np.finfo(original_dtype).max
            result = np.clip(result, 0, max_val).astype(original_dtype)
    
            # Keep aux bands
            for ai in self.config.AUX_BAND_INDICES:
                result[ai] = target_data[ai]
    
            # Save
            if output_path:
                print(f"\n💾 Step 8: Saving...")
                self._save_result(result, target_date, output_path)
    
            # Verify
            sample_bi = spectral_idx[0]
            clear_diff = np.sum(
                result[sample_bi][clear_mask].astype(np.float64) -
                target_data[sample_bi][clear_mask].astype(np.float64))
            print(f"   🔍 Clear pixel check: diff = {clear_diff}")
    
            # Stats
            final_unfilled = int(np.sum(cloud_mask & ~fill_success))
            self.stats[target_date] = {
                'initial_cloud_pixels': initial_cloud,
                'initial_cloud_percentage': cloud_pct,
                'temporal_filled': temporal_filled,
                'fusion_filled': fusion_filled,
                'spatial_filled': spatial_filled,
                'unfilled': final_unfilled,
                'success_rate': (
                    ((initial_cloud - final_unfilled) / initial_cloud * 100)
                    if initial_cloud > 0 else 100)}
    
            # Populate confidence fill metrics
            conf.temporal_filled = temporal_filled
            conf.fusion_filled = fusion_filled
            conf.spatial_filled = spatial_filled
            conf.unfilled_pixels = final_unfilled
    
            # Calculate confidence
            conf.calculate()
            self.confidence = conf
    
            print(f"\n{'='*70}")
            print(f"📊 PIPELINE SUMMARY:")
            print(f"   • Initial clouds: {initial_cloud:,} ({cloud_pct:.2f}%)")
            print(f"   • Temporal fill:  {temporal_filled:,}")
            print(f"   • S1-S2 fusion:   {fusion_filled:,}")
            print(f"   • Spatial fill:   {spatial_filled:,}")
            print(f"   • Unfilled:       {final_unfilled:,}")
            print(f"   • Success rate:   {self.stats[target_date]['success_rate']:.2f}%")
            print(f"{'='*70}")
    
            del target_data, cloud_mask, cloud_mask_buffered, fill_success, clear_mask
            gc.collect()
            if GPU_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
    
            return result
    
        def _save_result(self, data, date, output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with rasterio.open(self.s2_paths[date]) as src:
                profile = src.profile.copy()
            profile.update(dtype=data.dtype)
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data)
            print(f"   ✅ Saved: {output_path}")
    
    
    # ============================================================================
    # INITIALIZE AND RUN
    # ============================================================================
    pipeline = CloudRemovalPipeline(
        config, loader, S2_FILE_PATHS, S1_FILE_PATHS, BAND_NAMES)
    
    print(f"\n✅ Pipeline initialized!")
    print(f"\n🎯 Target: {TARGET_DATE}")
    print(f"📁 Output: {OUTPUT_PATH}")
    print(f"☁️ Cloud: {cloud_stats[TARGET_DATE]['cloud_percentage']:.2f}%")
    
    try:
        if 'CLOUD_PCT_PER_DATE' in dir():
            cl1 = CLOUD_PCT_PER_DATE.get(TARGET_DATE)
            nd1 = NODATA_PCT_S2_PER_DATE.get(TARGET_DATE)
            print(f"📊 Cell 1: Cloud={cl1}%, NoData={nd1}%")
        if 'INFERENCE_IS_BROKEN' in dir() and INFERENCE_IS_BROKEN:
            print(f"   ⚠️  BROKEN INFERENCE IMAGE")
    except:
        pass
    
    # RUN PIPELINE
    result = pipeline.process_date(TARGET_DATE, OUTPUT_PATH)
    
    print(f"\n✅ Processing complete!")
    print(f"   Result shape: {result.shape}")
    print(f"   Result dtype: {result.dtype}")
    
    # ============================================================================
    # Generate Vegetation Indices (NDRE and NDWI only)
    # ============================================================================
    generate_vegetation_indices(OUTPUT_PATH, OUTPUT_DIR, TARGET_DATE, BAND_NAMES)
    
    # ============================================================================
    # PRINT CONFIDENCE REPORT
    # ============================================================================
    if pipeline.confidence:
        pipeline.confidence.print_report()
    
    # ============================================================================
    # STORE CONFIDENCE VARIABLES
    # ============================================================================
    PIPELINE_CONFIDENCE = pipeline.confidence
    PIPELINE_STATS = pipeline.stats
    
    # Convenience variables
    CONFIDENCE_OVERALL = pipeline.confidence.confidence_overall if pipeline.confidence else 0.0
    CONFIDENCE_LEVEL = pipeline.confidence.confidence_level if pipeline.confidence else "UNKNOWN"
    CONFIDENCE_PREV_IMAGES = pipeline.confidence.confidence_prev_images if pipeline.confidence else 0.0
    CONFIDENCE_CLOUD_COVER = pipeline.confidence.confidence_cloud_cover if pipeline.confidence else 0.0
    CONFIDENCE_NODATA = pipeline.confidence.confidence_nodata if pipeline.confidence else 0.0
    CONFIDENCE_FILL_QUALITY = pipeline.confidence.confidence_fill_quality if pipeline.confidence else 0.0
    
    NUM_PREVIOUS_IMAGES = pipeline.confidence.num_previous_images if pipeline.confidence else 0
    INFERENCE_CLOUD_PCT = pipeline.confidence.inference_cloud_cover_pct if pipeline.confidence else 0.0
    INFERENCE_NODATA_PCT = pipeline.confidence.inference_nodata_pct if pipeline.confidence else 0.0
    
    FILL_TEMPORAL_PCT = (
        (pipeline.confidence.temporal_filled / max(pipeline.confidence.cloud_pixels, 1) * 100)
        if pipeline.confidence and pipeline.confidence.cloud_pixels > 0 else 0.0
    )
    FILL_FUSION_PCT = (
        (pipeline.confidence.fusion_filled / max(pipeline.confidence.cloud_pixels, 1) * 100)
        if pipeline.confidence and pipeline.confidence.cloud_pixels > 0 else 0.0
    )
    FILL_SPATIAL_PCT = (
        (pipeline.confidence.spatial_filled / max(pipeline.confidence.cloud_pixels, 1) * 100)
        if pipeline.confidence and pipeline.confidence.cloud_pixels > 0 else 0.0
    )
    FILL_UNFILLED_PCT = (
        (pipeline.confidence.unfilled_pixels / max(pipeline.confidence.cloud_pixels, 1) * 100)
        if pipeline.confidence and pipeline.confidence.cloud_pixels > 0 else 0.0
    )
    
    # ============================================================================
    # FINAL OUTPUT SUMMARY
    # ============================================================================
    print(f"\n{'='*70}")
    print(f"📦 FINAL OUTPUT")
    print(f"{'='*70}")
    
    output_dir_path = Path(OUTPUT_DIR)
    print(f"\n📁 {output_dir_path}/")
    for f in sorted(output_dir_path.glob("*.tif")):
        sz = f.stat().st_size / (1024 * 1024)
        print(f"   └── {f.name} ({sz:.1f} MB)")
    
    print(f"\n📁 {PAIRS_DIR}/")
    for folder in sorted(PAIRS_DIR.iterdir()):
        if folder.is_dir():
            print(f"   └── {folder.name}/")
            for f in sorted(folder.glob("*.tif")):
                sz = f.stat().st_size / (1024 * 1024)
                print(f"       └── {f.name} ({sz:.1f} MB)")
    
    print(f"\n{'='*70}")
    print(f"📦 CONFIDENCE VARIABLES")
    print(f"{'='*70}")
    print(f"""
      CONFIDENCE_OVERALL      : {CONFIDENCE_OVERALL:.1f}/100
      CONFIDENCE_LEVEL        : {CONFIDENCE_LEVEL}
      CONFIDENCE_PREV_IMAGES  : {CONFIDENCE_PREV_IMAGES:.1f}/100
      CONFIDENCE_CLOUD_COVER  : {CONFIDENCE_CLOUD_COVER:.1f}/100
      CONFIDENCE_NODATA       : {CONFIDENCE_NODATA:.1f}/100
      CONFIDENCE_FILL_QUALITY : {CONFIDENCE_FILL_QUALITY:.1f}/100
    
      NUM_PREVIOUS_IMAGES     : {NUM_PREVIOUS_IMAGES}
      INFERENCE_CLOUD_PCT     : {INFERENCE_CLOUD_PCT:.2f}%
      INFERENCE_NODATA_PCT    : {INFERENCE_NODATA_PCT:.2f}%
    
      FILL_TEMPORAL_PCT       : {FILL_TEMPORAL_PCT:.2f}%
      FILL_FUSION_PCT         : {FILL_FUSION_PCT:.2f}%
      FILL_SPATIAL_PCT        : {FILL_SPATIAL_PCT:.2f}%
      FILL_UNFILLED_PCT       : {FILL_UNFILLED_PCT:.2f}%
    
      PIPELINE_CONFIDENCE     : PipelineConfidence object (full details)
      PIPELINE_STATS          : {{date: stats_dict}}
    """)
    
    print("✅ ALL DONE")
    print(f"{'='*70}")

    return dict(
        INFERENCE_DATE=INFERENCE_DATE,
        CONFIDENCE_OVERALL=CONFIDENCE_OVERALL,
        CONFIDENCE_LEVEL=CONFIDENCE_LEVEL,
        CONFIDENCE_PREV_IMAGES=CONFIDENCE_PREV_IMAGES,
        CONFIDENCE_CLOUD_COVER=CONFIDENCE_CLOUD_COVER,
        CONFIDENCE_NODATA=CONFIDENCE_NODATA,
        CONFIDENCE_FILL_QUALITY=CONFIDENCE_FILL_QUALITY,
        NUM_PREVIOUS_IMAGES=NUM_PREVIOUS_IMAGES,
        INFERENCE_CLOUD_PCT=INFERENCE_CLOUD_PCT,
        INFERENCE_NODATA_PCT=INFERENCE_NODATA_PCT,
        FILL_TEMPORAL_PCT=FILL_TEMPORAL_PCT,
        FILL_FUSION_PCT=FILL_FUSION_PCT,
        FILL_SPATIAL_PCT=FILL_SPATIAL_PCT,
        FILL_UNFILLED_PCT=FILL_UNFILLED_PCT,
        OUTPUT_PATH=OUTPUT_PATH,
        OUTPUT_DIR=OUTPUT_DIR,
        TARGET_DATE=TARGET_DATE,
        BAND_NAMES=BAND_NAMES,
    )


if __name__ == "__main__":
    print(cfg.summary())
    # To run standalone, pass a result dict from 01_download.run()
    raise SystemExit("Run via run_pipeline.py or pass a download_result dict.")

"""
scripts/01_download.py
======================
Step 1 - Download Sentinel-1 & Sentinel-2 imagery into pairs/ folder.

Run standalone: python scripts/01_download.py
"""
import os
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
import re
import requests
import warnings
import json
import tempfile
import shutil
import copy
import time
warnings.filterwarnings('ignore')

try:
    import geopandas as gpd
except ImportError:
    print("⚠️  geopandas not installed")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from shapely.geometry import Point, Polygon, box, MultiPolygon, shape
from shapely.ops import unary_union
from shapely import wkt

try:
    import mgrs
    HAS_MGRS = True
except ImportError:
    HAS_MGRS = False

try:
    import openeo
    HAS_OPENEO = True
except ImportError:
    HAS_OPENEO = False

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import cfg

# Map cfg → variable names used by the original notebook Cell 1
AOI_GEOJSON_ENV   = cfg.AOI_GEOJSON
fecha_inicio_ENV  = cfg.fecha_inicio_dt
fecha_fin_ENV     = cfg.fecha_fin_dt
OUT_DIR_ENV       = cfg.OUT_DIR
NAME_CONFIG_ENV   = cfg.NAME_CONFIG


# Aliases + constants
AOI_GEOJSON            = AOI_GEOJSON_ENV
fecha_inicio           = fecha_inicio_ENV
fecha_fin              = fecha_fin_ENV
DOWNLOAD_BASE_DIR      = OUT_DIR_ENV
PAIRS_DIR              = DOWNLOAD_BASE_DIR / "pairs"

LOOKBACK_DAYS          = 30
PREVIOUS_IMAGES_COUNT  = 5
TARGET_CRS             = "EPSG:4326"
MARGIN_DEGREES         = 0.001
MIN_SPATIAL_COVERAGE_PCT = 95.0
MAX_NODATA_PCT         = 10.0
PREVIEW_RESOLUTION     = 100
OPENEO_BACKEND         = "https://openeo.dataspace.copernicus.eu"
S2_COLLECTION          = "SENTINEL2_L2A"
S2_BANDS               = ["B01","B02","B03","B04","B05","B06","B07",
                          "B08","B8A","B09","B11","B12","WVP","AOT","SCL"]
S2_EXPECTED_BANDS      = len(S2_BANDS)
TARGET_RESOLUTION      = 10
S1_COLLECTION          = "SENTINEL1_GRD"
S1_BANDS               = ["VV", "VH"]
S1_EXPECTED_BANDS      = len(S1_BANDS)
S1_DATE_BUFFER         = 60
MAX_RETRIES            = 3
POLL_INTERVAL          = 30
JOB_TIMEOUT            = 7200
CHUNK_SIZE             = 8 * 1024 * 1024
TIMEOUT                = 300
RETRY_DELAY            = 10


def get_band_count(filepath: Path) -> int:
    """Get band count of a raster file. Returns 0 on error."""
    if not HAS_RASTERIO or not filepath or not filepath.exists():
        return 0
    try:
        with rasterio.open(filepath) as src:
            return src.count
    except Exception:
        return 0


def is_valid_s1_file(filepath: Path) -> bool:
    """Check if file is a valid S1 file (2 bands, reasonable size)."""
    if not filepath or not filepath.exists():
        return False
    if filepath.stat().st_size < 1000:
        return False
    bands = get_band_count(filepath)
    return bands == S1_EXPECTED_BANDS


def is_valid_s2_file(filepath: Path) -> bool:
    """Check if file is a valid S2 file (15 bands, reasonable size)."""
    if not filepath or not filepath.exists():
        return False
    if filepath.stat().st_size < 1000:
        return False
    bands = get_band_count(filepath)
    return bands == S2_EXPECTED_BANDS


# =============================================================================
# METADATA TRACKER
# =============================================================================
class DateMetadata:
    """Track metadata for each downloaded date."""
    def __init__(self, date_str: str, satellite: str):
        self.date = date_str
        self.satellite = satellite
        self.cloud_cover_pct = None
        self.nodata_pct = None
        self.spatial_coverage_pct = None
        self.tiles = []
        self.is_complete = None
        self.is_inference = False
        self.is_broken_allowed = False
        self.download_path = None
        self.file_size_mb = None
        self.valid_pixel_pct = None
        self.band_count = None

    def to_dict(self):
        return {k: (str(v) if isinstance(v, Path) else v)
                for k, v in self.__dict__.items()}

    def __repr__(self):
        status = "✅" if self.is_complete else "⚠️"
        cloud = f"Cloud:{self.cloud_cover_pct:.1f}%" if self.cloud_cover_pct is not None else "Cloud:N/A"
        nodata = f"NoData:{self.nodata_pct:.1f}%" if self.nodata_pct is not None else "NoData:N/A"
        bands = f"Bands:{self.band_count}" if self.band_count is not None else ""
        role = "[INFERENCE]" if self.is_inference else "[PREVIOUS]"
        return f"{status} {self.satellite} {self.date} {role} {cloud} {nodata} {bands}"


# =============================================================================
# PAIRS FOLDER MANAGEMENT
# =============================================================================
def parse_pair_folder_name(folder_name: str) -> dict:
    m = re.match(r'^inference_(\d{4}-\d{2}-\d{2})$', folder_name)
    if m:
        return {'role': 'inference', 'index': None, 'date': m.group(1)}
    m = re.match(r'^prev(\d{2})_(\d{4}-\d{2}-\d{2})$', folder_name)
    if m:
        return {'role': 'previous', 'index': int(m.group(1)), 'date': m.group(2)}
    return None


def find_s2_file_in_folder(folder: Path) -> Path:
    """Find a valid S2 file in folder. Returns None if not found."""
    if not folder or not folder.exists():
        return None
    for f in sorted(folder.glob("S2_*.tif"), key=lambda x: x.stat().st_mtime, reverse=True):
        if is_valid_s2_file(f):
            return f
    for f in sorted(folder.glob("openEO_*.tif"), key=lambda x: x.stat().st_mtime, reverse=True):
        if is_valid_s2_file(f):
            return f
    return None


def find_s1_file_in_folder(folder: Path) -> Path:
    """
    Find a valid S1 file in folder (2 bands only).
    NEVER matches openEO_*.tif - those are S2.
    """
    if not folder or not folder.exists():
        return None

    s1_candidates = list(folder.glob("s1_*.tif")) + list(folder.glob("S1_*.tif"))

    for f in s1_candidates:
        if is_valid_s1_file(f):
            return f

    return None


def find_s1_raw_file_in_folder(folder: Path) -> Path:
    """Find a valid non-filled S1 file in folder."""
    if not folder or not folder.exists():
        return None
    s1_candidates = list(folder.glob("s1_*.tif")) + list(folder.glob("S1_*.tif"))
    for f in s1_candidates:
        if "_filled" not in f.name and is_valid_s1_file(f):
            return f
    return None


def scan_existing_pairs(pairs_dir: Path) -> dict:
    result = {
        'inference': None, 'previous': [], 'all_dates': {},
        'all_s2_dates': set(), 'all_s1_dates': set(),
    }
    if not pairs_dir.exists():
        return result

    for folder in sorted(pairs_dir.iterdir()):
        if not folder.is_dir():
            continue
        parsed = parse_pair_folder_name(folder.name)
        if not parsed:
            continue

        date_str = parsed['date']

        s2_file = find_s2_file_in_folder(folder)
        s1_file = find_s1_file_in_folder(folder)
        s2_exists = s2_file is not None
        s1_exists = s1_file is not None

        s2_date = None
        if s2_file:
            m = re.search(r'(\d{4})-?(\d{2})-?(\d{2})', s2_file.name)
            if m:
                s2_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

        s1_date = None
        if s1_file:
            m = re.search(r'(\d{4})-?(\d{2})-?(\d{2})', s1_file.name)
            if m:
                s1_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

        entry = {
            'date': date_str, 'folder': folder,
            's2_exists': s2_exists, 's1_exists': s1_exists,
            's2_file': s2_file, 's1_file': s1_file,
            's2_date': s2_date or date_str, 's1_date': s1_date,
            'role': parsed['role'], 'index': parsed.get('index'),
        }

        if parsed['role'] == 'inference':
            result['inference'] = entry
        else:
            result['previous'].append(entry)

        result['all_dates'][date_str] = folder
        if s2_exists:
            result['all_s2_dates'].add(date_str)
        if s1_exists and s1_date:
            result['all_s1_dates'].add(s1_date)

    result['previous'].sort(key=lambda x: x.get('index', 99))
    return result


def reorganize_pairs_for_new_inference(
    pairs_dir: Path, new_inference_date: str,
    new_previous_dates: list, existing_info: dict
) -> dict:
    print(f"\n🔄 Reorganizing pairs folder for new inference: {new_inference_date}")

    existing_inference = existing_info.get('inference')
    existing_previous = existing_info.get('previous', [])

    if existing_inference and existing_inference['date'] != new_inference_date:
        old_inf_date = existing_inference['date']
        old_inf_folder = existing_inference['folder']

        max_idx = 0
        for entry in existing_previous:
            if entry['index'] is not None:
                max_idx = max(max_idx, entry['index'])

        for idx in range(max_idx, 0, -1):
            old_name = None
            for entry in existing_previous:
                if entry['index'] == idx:
                    old_name = entry['folder']
                    break
            if old_name and old_name.exists():
                new_idx = idx + 1
                entry_date = parse_pair_folder_name(old_name.name)['date']
                new_name = pairs_dir / f"prev{new_idx:02d}_{entry_date}"
                if old_name != new_name:
                    print(f"   📁 Rename: {old_name.name} → {new_name.name}")
                    if new_name.exists():
                        shutil.rmtree(new_name)
                    old_name.rename(new_name)

        new_prev01_name = pairs_dir / f"prev01_{old_inf_date}"
        if old_inf_folder.exists():
            print(f"   📁 Rename: {old_inf_folder.name} → {new_prev01_name.name}")
            if new_prev01_name.exists():
                shutil.rmtree(new_prev01_name)
            old_inf_folder.rename(new_prev01_name)

    elif existing_inference and existing_inference['date'] == new_inference_date:
        print(f"   ✅ Inference folder already correct: {new_inference_date}")

    current_info = scan_existing_pairs(pairs_dir)
    needed_dates = set([new_inference_date] + new_previous_dates)

    for entry in current_info.get('previous', []):
        if entry['date'] not in needed_dates:
            if entry.get('index', 0) > PREVIOUS_IMAGES_COUNT:
                print(f"   🗑️  Removing excess: {entry['folder'].name}")
                shutil.rmtree(entry['folder'])

    current_info = scan_existing_pairs(pairs_dir)

    target_folders = {}
    target_folders[new_inference_date] = pairs_dir / f"inference_{new_inference_date}"
    for i, d in enumerate(new_previous_dates, 1):
        target_folders[d] = pairs_dir / f"prev{i:02d}_{d}"

    dates_already_have_s2 = set()
    dates_already_have_s1 = set()

    for folder in pairs_dir.iterdir():
        if not folder.is_dir():
            continue
        parsed = parse_pair_folder_name(folder.name)
        if not parsed:
            continue

        folder_date = parsed['date']

        if folder_date in target_folders:
            target_path = target_folders[folder_date]

            if folder != target_path:
                print(f"   📁 Rename: {folder.name} → {target_path.name}")
                if target_path.exists() and target_path != folder:
                    shutil.rmtree(target_path)
                folder.rename(target_path)
                folder = target_path

            if find_s2_file_in_folder(folder) is not None:
                dates_already_have_s2.add(folder_date)
            if find_s1_file_in_folder(folder) is not None:
                dates_already_have_s1.add(folder_date)

    all_needed_dates = [new_inference_date] + new_previous_dates
    s2_to_download = [d for d in all_needed_dates if d not in dates_already_have_s2]

    return {
        'dates_already_have_s2': dates_already_have_s2,
        'dates_already_have_s1': dates_already_have_s1,
        's2_dates_to_download': s2_to_download,
        'target_folders': target_folders,
    }


# =============================================================================
# DATE RANGE FUNCTIONS
# =============================================================================
def validate_and_build_date_range(start: datetime, end: datetime) -> list:
    delta = (end - start).days
    if delta < 0:
        raise ValueError(f"End date must be after start date")
    if delta > 4:
        print(f"⚠️  Range is {delta+1} days. Clamping to 5 from start.")
        end = start + timedelta(days=4)
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    while len(dates) < 5:
        dates.append(dates[-1] + timedelta(days=1))
    return dates


def get_lookback_range(ref: datetime, days: int = 30):
    return ref - timedelta(days=days), ref


# =============================================================================
# AOI & SPATIAL
# =============================================================================
def load_aoi_geometry():
    if not AOI_GEOJSON.exists():
        raise FileNotFoundError(f"AOI not found: {AOI_GEOJSON}")

    print(f"📍 Loading AOI: {AOI_GEOJSON}")
    gdf = gpd.read_file(AOI_GEOJSON)
    if gdf.crs is None:
        gdf = gdf.set_crs(TARGET_CRS)
    elif gdf.crs.to_string() != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)

    geom = gdf.geometry.iloc[0]
    bounds = geom.bounds
    print(f"✅ CRS: {gdf.crs}")
    print(f"📍 Bounds: W={bounds[0]:.6f} S={bounds[1]:.6f} "
          f"E={bounds[2]:.6f} N={bounds[3]:.6f}")

    centroid = geom.centroid
    utm_zone = int((centroid.x + 180) / 6) + 1
    utm_epsg = 32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone
    area_km2 = gdf.to_crs(epsg=utm_epsg).geometry.area.iloc[0] / 1e6
    print(f"📏 Area: {area_km2:.4f} km²")

    geojson_geom = json.loads(gdf.to_json())["features"][0]["geometry"]

    return gdf, {
        "west": bounds[0], "south": bounds[1],
        "east": bounds[2], "north": bounds[3], "crs": TARGET_CRS,
    }, geojson_geom


def add_margin(extent: dict, margin: float) -> dict:
    result = {
        "west": extent["west"] - margin, "south": extent["south"] - margin,
        "east": extent["east"] + margin, "north": extent["north"] + margin,
        "crs": "EPSG:4326",
    }
    print(f"📐 MARGIN: {margin}° (~{margin*111:.0f}m)")
    return result


def get_mgrs_tiles(aoi_geom) -> list:
    if not HAS_MGRS:
        return []
    m = mgrs.MGRS()
    tiles = set()
    b = aoi_geom.bounds
    step = 0.05
    x = b[0]
    while x <= b[2]:
        y = b[1]
        while y <= b[3]:
            try:
                tiles.add(m.toMGRS(y, x, MGRSPrecision=0)[:5])
            except:
                pass
            y += step
        x += step
    for lon, lat in [(b[0],b[1]),(b[0],b[3]),(b[2],b[1]),(b[2],b[3]),
                     ((b[0]+b[2])/2,(b[1]+b[3])/2)]:
        try:
            tiles.add(m.toMGRS(lat, lon, MGRSPrecision=0)[:5])
        except:
            pass
    return sorted(tiles)


def parse_footprint(fp_str: str):
    if not fp_str:
        return None
    try:
        fp_str = fp_str.strip()
        if fp_str.startswith("geography'"):
            m = re.search(r"SRID=\d+;(.+)'$", fp_str)
            if m:
                fp_str = m.group(1)
        if fp_str.upper().startswith(('POLYGON', 'MULTIPOLYGON')):
            return wkt.loads(fp_str)
        if fp_str.startswith('{'):
            return shape(json.loads(fp_str))
    except:
        pass
    return None


# =============================================================================
# ODATA QUERIES
# =============================================================================
def query_s2_products(spatial_extent: dict, start: str, end: str) -> dict:
    ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    w, s, e, n = spatial_extent["west"], spatial_extent["south"], spatial_extent["east"], spatial_extent["north"]

    filt = (
        "Collection/Name eq 'SENTINEL-2' and "
        "Attributes/OData.CSC.StringAttribute/any("
        "att:att/Name eq 'productType' and "
        "att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and "
        f"ContentDate/Start ge {start}T00:00:00.000Z and "
        f"ContentDate/Start le {end}T23:59:59.999Z and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;"
        f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))')"
    )

    print(f"\n📡 Querying S2 L2A: {start} → {end}")
    dates_products = defaultdict(dict)

    try:
        resp = requests.get(ODATA, params={
            "$filter": filt, "$top": 1000,
            "$orderby": "ContentDate/Start desc", "$expand": "Attributes",
        }, timeout=120)
        resp.raise_for_status()
        products = resp.json().get("value", [])
        print(f"📦 {len(products)} S2 products found")

        for prod in products:
            name = prod.get("Name", "")
            dm = re.search(r"_(\d{8})T\d{6}_", name)
            tm = re.search(r"_T(\d{2}[A-Z]{3})_", name)
            if not dm or not tm:
                continue
            d = dm.group(1)
            date_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            tile_id = tm.group(1)
            fp = parse_footprint(prod.get("Footprint","") or prod.get("GeoFootprint",""))

            cloud = None
            for attr in prod.get("Attributes", []):
                if "cloudcover" in attr.get("Name","").lower():
                    try:
                        cloud = float(attr.get("Value", 0))
                    except:
                        pass
                    break

            dates_products[date_str][tile_id] = {
                'product_name': name, 'product_id': prod.get("Id",""),
                'footprint': fp, 'cloud_cover': cloud,
            }
    except requests.exceptions.RequestException as ex:
        print(f"❌ OData error: {ex}")

    return dict(dates_products)


def query_s1_products(spatial_extent: dict, start: str, end: str) -> dict:
    ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    w, s, e, n = spatial_extent["west"], spatial_extent["south"], spatial_extent["east"], spatial_extent["north"]

    filt = (
        "Collection/Name eq 'SENTINEL-1' and "
        "Attributes/OData.CSC.StringAttribute/any("
        "att:att/Name eq 'productType' and "
        "att/OData.CSC.StringAttribute/Value eq 'IW_GRDH_1S') and "
        f"ContentDate/Start ge {start}T00:00:00.000Z and "
        f"ContentDate/Start le {end}T23:59:59.999Z and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;"
        f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))')"
    )

    print(f"\n📡 Querying S1 GRD: {start} → {end}")
    dates_products = defaultdict(list)

    try:
        resp = requests.get(ODATA, params={
            "$filter": filt, "$top": 1000,
            "$orderby": "ContentDate/Start desc",
        }, timeout=120)
        resp.raise_for_status()
        products = resp.json().get("value", [])
        print(f"📦 {len(products)} S1 products found")

        for prod in products:
            name = prod.get("Name", "")
            dm = re.search(r"_(\d{8})T\d{6}_", name)
            if dm:
                d = dm.group(1)
                date_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
                dates_products[date_str].append({
                    'product_name': name, 'product_id': prod.get("Id",""),
                })
    except requests.exceptions.RequestException as ex:
        print(f"❌ OData error: {ex}")

    return dict(dates_products)


# =============================================================================
# COVERAGE CHECK (NO cloud filtering)
# =============================================================================
def check_spatial_coverage(date: str, products: dict, aoi_geom) -> dict:
    if not products:
        return {'date': date, 'spatial_coverage_pct': 0.0, 'is_complete': False,
                'tiles': [], 'avg_cloud_cover': None, 'per_tile_cloud': {},
                'reason': 'No products'}

    aoi_area = aoi_geom.area
    valid_fps, tiles, clouds, ptc = [], [], [], {}

    for tid, info in products.items():
        fp = info.get('footprint')
        cc = info.get('cloud_cover')
        if fp is not None and fp.intersects(aoi_geom):
            valid_fps.append(fp)
            tiles.append(tid)
            ptc[tid] = cc
            if cc is not None:
                clouds.append(cc)

    cov = 0.0
    if valid_fps:
        try:
            cov = (unary_union(valid_fps).intersection(aoi_geom).area / aoi_area) * 100
        except:
            pass

    avg_cloud = np.mean(clouds) if clouds else None
    complete = cov >= MIN_SPATIAL_COVERAGE_PCT

    return {
        'date': date, 'spatial_coverage_pct': cov, 'is_complete': complete,
        'tiles': tiles, 'avg_cloud_cover': avg_cloud, 'per_tile_cloud': ptc,
        'reason': 'OK' if complete else f'Spatial: {cov:.1f}%'
    }


# =============================================================================
# PREVIEW VALIDATION (nodata + cloud measurement, NO filtering on cloud)
# =============================================================================
def validate_nodata_preview(date: str, spatial_extent: dict,
                            connection=None, resolution: int = PREVIEW_RESOLUTION) -> dict:
    if not HAS_OPENEO or not HAS_RASTERIO:
        return {'date': date, 'validated': False, 'nodata_pct': None,
                'reason': 'openEO/rasterio N/A'}
    try:
        if connection is None:
            connection = openeo.connect(OPENEO_BACKEND)
            connection.authenticate_oidc()

        scl = connection.load_collection(
            "SENTINEL2_L2A", spatial_extent=spatial_extent,
            temporal_extent=[date, date], bands=["SCL"]
        ).resample_spatial(resolution=resolution)

        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            scl.download(tmp_path, format="GTiff")
            with rasterio.open(tmp_path) as src:
                data = src.read(1)
                total = data.size
                nd_mask = (data == 0) | (data == -32768) | (data == 255)
                nd_pct = (np.sum(nd_mask) / total) * 100
                cl_mask = (data == 8) | (data == 9) | (data == 10)
                cl_pct = (np.sum(cl_mask) / total) * 100
                valid_mask = (data >= 4) & (data <= 7)
                valid_pct = (np.sum(valid_mask) / total) * 100

            os.unlink(tmp_path)
            return {
                'date': date, 'validated': True,
                'nodata_pct': nd_pct, 'cloud_pct_scl': cl_pct,
                'valid_pct': valid_pct,
                'is_complete': nd_pct < MAX_NODATA_PCT,
                'reason': 'OK' if nd_pct < MAX_NODATA_PCT else f'NoData:{nd_pct:.1f}%'
            }
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e
    except Exception as e:
        return {'date': date, 'validated': False, 'nodata_pct': None,
                'reason': f'Error: {str(e)[:80]}'}


# =============================================================================
# PROGRESS & DOWNLOAD HELPERS
# =============================================================================
def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        return f"{seconds // 3600:.0f}h {(seconds % 3600) // 60:.0f}m"


def print_progress_bar(progress: float, width: int = 40, status: str = ""):
    filled = int(width * progress)
    bar = "█" * filled + "░" * (width - filled)
    percentage = progress * 100
    sys.stdout.write(f"\r   [{bar}] {percentage:5.1f}% {status}")
    sys.stdout.flush()


def wait_for_job_with_progress(job, job_name: str, timeout: int = JOB_TIMEOUT) -> bool:
    start_time = time.time()
    last_status = None
    progress_map = {
        "created": 0.05, "queued": 0.10, "running": 0.15,
        "finished": 1.0, "error": -1, "canceled": -1
    }

    print(f"\n   ⏳ Job submitted. Tracking progress...")
    print(f"   Job ID: {job.job_id}")

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"\n   ❌ Timeout after {format_time(elapsed)}")
            return False

        try:
            status = job.status()
            job_info = job.describe()

            progress = job_info.get("progress", 0)
            if isinstance(progress, (int, float)):
                progress = progress / 100 if progress > 1 else progress
            else:
                progress = progress_map.get(status, 0.1)

            if status == "running":
                time_progress = min(0.9, elapsed / 2400)
                progress = max(0.15, min(0.95, time_progress))

            status_text = f"Status: {status} | Elapsed: {format_time(elapsed)}"

            if status != last_status:
                print()
                print(f"   📊 Status changed: {last_status} → {status}")
                last_status = status

            if status == "finished":
                print_progress_bar(1.0, status=status_text)
                print(f"\n   ✅ Job completed in {format_time(elapsed)}")
                return True

            elif status in ["error", "canceled"]:
                print(f"\n   ❌ Job {status}")
                try:
                    logs = job.logs()
                    if logs:
                        print("   Error logs:")
                        for log in logs[-5:]:
                            print(f"      {log}")
                except:
                    pass
                return False

            else:
                print_progress_bar(progress, status=status_text)

            time.sleep(POLL_INTERVAL)

        except Exception as e:
            print(f"\n   ⚠️ Error checking status: {e}")
            time.sleep(POLL_INTERVAL)


def download_with_retry_job(job, output_dir: Path, max_retries: int = MAX_RETRIES) -> bool:
    for attempt in range(max_retries):
        try:
            print(f"\n   📥 Downloading results to: {output_dir}")
            print(f"      Attempt {attempt + 1}/{max_retries}")

            start_time = time.time()
            results = job.get_results()
            results.download_files(output_dir)
            elapsed = time.time() - start_time

            downloaded = list(output_dir.glob("*.tif")) + list(output_dir.glob("*.nc"))
            if downloaded:
                for f in downloaded:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    bands = get_band_count(f)
                    print(f"   ✅ Downloaded: {f.name} ({size_mb:.2f} MB, {bands} bands) in {format_time(elapsed)}")
                return True

            print(f"   ⚠️ No files found after download")

        except Exception as e:
            print(f"   ❌ Download error: {e}")
            if attempt < max_retries - 1:
                wait_time = 30 * (attempt + 1)
                print(f"   ⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

    return False


def download_with_resume(url: str, output_path: Path, description: str = "", session=None):
    temp_path = output_path.with_suffix('.partial')
    resume_pos = 0
    if temp_path.exists():
        resume_pos = temp_path.stat().st_size
        print(f"            📥 Resuming from {resume_pos / 1024 / 1024:.1f} MB...")

    headers = {}
    if resume_pos > 0:
        headers['Range'] = f'bytes={resume_pos}-'
    mode = 'ab' if resume_pos > 0 else 'wb'

    try:
        if session is None:
            response = requests.get(url, headers=headers, stream=True, timeout=TIMEOUT)
        else:
            response = session.get(url, headers=headers, stream=True, timeout=TIMEOUT)
        response.raise_for_status()

        total_size = resume_pos
        if 'content-length' in response.headers:
            total_size += int(response.headers['content-length'])
        elif 'content-range' in response.headers:
            total_size = int(response.headers['content-range'].split('/')[-1])

        downloaded = resume_pos
        last_print = time.time()

        with open(temp_path, mode) as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if time.time() - last_print > 2:
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"            📊 Progress: {downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB ({percent:.1f}%)")
                        else:
                            print(f"            📊 Downloaded: {downloaded/1024/1024:.1f} MB")
                        last_print = time.time()

        if temp_path.exists():
            temp_path.rename(output_path)
        file_size = output_path.stat().st_size
        print(f"            ✅ {description} complete ({file_size / 1024 / 1024:.1f} MB)")
        return True

    except (requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout) as e:
        print(f"            ⚠️  Download interrupted: {type(e).__name__}")
        return False
    except Exception as e:
        print(f"            ❌ Download error: {type(e).__name__}: {str(e)}")
        if temp_path.exists():
            temp_path.unlink()
        raise


def download_cube_with_resume(conn, cube, output_path: Path, description: str = "", job_options: dict = None):
    print(f"         Creating batch job for {description}...")
    default_options = {}
    if job_options:
        default_options.update(job_options)

    job = cube.create_job(
        title=f"{description}_{output_path.stem}",
        out_format="GTiff",
        job_options=default_options if default_options else None
    )

    print(f"         Starting job {job.job_id}...")
    job.start_and_wait(
        max_poll_interval=60,
        connection_retry_interval=30,
        soft_error_max=10
    )

    print(f"         Job completed, downloading result...")
    results = job.get_results()
    assets = results.get_assets()

    if not assets:
        raise Exception("No assets found in job results")

    asset = assets[0]
    download_url = asset.href
    print(f"         Download URL obtained, starting chunked download...")

    max_attempts = MAX_RETRIES
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"         Attempt {attempt}/{max_attempts}")
            success = download_with_resume(
                download_url, output_path, description,
                session=conn._session if hasattr(conn, '_session') else None
            )
            if success:
                if not output_path.exists():
                    raise Exception("Download reported success but file not found")
                file_size = output_path.stat().st_size
                if file_size == 0:
                    output_path.unlink()
                    raise Exception("Downloaded file is empty")
                return True

            if attempt < max_attempts:
                print(f"         🔄 Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)

        except Exception as e:
            if attempt >= max_attempts:
                raise
            print(f"         ⚠️  Error on attempt {attempt}: {type(e).__name__}")
            print(f"         🔄 Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)

    raise Exception(f"Failed after {max_attempts} attempts")


# =============================================================================
# RASTER VALIDATION
# =============================================================================
def validate_raster(filepath: Path, expected_crs: str = None,
                    expected_bands: int = None) -> dict:
    with rasterio.open(filepath) as src:
        bounds = src.bounds
        crs = src.crs
        res = src.res
        shape_hw = (src.height, src.width)
        data = src.read()
        non_zero_ratio = np.count_nonzero(data) / data.size

        info = {
            "path": filepath, "crs": str(crs), "bounds": bounds,
            "resolution": res, "shape": shape_hw, "bands": src.count,
            "dtype": str(src.dtypes[0]), "non_zero_ratio": non_zero_ratio,
            "min": float(np.nanmin(data)), "max": float(np.nanmax(data)),
            "mean": float(np.nanmean(data)), "valid": True, "issues": []
        }

        if expected_bands is not None and src.count != expected_bands:
            info["issues"].append(
                f"CRITICAL: Expected {expected_bands} bands, got {src.count}")
            info["valid"] = False
        if non_zero_ratio < 0.01:
            info["issues"].append("CRITICAL: >99% zeros - likely data loss")
            info["valid"] = False
        if crs is None:
            info["issues"].append("CRITICAL: No CRS defined")
            info["valid"] = False
        elif expected_crs and str(crs) != expected_crs:
            info["issues"].append(f"WARNING: CRS mismatch - expected {expected_crs}, got {crs}")
        if crs and crs.is_projected:
            if abs(bounds.left) < 180 and abs(bounds.right) < 180:
                if abs(bounds.left) < 1000:
                    info["issues"].append("CRITICAL: Bounds appear to be in degrees but CRS is projected")
                    info["valid"] = False
            if res[0] < 1 or res[1] < 1:
                info["issues"].append(f"CRITICAL: Sub-meter resolution ({res}) - likely transform error")
                info["valid"] = False

        return info


def print_raster_validation(info: dict):
    print(f"\n   📊 Raster Validation: {info['path'].name}")
    print(f"      CRS: {info['crs']}")
    print(f"      Bounds: {info['bounds']}")
    print(f"      Resolution: {info['resolution']}")
    print(f"      Shape: {info['shape']}")
    print(f"      Bands: {info['bands']}")
    print(f"      Non-zero: {info['non_zero_ratio']*100:.1f}%")
    print(f"      Values: min={info['min']:.6f}, max={info['max']:.6f}, mean={info['mean']:.6f}")
    if info['valid']:
        print(f"      ✅ VALID")
    else:
        print(f"      ❌ INVALID")
        for issue in info['issues']:
            print(f"         ⚠️  {issue}")


# =============================================================================
# S2 DOWNLOAD - BATCH JOB METHOD
# =============================================================================
def download_s2_to_folder(date: str, target_folder: Path,
                          spatial_extent: dict, connection=None,
                          date_index: int = 1, total_dates: int = 1) -> Path:
    """Download S2 data using batch job with progress tracking and retry."""
    target_folder.mkdir(parents=True, exist_ok=True)

    existing = find_s2_file_in_folder(target_folder)
    if existing:
        print(f"   ⏭️  S2 exists: {existing.name} ({get_band_count(existing)} bands)")
        return existing

    print("\n" + "=" * 70)
    print(f"🛰️  SENTINEL-2 L2A DOWNLOAD [{date_index}/{total_dates}]")
    print("=" * 70)
    print(f"   Date: {date}")
    print(f"   Output Dir: {target_folder}")
    print(f"   Bands: {S2_EXPECTED_BANDS} bands ({', '.join(S2_BANDS)})")
    print(f"   Resolution: {TARGET_RESOLUTION}m")
    print(f"   CRS: {TARGET_CRS}")
    print(f"   Margin: {MARGIN_DEGREES} degrees (~{MARGIN_DEGREES * 111:.0f}m)")

    temporal_extent = [date, date]

    print(f"\n   📍 Spatial Extent (with margin):")
    print(f"      West:  {spatial_extent['west']:.6f}")
    print(f"      South: {spatial_extent['south']:.6f}")
    print(f"      East:  {spatial_extent['east']:.6f}")
    print(f"      North: {spatial_extent['north']:.6f}")

    try:
        if connection is None:
            connection = openeo.connect(OPENEO_BACKEND)
            connection.authenticate_oidc()

        print("\n   📦 Building S2 data cube...")
        s2_cube = connection.load_collection(
            S2_COLLECTION,
            spatial_extent=spatial_extent,
            temporal_extent=temporal_extent,
            bands=S2_BANDS
        )

        print(f"   🔄 Resampling to {TARGET_RESOLUTION}m...")
        s2_cube = s2_cube.resample_spatial(resolution=TARGET_RESOLUTION)

        print("\n   🚀 Creating batch job...")
        job = s2_cube.create_job(
            title=f"S2_L2A_{date}",
            description=f"Sentinel-2 L2A {S2_EXPECTED_BANDS} bands for {date}",
            out_format="GTiff",
            job_options={
                "driver-memory": "4g",
                "executor-memory": "4g"
            }
        )

        job.start_job()
        print(f"   ✅ Job started: {job.job_id}")

        if not wait_for_job_with_progress(job, "S2"):
            return None

        if not download_with_retry_job(job, target_folder):
            return None

        actual_file = find_s2_file_in_folder(target_folder)
        if actual_file:
            bands = get_band_count(actual_file)
            print(f"\n   🎉 S2 download complete for {date}!")
            print(f"   📁 File: {actual_file.name} "
                  f"({actual_file.stat().st_size/(1024*1024):.2f} MB, {bands} bands)")
            if bands != S2_EXPECTED_BANDS:
                print(f"   ⚠️  WARNING: Expected {S2_EXPECTED_BANDS} bands, got {bands}")
            return actual_file
        else:
            all_tifs = list(target_folder.glob("*.tif"))
            for f in sorted(all_tifs, key=lambda x: x.stat().st_mtime, reverse=True):
                bands = get_band_count(f)
                if bands == S2_EXPECTED_BANDS:
                    print(f"\n   🎉 S2 download complete for {date}!")
                    print(f"   📁 File: {f.name} "
                          f"({f.stat().st_size/(1024*1024):.2f} MB, {bands} bands)")
                    return f
            print(f"   ❌ No valid S2 file ({S2_EXPECTED_BANDS} bands) found after download")
            return None

    except Exception as e:
        print(f"\n   ❌ S2 Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# S1 DOWNLOAD - RAW GRD WITH 60-DAY MEDIAN COMPOSITE
# =============================================================================
def download_s1_to_folder(date: str, target_folder: Path,
                          spatial_extent: dict, connection=None,
                          date_index: int = 1, total_dates: int = 1) -> Path:
    """
    Download S1 data as RAW GRD with 60-day median composite.
    Uses ascending orbit, no sar_backscatter to avoid LUT errors.
    ONLY checks for s1_*/S1_* prefixed files. NEVER matches openEO_* (those are S2).
    """
    target_folder.mkdir(parents=True, exist_ok=True)

    existing = find_s1_raw_file_in_folder(target_folder)
    if existing:
        print(f"   ⏭️  S1 exists: {existing.name} ({get_band_count(existing)} bands)")
        return existing

    existing_any = find_s1_file_in_folder(target_folder)
    if existing_any:
        print(f"   ⏭️  S1 exists: {existing_any.name} ({get_band_count(existing_any)} bands)")
        return existing_any

    date_compact = date.replace('-', '')

    s1_dt = datetime.strptime(date, "%Y-%m-%d")
    s1_start = (s1_dt - timedelta(days=S1_DATE_BUFFER)).strftime("%Y-%m-%d")
    s1_end = date

    print(f"\n{'='*70}")
    print(f"📡 S1 Download - RAW GRD Method (no sar_backscatter) [{date_index}/{total_dates}]")
    print(f"{'='*70}")
    print(f"   ⚠️  Bypassing calibration to avoid LUT metadata errors")
    print(f"   📅 S1 Target Date: {date}")
    print(f"   📅 Search Window: {s1_start} to {s1_end} ({S1_DATE_BUFFER} days composite)")
    print(f"   📊 Bands: {S1_BANDS} ({S1_EXPECTED_BANDS} bands)")
    print(f"   🎯 Target Resolution: {TARGET_RESOLUTION}m")
    print(f"   📐 CRS: {TARGET_CRS}")
    print(f"   📁 Output Dir: {target_folder}")

    print(f"\n   📍 Spatial Extent (with margin):")
    print(f"      West:  {spatial_extent['west']:.6f}")
    print(f"      South: {spatial_extent['south']:.6f}")
    print(f"      East:  {spatial_extent['east']:.6f}")
    print(f"      North: {spatial_extent['north']:.6f}")

    final_output = target_folder / f"s1_{date_compact}.tif"

    if final_output.exists():
        print(f"   🗑️  Removing existing file: {final_output.name}")
        final_output.unlink()

    try:
        if connection is None:
            connection = openeo.connect(OPENEO_BACKEND)
            connection.authenticate_oidc()

        print(f"\n   📦 Loading S1 GRD collection (raw, no calibration)...")

        s1_cube = connection.load_collection(
            S1_COLLECTION,
            spatial_extent=spatial_extent,
            temporal_extent=[s1_start, s1_end],
            bands=S1_BANDS,
            properties={
                "sat:orbit_state": lambda x: x == "ascending"
            }
        )

        print(f"   🔄 Applying nodata mask and temporal median composite ({S1_DATE_BUFFER} days)...")
        s1_cube = s1_cube.reduce_dimension(
            dimension="t",
            reducer="median"
        )

        print(f"   🔄 Resampling to {TARGET_RESOLUTION}m...")
        s1_cube = s1_cube.resample_spatial(resolution=TARGET_RESOLUTION)

        job_options = {
            "soft-errors": "true",
            "tile-size": 512
        }

        download_cube_with_resume(
            connection, s1_cube, final_output,
            f"S1 RAW GRD {date}",
            job_options=job_options
        )

        if final_output.exists():
            info = validate_raster(final_output, expected_bands=S1_EXPECTED_BANDS)
            print_raster_validation(info)

            if info['bands'] != S1_EXPECTED_BANDS:
                print(f"   ❌ CRITICAL: Expected {S1_EXPECTED_BANDS} bands, got {info['bands']}")
                print(f"   ❌ This file is NOT a valid S1 product!")
                final_output.unlink()
                return None

            if info['non_zero_ratio'] < 0.01:
                print(f"   ⚠️  Very low valid data percentage")
                return final_output

            print(f"\n   🎉 S1 download complete for {date}!")
            print(f"   📁 File: {final_output.name} "
                  f"({final_output.stat().st_size/(1024*1024):.2f} MB, "
                  f"{info['bands']} bands)")
            return final_output
        else:
            print(f"   ❌ S1 file not found after download")
            return None

    except Exception as e:
        print(f"\n   ❌ S1 Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# NODATA MEASUREMENT
# =============================================================================
def measure_nodata(filepath: Path) -> dict:
    if not HAS_RASTERIO or not filepath or not filepath.exists():
        return {'nodata_pct': None, 'per_band': {}, 'band_count': 0}
    try:
        with rasterio.open(filepath) as src:
            total, total_nd, per_band = 0, 0, {}
            band_count = src.count
            for b in range(1, src.count + 1):
                data = src.read(b)
                n = data.size
                total += n
                nd_val = src.nodata
                if nd_val is not None:
                    bnd = np.sum(data == nd_val)
                else:
                    bnd = np.sum(np.isnan(data) | (data == 0) | (data == -9999))
                total_nd += bnd
                per_band[f"band_{b}"] = {
                    'nodata_pct': (bnd / n) * 100,
                    'nodata_pixels': int(bnd), 'total_pixels': int(n)
                }
            return {
                'nodata_pct': (total_nd / total) * 100 if total > 0 else 0,
                'per_band': per_band,
                'band_count': band_count
            }
    except Exception as e:
        print(f"   ⚠️  Nodata measure error: {e}")
        return {'nodata_pct': None, 'per_band': {}, 'band_count': 0}


# =============================================================================
# NEAREST S1
# =============================================================================
def find_nearest_s1(s1_dates: list, target: str):
    t = datetime.strptime(target, "%Y-%m-%d")
    best, best_diff = None, float("inf")
    for d in s1_dates:
        diff = abs((datetime.strptime(d, "%Y-%m-%d") - t).days)
        if diff < best_diff:
            best_diff = diff
            best = d
    return best, best_diff


# =============================================================================


def run() -> dict:
    """Execute download pipeline. Returns dict of all variables for step 2."""
    # =============================================================================
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  S1/S2 Downloader - PAIRS ONLY                                             ║
    ║                                                                             ║
    ║  • Downloads directly into pairs/ folder                                    ║
    ║  • Auto-renames on re-run (inference→prev01, shift up)                     ║
    ║  • Checks existing folders to skip re-downloads                             ║
    ║  • No cloud filtering - only spatial completeness                           ║
    ║  • Broken image allowed for inference if no complete exists                 ║
    ║  • S2: Batch job download with progress + retry                             ║
    ║  • S1: RAW GRD 60-day median composite (no sar_backscatter)                ║
    ║  • Band validation: S2=15 bands, S1=2 bands (VV+VH)                        ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # ---- STEP 0: Dates ----
    print("=" * 80)
    print("STEP 0: DATE CONFIGURATION")
    print("=" * 80)
    
    inference_window = validate_and_build_date_range(fecha_inicio, fecha_fin)
    inference_window_strs = [d.strftime("%Y-%m-%d") for d in inference_window]
    
    lookback_start, lookback_end = get_lookback_range(fecha_fin, LOOKBACK_DAYS)
    lb_start_str = lookback_start.strftime("%Y-%m-%d")
    lb_end_str = lookback_end.strftime("%Y-%m-%d")
    
    print(f"\n📅 INFERENCE WINDOW (5 days):")
    for d in inference_window_strs:
        print(f"   {d}")
    print(f"\n🔍 LOOKBACK: {lb_start_str} → {lb_end_str}")
    
    print(f"\n📊 BAND CONFIGURATION:")
    print(f"   S2: {S2_EXPECTED_BANDS} bands → {', '.join(S2_BANDS)}")
    print(f"   S1: {S1_EXPECTED_BANDS} bands → {', '.join(S1_BANDS)}")
    
    
    # ---- STEP 1: AOI ----
    print("\n" + "=" * 80)
    print("STEP 1: LOAD AOI")
    print("=" * 80)
    
    gdf, spatial_extent, geojson_geom = load_aoi_geometry()
    aoi_geom = gdf.geometry.iloc[0]
    
    bounds_original = {
        "west": spatial_extent["west"], "south": spatial_extent["south"],
        "east": spatial_extent["east"], "north": spatial_extent["north"]
    }
    
    spatial_extent = add_margin(spatial_extent, MARGIN_DEGREES)
    aoi_with_margin = box(
        spatial_extent["west"], spatial_extent["south"],
        spatial_extent["east"], spatial_extent["north"]
    )
    
    print(f"\n📋 BOUNDS COMPARISON ({TARGET_CRS}):")
    print(f"   {'Coordinate':<12} {'Original':<18} {'With Margin':<18} {'Difference':<12}")
    print(f"   {'-'*60}")
    print(f"   {'West':<12} {bounds_original['west']:<18.6f} {spatial_extent['west']:<18.6f} {MARGIN_DEGREES:<12}")
    print(f"   {'South':<12} {bounds_original['south']:<18.6f} {spatial_extent['south']:<18.6f} {MARGIN_DEGREES:<12}")
    print(f"   {'East':<12} {bounds_original['east']:<18.6f} {spatial_extent['east']:<18.6f} {MARGIN_DEGREES:<12}")
    print(f"   {'North':<12} {bounds_original['north']:<18.6f} {spatial_extent['north']:<18.6f} {MARGIN_DEGREES:<12}")
    
    
    # ---- STEP 2: Scan Existing Pairs ----
    print("\n" + "=" * 80)
    print("STEP 2: SCAN EXISTING PAIRS FOLDER")
    print("=" * 80)
    
    existing_info = scan_existing_pairs(PAIRS_DIR)
    
    print(f"\n📂 Pairs directory: {PAIRS_DIR.absolute()}")
    if existing_info['inference']:
        ei = existing_info['inference']
        s2_bands = get_band_count(ei.get('s2_file')) if ei.get('s2_file') else 0
        s1_bands = get_band_count(ei.get('s1_file')) if ei.get('s1_file') else 0
        print(f"   Current inference: {ei['date']} "
              f"(S2:{'✅' if ei['s2_exists'] else '❌'}{f'[{s2_bands}b]' if s2_bands else ''} "
              f"S1:{'✅' if ei['s1_exists'] else '❌'}{f'[{s1_bands}b]' if s1_bands else ''})")
    else:
        print("   No existing inference folder")
    
    if existing_info['previous']:
        print(f"   Previous folders: {len(existing_info['previous'])}")
        for ep in existing_info['previous']:
            s2_bands = get_band_count(ep.get('s2_file')) if ep.get('s2_file') else 0
            s1_bands = get_band_count(ep.get('s1_file')) if ep.get('s1_file') else 0
            print(f"      prev{ep['index']:02d}_{ep['date']} "
                  f"(S2:{'✅' if ep['s2_exists'] else '❌'}{f'[{s2_bands}b]' if s2_bands else ''} "
                  f"S1:{'✅' if ep['s1_exists'] else '❌'}{f'[{s1_bands}b]' if s1_bands else ''})")
    else:
        print("   No existing previous folders")
    
    print(f"\n   All dates with valid S2 ({S2_EXPECTED_BANDS}b): {sorted(existing_info['all_s2_dates'])}")
    print(f"   All dates with valid S1 ({S1_EXPECTED_BANDS}b): {sorted(existing_info['all_s1_dates'])}")
    
    
    # ---- STEP 3: Query S2 ----
    print("\n" + "=" * 80)
    print("STEP 3: QUERY S2 PRODUCTS")
    print("=" * 80)
    
    s2_products = query_s2_products(spatial_extent, lb_start_str, lb_end_str)
    if not s2_products:
        raise SystemExit("❌ No S2 products found!")
    print(f"📊 S2 on {len(s2_products)} dates")
    
    
    # ---- STEP 4: Spatial Check ----
    print("\n" + "=" * 80)
    print("STEP 4: SPATIAL COVERAGE (NO CLOUD FILTER)")
    print("=" * 80)
    
    spatial_results = {}
    print(f"\n{'Date':<12} {'Coverage':<10} {'Tiles':<6} {'Cloud%':<10} "
          f"{'InWin':<7} {'Complete'}")
    print("-" * 65)
    
    for date in sorted(s2_products.keys(), reverse=True):
        r = check_spatial_coverage(date, s2_products[date], aoi_with_margin)
        spatial_results[date] = r
        iw = "✅" if date in inference_window_strs else "  "
        cl = f"{r['avg_cloud_cover']:.1f}%" if r['avg_cloud_cover'] is not None else "N/A"
        st = "✅" if r['is_complete'] else "❌"
        print(f"{date:<12} {r['spatial_coverage_pct']:.1f}%{'':>4} "
              f"{len(r['tiles']):<6} {cl:<10} {iw:<7} {st} {r['reason']}")
    
    complete_dates = sorted([d for d, r in spatial_results.items() if r['is_complete']], reverse=True)
    print(f"\n📊 Complete: {len(complete_dates)} | "
          f"Incomplete: {len(spatial_results) - len(complete_dates)}")
    
    
    # ---- STEP 5: Determine Inference Date ----
    print("\n" + "=" * 80)
    print("STEP 5: DETERMINE INFERENCE DATE")
    print("=" * 80)
    
    inf_candidates = [d for d in complete_dates if d in inference_window_strs]
    inference_date = None
    inference_is_broken = False
    
    if inf_candidates:
        inference_date = max(inf_candidates)
        print(f"\n✅ INFERENCE: {inference_date} (latest complete in window)")
    else:
        print(f"\n⚠️  No complete image in inference window!")
        window_avail = [d for d in inference_window_strs if d in spatial_results]
        if window_avail:
            best = max(window_avail, key=lambda d: spatial_results[d]['spatial_coverage_pct'])
            inference_date = best
            inference_is_broken = True
            print(f"   🔧 Using BROKEN: {inference_date} "
                  f"(coverage: {spatial_results[best]['spatial_coverage_pct']:.1f}%)")
        else:
            pre = [d for d in complete_dates if d < inference_window_strs[0]]
            if pre:
                inference_date = pre[0]
                print(f"   ⚠️  Using nearest before window: {inference_date}")
            else:
                raise SystemExit("❌ No inference date available!")
    
    print(f"\n🎯 INFERENCE DATE: {inference_date} "
          f"{'⚠️ BROKEN' if inference_is_broken else '✅ COMPLETE'}")
    
    
    # ---- STEP 6: Select Previous 5 ----
    print("\n" + "=" * 80)
    print("STEP 6: SELECT PREVIOUS 5 COMPLETE DATES")
    print("=" * 80)
    
    prev_candidates = [d for d in complete_dates if d != inference_date and d <= inference_date]
    previous_dates = prev_candidates[:PREVIOUS_IMAGES_COUNT]
    
    print(f"\n📋 Selected {len(previous_dates)} previous dates:")
    for i, d in enumerate(previous_dates, 1):
        sr = spatial_results[d]
        cl = f"{sr['avg_cloud_cover']:.1f}%" if sr['avg_cloud_cover'] is not None else "N/A"
        print(f"   {i}. {d} | Coverage: {sr['spatial_coverage_pct']:.1f}% | Cloud: {cl}")
    
    
    # ---- STEP 7: Preview Validation ----
    print("\n" + "=" * 80)
    print("STEP 7: OPENEO PREVIEW VALIDATION (NoData + Cloud %)")
    print("=" * 80)
    
    all_target_dates = [inference_date] + previous_dates
    
    dates_needing_validation = [
        d for d in all_target_dates
        if d not in existing_info['all_s2_dates']
    ]
    if inference_date not in dates_needing_validation:
        dates_needing_validation.insert(0, inference_date)
    dates_needing_validation = sorted(list(set(dates_needing_validation)))
    
    preview_results = {}
    DATE_METADATA_S2 = {}
    
    if HAS_OPENEO and dates_needing_validation:
        print(f"\n🔍 Validating {len(dates_needing_validation)} dates...")
        try:
            val_conn = openeo.connect(OPENEO_BACKEND)
            val_conn.authenticate_oidc()
            print("✅ Connected")
    
            for i, date in enumerate(dates_needing_validation, 1):
                print(f"   [{i}/{len(dates_needing_validation)}] {date}...", end=" ")
                r = validate_nodata_preview(date, spatial_extent, val_conn)
                preview_results[date] = r
                if r.get('validated'):
                    nd = r.get('nodata_pct', 0)
                    cl = r.get('cloud_pct_scl', 0)
                    print(f"NoData:{nd:.1f}% Cloud:{cl:.1f}% "
                          f"{'✅' if r.get('is_complete') else '⚠️'}")
                else:
                    print(f"⚠️  {r.get('reason')}")
        except Exception as e:
            print(f"\n❌ OpenEO error: {e}")
    else:
        print("   Skipping validation (all validated or openEO N/A)")
    
    # Build S2 metadata
    for date in all_target_dates:
        meta = DateMetadata(date, "S2")
        meta.is_inference = (date == inference_date)
        meta.is_broken_allowed = (date == inference_date and inference_is_broken)
        meta.band_count = S2_EXPECTED_BANDS
    
        if date in spatial_results:
            sr = spatial_results[date]
            meta.spatial_coverage_pct = sr['spatial_coverage_pct']
            meta.tiles = sr['tiles']
            meta.cloud_cover_pct = sr['avg_cloud_cover']
            meta.is_complete = sr['is_complete']
    
        if date in preview_results:
            pr = preview_results[date]
            if pr.get('validated'):
                meta.nodata_pct = pr.get('nodata_pct')
                meta.valid_pixel_pct = pr.get('valid_pct')
                if pr.get('cloud_pct_scl') is not None:
                    meta.cloud_cover_pct = pr.get('cloud_pct_scl')
                if not meta.is_inference:
                    meta.is_complete = pr.get('is_complete', meta.is_complete)
    
        DATE_METADATA_S2[date] = meta
    
    # Re-evaluate previous dates
    valid_prev = [d for d in previous_dates
                  if DATE_METADATA_S2[d].is_complete or DATE_METADATA_S2[d].is_complete is None]
    invalid_prev = [d for d in previous_dates if d not in valid_prev]
    
    if len(valid_prev) < PREVIOUS_IMAGES_COUNT:
        extra = [d for d in complete_dates
                 if d != inference_date and d not in valid_prev
                 and d not in invalid_prev and d <= inference_date]
        for d in extra[:PREVIOUS_IMAGES_COUNT - len(valid_prev)]:
            valid_prev.append(d)
            if d not in DATE_METADATA_S2:
                m = DateMetadata(d, "S2")
                m.band_count = S2_EXPECTED_BANDS
                sr = spatial_results.get(d, {})
                m.spatial_coverage_pct = sr.get('spatial_coverage_pct')
                m.tiles = sr.get('tiles', [])
                m.cloud_cover_pct = sr.get('avg_cloud_cover')
                m.is_complete = True
                DATE_METADATA_S2[d] = m
    
    previous_dates = sorted(valid_prev, reverse=True)[:PREVIOUS_IMAGES_COUNT]
    
    print(f"\n📋 FINAL SELECTION:")
    print(f"   Inference: {inference_date} {'⚠️ BROKEN' if inference_is_broken else '✅'}")
    for i, d in enumerate(previous_dates, 1):
        m = DATE_METADATA_S2.get(d)
        cl = f"{m.cloud_cover_pct:.1f}%" if m and m.cloud_cover_pct is not None else "N/A"
        nd = f"{m.nodata_pct:.1f}%" if m and m.nodata_pct is not None else "N/A"
        print(f"   Prev {i}: {d} | Cloud: {cl} | NoData: {nd}")
    
    
    # ---- STEP 8: Query & Match S1 ----
    print("\n" + "=" * 80)
    print("STEP 8: QUERY & MATCH S1")
    print("=" * 80)
    
    s1_start = (lookback_start - timedelta(days=12)).strftime("%Y-%m-%d")
    s1_end = (datetime.strptime(lb_end_str, "%Y-%m-%d") + timedelta(days=12)).strftime("%Y-%m-%d")
    
    s1_products = query_s1_products(spatial_extent, s1_start, s1_end)
    all_s1_dates = sorted(s1_products.keys(), reverse=True)
    print(f"📊 S1 dates: {len(all_s1_dates)}")
    
    # Build pairs
    download_pairs = []
    DATE_METADATA_S1 = {}
    
    all_download_dates = [inference_date] + previous_dates
    
    print(f"\n{'Role':<12} {'S2 Date':<12} {'S1 Date':<12} {'Δ Days'}")
    print("-" * 48)
    
    for i, s2d in enumerate(all_download_dates):
        s1d, s1diff = find_nearest_s1(all_s1_dates, s2d) if all_s1_dates else (None, None)
        role = "inference" if s2d == inference_date else "previous"
        pidx = 0 if role == "inference" else (previous_dates.index(s2d) + 1 if s2d in previous_dates else i)
    
        pair = {'s2_date': s2d, 's1_date': s1d, 's1_diff_days': s1diff,
                'role': role, 'prev_index': pidx}
        download_pairs.append(pair)
    
        if s1d and s1d not in DATE_METADATA_S1:
            m = DateMetadata(s1d, "S1")
            m.is_inference = (role == "inference")
            m.band_count = S1_EXPECTED_BANDS
            DATE_METADATA_S1[s1d] = m
    
        r = "INFERENCE" if role == "inference" else f"PREV {pidx:02d}"
        print(f"{r:<12} {s2d:<12} {s1d or 'N/A':<12} {s1diff if s1diff is not None else 'N/A'}")
    
    
    # ---- STEP 9: Reorganize Existing Folders ----
    print("\n" + "=" * 80)
    print("STEP 9: REORGANIZE PAIRS FOLDER")
    print("=" * 80)
    
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    
    reorg = reorganize_pairs_for_new_inference(
        PAIRS_DIR, inference_date, previous_dates, existing_info
    )
    
    target_folders = reorg['target_folders']
    
    # Determine S1 downloads needed - use validated file detection
    s1_dates_needed = {}  # s1_date -> s2_date (for folder lookup)
    for pair in download_pairs:
        if pair['s1_date']:
            s1_dates_needed[pair['s1_date']] = pair['s2_date']
    
    # Check which S1 dates already have valid files (2 bands)
    s1_already = set()
    for s1_date, s2_date in s1_dates_needed.items():
        folder = target_folders.get(s2_date)
        if folder and find_s1_file_in_folder(folder) is not None:
            s1_already.add(s1_date)
    
    # Also scan all pair folders for S1 files that might be in wrong folder
    for folder in PAIRS_DIR.iterdir():
        if folder.is_dir():
            s1_file = find_s1_raw_file_in_folder(folder)
            if s1_file:
                m_match = re.search(r'(\d{4})-?(\d{2})-?(\d{2})', s1_file.name)
                if m_match:
                    s1_date_str = f"{m_match.group(1)}-{m_match.group(2)}-{m_match.group(3)}"
                    s1_already.add(s1_date_str)
    
    s2_to_download = reorg['s2_dates_to_download']
    s1_to_download = sorted([d for d in s1_dates_needed.keys() if d not in s1_already])
    
    print(f"\n📋 DOWNLOAD PLAN:")
    print(f"   S2 to download : {len(s2_to_download)} {s2_to_download}")
    print(f"   S2 already have: {sorted(reorg['dates_already_have_s2'])}")
    print(f"   S1 to download : {len(s1_to_download)} {s1_to_download}")
    print(f"   S1 already have: {sorted(s1_already)}")
    print(f"   S2 expected bands: {S2_EXPECTED_BANDS}")
    print(f"   S1 expected bands: {S1_EXPECTED_BANDS}")
    
    
    # ---- STEP 10: Execute Downloads ----
    print("\n" + "=" * 80)
    print("STEP 10: EXECUTE DOWNLOADS")
    print("=" * 80)
    
    download_results = {
        's2_downloaded': [], 's2_skipped': [], 's2_failed': [],
        's1_downloaded': [], 's1_skipped': [], 's1_failed': [],
    }
    
    total_start_time = time.time()
    
    if s2_to_download or s1_to_download:
        print(f"\n🔌 Connecting to openEO...")
        try:
            dl_conn = openeo.connect(OPENEO_BACKEND)
            dl_conn.authenticate_oidc()
            print("✅ Connected and authenticated!")
            user_info = dl_conn.describe_account()
            print(f"   User: {user_info.get('user_id', 'N/A')}")
    
            # =========================================================
            # Download S2 using BATCH JOB method
            # =========================================================
            if s2_to_download:
                print(f"\n{'='*70}")
                print(f"📥 STARTING S2 DOWNLOADS ({len(s2_to_download)} dates)")
                print(f"   Expected: {S2_EXPECTED_BANDS} bands per file")
                print(f"{'='*70}")
    
                for i, date in enumerate(s2_to_download, 1):
                    folder = target_folders.get(date)
                    if not folder:
                        print(f"   ⚠️  No target folder for S2 {date}")
                        download_results['s2_failed'].append(date)
                        continue
    
                    print(f"\n{'#'*70}")
                    print(f"# Processing S2 date {i}/{len(s2_to_download)}: {date}")
                    print(f"# Output folder: {folder}")
                    print(f"{'#'*70}")
    
                    result = download_s2_to_folder(
                        date, folder, spatial_extent, dl_conn,
                        date_index=i, total_dates=len(s2_to_download)
                    )
    
                    if result:
                        download_results['s2_downloaded'].append(date)
                        if date in DATE_METADATA_S2:
                            DATE_METADATA_S2[date].download_path = result
                            DATE_METADATA_S2[date].file_size_mb = result.stat().st_size/(1024*1024)
                            DATE_METADATA_S2[date].band_count = get_band_count(result)
                    else:
                        download_results['s2_failed'].append(date)
    
                    elapsed = time.time() - total_start_time
                    print(f"\n   📊 S2 Progress: {i}/{len(s2_to_download)} | "
                          f"Success: {len(download_results['s2_downloaded'])} | "
                          f"Failed: {len(download_results['s2_failed'])} | "
                          f"Elapsed: {format_time(elapsed)}")
    
            # =========================================================
            # Download S1 using RAW GRD + 60-day median composite
            # =========================================================
            if s1_to_download:
                print(f"\n{'='*70}")
                print(f"📥 STARTING S1 DOWNLOADS ({len(s1_to_download)} dates)")
                print(f"   Method: RAW GRD with {S1_DATE_BUFFER}-day median composite")
                print(f"   No sar_backscatter (avoids LUT errors)")
                print(f"   Expected: {S1_EXPECTED_BANDS} bands per file (VV + VH)")
                print(f"{'='*70}")
    
                for i, s1_date in enumerate(s1_to_download, 1):
                    # Find which pair folder this S1 goes into
                    target_folder = None
                    for pair in download_pairs:
                        if pair['s1_date'] == s1_date:
                            target_folder = target_folders.get(pair['s2_date'])
                            break
    
                    if not target_folder:
                        print(f"   ⚠️  No target folder for S1 {s1_date}")
                        download_results['s1_failed'].append(s1_date)
                        continue
    
                    print(f"\n{'#'*70}")
                    print(f"# Processing S1 {i}/{len(s1_to_download)}: {s1_date}")
                    print(f"# Output folder: {target_folder}")
                    print(f"{'#'*70}")
    
                    result = download_s1_to_folder(
                        s1_date, target_folder, spatial_extent, dl_conn,
                        date_index=i, total_dates=len(s1_to_download)
                    )
    
                    if result:
                        bands = get_band_count(result)
                        if bands == S1_EXPECTED_BANDS:
                            download_results['s1_downloaded'].append(s1_date)
                            if s1_date in DATE_METADATA_S1:
                                DATE_METADATA_S1[s1_date].download_path = result
                                DATE_METADATA_S1[s1_date].file_size_mb = result.stat().st_size/(1024*1024)
                                DATE_METADATA_S1[s1_date].band_count = bands
                        else:
                            print(f"   ❌ Downloaded file has {bands} bands, expected {S1_EXPECTED_BANDS}")
                            download_results['s1_failed'].append(s1_date)
                    else:
                        download_results['s1_failed'].append(s1_date)
    
                    elapsed = time.time() - total_start_time
                    print(f"\n   📊 S1 Progress: {i}/{len(s1_to_download)} | "
                          f"Success: {len(download_results['s1_downloaded'])} | "
                          f"Failed: {len(download_results['s1_failed'])} | "
                          f"Elapsed: {format_time(elapsed)}")
    
        except Exception as e:
            print(f"\n❌ Connection error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n✅ All files already downloaded!")
    
    total_download_time = time.time() - total_start_time
    
    # Update paths for existing files using validated finders
    for pair in download_pairs:
        folder = target_folders.get(pair['s2_date'])
        if not folder or not folder.exists():
            continue
    
        # S2 files - use validated finder
        s2_file = find_s2_file_in_folder(folder)
        if s2_file and pair['s2_date'] in DATE_METADATA_S2:
            DATE_METADATA_S2[pair['s2_date']].download_path = s2_file
            DATE_METADATA_S2[pair['s2_date']].file_size_mb = s2_file.stat().st_size/(1024*1024)
            DATE_METADATA_S2[pair['s2_date']].band_count = get_band_count(s2_file)
    
        # S1 files - use validated finder (NEVER matches openEO_*)
        if pair['s1_date']:
            s1_file = find_s1_file_in_folder(folder)
            if s1_file and pair['s1_date'] in DATE_METADATA_S1:
                DATE_METADATA_S1[pair['s1_date']].download_path = s1_file
                DATE_METADATA_S1[pair['s1_date']].file_size_mb = s1_file.stat().st_size/(1024*1024)
                DATE_METADATA_S1[pair['s1_date']].band_count = get_band_count(s1_file)
    
    
    # ---- STEP 11: Measure NoData ----
    print("\n" + "=" * 80)
    print("STEP 11: MEASURE NODATA IN DOWNLOADED FILES")
    print("=" * 80)
    
    print("\n📊 S2 NoData:")
    print(f"{'Date':<12} {'Role':<12} {'Bands':<7} {'NoData%':<10} {'Cloud%':<10} {'Size MB':<10} {'Status'}")
    print("-" * 75)
    
    for date in [inference_date] + previous_dates:
        meta = DATE_METADATA_S2.get(date)
        if not meta:
            continue
        role = "INFERENCE" if meta.is_inference else "PREVIOUS"
    
        if meta.download_path and meta.download_path.exists():
            nd = measure_nodata(meta.download_path)
            if nd['nodata_pct'] is not None:
                meta.nodata_pct = nd['nodata_pct']
            meta.band_count = nd.get('band_count', meta.band_count)
    
        bands_s = str(meta.band_count) if meta.band_count else "N/A"
        nd_s = f"{meta.nodata_pct:.2f}%" if meta.nodata_pct is not None else "N/A"
        cl_s = f"{meta.cloud_cover_pct:.1f}%" if meta.cloud_cover_pct is not None else "N/A"
        sz_s = f"{meta.file_size_mb:.1f}" if meta.file_size_mb else "N/A"
        st = "✅" if meta.is_complete else "⚠️ BROKEN"
        print(f"{date:<12} {role:<12} {bands_s:<7} {nd_s:<10} {cl_s:<10} {sz_s:<10} {st}")
    
    print(f"\n📊 S1 NoData:")
    print(f"{'Date':<12} {'Role':<12} {'Bands':<7} {'NoData%':<10} {'Size MB':<10} {'File':<30} {'Status'}")
    print("-" * 95)
    
    for date in sorted(DATE_METADATA_S1.keys(), reverse=True):
        meta = DATE_METADATA_S1[date]
        role = "INFERENCE" if meta.is_inference else "PREVIOUS"
    
        if meta.download_path and meta.download_path.exists():
            # Verify it's actually S1
            actual_bands = get_band_count(meta.download_path)
            if actual_bands != S1_EXPECTED_BANDS:
                print(f"{date:<12} {role:<12} {actual_bands:<7} {'N/A':<10} {'N/A':<10} "
                      f"{meta.download_path.name:<30} ❌ WRONG FILE ({actual_bands}b≠{S1_EXPECTED_BANDS}b)")
                meta.download_path = None
                meta.band_count = actual_bands
                continue
    
            nd = measure_nodata(meta.download_path)
            if nd['nodata_pct'] is not None:
                meta.nodata_pct = nd['nodata_pct']
            meta.band_count = nd.get('band_count', S1_EXPECTED_BANDS)
            filename = meta.download_path.name
        else:
            filename = "NOT FOUND"
    
        nd_s = f"{meta.nodata_pct:.2f}%" if meta.nodata_pct is not None else "N/A"
        sz_s = f"{meta.file_size_mb:.1f}" if meta.file_size_mb else "N/A"
        bands_s = str(meta.band_count) if meta.band_count else "N/A"
        has_nd = meta.nodata_pct is not None and meta.nodata_pct > 0.1
        has_file = meta.download_path is not None and meta.download_path.exists()
        meta.is_complete = has_file and not has_nd
        st = "⚠️ HAS NODATA" if has_nd else ("✅" if has_file else "❌ MISSING")
        print(f"{date:<12} {role:<12} {bands_s:<7} {nd_s:<10} {sz_s:<10} {filename:<30} {st}")
    
    
    # ---- STEP 12: S1 NoData Summary (no fill) ----
    print("\n" + "=" * 80)
    print("STEP 12: S1 NODATA SUMMARY")
    print("=" * 80)
    
    s1_with_nodata = [
        d for d, m in DATE_METADATA_S1.items()
        if m.nodata_pct is not None and m.nodata_pct > 0.1
        and m.download_path and m.download_path.exists()
        and is_valid_s1_file(m.download_path)
    ]
    
    if s1_with_nodata:
        print(f"\n📊 S1 dates with nodata > 0.1%: {s1_with_nodata}")
        for date in s1_with_nodata:
            meta = DATE_METADATA_S1[date]
            print(f"   • {date}: {meta.nodata_pct:.2f}% nodata ({meta.band_count} bands)")
        print(f"   ℹ️  No median fill applied (disabled)")
    else:
        print("✅ All S1 files have minimal nodata")
    
    s1_missing = [
        d for d, m in DATE_METADATA_S1.items()
        if m.download_path is None or not m.download_path.exists()
    ]
    if s1_missing:
        print(f"\n⚠️  S1 dates with NO valid file ({S1_EXPECTED_BANDS} bands): {s1_missing}")
    
    
    # ---- FINAL SUMMARY ----
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    s2_success_rate = (len(download_results['s2_downloaded']) / len(s2_to_download) * 100) if s2_to_download else 100
    s1_success_rate = (len(download_results['s1_downloaded']) / len(s1_to_download) * 100) if s1_to_download else 100
    
    # Count actual valid files
    s2_valid_count = sum(1 for p in download_pairs if find_s2_file_in_folder(target_folders.get(p['s2_date'])))
    s1_valid_count = sum(1 for p in download_pairs if p['s1_date'] and find_s1_file_in_folder(target_folders.get(p['s2_date'])))
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  📊 COMPLETE                                                                    ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  Inference Window       : {inference_window_strs[0]} to {inference_window_strs[-1]:<33} ║
    ║  Inference Date         : {inference_date:<49} ║
    ║  Inference Broken?      : {'YES ⚠️' if inference_is_broken else 'NO  ✅':<49} ║
    ║  Previous Images        : {len(previous_dates):<49} ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  S2 Downloaded (new)    : {len(download_results['s2_downloaded']):<49} ║
    ║  S2 Skipped (existing)  : {len(all_target_dates) - len(s2_to_download):<49} ║
    ║  S2 Failed              : {len(download_results['s2_failed']):<49} ║
    ║  S2 Success Rate        : {f'{s2_success_rate:.1f}%':<49} ║
    ║  S2 Valid Files (total) : {f'{s2_valid_count}/{len(download_pairs)} ({S2_EXPECTED_BANDS} bands each)':<49} ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  S1 Downloaded (new)    : {len(download_results['s1_downloaded']):<49} ║
    ║  S1 Skipped (existing)  : {len(s1_dates_needed) - len(s1_to_download):<49} ║
    ║  S1 Failed              : {len(download_results['s1_failed']):<49} ║
    ║  S1 Success Rate        : {f'{s1_success_rate:.1f}%':<49} ║
    ║  S1 Valid Files (total) : {f'{s1_valid_count}/{len(download_pairs)} ({S1_EXPECTED_BANDS} bands each)':<49} ║
    ║  S1 NoData Filled       : {'N/A (disabled)':<49} ║
    ║  S1 Composite Window    : {f'{S1_DATE_BUFFER} days median':<49} ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  CRS                    : {TARGET_CRS:<49} ║
    ║  Margin                 : {f'{MARGIN_DEGREES} degrees (~{MARGIN_DEGREES * 111:.0f}m)':<49} ║
    ║  S1 Method              : {'RAW GRD (no sar_backscatter)':<49} ║
    ║  Total Download Time    : {format_time(total_download_time):<49} ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Per-date metadata table
    print("=" * 115)
    print("📊 PER-DATE METADATA")
    print("=" * 115)
    print(f"\n{'Role':<12} {'S2 Date':<12} {'S2 Bands':<10} {'S2 Cloud%':<12} {'S2 NoData%':<12} "
          f"{'S1 Date':<12} {'S1 Bands':<10} {'S1 NoData%':<12} {'Status'}")
    print("-" * 105)
    
    for pair in download_pairs:
        s2d = pair['s2_date']
        s1d = pair.get('s1_date', 'N/A')
        role = pair['role'].upper()
        if role == "PREVIOUS":
            role = f"PREV {pair.get('prev_index', '?'):02d}"
    
        s2m = DATE_METADATA_S2.get(s2d)
        s1m = DATE_METADATA_S1.get(s1d) if s1d != 'N/A' else None
    
        s2b = str(s2m.band_count) if s2m and s2m.band_count else "N/A"
        s2cl = f"{s2m.cloud_cover_pct:.1f}%" if s2m and s2m.cloud_cover_pct is not None else "N/A"
        s2nd = f"{s2m.nodata_pct:.2f}%" if s2m and s2m.nodata_pct is not None else "N/A"
    
        s1b = str(s1m.band_count) if s1m and s1m.band_count else "N/A"
        s1nd = f"{s1m.nodata_pct:.2f}%" if s1m and s1m.nodata_pct is not None else "N/A"
    
        # Status check
        s2_ok = s2m and s2m.download_path and s2m.download_path.exists() and s2m.band_count == S2_EXPECTED_BANDS
        s1_ok = s1m and s1m.download_path and s1m.download_path.exists() and s1m.band_count == S1_EXPECTED_BANDS
        status = "✅" if (s2_ok and s1_ok) else ("⚠️ S1 MISSING" if s2_ok and not s1_ok else "❌")
        broken = " ⚠️BROKEN" if s2m and s2m.is_broken_allowed else ""
    
        print(f"{role:<12} {s2d:<12} {s2b:<10} {s2cl:<12} {s2nd:<12} "
              f"{s1d or 'N/A':<12} {s1b:<10} {s1nd:<12} {status}{broken}")
    
    print("-" * 105)
    
    # Print failed dates if any
    if download_results['s2_failed']:
        print(f"\n❌ S2 FAILED DATES ({len(download_results['s2_failed'])}):")
        for fecha in download_results['s2_failed']:
            print(f"   - {fecha}")
    
    if download_results['s1_failed']:
        print(f"\n❌ S1 FAILED DATES ({len(download_results['s1_failed'])}):")
        for fecha in download_results['s1_failed']:
            print(f"   - {fecha}")
    
    # Folder structure
    print("\n📁 FOLDER STRUCTURE:")
    print(f"   {PAIRS_DIR}/")
    if PAIRS_DIR.exists():
        for folder in sorted(PAIRS_DIR.iterdir()):
            if folder.is_dir():
                parsed = parse_pair_folder_name(folder.name)
                if parsed:
                    files = list(folder.glob("*.tif"))
                    print(f"   └── {folder.name}/")
                    for f in sorted(files):
                        sz = f.stat().st_size / (1024*1024)
                        bands = get_band_count(f)
                        sat_type = "S2" if bands == S2_EXPECTED_BANDS else ("S1" if bands == S1_EXPECTED_BANDS else f"?{bands}b")
                        print(f"       └── {f.name} ({sz:.1f} MB, {bands}b={sat_type})")
    
    # List all files summary
    print(f"\n📁 ALL S2 FILES (expected {S2_EXPECTED_BANDS} bands):")
    for pair in download_pairs:
        folder = target_folders.get(pair['s2_date'])
        if folder and folder.exists():
            s2_file = find_s2_file_in_folder(folder)
            if s2_file:
                size_mb = s2_file.stat().st_size / (1024 * 1024)
                bands = get_band_count(s2_file)
                ok = "✅" if bands == S2_EXPECTED_BANDS else f"⚠️ {bands}b"
                print(f"   {ok} {s2_file.relative_to(PAIRS_DIR)}: {size_mb:.2f} MB, {bands} bands")
            else:
                print(f"   ❌ {folder.name}/: No valid S2 file")
        else:
            print(f"   ❌ {pair['s2_date']}: Folder not found")
    
    print(f"\n📁 ALL S1 FILES (expected {S1_EXPECTED_BANDS} bands):")
    for pair in download_pairs:
        folder = target_folders.get(pair['s2_date'])
        if folder and folder.exists():
            s1_file = find_s1_file_in_folder(folder)
            if s1_file:
                size_mb = s1_file.stat().st_size / (1024 * 1024)
                bands = get_band_count(s1_file)
                ok = "✅" if bands == S1_EXPECTED_BANDS else f"⚠️ {bands}b"
                print(f"   {ok} {s1_file.relative_to(PAIRS_DIR)}: {size_mb:.2f} MB, {bands} bands")
            else:
                s1d = pair.get('s1_date', 'N/A')
                print(f"   ❌ {folder.name}/: No valid S1 file (S1 date: {s1d})")
        elif pair.get('s1_date'):
            print(f"   ❌ {pair['s2_date']}: Folder not found")
    
    
    # =============================================================================
    # STORE VARIABLES
    # =============================================================================
    INFERENCE_DATE = inference_date
    INFERENCE_IS_BROKEN = inference_is_broken
    PREVIOUS_DATES = previous_dates
    DOWNLOAD_PAIRS = download_pairs
    INFERENCE_WINDOW = inference_window_strs
    
    CLOUD_PCT_PER_DATE = {d: m.cloud_cover_pct for d, m in DATE_METADATA_S2.items()}
    NODATA_PCT_S2_PER_DATE = {d: m.nodata_pct for d, m in DATE_METADATA_S2.items()}
    NODATA_PCT_S1_PER_DATE = {d: m.nodata_pct for d, m in DATE_METADATA_S1.items()}
    
    ALL_S2_DATES = [inference_date] + previous_dates
    ALL_S1_DATES = sorted(list(set([p['s1_date'] for p in download_pairs if p['s1_date']])), reverse=True)
    
    SPATIAL_EXTENT = spatial_extent
    AOI_GEOMETRY = aoi_geom
    TARGET_FOLDERS = target_folders
    DOWNLOAD_RESULTS = download_results
    
    # Build file path dictionaries using VALIDATED finders
    S2_FILE_PATHS = {}
    S1_FILE_PATHS = {}
    for pair in download_pairs:
        folder = target_folders.get(pair['s2_date'])
        if not folder or not folder.exists():
            continue
    
        # S2 files - validated finder
        s2_file = find_s2_file_in_folder(folder)
        if s2_file:
            S2_FILE_PATHS[pair['s2_date']] = s2_file
    
        # S1 files - validated finder (NEVER matches openEO_*)
        if pair['s1_date']:
            s1_file = find_s1_file_in_folder(folder)
            if s1_file:
                S1_FILE_PATHS[pair['s2_date']] = {
                    's1_date': pair['s1_date'],
                    's1_file': s1_file
                }
    
    print("\n" + "=" * 80)
    print("📦 VARIABLES FOR DOWNSTREAM USE")
    print("=" * 80)
    print(f"""
      INFERENCE_DATE         : {INFERENCE_DATE}
      INFERENCE_IS_BROKEN    : {INFERENCE_IS_BROKEN}
      PREVIOUS_DATES         : {PREVIOUS_DATES}
      INFERENCE_WINDOW       : {INFERENCE_WINDOW}
    
      CLOUD_PCT_PER_DATE     :""")
    for d, v in sorted(CLOUD_PCT_PER_DATE.items()):
        r = "INF" if d == INFERENCE_DATE else "PRV"
        print(f"    {d}: {f'{v:.1f}%' if v is not None else 'N/A':>8}  [{r}]")
    
    print(f"\n  NODATA_PCT_S2_PER_DATE :")
    for d, v in sorted(NODATA_PCT_S2_PER_DATE.items()):
        r = "INF" if d == INFERENCE_DATE else "PRV"
        print(f"    {d}: {f'{v:.2f}%' if v is not None else 'N/A':>8}  [{r}]")
    
    print(f"\n  NODATA_PCT_S1_PER_DATE :")
    for d, v in sorted(NODATA_PCT_S1_PER_DATE.items()):
        print(f"    {d}: {f'{v:.2f}%' if v is not None else 'N/A':>8}")
    
    print(f"""
      ALL_S2_DATES           : {ALL_S2_DATES}
      ALL_S1_DATES           : {ALL_S1_DATES}
      PAIRS_DIR              : {PAIRS_DIR.absolute()}
      TARGET_FOLDERS         :""")
    for d, f in sorted(TARGET_FOLDERS.items()):
        r = "INF" if d == INFERENCE_DATE else "PRV"
        print(f"    {d} → {f.name}  [{r}]")
    
    print(f"""
      DATE_METADATA_S2       : {len(DATE_METADATA_S2)} dates
      DATE_METADATA_S1       : {len(DATE_METADATA_S1)} dates
    """)
    
    for d in sorted(DATE_METADATA_S2.keys(), reverse=True):
        print(f"   {DATE_METADATA_S2[d]}")
    for d in sorted(DATE_METADATA_S1.keys(), reverse=True):
        print(f"   {DATE_METADATA_S1[d]}")
    
    # S2 and S1 file path summaries with band validation
    print(f"\n📁 S2_FILE_PATHS dictionary ({len(S2_FILE_PATHS)} files):")
    for fecha, path in S2_FILE_PATHS.items():
        bands = get_band_count(path)
        ok = "✅" if bands == S2_EXPECTED_BANDS else f"⚠️ {bands}b"
        print(f"   {ok} '{fecha}': '{path}'")
    
    print(f"\n📁 S1_FILE_PATHS dictionary ({len(S1_FILE_PATHS)} files):")
    for s2_date, info in S1_FILE_PATHS.items():
        bands = get_band_count(info['s1_file'])
        ok = "✅" if bands == S1_EXPECTED_BANDS else f"⚠️ {bands}b"
        print(f"   {ok} '{s2_date}': {{'s1_date': '{info['s1_date']}', 's1_file': '{info['s1_file']}'}}")
    
    # Report completeness
    print(f"\n📊 PAIR COMPLETENESS:")
    print(f"   S2 files: {len(S2_FILE_PATHS)}/{len(download_pairs)} pairs")
    print(f"   S1 files: {len(S1_FILE_PATHS)}/{len(download_pairs)} pairs")
    missing_s1_pairs = [p['s2_date'] for p in download_pairs if p['s2_date'] not in S1_FILE_PATHS and p['s1_date']]
    if missing_s1_pairs:
        print(f"   ⚠️  Pairs missing S1: {missing_s1_pairs}")
        print(f"   💡 These S1 files need to be downloaded (re-run may fix)")
    
    print("\n✅ ALL DONE")
    print("=" * 80)
    
    # =============================================================================
    # EXPLICIT GLOBAL EXPORT FOR CELL 2
    # =============================================================================
    print("\n" + "=" * 80)
    print("🔗 EXPORTING VARIABLES TO GLOBAL SCOPE")
    print("=" * 80)
    
    globals().update({
        'INFERENCE_DATE': INFERENCE_DATE,
        'INFERENCE_IS_BROKEN': INFERENCE_IS_BROKEN,
        'PREVIOUS_DATES': PREVIOUS_DATES,
        'DOWNLOAD_PAIRS': DOWNLOAD_PAIRS,
        'INFERENCE_WINDOW': INFERENCE_WINDOW,
        'TARGET_FOLDERS': TARGET_FOLDERS,
        'PAIRS_DIR': PAIRS_DIR,
        'CLOUD_PCT_PER_DATE': CLOUD_PCT_PER_DATE,
        'NODATA_PCT_S2_PER_DATE': NODATA_PCT_S2_PER_DATE,
        'NODATA_PCT_S1_PER_DATE': NODATA_PCT_S1_PER_DATE,
        'ALL_S2_DATES': ALL_S2_DATES,
        'ALL_S1_DATES': ALL_S1_DATES,
        'SPATIAL_EXTENT': SPATIAL_EXTENT,
        'AOI_GEOMETRY': AOI_GEOMETRY,
        'DOWNLOAD_RESULTS': DOWNLOAD_RESULTS,
        'DATE_METADATA_S2': DATE_METADATA_S2,
        'DATE_METADATA_S1': DATE_METADATA_S1,
        'S2_FILE_PATHS': S2_FILE_PATHS,
        'S1_FILE_PATHS': S1_FILE_PATHS,
        'S2_EXPECTED_BANDS': S2_EXPECTED_BANDS,
        'S1_EXPECTED_BANDS': S1_EXPECTED_BANDS,
    })
    
    print(f"✅ Exported variables:")
    print(f"   INFERENCE_DATE       = {INFERENCE_DATE}")
    print(f"   PREVIOUS_DATES       = {len(PREVIOUS_DATES)} dates")
    print(f"   DOWNLOAD_PAIRS       = {len(DOWNLOAD_PAIRS)} pairs")
    print(f"   TARGET_FOLDERS       = {len(TARGET_FOLDERS)} folders")
    print(f"   PAIRS_DIR            = {PAIRS_DIR}")
    print(f"   DATE_METADATA_S2     = {len(DATE_METADATA_S2)} dates")
    print(f"   DATE_METADATA_S1     = {len(DATE_METADATA_S1)} dates")
    print(f"   S2_FILE_PATHS        = {len(S2_FILE_PATHS)} files (expected {S2_EXPECTED_BANDS}b each)")
    print(f"   S1_FILE_PATHS        = {len(S1_FILE_PATHS)} files (expected {S1_EXPECTED_BANDS}b each)")
    print("\n✅ Variables ready for Cell 2!")
    print("=" * 80)
    
    

    return dict(
        INFERENCE_DATE=INFERENCE_DATE,
        INFERENCE_IS_BROKEN=INFERENCE_IS_BROKEN,
        PREVIOUS_DATES=PREVIOUS_DATES,
        DOWNLOAD_PAIRS=DOWNLOAD_PAIRS,
        INFERENCE_WINDOW=INFERENCE_WINDOW,
        TARGET_FOLDERS=TARGET_FOLDERS,
        PAIRS_DIR=PAIRS_DIR,
        CLOUD_PCT_PER_DATE=CLOUD_PCT_PER_DATE,
        NODATA_PCT_S2_PER_DATE=NODATA_PCT_S2_PER_DATE,
        NODATA_PCT_S1_PER_DATE=NODATA_PCT_S1_PER_DATE,
        ALL_S2_DATES=ALL_S2_DATES,
        ALL_S1_DATES=ALL_S1_DATES,
        SPATIAL_EXTENT=SPATIAL_EXTENT,
        AOI_GEOMETRY=AOI_GEOMETRY,
        DOWNLOAD_RESULTS=DOWNLOAD_RESULTS,
        DATE_METADATA_S2=DATE_METADATA_S2,
        DATE_METADATA_S1=DATE_METADATA_S1,
        S2_FILE_PATHS=S2_FILE_PATHS,
        S1_FILE_PATHS=S1_FILE_PATHS,
    )


if __name__ == "__main__":
    print(cfg.summary())
    result = run()
    print(f"\n✅ Download complete. Inference date: {result['INFERENCE_DATE']}")

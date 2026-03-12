"""
config/settings.py
==================
Single source of truth for the entire pipeline.
All scripts import:  from config.settings import cfg

Never call os.getenv() directly in scripts — use cfg.* instead.

To run for a different mill, change only these 4 lines in .env:
    INGENIO=Amajac
    FECHA_INICIO=2026-02-15
    FECHA_FIN=2026-02-20
    FECHAS=2026-2-19
Everything else (paths, AOI, prefix, CRS...) is auto-derived.
"""

import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List

# -- Auto-load .env from project root -----------------------------------------
try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parent.parent / ".env"
    if _env.exists():
        load_dotenv(_env)
        print(f"[config] Loaded: {_env}")
    else:
        print(f"[config] WARNING: .env not found at {_env}. Copy .env.template -> .env")
except ImportError:
    print("[config] WARNING: python-dotenv not installed. Run: pip install python-dotenv")


# -- Static mill metadata ------------------------------------------------------
# All path slugs are taken DIRECTLY from the original notebook configs.
# Several mills have irregular names that cannot be derived by formula:
#   Pantaleon  -> OUTPUT_DIR = "Output-gt"   (not "Output-pantaleon")
#   Monte_Rosa -> slug = "monterosa"         (no underscore or space)
#   IPSA       -> OUT_DIR = "IPSA-auto"      (uppercase)
#
# NAME_LOCATION = value of NAME_CONFIG_ENV["location"] in original notebook.
#   Monte_Rosa uses underscore here (not the display "Monte Rosa").

INGENIOS_META = {
    "Pantaleon": {
        "DISPLAY":        "Pantaleon",
        "EMPRESA":        "Grupo Pantaleon",
        "PAIS":           "GT",
        "AOI_GEOJSON":    "GT01_BoundingBox_margin001.geojson",
        "AOI_BOUNDS":     {"west": -91.530544, "south": 13.947144,
                           "east": -90.590395, "north": 14.445327},
        "CRS":            "EPSG:32615",
        "OUT_DIR":        "pantaleon-auto",
        "INPUT_DIR":      "inputs-pantaleon-auto",
        "OUTPUT_DIR":     "Output-gt",           # irregular
        "NAME_LOCATION":  "Pantaleon",
    },
    "Monte_Rosa": {
        "DISPLAY":        "Monte Rosa",
        "EMPRESA":        "Grupo Pantaleon",
        "PAIS":           "NI",
        "AOI_GEOJSON":    "MR01_BoundingBox_margin001.geojson",
        "AOI_BOUNDS":     {"west": -87.502239, "south": 12.279927,
                           "east": -86.79237,  "north": 12.948203},
        "CRS":            "EPSG:32616",
        "OUT_DIR":        "monterosa-auto",      # irregular (no underscore/space)
        "INPUT_DIR":      "inputs-monterosa-auto",
        "OUTPUT_DIR":     "Output-monterosa",
        "NAME_LOCATION":  "Monte_Rosa",          # underscore, not space
    },
    "Amajac": {
        "DISPLAY":        "Amajac",
        "EMPRESA":        "Grupo Pantaleon",
        "PAIS":           "MX06",
        "AOI_GEOJSON":    "AM01_BoundingBox_margin001.geojson",
        "AOI_BOUNDS":     {"west": -98.543185, "south": 21.662828,
                           "east": -98.021094, "north": 22.280718},
        "CRS":            "EPSG:32614",
        "OUT_DIR":        "amajac-auto",
        "INPUT_DIR":      "inputs-amajac-auto",
        "OUTPUT_DIR":     "Output-amajac",
        "NAME_LOCATION":  "Amajac",
    },
    "EMSA": {
        "DISPLAY":        "EMSA",
        "EMPRESA":        "Grupo Pantaleon",
        "PAIS":           "MX07",
        "AOI_GEOJSON":    "EM01_BoundingBox_margin001.geojson",
        "AOI_BOUNDS":     {"west": -99.353038, "south": 22.437112,
                           "east": -98.49084,  "north": 23.31672},
        "CRS":            "EPSG:32614",
        "OUT_DIR":        "emsa-auto",
        "INPUT_DIR":      "inputs-emsa-auto",
        "OUTPUT_DIR":     "Output-emsa",
        "NAME_LOCATION":  "EMSA",
    },
    "IPSA": {
        "DISPLAY":        "IPSA",
        "EMPRESA":        "Grupo Pantaleon",
        "PAIS":           "MX02",
        "AOI_GEOJSON":    "IP01_BoundingBox_margin001.geojson",
        "AOI_BOUNDS":     {"west": -98.623859, "south": 21.65155,
                           "east": -97.987145, "north": 22.508884},
        "CRS":            "EPSG:32614",
        "OUT_DIR":        "IPSA-auto",           # irregular (uppercase)
        "INPUT_DIR":      "inputs-ipsa-auto",
        "OUTPUT_DIR":     "Output-ipsa",
        "NAME_LOCATION":  "IPSA",
    },
}


def _require(key: str) -> str:
    v = os.getenv(key, "").strip()
    if not v:
        raise EnvironmentError(
            f"Required variable '{key}' is missing from .env. "
            "Copy .env.template -> .env and fill it in."
        )
    return v


def _bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).strip().lower() in ("true", "1", "yes")


@dataclass
class PipelineConfig:
    # Required: just these 4 in .env
    INGENIO:       str = field(default_factory=lambda: _require("INGENIO"))
    SUPABASE_URL:  str = field(default_factory=lambda: _require("SUPABASE_URL"))
    SUPABASE_KEY:  str = field(default_factory=lambda: _require("SUPABASE_KEY"))

    # Dates
    _FECHA_INICIO: str = field(default_factory=lambda: _require("FECHA_INICIO"))
    _FECHA_FIN:    str = field(default_factory=lambda: _require("FECHA_FIN"))
    _FECHAS_RAW:   str = field(default_factory=lambda: _require("FECHAS"))

    # Products (optional, defaults to all 4)
    _PRODUCTOS_RAW: str = field(default_factory=lambda:
        os.getenv("PRODUCTOS", "NDVI,NDWI,SMART_GROWTH,WEED"))

    # Excel / DB
    EXCEL_PATH:   str  = field(default_factory=lambda: os.getenv("EXCEL_PATH", "Plantillas-new.xlsx"))
    BD_INSERT:    bool = field(default_factory=lambda: _bool("BD_INSERT", False))
    DRY_RUN_SYNC: bool = field(default_factory=lambda: _bool("DRY_RUN_SYNC", False))

    # Skip flags
    SKIP_DOWNLOAD:      bool = field(default_factory=lambda: _bool("SKIP_DOWNLOAD"))
    SKIP_CLOUD_REMOVAL: bool = field(default_factory=lambda: _bool("SKIP_CLOUD_REMOVAL"))
    SKIP_RENAME:        bool = field(default_factory=lambda: _bool("SKIP_RENAME"))
    SKIP_PROCESSING:    bool = field(default_factory=lambda: _bool("SKIP_PROCESSING"))
    SKIP_DB_PUSH:       bool = field(default_factory=lambda: _bool("SKIP_DB_PUSH"))

    def __post_init__(self):
        if self.INGENIO not in INGENIOS_META:
            raise EnvironmentError(
                f"INGENIO='{self.INGENIO}' is not valid. "
                f"Choose one of: {list(INGENIOS_META.keys())}"
            )
        # Expose to os.environ so legacy Supabase client code picks them up
        os.environ["SUPABASE_URL"] = self.SUPABASE_URL
        os.environ["SUPABASE_KEY"] = self.SUPABASE_KEY

    # -- Mill metadata shortcuts -----------------------------------------------

    @property
    def meta(self) -> dict:
        return INGENIOS_META[self.INGENIO]

    @property
    def ingenio_display(self) -> str:
        """'Monte Rosa' (with space) -- used in Supabase queries."""
        return self.meta["DISPLAY"]

    @property
    def PAIS(self) -> str:
        return self.meta["PAIS"]

    @property
    def EMPRESA(self) -> str:
        return self.meta["EMPRESA"]

    @property
    def CRS(self) -> str:
        return self.meta["CRS"]

    @property
    def AOI_GEOJSON(self) -> Path:
        return Path(self.meta["AOI_GEOJSON"])

    @property
    def AOI_BOUNDS(self) -> dict:
        return self.meta["AOI_BOUNDS"]

    # -- Date helpers ----------------------------------------------------------

    @property
    def fecha_inicio_dt(self) -> datetime:
        return datetime.strptime(self._FECHA_INICIO, "%Y-%m-%d")

    @property
    def fecha_fin_dt(self) -> datetime:
        return datetime.strptime(self._FECHA_FIN, "%Y-%m-%d")

    @property
    def FECHAS(self) -> List[str]:
        return [f.strip() for f in self._FECHAS_RAW.split(",") if f.strip()]

    @property
    def PRODUCTOS(self) -> List[str]:
        return [p.strip() for p in self._PRODUCTOS_RAW.split(",") if p.strip()]

    # -- Paths: ALL auto-derived from INGENIOS_META, nothing from .env ---------

    @property
    def OUT_DIR(self) -> Path:
        """Download root. e.g. pantaleon-auto / IPSA-auto"""
        return Path(self.meta["OUT_DIR"])

    @property
    def INPUT_DIR(self) -> Path:
        """Where renamed cloudfill TIFs go. e.g. inputs-pantaleon-auto"""
        return Path(self.meta["INPUT_DIR"])

    @property
    def OUTPUT_DIR(self) -> Path:
        """Product output folder. e.g. Output-gt / Output-monterosa"""
        return Path(self.meta["OUTPUT_DIR"])

    @property
    def PAIRS_DIR(self) -> Path:
        return self.OUT_DIR / "pairs"

    @property
    def CLOUD_FREE_DIR(self) -> Path:
        return self.OUT_DIR / "cloud_free_output"

    @property
    def NAME_CONFIG(self) -> dict:
        """
        Backward-compatible dict for rename_and_move_results().
        Matches NAME_CONFIG_ENV from the original notebook exactly.

        e.g. for Monte_Rosa:
            inference_folder : "monterosa-auto"
            output_folder    : "inputs-monterosa-auto"
            prefix           : "NI_Grupo Pantaleon"
            location         : "Monte_Rosa"     <- underscore, not space
        """
        return {
            "inference_folder": str(self.OUT_DIR),
            "output_folder":    str(self.INPUT_DIR),
            "prefix":           f"{self.PAIS}_{self.EMPRESA}",
            "location":         self.meta["NAME_LOCATION"],
        }

    def summary(self) -> str:
        nc = self.NAME_CONFIG
        lines = [
            "=" * 60,
            "  PIPELINE CONFIG",
            "=" * 60,
            f"  INGENIO        : {self.INGENIO}",
            f"  Display name   : {self.ingenio_display}",
            f"  PAIS / EMPRESA : {self.PAIS} / {self.EMPRESA}",
            f"  CRS            : {self.CRS}",
            f"  AOI GeoJSON    : {self.AOI_GEOJSON}",
            "",
            f"  FECHA_INICIO   : {self._FECHA_INICIO}",
            f"  FECHA_FIN      : {self._FECHA_FIN}",
            f"  FECHAS         : {self.FECHAS}",
            f"  PRODUCTOS      : {self.PRODUCTOS}",
            "",
            f"  OUT_DIR        : {self.OUT_DIR}",
            f"  PAIRS_DIR      : {self.PAIRS_DIR}",
            f"  CLOUD_FREE_DIR : {self.CLOUD_FREE_DIR}",
            f"  INPUT_DIR      : {self.INPUT_DIR}",
            f"  OUTPUT_DIR     : {self.OUTPUT_DIR}",
            "",
            f"  NAME_CONFIG    : prefix='{nc['prefix']}'  location='{nc['location']}'",
            "",
            f"  EXCEL_PATH     : {self.EXCEL_PATH}",
            f"  BD_INSERT      : {self.BD_INSERT}",
            f"  DRY_RUN_SYNC   : {self.DRY_RUN_SYNC}",
            "",
            f"  SKIP_DOWNLOAD      : {self.SKIP_DOWNLOAD}",
            f"  SKIP_CLOUD_REMOVAL : {self.SKIP_CLOUD_REMOVAL}",
            f"  SKIP_RENAME        : {self.SKIP_RENAME}",
            f"  SKIP_PROCESSING    : {self.SKIP_PROCESSING}",
            f"  SKIP_DB_PUSH       : {self.SKIP_DB_PUSH}",
            "=" * 60,
        ]
        return "\n".join(lines)


# -- Module-level singleton -- import this everywhere -------------------------
cfg = PipelineConfig()
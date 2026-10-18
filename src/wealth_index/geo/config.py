"""Feature-build configuration, loaded from configs/features.yaml."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class PathsConfig:
    boundaries: str = "data/spatial_data/subdistricts.json"
    osm_pbf: str = "data/raw/osm/java-250101.osm.pbf"
    # Globs: GEE splits large exports into tiles, and SDGSAT comes as several scenes.
    s2_composite: str = "data/raw/s2/s2_l2a_2024_median*.tif"
    sdgsat: str = "data/raw/sdgsat/**/*.tif"
    interim: str = "data/interim"
    processed: str = "data/processed"


@dataclass
class GridConfig:
    buffer_m: float = 2000  # around the study area, so KDE and distances aren't cut at the edge
    snap_m: float = 300  # grid origins are multiples of this, so the 10/30/100 m grids nest
    res_image: float = 10
    res_kde: float = 100
    res_distance: float = 30


# Reconstructed mapping of the thesis' nine POI categories (spec 8.6), as quackosm tag filters.
POI_CATEGORIES: dict[str, dict[str, Any]] = {
    "Pub": {
        "amenity": ["townhall", "community_centre", "library", "post_office", "courthouse"],
        "office": ["government"],
    },
    "Cmer": {
        "shop": True,
        "amenity": ["marketplace", "restaurant", "cafe", "fast_food"],
        "building": ["retail", "commercial"],
    },
    "Cmun": {
        "man_made": ["mast", "communications_tower"],
        "telecom": True,
        "office": ["telecommunication"],
    },
    "Edu": {"amenity": ["school", "university", "college", "kindergarten"]},
    # Not emergency=*: that key is also a road access tag.
    "Emr": {
        "amenity": ["police", "fire_station"],
        "emergency": ["ambulance_station", "assembly_point", "siren"],
    },
    "Fin": {"amenity": ["bank", "atm", "bureau_de_change"]},
    "HC": {"amenity": ["hospital", "clinic", "doctors", "dentist", "pharmacy"], "healthcare": True},
    "Pow": {"power": ["plant", "substation", "generator", "tower"]},
    "Tra": {
        "public_transport": ["station", "stop_position"],
        "railway": ["station", "halt"],
        "amenity": ["bus_station"],
    },
}
ROAD_CLASSES: dict[str, list[str]] = {
    "Pri": ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"],
    "Sec": ["secondary", "secondary_link"],
    "Ter": ["tertiary", "tertiary_link"],
}
ROAD_DRIVABLE = [
    *(v for values in ROAD_CLASSES.values() for v in values),
    "unclassified",
    "residential",
    "living_street",
    "service",
    "road",
]


@dataclass
class OsmConfig:
    aoi_buffer_m: float = 10_000  # extraction area, so edge villages see POIs just outside
    kde_radius_m: float = 1000
    count_radii_km: list[float] = field(default_factory=lambda: [1, 5, 10])
    buildings: bool = False
    cpu_limit: int | None = None  # quackosm threads; fewer threads use less memory
    clip_pbf: bool = True  # cut the PBF to the study area first (needs pyosmium); saves memory
    # Category -> quackosm tag filter ({key: [values]} or {key: true} for any value).
    poi_categories: dict[str, dict[str, Any]] = field(default_factory=lambda: dict(POI_CATEGORIES))
    road_classes: dict[str, list[str]] = field(default_factory=lambda: dict(ROAD_CLASSES))
    road_drivable: list[str] = field(
        default_factory=lambda: list(ROAD_DRIVABLE)
    )  # counted in Road_*


@dataclass
class GeeConfig:
    project: str | None = None
    start: str = "2024-01-01"
    end: str = "2025-01-01"  # exclusive
    max_cloud_pct: float = 60
    cld_prb_thresh: float = 50
    nir_drk_thresh: float = 0.15
    cld_prj_dist_km: float = 1
    buffer_m: float = 50
    statistic: str = "median"  # median | mean
    drive_folder: str = "wealth_index"
    file_prefix: str = "s2_l2a_2024_median"


@dataclass
class S2Config:
    bands: list[str] = field(
        default_factory=lambda: ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    )
    scale: float = 1e-4  # GEE exports reflectance x 10,000 as uint16
    nodata: float = 0
    ndbi_swir: str = "B11"  # B11 is the standard formula; B12 is what the thesis text says
    gee: GeeConfig = field(default_factory=GeeConfig)


@dataclass
class TextureConfig:
    bands: list[str] = field(default_factory=lambda: ["B2", "B4", "B8"])  # superset: all S2 bands
    levels: int = 32
    percentiles: list[float] = field(default_factory=lambda: [2, 98])
    distances: list[int] = field(default_factory=lambda: [1, 2])


@dataclass
class GreenConfig:
    settlement: str = "ntl"  # ntl | ndbi
    threshold: float = 0  # settlement where NTL (or NDBI) is above this
    radius_m: float = 200


@dataclass
class StripeConfig:
    enabled: bool = True
    direction: str = "auto"  # auto | vertical | horizontal (stripes run along this axis)
    k: float = 3  # flag pixels above the cross-stripe median by more than k * MAD
    min_run: int = 10  # ... and only in runs this long, so single bright lights survive
    filter_width: int = 5


@dataclass
class NtlConfig:
    band: int = 1
    nodata: float | None = None  # None: the file's own nodata, else fill found by footprint
    overlap: str = "median"  # median | first | latest (scenes sorted by file name)
    positive_threshold: float = 0  # NTLPos_* and NTLBin_Ratio use DN > this
    stripe: StripeConfig = field(default_factory=StripeConfig)


@dataclass
class BuildConfig:
    max_nan_share: float = 0.2


@dataclass
class FeatureBuildConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    crs: str = "EPSG:32748"
    id_col: str = "KDEPUM"
    feature_set: str = "curated"  # curated | superset
    grid: GridConfig = field(default_factory=GridConfig)
    osm: OsmConfig = field(default_factory=OsmConfig)
    s2: S2Config = field(default_factory=S2Config)
    texture: TextureConfig = field(default_factory=TextureConfig)
    green: GreenConfig = field(default_factory=GreenConfig)
    ntl: NtlConfig = field(default_factory=NtlConfig)
    build: BuildConfig = field(default_factory=BuildConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def interim(self) -> Path:
        return Path(self.paths.interim)

    @property
    def processed(self) -> Path:
        return Path(self.paths.processed)


def _from_dict(cls, raw: dict[str, Any]):
    hints = get_type_hints(cls)
    unknown = set(raw) - set(hints)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for key, value in raw.items():
        hint = hints[key]
        kwargs[key] = _from_dict(hint, value or {}) if dataclasses.is_dataclass(hint) else value
    return cls(**kwargs)


def load_feature_config(path: str | Path) -> FeatureBuildConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _from_dict(FeatureBuildConfig, raw)

"""Village boundaries: repair, reproject, area and centroid."""

from __future__ import annotations

import logging

import geopandas as gpd
import shapely
from shapely.geometry import MultiPolygon, Polygon

from wealth_index.geo.config import FeatureBuildConfig

log = logging.getLogger(__name__)

ADMIN_COLS = ["NAMOBJ", "WADMKC", "KDCPUM", "WADMKK", "KDPKAB", "WADMPR", "TIPADM"]


def _polygonal(geom):
    """make_valid can return a GeometryCollection; keep only its polygon parts."""
    if isinstance(geom, Polygon | MultiPolygon):
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon | MultiPolygon)]
    polys = [p for g in parts for p in (g.geoms if isinstance(g, MultiPolygon) else [g])]
    return MultiPolygon(polys) if len(polys) > 1 else polys[0]


def clean_villages(gdf: gpd.GeoDataFrame, cfg: FeatureBuildConfig) -> gpd.GeoDataFrame:
    """Repair geometries, reproject to the metric CRS, add area_km2 and lon/lat; index by id."""
    ids = gdf[cfg.id_col]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError(f"'{cfg.id_col}' must be unique and non-null")
    invalid = ~gdf.is_valid
    if invalid.any():
        log.info("Repairing %d invalid geometries", invalid.sum())
        gdf.loc[invalid, "geometry"] = [
            _polygonal(g) for g in shapely.make_valid(gdf.geometry[invalid])
        ]

    out = gdf[[cfg.id_col, *[c for c in ADMIN_COLS if c in gdf], "geometry"]].to_crs(cfg.crs)
    out["area_km2"] = out.area / 1e6
    # The centroid of an L-shaped or multi-part village can fall outside it; use a point inside.
    centroid = out.centroid
    point = centroid.where(centroid.within(out.geometry), out.representative_point())
    lonlat = point.to_crs("EPSG:4326")
    out["lon"], out["lat"] = lonlat.x, lonlat.y
    return out.set_index(cfg.id_col)


def run(cfg: FeatureBuildConfig) -> gpd.GeoDataFrame:
    villages = clean_villages(gpd.read_file(cfg.paths.boundaries), cfg)
    cfg.interim.mkdir(parents=True, exist_ok=True)
    cfg.processed.mkdir(parents=True, exist_ok=True)
    villages.to_parquet(cfg.interim / "boundaries.parquet")
    villages.drop(columns="geometry").to_csv(cfg.processed / "villages.csv")
    log.info("Wrote %d villages", len(villages))
    return villages


def read_villages(cfg: FeatureBuildConfig) -> gpd.GeoDataFrame:
    path = cfg.interim / "boundaries.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `wi features boundaries` first")
    return gpd.read_parquet(path)

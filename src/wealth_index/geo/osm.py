"""OpenStreetMap features: POI density, proximity and counts; road length, density and distance."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from shapely.geometry import box

from wealth_index.geo.catalogue import ROAD_CLASSES, osm_columns
from wealth_index.geo.config import FeatureBuildConfig
from wealth_index.geo.distance import count_within, distance_raster, nearest_distance
from wealth_index.geo.grid import Grid, study_grid
from wealth_index.geo.kde import line_density, point_density
from wealth_index.geo.zonal import zonal

log = logging.getLogger(__name__)

ROAD = "Road"  # quackosm group name for drivable highways
LINES = ("LineString", "MultiLineString")


def clip_pbf(pbf: Path, bounds: tuple[float, float, float, float], out: Path) -> Path:
    """Cut a PBF to a lon/lat box, so quackosm does not hold a whole region in memory.

    Keeps the nodes inside the box, the ways and relations that touch it, and everything those
    reference, so features crossing the edge stay complete.
    """
    import osmium

    if out.exists():
        return out
    log.info("Clipping %s to %s (one-off, a few minutes)", pbf.name, np.round(bounds, 3))
    minx, miny, maxx, maxy = bounds
    ids = osmium.IdTracker()
    for obj in osmium.FileProcessor(str(pbf)):  # PBFs are sorted: nodes, ways, relations
        if obj.is_node():
            if minx <= obj.location.lon <= maxx and miny <= obj.location.lat <= maxy:
                ids.add_node(obj.id)
        elif ids.contains_any_references(obj):
            (ids.add_way if obj.is_way() else ids.add_relation)(obj.id)
    ids.complete_backward_references(str(pbf), relation_depth=1)
    tmp = out.with_name(out.name + ".part")
    with osmium.SimpleWriter(osmium.io.File(str(tmp), "pbf"), overwrite=True) as writer:
        for obj in osmium.FileProcessor(str(pbf)).with_filter(ids.id_filter()):
            writer.add(obj)
    tmp.replace(out)
    return out


def _extract(
    pbf: Path, tags: dict, aoi, out_dir: Path, name: str, cpu_limit: int | None = None
) -> Path:
    """Run quackosm once per (file, tags, area); later runs reuse the GeoParquet."""
    import quackosm as qosm

    stat = pbf.stat()
    key = json.dumps([pbf.name, stat.st_size, stat.st_mtime_ns, tags, aoi.bounds], sort_keys=True)
    out = out_dir / f"osm_{name}_{hashlib.sha1(key.encode()).hexdigest()[:12]}.parquet"
    if not out.exists():
        log.info("Extracting %s from %s (a few minutes for all of Java)", name, pbf.name)
        qosm.convert_pbf_to_parquet(
            pbf,
            tags_filter=tags,
            geometry_filter=aoi,
            result_file_path=out,
            working_directory=out_dir / "quackosm",
            verbosity_mode="transient",
            cpu_limit=cpu_limit,
        )
    return out


def extract(cfg: FeatureBuildConfig, villages: gpd.GeoDataFrame) -> tuple[Path, Path | None]:
    """GeoParquet of POIs + roads (and buildings if enabled) around the study area."""
    pbf = Path(cfg.paths.osm_pbf)
    if not pbf.exists():
        raise FileNotFoundError(f"{pbf} not found (see paths.osm_pbf in the features config)")
    minx, miny, maxx, maxy = villages.total_bounds
    b = cfg.osm.aoi_buffer_m
    aoi = gpd.GeoSeries([box(minx - b, miny - b, maxx + b, maxy + b)], crs=villages.crs)
    aoi = aoi.to_crs("EPSG:4326").iloc[0]
    if cfg.osm.clip_pbf:
        stat = pbf.stat()
        key = json.dumps([pbf.name, stat.st_size, stat.st_mtime_ns, aoi.bounds])
        clip_name = f"{pbf.name.split('.')[0]}_clip_{hashlib.sha1(key.encode()).hexdigest()[:12]}"
        pbf = clip_pbf(pbf, aoi.bounds, cfg.interim / f"{clip_name}.osm.pbf")
    tags = {**cfg.osm.poi_categories, ROAD: {"highway": cfg.osm.road_drivable}}
    cpu = cfg.osm.cpu_limit
    pois_roads = _extract(pbf, tags, aoi, cfg.interim, "pois_roads", cpu)
    buildings = (
        _extract(pbf, {"building": True}, aoi, cfg.interim, "buildings", cpu)
        if cfg.osm.buildings
        else None
    )
    return pois_roads, buildings


def poi_points(osm: gpd.GeoDataFrame, categories: list[str]) -> gpd.GeoDataFrame:
    """One point per POI and category, from quackosm output with one column per category.

    Areas become representative points. A place mapped both as a node and as an area
    (a bank node inside a bank building) counts once: the area is dropped when a node of the
    same category lies inside it.
    """
    frames = [
        osm.loc[osm[c].notna(), ["feature_id", "geometry"]].assign(category=c)
        for c in categories
        if c in osm
    ]
    if not frames:
        return gpd.GeoDataFrame(
            columns=["feature_id", "category", "geometry"], geometry="geometry", crs=osm.crs
        )
    pois = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=osm.crs)
    is_node = pois["feature_id"].str.startswith("node/")
    areas = pois[~is_node & pois.geom_type.isin(["Polygon", "MultiPolygon"])]
    nodes = pois[is_node]
    if len(areas) and len(nodes):
        a_idx, n_idx = nodes.sindex.query(areas.geometry, predicate="contains")
        same = areas["category"].to_numpy()[a_idx] == nodes["category"].to_numpy()[n_idx]
        pois = pois.drop(index=areas.index[np.unique(a_idx[same])])
    pois["geometry"] = pois.representative_point()
    return pois.reset_index(drop=True)


def road_lines(osm: gpd.GeoDataFrame, classes: dict[str, list[str]]) -> gpd.GeoDataFrame:
    """Drivable road lines with a `cls` column: Pri/Sec/Ter, or None for other drivable roads."""
    roads = osm.loc[osm[ROAD].notna() & osm.geom_type.isin(LINES), ["feature_id", "geometry"]]
    highway = osm.loc[roads.index, ROAD].str.split("=", n=1).str[1]
    lookup = {v: c for c, values in classes.items() for v in values}
    return roads.assign(highway=highway, cls=highway.map(lookup)).reset_index(drop=True)


def clipped_length(villages: gpd.GeoDataFrame, lines: gpd.GeoSeries) -> pd.Series:
    """Metres of line inside each village."""
    v_idx, l_idx = lines.sindex.query(villages.geometry, predicate="intersects")
    polys = np.asarray(villages.geometry)
    segs = np.asarray(lines)
    shapely.prepare(polys)
    lengths = shapely.length(segs[l_idx])
    crossing = ~shapely.contains_properly(polys[v_idx], segs[l_idx])  # only these need clipping
    lengths[crossing] = shapely.length(
        shapely.intersection(segs[l_idx][crossing], polys[v_idx][crossing])
    )
    return pd.Series(np.bincount(v_idx, lengths, minlength=len(villages)), index=villages.index)


def building_features(villages: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> pd.DataFrame:
    """Count, density, footprint share and mean footprint, assigning each building to the
    village that holds its representative point."""
    footprints = buildings[buildings.geom_type.isin(["Polygon", "MultiPolygon"])]
    b_idx, v_idx = villages.sindex.query(footprints.representative_point(), predicate="within")
    area = footprints.area.to_numpy()[b_idx]
    n = np.bincount(v_idx, minlength=len(villages)).astype(float)
    total = np.bincount(v_idx, area, minlength=len(villages))
    km2 = villages["area_km2"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_area = np.where(n > 0, total / n, np.nan)
    return pd.DataFrame(
        {
            "Bldg_Count": n,
            "Bldg_Dens": n / km2,
            "Bldg_AreaShare": total / (km2 * 1e6),
            "Bldg_MeanArea": mean_area,
        },
        index=villages.index,
    )


def _write_bands(path: Path, grid: Grid, bands: dict[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **grid.profile(count=len(bands))) as dst:
        for i, (name, arr) in enumerate(bands.items(), start=1):
            dst.write(arr.astype("float32"), i)
            dst.set_band_description(i, name)
    return path


def osm_features(
    cfg: FeatureBuildConfig,
    villages: gpd.GeoDataFrame,
    osm: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    osm = osm.to_crs(cfg.crs)
    cats = list(cfg.osm.poi_categories)
    pois = poi_points(osm, cats)
    unique_pois = pois.drop_duplicates("feature_id")
    roads = road_lines(osm, cfg.osm.road_classes)
    log.info("OSM: %d POIs (%d unique), %d road lines", len(pois), len(unique_pois), len(roads))

    g = cfg.grid
    kde_grid = study_grid(villages.total_bounds, g.res_kde, g.buffer_m, g.snap_m, cfg.crs)
    dist_grid = study_grid(villages.total_bounds, g.res_distance, g.buffer_m, g.snap_m, cfg.crs)
    r = cfg.osm.kde_radius_m

    def xy(gdf):
        return shapely.get_coordinates(gdf.geometry.to_numpy())

    density = {
        f"POI{c}_Dens": point_density(xy(pois[pois.category == c]), kde_grid, r) for c in cats
    }
    density["POI_Mean"] = point_density(xy(unique_pois), kde_grid, r)
    density["Road_Dens"] = line_density(roads.geometry, kde_grid, r)
    for c in ROAD_CLASSES:
        density[f"Road{c}_Dens"] = line_density(roads.geometry[roads.cls == c], kde_grid, r)
    path = _write_bands(cfg.interim / "rasters" / "osm_density.tif", kde_grid, density)
    z = zonal(path, villages, ["mean"])
    feats = {name: z[f"{i}:mean"] for i, name in enumerate(density, start=1)}

    dist = {
        f"Road{c}_MeanDist": distance_raster(roads.geometry[roads.cls == c], dist_grid)
        for c in ROAD_CLASSES
    }
    path = _write_bands(cfg.interim / "rasters" / "osm_distance.tif", dist_grid, dist)
    z = zonal(path, villages, ["mean"])
    feats |= {name: z[f"{i}:mean"] for i, name in enumerate(dist, start=1)}

    for c in cats:
        feats[f"POI{c}_Prox"] = nearest_distance(villages, pois.geometry[pois.category == c])
    for c in ROAD_CLASSES:
        feats[f"Road{c}_Prox"] = nearest_distance(villages, roads.geometry[roads.cls == c])
    for km in cfg.osm.count_radii_km:
        feats[f"POI_Count{km:g}KM"] = count_within(villages, unique_pois.geometry, km * 1000)
    feats["Road_Length"] = clipped_length(villages, roads.geometry) / 1000
    feats["Road_RLength"] = feats["Road_Length"] / villages["area_km2"]

    out = pd.DataFrame(feats)
    if buildings is not None:
        out = out.join(building_features(villages, buildings.to_crs(cfg.crs)))
    return out[osm_columns(cfg)]


def run(cfg: FeatureBuildConfig, villages: gpd.GeoDataFrame) -> pd.DataFrame:
    pois_roads, buildings = extract(cfg, villages)
    return osm_features(
        cfg,
        villages,
        gpd.read_parquet(pois_roads),
        gpd.read_parquet(buildings) if buildings else None,
    )

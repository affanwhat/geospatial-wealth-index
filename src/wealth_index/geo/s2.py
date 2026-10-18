"""Sentinel-2 L2A: cloud-masked composite on GEE, then spectral indices computed locally."""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from wealth_index.geo.catalogue import INDICES, s2_columns
from wealth_index.geo.config import FeatureBuildConfig
from wealth_index.geo.grid import iter_tiles, mosaic_vrt, study_grid
from wealth_index.geo.zonal import zonal

log = logging.getLogger(__name__)

# Bands of the indices raster: the seven indices, then NDBI on positive pixels only, then
# NDBI > 0 as 1/0 (NaN stays NaN), so Pos_MEAN and Bin_Ratio are plain zonal means.
INDEX_BANDS = [*INDICES, "NDBI_POS", "NDBI_BIN"]


# --- Composite on Google Earth Engine -------------------------------------------------------
# Cloud and shadow masking follows Braaten's s2cloudless tutorial for the GEE Python API.


def _masked_collection(ee, aoi, gee):
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(gee.start, gee.end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", gee.max_cloud_pct))
    )
    prob = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(gee.start, gee.end)
    )
    joined = ee.ImageCollection(
        ee.Join.saveFirst("s2cloudless").apply(
            primary=s2,
            secondary=prob,
            condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
        )
    )

    def mask(img):
        cld_prb = ee.Image(img.get("s2cloudless")).select("probability")
        clouds = cld_prb.gt(gee.cld_prb_thresh)
        not_water = img.select("SCL").neq(6)
        dark = img.select("B8").lt(gee.nir_drk_thresh * 1e4).multiply(not_water)
        azimuth = ee.Number(90).subtract(ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE")))
        projection = (
            clouds.directionalDistanceTransform(azimuth, gee.cld_prj_dist_km * 10)
            .reproject(crs=img.select(0).projection(), scale=100)
            .select("distance")
            .mask()
        )
        shadows = projection.multiply(dark)
        cld_shdw = (
            clouds.add(shadows)
            .gt(0)
            .focalMin(2)
            .focalMax(gee.buffer_m * 2 / 20)
            .reproject(crs=img.select(0).projection(), scale=20)
        )
        return img.select("B.*").updateMask(cld_shdw.Not())

    return joined.map(mask)


def export_composite(cfg: FeatureBuildConfig, villages: gpd.GeoDataFrame):
    """Submit the composite export to Google Drive on the 10 m study grid; returns the task."""
    import ee

    gee = cfg.s2.gee
    ee.Initialize(project=gee.project)
    grid = study_grid(
        villages.total_bounds, cfg.grid.res_image, cfg.grid.buffer_m, cfg.grid.snap_m, cfg.crs
    )
    region = ee.Geometry.Rectangle(list(grid.bounds), proj=cfg.crs, geodesic=False)
    images = _masked_collection(ee, region, gee).map(lambda img: img.resample("bilinear"))
    composite = images.median() if gee.statistic == "median" else images.mean()
    image = composite.select(cfg.s2.bands).round().toUint16()  # reflectance x 1e4, 0 = no data
    t = grid.transform
    task = ee.batch.Export.image.toDrive(
        image=image.unmask(0),
        description=gee.file_prefix,
        folder=gee.drive_folder,
        fileNamePrefix=gee.file_prefix,
        crs=cfg.crs,
        crsTransform=[t.a, t.b, t.c, t.d, t.e, t.f],
        dimensions=f"{grid.width}x{grid.height}",
        maxPixels=int(1e11),
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
        skipEmptyTiles=True,
    )
    task.start()
    log.info(
        "Submitted GEE export %s (%dx%d px); download the file(s) into %s",
        task.id,
        grid.width,
        grid.height,
        Path(cfg.paths.s2_composite).parent,
    )
    return task


# --- Local index computation ----------------------------------------------------------------


def composite_path(cfg: FeatureBuildConfig) -> Path:
    paths = sorted(Path(p) for p in glob.glob(cfg.paths.s2_composite))
    if not paths:
        raise FileNotFoundError(f"No Sentinel-2 composite matches {cfg.paths.s2_composite}")
    return mosaic_vrt(paths, cfg.interim / "rasters" / "s2_composite.vrt")


def compute_indices(bands: dict[str, np.ndarray], ndbi_swir: str = "B11") -> dict[str, np.ndarray]:
    """Spectral indices from surface reflectance (0-1). Division by zero gives NaN."""
    blue, red, nir, swir = bands["B2"], bands["B4"], bands["B8"], bands[ndbi_swir]
    with np.errstate(divide="ignore", invalid="ignore"):

        def ratio(num, den):
            return np.where(den != 0, num / den, np.nan)

        ndvi = ratio(nir - red, nir + red)
        ndbi = ratio(swir - nir, swir + nir)
        evi = 2.5 * ratio(nir - red, nir + 6 * red - 7.5 * blue + 1)
        out = {
            "NDVI": ndvi,
            # The EVI denominator crosses zero on blue-bright pixels (blue roofs, haze), where
            # EVI runs off to +-infinity; outside [-1, 1] it carries no vegetation signal.
            "EVI": np.where(np.abs(evi) <= 1, evi, np.nan),
            "SAVI": 1.5 * ratio(nir - red, nir + red + 0.5),
            "MSAVI": (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2,
            "TVI": np.sqrt(ndvi + 0.5),  # NaN where NDVI < -0.5
            "GR": ratio(nir, red),
            "NDBI": ndbi,
            "NDBI_POS": np.where(ndbi > 0, ndbi, np.nan),
            "NDBI_BIN": np.where(np.isnan(ndbi), np.nan, (ndbi > 0).astype(float)),
        }
    return out


def write_indices(cfg: FeatureBuildConfig, composite: Path, out: Path) -> Path:
    """Index raster on the composite's grid, computed tile by tile (the full grid is ~10^8 px)."""
    needed = sorted({"B2", "B4", "B8", cfg.s2.ndbi_swir}, key=cfg.s2.bands.index)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(composite) as src:
        prof = src.profile | {
            "driver": "GTiff",
            "count": len(INDEX_BANDS),
            "dtype": "float32",
            "nodata": np.nan,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "deflate",
            "predictor": 3,
            "BIGTIFF": "IF_SAFER",
        }
        with rasterio.open(out, "w", **prof) as dst:
            for tile in iter_tiles(src.width, src.height):
                bands = {}
                for b in needed:
                    raw = src.read(cfg.s2.bands.index(b) + 1, window=tile.inner).astype("float32")
                    bands[b] = np.where(raw == cfg.s2.nodata, np.nan, raw * cfg.s2.scale)
                indices = compute_indices(bands, cfg.s2.ndbi_swir)
                for i, name in enumerate(INDEX_BANDS, start=1):
                    dst.write(indices[name].astype("float32"), i, window=tile.inner)
            for i, name in enumerate(INDEX_BANDS, start=1):
                dst.set_band_description(i, name)
    return out


def s2_features(cfg: FeatureBuildConfig, villages: gpd.GeoDataFrame, indices: Path) -> pd.DataFrame:
    ops = ["mean", "stdev", "min", "max", "sum"]
    z = zonal(indices, villages, ops)
    feats = {}
    for i, name in enumerate(INDEX_BANDS, start=1):
        feats[f"{name}_MEAN"] = z[f"{i}:mean"]
        feats[f"{name}_STD"] = z[f"{i}:stdev"]
        feats[f"{name}_MIN"] = z[f"{i}:min"]
        feats[f"{name}_MAX"] = z[f"{i}:max"]
        feats[f"{name}_RANGE"] = z[f"{i}:max"] - z[f"{i}:min"]
        feats[f"{name}_SUM"] = z[f"{i}:sum"]
    feats["NDBIPos_MEAN"] = feats["NDBI_POS_MEAN"]
    feats["NDBIBin_Ratio"] = feats["NDBI_BIN_MEAN"]
    return pd.DataFrame(feats)[s2_columns(cfg)]


def run(cfg: FeatureBuildConfig, villages: gpd.GeoDataFrame) -> pd.DataFrame:
    indices = write_indices(cfg, composite_path(cfg), cfg.interim / "rasters" / "s2_indices.tif")
    return s2_features(cfg, villages, indices)

"""Green exposure around settlements (thesis section 3.4.2.2), at 0 m and within a radius.

Settlement pixels are where night-time light (or NDBI) is above a threshold. Exposure at 0 m
is the NDVI of the settlement pixel itself; at 200 m it is the mean NDVI in a disk around it.
Village statistics use settlement pixels only, so villages without any get NaN.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy.signal import fftconvolve

from wealth_index.geo.catalogue import green_columns
from wealth_index.geo.config import FeatureBuildConfig
from wealth_index.geo.grid import iter_tiles
from wealth_index.geo.zonal import zonal

NDVI_BAND, NDBI_BAND = 1, 7  # in the s2 indices raster


def disk(radius_m: float, res: float) -> np.ndarray:
    n = int(math.floor(radius_m / res))
    yy, xx = np.mgrid[-n : n + 1, -n : n + 1] * res
    return (xx**2 + yy**2 <= radius_m**2).astype(float)


def focal_mean(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Mean of the non-NaN values under the kernel at each pixel."""
    valid = ~np.isnan(arr)
    total = fftconvolve(np.where(valid, arr, 0), kernel, mode="same")
    count = fftconvolve(valid.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0.5, total / count, np.nan)


def write_green(cfg: FeatureBuildConfig, indices: Path, ntl: Path | None, out: Path) -> Path:
    with ExitStack() as stack:
        idx = stack.enter_context(rasterio.open(indices))
        if cfg.green.settlement == "ntl":
            ntl_src = stack.enter_context(rasterio.open(ntl))
            settle_src, settle_band = (
                stack.enter_context(
                    WarpedVRT(
                        ntl_src,
                        crs=idx.crs,
                        transform=idx.transform,
                        width=idx.width,
                        height=idx.height,
                        resampling=Resampling.nearest,
                    )
                ),
                1,
            )
        else:
            settle_src, settle_band = idx, NDBI_BAND
        kernel = disk(cfg.green.radius_m, idx.res[0])
        halo = kernel.shape[0] // 2
        prof = idx.profile | {"count": 2}
        dst = stack.enter_context(rasterio.open(out, "w", **prof))
        for tile in iter_tiles(idx.width, idx.height, halo=halo):
            ndvi = idx.read(NDVI_BAND, window=tile.outer)
            settle = (
                settle_src.read(settle_band, window=tile.inner) > cfg.green.threshold
            )  # NaN -> False
            dst.write(
                np.where(settle, ndvi[tile.crop], np.nan).astype("float32"), 1, window=tile.inner
            )
            dst.write(
                np.where(settle, focal_mean(ndvi, kernel)[tile.crop], np.nan).astype("float32"),
                2,
                window=tile.inner,
            )
        dst.descriptions = ("GREEN_0M", f"GREEN_{cfg.green.radius_m:g}M")
    return out


def green_features(
    cfg: FeatureBuildConfig, villages: gpd.GeoDataFrame, green: Path
) -> pd.DataFrame:
    z = zonal(green, villages, ["mean", "stdev", "quantile(q=0.9)"])
    feats = {
        "Green_MEAN": z["1:mean"],
        "Green_STD": z["1:stdev"],
        "Green200M_MEAN": z["2:mean"],
        "Green200M_STD": z["2:stdev"],
        "Green200M_PCT90": z["2:quantile_90"],
    }
    return pd.DataFrame(feats)[green_columns(cfg)]


def run(cfg: FeatureBuildConfig, villages: gpd.GeoDataFrame) -> pd.DataFrame:
    rasters = cfg.interim / "rasters"
    ntl = rasters / "ntl.tif" if cfg.green.settlement == "ntl" else None
    green = write_green(cfg, rasters / "s2_indices.tif", ntl, rasters / "green.tif")
    return green_features(cfg, villages, green)

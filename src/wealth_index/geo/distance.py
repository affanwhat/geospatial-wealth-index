"""Polygon-to-feature distances, distance rasters, and counts within a radius."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from scipy import ndimage

from wealth_index.geo.grid import Grid


def nearest_distance(villages: gpd.GeoDataFrame, features: gpd.GeoSeries) -> pd.Series:
    """Metres from each village polygon to the nearest feature; 0 if one touches or lies inside."""
    if len(features) == 0:
        return pd.Series(np.nan, index=villages.index)
    (src, _), dist = features.sindex.nearest(
        villages.geometry, return_all=False, return_distance=True
    )
    out = np.full(len(villages), np.nan)
    out[src] = dist
    return pd.Series(out, index=villages.index)


def distance_raster(features: gpd.GeoSeries, grid: Grid) -> np.ndarray:
    """Metres from each cell to the nearest cell a feature touches (NaN if there are none)."""
    if len(features) == 0:
        return np.full(grid.shape, np.nan, dtype="float32")
    hit = rasterize(
        ((g, 1) for g in features),
        out_shape=grid.shape,
        transform=grid.transform,
        all_touched=True,
        dtype="uint8",
    )
    return (ndimage.distance_transform_edt(hit == 0) * grid.res).astype("float32")


def count_within(villages: gpd.GeoDataFrame, points: gpd.GeoSeries, distance_m: float) -> pd.Series:
    """Number of points within `distance_m` of each village polygon (inside counts as 0 m)."""
    village_idx, _ = points.sindex.query(
        villages.geometry, predicate="dwithin", distance=distance_m
    )
    return pd.Series(np.bincount(village_idx, minlength=len(villages)), index=villages.index)

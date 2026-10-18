"""Zonal statistics with partial-pixel weighting (exactextract).

A pixel on a village edge counts by the share of it inside the polygon, which matters for
small Jakarta kelurahan. NaN pixels are skipped.
"""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from exactextract import exact_extract


def _suffix(op: str) -> str:
    """exactextract's column suffix for an op, e.g. 'quantile(q=0.9)' -> 'quantile_90'."""
    m = re.fullmatch(r"quantile\(q=([\d.]+)\)", op)
    return f"quantile_{round(float(m.group(1)) * 100)}" if m else op


def zonal(raster: str | Path, villages: gpd.GeoDataFrame, ops: list[str]) -> pd.DataFrame:
    """One row per village (same index), columns '<band>:<op>', e.g. '1:mean', '2:quantile_90'."""
    with rasterio.open(raster) as src:
        count, crs = src.count, src.crs
    geoms = villages[["geometry"]].to_crs(crs).reset_index(drop=True)
    raw = exact_extract(str(raster), geoms, ops, output="pandas")
    if len(raw) != len(villages):
        raise RuntimeError(f"exactextract returned {len(raw)} rows for {len(villages)} villages")
    cols = {}
    for b in range(1, count + 1):
        for op in ops:
            s = _suffix(op)
            cols[f"{b}:{s}"] = raw[f"band_{b}_{s}" if count > 1 else s].to_numpy()
    return pd.DataFrame(cols, index=villages.index)

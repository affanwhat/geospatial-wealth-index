"""Grey-level co-occurrence (GLCM) texture per village, replacing ENVI FETEX 2.0.

Each band is quantised to `levels` grey levels (1..levels) with study-wide percentile limits.
Pixels outside the village get the reserved level 0, and row and column 0 are dropped from
the co-occurrence matrix, so pixel pairs that leave the polygon never count.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, geometry_window
from scipy import stats
from skimage.feature import graycomatrix, graycoprops

from wealth_index.geo.catalogue import TEXTURE_PROPS, texture_bands, texture_columns
from wealth_index.geo.config import FeatureBuildConfig
from wealth_index.geo.s2 import composite_path

log = logging.getLogger(__name__)

ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def quantize(x: np.ndarray, lo: float, hi: float, levels: int) -> np.ndarray:
    """Map values to grey levels 1..levels (0 is reserved for 'outside')."""
    q = np.floor((x - lo) / (hi - lo) * levels)
    return (np.clip(np.nan_to_num(q, nan=0), 0, levels - 1) + 1).astype(np.uint8)


def masked_glcm(
    q: np.ndarray, inside: np.ndarray, levels: int, distances: list[int]
) -> dict[str, float]:
    """GLCM properties over pixel pairs with both pixels inside, averaged over distances
    and angles."""
    q = np.where(inside, q, 0).astype(np.uint8)
    P = graycomatrix(q, distances, ANGLES, levels=levels + 1, symmetric=True)[1:, 1:]
    has_pairs = P.sum(axis=(0, 1)) > 0  # a thin village may have no pairs at distance 2
    if not has_pairs.any():
        return {p: np.nan for p in TEXTURE_PROPS}
    return {
        p: float(graycoprops(P, p.lower() if p != "ASM" else p)[has_pairs].mean())
        for p in TEXTURE_PROPS
    }


def _limits(src, band: int, percentiles, nodata) -> tuple[float, float]:
    """Study-wide percentiles from a decimated read (about 4 million pixels)."""
    f = max(1, int(np.sqrt(src.width * src.height / 4e6)))
    arr = src.read(band, out_shape=(src.height // f, src.width // f), resampling=Resampling.nearest)
    lo, hi = np.percentile(arr[arr != nodata], percentiles)
    return float(lo), float(hi)


def texture_features(
    cfg: FeatureBuildConfig, villages: gpd.GeoDataFrame, composite
) -> pd.DataFrame:
    t, s2 = cfg.texture, cfg.s2
    bands = texture_bands(cfg)
    rows = {}
    with rasterio.open(composite) as src:
        idx = [s2.bands.index(b) + 1 for b in bands]
        limits = {
            b: _limits(src, i, t.percentiles, s2.nodata) for b, i in zip(bands, idx, strict=True)
        }
        log.info("Texture: %d villages x %d bands", len(villages), len(bands))
        for vid, geom in villages.to_crs(src.crs).geometry.items():
            win = geometry_window(src, [geom], pad_x=1, pad_y=1)
            data = src.read(idx, window=win)
            inside = geometry_mask(
                [geom], out_shape=data.shape[1:], transform=src.window_transform(win), invert=True
            )
            row = {}
            for b, raw in zip(bands, data, strict=True):
                valid = inside & (raw != s2.nodata)
                props = masked_glcm(
                    quantize(raw, *limits[b], t.levels), valid, t.levels, t.distances
                )
                values = raw[valid] * s2.scale
                varied = (
                    values.size > 3 and np.ptp(values) > 0
                )  # moments of a constant are undefined
                props["SKEW"] = float(stats.skew(values)) if varied else np.nan
                props["KURT"] = float(stats.kurtosis(values)) if varied else np.nan
                row |= {f"TEX_{b}_{p}": v for p, v in props.items()}
            rows[vid] = row
    return pd.DataFrame.from_dict(rows, orient="index").reindex(villages.index)[
        texture_columns(cfg)
    ]


def run(cfg: FeatureBuildConfig, villages: gpd.GeoDataFrame) -> pd.DataFrame:
    return texture_features(cfg, villages, composite_path(cfg))

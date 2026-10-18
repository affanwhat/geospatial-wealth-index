"""SDGSAT-1 GIU night-time lights: stripe correction, mosaic on the study grid, statistics.

Stripe correction reconstructs Wu et al. (2024): stripe pixels are replaced by the mean of
their non-stripe 8-neighbours. Detection (not specified in the paper) flags pixels brighter
than the median across the stripe direction by more than k * MAD, and keeps only flags that
form runs of at least `min_run` pixels along the stripe, so single bright lights are kept.
"""

from __future__ import annotations

import glob
import logging
import warnings
from contextlib import ExitStack
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from scipy import ndimage

from wealth_index.geo.catalogue import ntl_columns
from wealth_index.geo.config import FeatureBuildConfig, StripeConfig
from wealth_index.geo.grid import Grid, iter_tiles, study_grid
from wealth_index.geo.zonal import zonal

log = logging.getLogger(__name__)

NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=float)


def stripe_direction(img: np.ndarray) -> str:
    """'vertical' if stripes run down the columns, else 'horizontal'.

    A stripe shifts the median of the whole line it runs along, while sparse bright lights
    barely move any median; so the profile of column medians has a spike for every vertical
    stripe. Compare the largest spike in each profile against its typical variation.
    """

    def spikiness(profile):
        d2 = np.abs(np.diff(profile, 2))
        return np.nanmax(d2) / (np.nanmedian(d2) + 1e-9)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows or columns
        cols, rows = np.nanmedian(img, axis=0), np.nanmedian(img, axis=1)
    return "vertical" if spikiness(cols) > spikiness(rows) else "horizontal"


def stripe_mask(img: np.ndarray, direction: str, k: float, min_run: int, width: int) -> np.ndarray:
    vertical = direction == "vertical"
    valid = ~np.isnan(img)
    filled = np.where(valid, img, 0)
    resid = filled - ndimage.median_filter(filled, size=(1, width) if vertical else (width, 1))
    r = resid[valid]
    mad = 1.4826 * np.median(np.abs(r - np.median(r))) if r.size else 0.0
    candidates = valid & (resid > k * max(mad, 1e-6))
    line = np.ones((min_run, 1) if vertical else (1, min_run), dtype=bool)
    return ndimage.binary_opening(candidates, structure=line)


def fill_from_neighbours(img: np.ndarray, mask: np.ndarray, max_iter: int = 3) -> np.ndarray:
    """Replace masked pixels by the mean of their valid, unmasked 8-neighbours."""
    out = img.astype("float32", copy=True)
    todo = mask.copy()
    for _ in range(max_iter):
        if not todo.any():
            break
        good = ~todo & ~np.isnan(out)
        total = ndimage.convolve(np.where(good, out, 0), NEIGHBOURS, mode="constant")
        count = ndimage.convolve(good.astype(float), NEIGHBOURS, mode="constant")
        fill = todo & (count > 0)
        out[fill] = total[fill] / count[fill]
        todo &= ~fill
    out[todo] = np.nan
    return out


def destripe(img: np.ndarray, cfg: StripeConfig, direction: str | None = None) -> np.ndarray:
    direction = direction or (stripe_direction(img) if cfg.direction == "auto" else cfg.direction)
    return fill_from_neighbours(
        img, stripe_mask(img, direction, cfg.k, cfg.min_run, cfg.filter_width)
    )


def _footprint(src, band: int, factor: int = 32) -> np.ndarray:
    """Coarse data footprint for scenes without a nodata value: zero areas connected to the
    scene border are fill; zeros enclosed by data are dark but valid pixels. One cell per
    `factor` x `factor` block, read in strips so a full scene never sits in memory."""
    rows, cols = -(-src.height // factor), -(-src.width // factor)
    coarse = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        strip = src.read(
            band, window=Window(0, r * factor, src.width, min(factor, src.height - r * factor))
        )
        lit = np.pad(strip != 0, ((0, 0), (0, cols * factor - src.width)))
        coarse[r] = lit.reshape(lit.shape[0], cols, factor).any(axis=(0, 2))
    return ndimage.binary_fill_holes(coarse)


def correct_scene(path: Path, grid: Grid, cfg: FeatureBuildConfig, out: Path) -> Path | None:
    """Crop a scene to the study area, mark no-data as NaN, remove stripes; float32 GeoTIFF."""
    n = cfg.ntl
    with rasterio.open(path) as src:
        bounds = transform_bounds(grid.crs, src.crs, *grid.bounds)
        win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, src.width, src.height)) if _overlaps(win, src) else None
        if win is None:
            log.info("Scene %s does not cover the study area; skipped", path.name)
            return None
        nodata = n.nodata if n.nodata is not None else src.nodata
        footprint, factor = (_footprint(src, n.band), 32) if nodata is None else (None, 1)
        halo = n.stripe.min_run + n.stripe.filter_width
        direction = None
        if n.stripe.enabled and n.stripe.direction == "auto":
            c = Window(
                win.col_off + max(win.width // 2 - 1024, 0),
                win.row_off + max(win.height // 2 - 1024, 0),
                min(2048, win.width),
                min(2048, win.height),
            )
            direction = stripe_direction(src.read(n.band, window=c).astype("float32"))
            log.info("Scene %s: stripes run %s", path.name, direction)
        prof = {
            "driver": "GTiff",
            "width": win.width,
            "height": win.height,
            "count": 1,
            "dtype": "float32",
            "crs": src.crs,
            "transform": src.window_transform(win),
            "nodata": np.nan,
            "tiled": True,
            "compress": "deflate",
            "predictor": 3,
            "BIGTIFF": "IF_SAFER",
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out, "w", **prof) as dst:
            for tile in iter_tiles(win.width, win.height, halo=halo):
                o = tile.outer
                arr = src.read(
                    n.band,
                    window=Window(
                        win.col_off + o.col_off, win.row_off + o.row_off, o.width, o.height
                    ),
                ).astype("float32")
                if nodata is not None:
                    arr[arr == nodata] = np.nan
                else:
                    rows = (win.row_off + o.row_off + np.arange(o.height)) // factor
                    cols = (win.col_off + o.col_off + np.arange(o.width)) // factor
                    arr[~footprint[np.ix_(rows, cols)]] = np.nan
                if n.stripe.enabled:
                    arr = destripe(arr, n.stripe, direction)
                dst.write(arr[tile.crop], 1, window=tile.inner)
    return out


def _overlaps(win: Window, src) -> bool:
    return (
        win.col_off < src.width
        and win.row_off < src.height
        and win.col_off + win.width > 0
        and win.row_off + win.height > 0
    )


def combine(stack: np.ndarray, how: str) -> np.ndarray:
    """Merge co-registered scenes (first axis) where they overlap; NaN means no data."""
    if how == "median":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels
            return np.nanmedian(stack, axis=0)
    order = stack if how == "first" else stack[::-1]
    has = ~np.isnan(order)
    pick = np.argmax(has, axis=0)
    out = np.take_along_axis(order, pick[None], axis=0)[0]
    return np.where(has.any(axis=0), out, np.nan)


def write_mosaic(cfg: FeatureBuildConfig, scenes: list[Path], grid: Grid, out: Path) -> Path:
    """Three bands on the study grid: DN, DN where above the threshold (else NaN), and 1/0."""
    thr = cfg.ntl.positive_threshold
    with ExitStack() as stack:
        vrts = []
        for p in scenes:
            src = stack.enter_context(rasterio.open(p))
            vrt = stack.enter_context(
                WarpedVRT(
                    src,
                    crs=grid.crs,
                    transform=grid.transform,
                    width=grid.width,
                    height=grid.height,
                    resampling=Resampling.bilinear,
                    src_nodata=np.nan,
                    nodata=np.nan,
                )
            )
            vrts.append((vrt, transform_bounds(src.crs, grid.crs, *src.bounds)))
        dst = stack.enter_context(rasterio.open(out, "w", **grid.profile(count=3)))
        for tile in iter_tiles(grid.width, grid.height):
            left, bottom, right, top = rasterio.windows.bounds(tile.inner, grid.transform)
            layers = [
                vrt.read(1, window=tile.inner)
                for vrt, (vl, vb, vr, vt) in vrts
                if vl < right and vr > left and vb < top and vt > bottom
            ]
            dn = (
                combine(np.stack(layers), cfg.ntl.overlap)
                if layers
                else np.full((tile.inner.height, tile.inner.width), np.nan)
            )
            dst.write(dn.astype("float32"), 1, window=tile.inner)
            dst.write(np.where(dn > thr, dn, np.nan).astype("float32"), 2, window=tile.inner)
            dst.write(
                np.where(np.isnan(dn), np.nan, (dn > thr).astype("float32")), 3, window=tile.inner
            )
        dst.descriptions = ("NTL", "NTL_POS", "NTL_BIN")
    return out


def ntl_features(cfg: FeatureBuildConfig, villages: gpd.GeoDataFrame, mosaic: Path) -> pd.DataFrame:
    z = zonal(mosaic, villages, ["mean", "stdev", "median", "quantile(q=0.9)", "max", "sum"])
    feats = {
        "NTL_MEAN": z["1:mean"],
        "NTL_STD": z["1:stdev"],
        "NTL_MEDIAN": z["1:median"],
        "NTL_PCT90": z["1:quantile_90"],
        "NTL_MAX": z["1:max"],
        "NTL_SUM": z["1:sum"],
        "NTLPos_MEAN": z["2:mean"],
        "NTLBin_Ratio": z["3:mean"],
    }
    return pd.DataFrame(feats)[ntl_columns(cfg)]


def scene_paths(cfg: FeatureBuildConfig) -> list[Path]:
    paths = sorted(Path(p) for p in glob.glob(cfg.paths.sdgsat, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No SDGSAT-1 scenes match {cfg.paths.sdgsat}")
    return paths


def run(cfg: FeatureBuildConfig, villages: gpd.GeoDataFrame) -> pd.DataFrame:
    g = cfg.grid
    grid = study_grid(villages.total_bounds, g.res_image, g.buffer_m, g.snap_m, cfg.crs)
    work = cfg.interim / "ntl_scenes"
    corrected = [correct_scene(p, grid, cfg, work / f"{p.stem}.tif") for p in scene_paths(cfg)]
    corrected = [p for p in corrected if p is not None]
    if not corrected:
        raise ValueError("No SDGSAT-1 scene covers the study area")
    mosaic = write_mosaic(cfg, corrected, grid, cfg.interim / "rasters" / "ntl.tif")
    return ntl_features(cfg, villages, mosaic)

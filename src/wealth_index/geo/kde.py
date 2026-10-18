"""Quartic (biweight) kernel density on a grid, the kernel ArcGIS Kernel Density uses."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import shapely
from scipy.signal import fftconvolve

from wealth_index.geo.grid import Grid


def quartic_kernel(radius_m: float, res: float) -> np.ndarray:
    """Discrete kernel (1 - (d/r)^2)^2 inside the radius, normalised to sum to 1."""
    n = int(np.ceil(radius_m / res))
    offsets = np.arange(-n, n + 1) * res
    d2 = (offsets[None, :] ** 2 + offsets[:, None] ** 2) / radius_m**2
    k = np.where(d2 < 1, (1 - d2) ** 2, 0.0)
    return k / k.sum()


def rasterize_points(xy: np.ndarray, grid: Grid, weights: np.ndarray | None = None) -> np.ndarray:
    """Sum of point weights (default 1) per cell; points outside the grid are dropped."""
    col = np.floor((xy[:, 0] - grid.left) / grid.res).astype(np.int64)
    row = np.floor((grid.top - xy[:, 1]) / grid.res).astype(np.int64)
    inside = (col >= 0) & (col < grid.width) & (row >= 0) & (row < grid.height)
    w = None if weights is None else weights[inside]
    flat = np.bincount(row[inside] * grid.width + col[inside], w, grid.width * grid.height)
    return flat.reshape(grid.shape)


def point_density(xy: np.ndarray, grid: Grid, radius_m: float, weights=None) -> np.ndarray:
    """Points (or summed weights) per km2 in each cell."""
    if len(xy) == 0:
        return np.zeros(grid.shape)
    counts = rasterize_points(xy, grid, weights)
    density = fftconvolve(counts, quartic_kernel(radius_m, grid.res), mode="same")
    return np.clip(density, 0, None) / (grid.res**2 / 1e6)  # FFT leaves tiny negatives


def sample_lines(lines: gpd.GeoSeries, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Midpoints of segments at most `step` long, and each segment's length in km."""
    parts = shapely.segmentize(lines.explode(index_parts=False).to_numpy(), step)
    coords, idx = shapely.get_coordinates(parts, return_index=True)
    same = idx[1:] == idx[:-1]  # consecutive vertices of the same line
    a, b = coords[:-1][same], coords[1:][same]
    return (a + b) / 2, np.hypot(*(b - a).T) / 1000


def line_density(lines: gpd.GeoSeries, grid: Grid, radius_m: float) -> np.ndarray:
    """Line length (km) per km2, from length rasterised per cell and the same quartic kernel."""
    if len(lines) == 0:
        return np.zeros(grid.shape)
    mid, km = sample_lines(lines, step=grid.res / 10)
    return point_density(mid, grid, radius_m, weights=km)

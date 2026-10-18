"""Small rasters and polygons with known answers for the geo tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

import wealth_index.geo  # noqa: F401  (points rasterio at its bundled PROJ data)

UTM = "EPSG:32748"


def write_raster(
    path: Path,
    bands: np.ndarray,
    left: float,
    top: float,
    res: float,
    crs: str = UTM,
    nodata=np.nan,
) -> Path:
    bands = bands[None] if bands.ndim == 2 else bands
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = {
        "driver": "GTiff",
        "width": bands.shape[2],
        "height": bands.shape[1],
        "count": bands.shape[0],
        "dtype": str(bands.dtype),
        "crs": crs,
        "transform": from_origin(left, top, res, res),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(bands)
    return path

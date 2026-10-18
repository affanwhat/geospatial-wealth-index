"""Phase 2: village feature engineering from boundaries, OSM, Sentinel-2 and SDGSAT-1.

Needs the optional `geo` extra: `pip install -e ".[geo]"`.
"""

from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path


def _use_bundled_gdal_data() -> None:
    """Point GDAL/PROJ at the data shipped inside the rasterio wheel.

    A PROJ_LIB or GDAL_DATA left by another install (PostGIS, QGIS, ...) carries an older
    proj.db that rasterio's PROJ refuses, so every CRS lookup fails. Only this process is
    affected; pyproj already uses its own copy.
    """
    spec = find_spec("rasterio")
    if spec is None or spec.origin is None:
        return
    root = Path(spec.origin).parent
    for var, sub in (
        ("PROJ_DATA", "proj_data"),
        ("PROJ_LIB", "proj_data"),
        ("GDAL_DATA", "gdal_data"),
    ):
        if (root / sub).is_dir():
            os.environ[var] = str(root / sub)


_use_bundled_gdal_data()

"""Analysis grids, windowed iteration, and mosaicking of aligned tiles."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window

GDAL_TYPES = {
    "uint8": "Byte",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "float32": "Float32",
    "float64": "Float64",
}


@dataclass(frozen=True)
class Grid:
    crs: str
    left: float
    top: float
    res: float
    width: int
    height: int

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0, self.left, 0, -self.res, self.top)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.left,
            self.top - self.height * self.res,
            self.left + self.width * self.res,
            self.top,
        )

    def window_transform(self, window: Window) -> Affine:
        return rasterio.windows.transform(window, self.transform)

    def profile(
        self, count: int = 1, dtype: str = "float32", nodata: float | None = np.nan
    ) -> dict:
        return {
            "driver": "GTiff",
            "width": self.width,
            "height": self.height,
            "count": count,
            "dtype": dtype,
            "crs": self.crs,
            "transform": self.transform,
            "nodata": nodata,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "deflate",
            "predictor": 3 if dtype.startswith("float") else 2,
            "BIGTIFF": "IF_SAFER",
        }


def study_grid(bounds, res: float, buffer_m: float, snap_m: float, crs: str) -> Grid:
    """Grid over `bounds` (in `crs` metres) plus a buffer, with its origin snapped to `snap_m`."""
    minx, miny, maxx, maxy = bounds
    left = math.floor((minx - buffer_m) / snap_m) * snap_m
    bottom = math.floor((miny - buffer_m) / snap_m) * snap_m
    right = math.ceil((maxx + buffer_m) / snap_m) * snap_m
    top = math.ceil((maxy + buffer_m) / snap_m) * snap_m
    return Grid(crs, left, top, res, round((right - left) / res), round((top - bottom) / res))


@dataclass(frozen=True)
class Tile:
    inner: Window  # the part this tile writes
    outer: Window  # inner plus a halo, for neighbourhood operations
    crop: tuple[slice, slice]  # inner, indexed inside an array read with `outer`


def iter_tiles(width: int, height: int, size: int = 2048, halo: int = 0) -> Iterator[Tile]:
    for row in range(0, height, size):
        for col in range(0, width, size):
            h, w = min(size, height - row), min(size, width - col)
            r0, c0 = max(row - halo, 0), max(col - halo, 0)
            r1, c1 = min(row + h + halo, height), min(col + w + halo, width)
            yield Tile(
                Window(col, row, w, h),
                Window(c0, r0, c1 - c0, r1 - r0),
                (slice(row - r0, row - r0 + h), slice(col - c0, col - c0 + w)),
            )


def mosaic_vrt(paths: list[Path], out: Path) -> Path:
    """Stitch tiles that share a CRS and pixel grid (e.g. a split GEE export) into one VRT."""
    if len(paths) == 1:
        return paths[0]
    metas = []
    for p in paths:
        with rasterio.open(p) as src:
            metas.append((Path(p).resolve(), src.profile, src.transform))
    _, prof, t0 = metas[0]
    res = t0.a
    for p, other, t in metas:
        if other["crs"] != prof["crs"] or other["count"] != prof["count"] or t.a != res:
            raise ValueError(
                f"{p} does not share CRS, band count and resolution with {metas[0][0]}"
            )
    left = min(t.c for _, _, t in metas)
    top = max(t.f for _, _, t in metas)
    right = max(t.c + m["width"] * res for _, m, t in metas)
    bottom = min(t.f - m["height"] * res for _, m, t in metas)
    width, height = round((right - left) / res), round((top - bottom) / res)

    nodata = prof["nodata"]
    bands = []
    for b in range(1, prof["count"] + 1):
        sources = "".join(
            f'<SimpleSource><SourceFilename relativeToVRT="0">{escape(str(p))}</SourceFilename>'
            f"<SourceBand>{b}</SourceBand>"
            f'<SrcRect xOff="0" yOff="0" xSize="{m["width"]}" ySize="{m["height"]}"/>'
            f'<DstRect xOff="{round((t.c - left) / res)}" yOff="{round((top - t.f) / res)}" '
            f'xSize="{m["width"]}" ySize="{m["height"]}"/></SimpleSource>'
            for p, m, t in metas
        )
        nd = f"<NoDataValue>{nodata}</NoDataValue>" if nodata is not None else ""
        bands.append(
            f'<VRTRasterBand dataType="{GDAL_TYPES[prof["dtype"]]}" band="{b}">{nd}{sources}'
            "</VRTRasterBand>"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">'
        f"<SRS>{escape(prof['crs'].to_wkt())}</SRS>"
        f"<GeoTransform>{left}, {res}, 0, {top}, 0, {-res}</GeoTransform>"
        + "".join(bands)
        + "</VRTDataset>",
        encoding="utf-8",
    )
    return out

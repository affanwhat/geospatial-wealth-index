import numpy as np
import pytest

pytest.importorskip("rasterio")

import geopandas as gpd  # noqa: E402
from geo_helpers import UTM  # noqa: E402
from shapely.geometry import LineString  # noqa: E402

from wealth_index.geo.grid import Grid  # noqa: E402
from wealth_index.geo.kde import line_density, point_density, quartic_kernel  # noqa: E402

GRID = Grid(UTM, left=0, top=10_000, res=100, width=100, height=100)  # 10 x 10 km
CELL_KM2 = 0.01


def test_quartic_kernel_is_normalised_and_bounded():
    k = quartic_kernel(1000, 100)
    assert k.sum() == pytest.approx(1.0)
    assert k[10, 10] == k.max()  # peak at the centre
    assert k[10, 0] == 0  # exactly one radius away


def test_point_density_integrates_to_count_and_peaks_at_points():
    xy = np.array([[5050.0, 5050.0], [2050.0, 7050.0], [7050.0, 2050.0]])
    d = point_density(xy, GRID, radius_m=1000)
    assert (d * CELL_KM2).sum() == pytest.approx(3, rel=0.01)
    row, col = np.unravel_index(d.argmax(), d.shape)
    assert (col, row) in {(50, 49), (20, 29), (70, 79)}
    assert d[49, 50 + 11] == pytest.approx(0, abs=1e-9)  # beyond the radius of every point


def test_points_outside_the_grid_are_dropped():
    d = point_density(np.array([[-500.0, 5000.0]]), GRID, radius_m=1000)
    assert (d * CELL_KM2).sum() == pytest.approx(0, abs=1e-9)


def test_line_density_integrates_to_line_length():
    line = gpd.GeoSeries([LineString([(4000, 5000), (5000, 5000)])], crs=UTM)  # 1 km
    d = line_density(line, GRID, radius_m=1000)
    assert (d * CELL_KM2).sum() == pytest.approx(1.0, rel=0.01)

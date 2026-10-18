import numpy as np
import pytest

pytest.importorskip("rasterio")

import geopandas as gpd  # noqa: E402
from geo_helpers import UTM  # noqa: E402
from shapely.geometry import LineString, Point, box  # noqa: E402

from wealth_index.geo.distance import count_within, distance_raster, nearest_distance  # noqa: E402
from wealth_index.geo.grid import Grid  # noqa: E402


def test_distance_raster_matches_analytic_distance_within_one_cell():
    grid = Grid(UTM, left=0, top=3000, res=30, width=100, height=100)
    road = gpd.GeoSeries([LineString([(0, 1500), (3000, 1500)])], crs=UTM)
    d = distance_raster(road, grid)
    centre_y = grid.top - (np.arange(grid.height) + 0.5) * grid.res
    expected = np.abs(centre_y - 1500)[:, None] * np.ones(grid.width)
    assert np.abs(d - expected).max() <= grid.res


def test_distance_raster_without_features_is_nan():
    grid = Grid(UTM, left=0, top=300, res=30, width=10, height=10)
    assert np.isnan(distance_raster(gpd.GeoSeries([], crs=UTM), grid)).all()


def test_nearest_distance_is_zero_inside_and_exact_outside():
    villages = gpd.GeoDataFrame(
        geometry=[box(0, 0, 100, 100), box(500, 0, 600, 100)], index=["a", "b"], crs=UTM
    )
    pois = gpd.GeoSeries([Point(50, 50)], crs=UTM)
    d = nearest_distance(villages, pois)
    assert d["a"] == 0
    assert d["b"] == pytest.approx(450)


def test_count_within_includes_points_inside_and_within_the_radius():
    villages = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], index=["a"], crs=UTM)
    pois = gpd.GeoSeries([Point(50, 50), Point(150, 50), Point(1200, 50)], crs=UTM)
    assert count_within(villages, pois, 1000)["a"] == 2

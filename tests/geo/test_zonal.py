import numpy as np
import pytest

pytest.importorskip("exactextract")

import geopandas as gpd  # noqa: E402
from geo_helpers import UTM, write_raster  # noqa: E402
from shapely.geometry import box  # noqa: E402

from wealth_index.geo.zonal import zonal  # noqa: E402


def test_zonal_stats_skip_nan_and_weight_partial_pixels(tmp_path):
    a = np.arange(16, dtype="float32").reshape(4, 4)
    a[0, 0] = np.nan
    path = write_raster(tmp_path / "r.tif", np.stack([a, a > 5]).astype("float32"), 0, 4, 1)
    villages = gpd.GeoDataFrame(
        geometry=[box(0, 2, 2, 4), box(2, 0, 4, 2), box(0.5, 0, 1, 1)],
        index=["A", "B", "C"],
        crs=UTM,
    )
    z = zonal(path, villages, ["mean", "stdev", "count", "quantile(q=0.9)"])

    assert list(z.index) == ["A", "B", "C"]
    assert z.loc["A", "1:mean"] == pytest.approx(10 / 3)  # 1, 4, 5; the NaN pixel is skipped
    assert z.loc["A", "1:stdev"] == pytest.approx(np.std([1, 4, 5]))
    assert z.loc["B", "1:mean"] == pytest.approx(12.5)  # 10, 11, 14, 15
    assert z.loc["B", "2:mean"] == pytest.approx(1.0)  # share of pixels above 5
    assert z.loc["C", "1:count"] == pytest.approx(0.5)  # half of one pixel
    assert z.loc["C", "1:mean"] == pytest.approx(12.0)
    assert "1:quantile_90" in z

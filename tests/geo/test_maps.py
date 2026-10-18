import numpy as np
import pandas as pd
import pytest

pytest.importorskip("geopandas")

import geopandas as gpd  # noqa: E402
from geo_helpers import UTM  # noqa: E402
from shapely.geometry import box  # noqa: E402

from wealth_index.geo.maps import join_predictions, make_map  # noqa: E402


@pytest.fixture
def villages():
    return gpd.GeoDataFrame(
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100), box(200, 0, 300, 100)],
        index=pd.Index(["31.71.01.1001", "31.71.01.1002", "31.71.01.1003"], name="KDEPUM"),
        crs=UTM,
    )


def test_join_matches_ids_and_leaves_missing_villages_nan(villages):
    preds = pd.DataFrame(
        {
            "id": ["31.71.01.1002 ", "31.71.01.1001", "99.99.99.9999"],
            "wealth_index_pred": [20.0, 10.0, 5.0],
        }
    )
    out = join_predictions(villages, preds, id_col="id", value_col="wealth_index_pred")
    assert out.index.equals(villages.index)
    np.testing.assert_array_equal(out["wealth_index_pred"].to_numpy(), [10.0, 20.0, np.nan])


def test_join_rejects_duplicates_and_no_overlap(villages):
    dupes = pd.DataFrame({"id": ["31.71.01.1001"] * 2, "wealth_index_pred": [1.0, 2.0]})
    with pytest.raises(ValueError, match="duplicate"):
        join_predictions(villages, dupes, id_col="id", value_col="wealth_index_pred")
    none = pd.DataFrame({"id": ["x"], "wealth_index_pred": [1.0]})
    with pytest.raises(ValueError, match="no prediction id"):
        join_predictions(villages, none, id_col="id", value_col="wealth_index_pred")


def test_make_map_writes_png_and_geopackage(villages, tmp_path):
    preds = tmp_path / "preds.csv"
    preds_df = pd.DataFrame({"id": villages.index[:2], "wealth_index_pred": [1.0, 2.0]})
    preds_df.to_csv(preds, index=False)
    png, gpkg = tmp_path / "out" / "map.png", tmp_path / "out" / "map.gpkg"
    make_map(villages, preds, png, vector=gpkg)
    assert png.stat().st_size > 0
    back = gpd.read_file(gpkg)
    assert list(back["KDEPUM"]) == list(villages.index)
    assert back["wealth_index_pred"].isna().sum() == 1

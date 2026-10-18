import json

import pandas as pd
import pytest

from wealth_index.config import DataConfig
from wealth_index.data import DataValidationError, load_dataset
from wealth_index.labels import join_labels, run_join


def features():
    return pd.DataFrame(
        {
            "id": ["31.71.01.1001", "31.71.01.1002", "31.71.01.1003"],
            "lon": [106.80, 106.81, 106.82],
            "lat": [-6.20, -6.21, -6.22],
            "NDVI_MEAN": [0.1, 0.2, 0.3],
            "NTL_MEAN": [5.0, 6.0, 7.0],
        }
    )


def labels(**overrides):
    df = pd.DataFrame(
        {"KDEPUM": ["31.71.01.1002", "31.71.01.1001 ", "99.99.99.9999"], "wi": [40.0, 20.0, 50.0]}
    )
    return df.assign(**overrides)


def test_join_orders_columns_for_training_and_reports_mismatches():
    out, report = join_labels(features(), labels(), label_col="wi", label_id_col="KDEPUM")
    assert list(out.columns) == ["id", "lon", "lat", "wealth_index", "NDVI_MEAN", "NTL_MEAN"]
    # Label ids are matched after stripping whitespace; feature order is kept.
    assert out["id"].tolist() == ["31.71.01.1001", "31.71.01.1002"]
    assert out["wealth_index"].tolist() == [20.0, 40.0]
    assert (report.matched, report.unlabelled, report.unknown) == (
        2,
        ["31.71.01.1003"],
        ["99.99.99.9999"],
    )


@pytest.mark.parametrize(
    "bad, message",
    [
        (labels(KDEPUM=["31.71.01.1001"] * 3), "duplicate label ids"),
        (labels(wi=[1.0, None, 2.0]), "missing a value"),
        (labels(wi=["low", "mid", "high"]), "not numeric"),
        (labels(KDEPUM=["3171011001", "3171011002", "x"]), "no label id matches"),
    ],
)
def test_join_rejects_bad_labels(bad, message):
    with pytest.raises(DataValidationError, match=message):
        join_labels(features(), bad, label_col="wi", label_id_col="KDEPUM")


def test_run_join_writes_a_dataset_training_can_load(tmp_path):
    features().to_csv(tmp_path / "features.csv", index=False)
    labels().to_csv(tmp_path / "labels.csv", index=False)
    out = tmp_path / "processed" / "dataset.csv"
    run_join(
        tmp_path / "features.csv",
        tmp_path / "labels.csv",
        out,
        label_col="wi",
        label_id_col="KDEPUM",
    )

    ds = load_dataset(DataConfig(path=str(out)))
    assert list(ds.X.columns) == ["NDVI_MEAN", "NTL_MEAN"] and ds.y.tolist() == [20.0, 40.0]
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["rows"] == 2 and manifest["labels"]["column"] == "wi"
    assert manifest["join"]["unlabelled"] == ["31.71.01.1003"]

import json

import pandas as pd
import pytest

from wealth_index.config import TpotConfig
from wealth_index.data import DataValidationError
from wealth_index.predict import predict, resolve_model_path
from wealth_index.report import collect_runs
from wealth_index.train import train


def test_train_then_predict_end_to_end(small_config, sample_csv, tmp_path):
    run_dir = train(small_config)

    for name in ["model.joblib", "metrics.json", "metadata.json", "predictions_test.csv"]:
        assert (run_dir / name).exists()
    assert (run_dir / "plots" / "actual_vs_predicted.png").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics["test"]) == {"linear_regression", "rf_default", "tuned"}
    assert set(metrics["cv_comparison"]) == {"random", "spatial"}

    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert "lon" not in metadata["features_in"]  # coordinates are never model features
    assert 0 < len(metadata["selected_features"]) <= small_config.features.top_k

    tables = collect_runs(small_config.output_dir)
    assert len(tables["runs"]) == 1
    assert len(tables["model_metrics"]) == 6  # 3 models x train/test
    assert {"lon", "lat", "residual"} <= set(tables["predictions"].columns)
    assert tables["feature_importance"]["selected"].sum() == len(metadata["selected_features"])

    model_path = resolve_model_path(small_config.output_dir)
    out = predict(model_path, sample_csv, tmp_path / "preds.csv")
    assert len(out) == 300
    assert list(out.columns) == ["id", "wealth_index_pred"]


def test_tpot_search_produces_a_predicting_pipeline(small_config, sample_csv, tmp_path):
    pytest.importorskip("tpot")
    small_config.search.method = "tpot"
    small_config.search.tpot = TpotConfig(generations=2, population_size=4, max_time_mins=5)
    run_dir = train(small_config)

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["search"]["method"] == "tpot"
    assert "RandomForestRegressor" in metrics["search"]["best_pipeline"]
    out = predict(run_dir / "model.joblib", sample_csv, tmp_path / "preds.csv")
    assert len(out) == 300


def test_predict_rejects_missing_feature_columns(small_config, sample_csv, tmp_path):
    train(small_config)
    broken = tmp_path / "broken.csv"
    pd.read_csv(sample_csv).drop(columns=["NTL_MEAN"]).to_csv(broken, index=False)
    with pytest.raises(DataValidationError, match="NTL_MEAN"):
        predict(resolve_model_path(small_config.output_dir), broken, tmp_path / "p.csv")

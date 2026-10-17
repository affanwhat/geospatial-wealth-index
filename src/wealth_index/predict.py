"""Batch scoring with a trained pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from wealth_index.data import DataValidationError
from wealth_index.train import LATEST_POINTER


def resolve_model_path(artifacts_dir: str | Path) -> Path:
    """Return the model of the most recent training run."""
    pointer = Path(artifacts_dir) / LATEST_POINTER
    if not pointer.exists():
        raise FileNotFoundError(f"No training run found in {artifacts_dir}; run `wi train` first")
    return Path(artifacts_dir) / pointer.read_text().strip() / "model.joblib"


def predict(
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    *,
    sep: str = ",",
    id_col: str = "id",
) -> pd.DataFrame:
    model = joblib.load(model_path)
    df = pd.read_csv(input_path, sep=sep)

    required = [id_col, *model.feature_names_in_]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Input is missing columns the model was trained on: {missing}")

    out = pd.DataFrame(
        {id_col: df[id_col], "wealth_index_pred": model.predict(df[model.feature_names_in_])}
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out

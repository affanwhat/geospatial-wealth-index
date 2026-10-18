"""Join wealth index labels to the village feature table, producing the training dataset."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from wealth_index.config import DataConfig
from wealth_index.data import DataValidationError, file_sha256, validate

log = logging.getLogger(__name__)


@dataclass
class JoinReport:
    features: int
    labels: int
    matched: int
    unlabelled: list[str]  # villages with features but no label (dropped)
    unknown: list[str]  # labels whose id is not in the feature table (ignored)


def _ids(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def join_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    label_col: str,
    label_id_col: str,
    data: DataConfig | None = None,
) -> tuple[pd.DataFrame, JoinReport]:
    """Dataset rows for every village that has both features and a label.

    Columns come out as the training config expects: id, coordinates, target, then features.
    Villages without a label are dropped and labels without a village are ignored; both are
    listed in the report. Raises DataValidationError on duplicate or missing ids, a missing or
    non-numeric label, or no overlap at all.
    """
    data = data or DataConfig()
    for name, df, col in [("labels", labels, label_id_col), ("features", features, data.id_col)]:
        if col not in df:
            raise DataValidationError(f"{name} has no id column '{col}'")
    if label_col not in labels:
        raise DataValidationError(f"labels have no column '{label_col}'")
    if not is_numeric_dtype(labels[label_col]):
        raise DataValidationError(f"label column '{label_col}' is not numeric")

    lab = pd.Series(labels[label_col].to_numpy(), index=_ids(labels[label_id_col]))
    if lab.index.duplicated().any():
        dupes = lab.index[lab.index.duplicated()].unique().tolist()
        raise DataValidationError(f"duplicate label ids: {dupes[:10]}")
    if lab.isna().any():
        raise DataValidationError(f"{int(lab.isna().sum())} labels are missing a value")

    feats = features.assign(**{data.id_col: _ids(features[data.id_col])})
    has_label = feats[data.id_col].isin(lab.index)
    report = JoinReport(
        features=len(feats),
        labels=len(lab),
        matched=int(has_label.sum()),
        unlabelled=feats.loc[~has_label, data.id_col].tolist(),
        unknown=sorted(set(lab.index) - set(feats[data.id_col])),
    )
    if not report.matched:
        raise DataValidationError(
            f"no label id matches a feature id (e.g. label {lab.index[0]!r} vs feature "
            f"{feats[data.id_col].iloc[0]!r})"
        )

    out = feats[has_label].copy()
    out[data.target_col] = out[data.id_col].map(lab).to_numpy()
    front = [data.id_col, *data.coord_cols, data.target_col]
    out = out[front + [c for c in out.columns if c not in front]].reset_index(drop=True)
    validate(out, data)
    return out, report


def run_join(
    features_path: Path,
    labels_path: Path,
    output: Path,
    *,
    label_col: str,
    label_id_col: str,
    data: DataConfig | None = None,
) -> JoinReport:
    """Join two CSV files and write the dataset plus a manifest next to it."""
    data = data or DataConfig()
    features = pd.read_csv(features_path, dtype={data.id_col: str})
    labels = pd.read_csv(labels_path, dtype={label_id_col: str})
    dataset, report = join_labels(
        features, labels, label_col=label_col, label_id_col=label_id_col, data=data
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False, sep=data.sep)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "features": {"path": str(features_path), "sha256": file_sha256(features_path)},
        "labels": {
            "path": str(labels_path),
            "sha256": file_sha256(labels_path),
            "column": label_col,
            "id_column": label_id_col,
        },
        "rows": len(dataset),
        "target": dataset[data.target_col].describe().round(4).to_dict(),
        "join": asdict(report),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    if report.unlabelled:
        log.warning("%d villages have no label and were dropped", len(report.unlabelled))
    if report.unknown:
        log.warning("%d labels match no village and were ignored", len(report.unknown))
    log.info("Wrote %s: %d villages", output, len(dataset))
    return report

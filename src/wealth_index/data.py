"""Dataset loading and schema validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from wealth_index.config import DataConfig


class DataValidationError(ValueError):
    """Raised when an input table does not match the expected schema."""


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    ids: pd.Series
    coords: pd.DataFrame


def feature_columns(df: pd.DataFrame, cfg: DataConfig) -> list[str]:
    reserved = {cfg.id_col, cfg.target_col, *cfg.coord_cols}
    return [c for c in df.columns if c not in reserved]


def validate(df: pd.DataFrame, cfg: DataConfig, *, require_target: bool = True) -> None:
    required = [cfg.id_col, *cfg.coord_cols] + ([cfg.target_col] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if df[cfg.id_col].duplicated().any():
        raise DataValidationError(f"Duplicate values in id column '{cfg.id_col}'")
    if require_target and df[cfg.target_col].isna().any():
        raise DataValidationError(f"Target column '{cfg.target_col}' contains missing values")
    non_numeric = [c for c in feature_columns(df, cfg) if not is_numeric_dtype(df[c])]
    if non_numeric:
        raise DataValidationError(f"Non-numeric feature columns: {non_numeric}")
    if not feature_columns(df, cfg):
        raise DataValidationError("No feature columns found")


def load_dataset(cfg: DataConfig) -> Dataset:
    df = pd.read_csv(cfg.path, sep=cfg.sep)
    validate(df, cfg)
    return Dataset(
        X=df[feature_columns(df, cfg)],
        y=df[cfg.target_col],
        ids=df[cfg.id_col],
        coords=df[cfg.coord_cols],
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

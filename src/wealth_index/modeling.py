"""Model pipeline definitions."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wealth_index.config import FeatureConfig
from wealth_index.features import CorrelationFilter, PermutationImportanceSelector


def build_pipeline(features: FeatureConfig, seed: int) -> Pipeline:
    """Impute -> drop collinear features -> keep top-k by PFI -> random forest."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="mean")),
            ("corr", CorrelationFilter(threshold=features.corr_threshold)),
            (
                "select",
                PermutationImportanceSelector(
                    top_k=features.top_k,
                    n_repeats=features.pfi_repeats,
                    n_estimators=features.pfi_estimators,
                    random_state=seed,
                ),
            ),
            ("model", RandomForestRegressor(random_state=seed, n_jobs=1)),
        ]
    ).set_output(transform="pandas")


def build_linear_baseline() -> Pipeline:
    """Multiple linear regression on all features, the thesis' statistical benchmark."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]
    ).set_output(transform="pandas")

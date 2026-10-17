"""Feature selection steps as scikit-learn transformers.

Wrapping selection in transformers lets it live inside the Pipeline, so it is refit on
every training fold and never sees validation or test rows.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectorMixin
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted, validate_data


class CorrelationFilter(SelectorMixin, BaseEstimator):
    """Drop each feature whose |Pearson r| with an earlier feature exceeds `threshold`."""

    def __init__(self, threshold: float | None = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = validate_data(self, X)
        n_features = X.shape[1]
        self.support_ = np.ones(n_features, dtype=bool)
        if self.threshold is not None and n_features > 1:
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.abs(np.corrcoef(X, rowvar=False))
            upper = np.triu(np.nan_to_num(corr, nan=0.0), k=1)
            self.support_ = ~(upper > self.threshold).any(axis=0)
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_


class PermutationImportanceSelector(SelectorMixin, BaseEstimator):
    """Keep the `top_k` features ranked by random-forest permutation importance.

    Importance is measured on a validation split carved out of the data passed to `fit`,
    so the ranking is not inflated by the forest memorising its own training rows.
    """

    def __init__(
        self,
        top_k: int = 25,
        n_repeats: int = 5,
        val_size: float = 0.25,
        n_estimators: int = 100,
        random_state: int | None = None,
    ):
        self.top_k = top_k
        self.n_repeats = n_repeats
        self.val_size = val_size
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=self.val_size, random_state=self.random_state
        )
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=1
        ).fit(X_fit, y_fit)
        result = permutation_importance(
            forest, X_val, y_val, n_repeats=self.n_repeats, random_state=self.random_state
        )
        self.importances_ = result.importances_mean
        k = min(self.top_k, X.shape[1])
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[np.argsort(self.importances_)[::-1][:k]] = True
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_

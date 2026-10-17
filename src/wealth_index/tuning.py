"""Hyperparameter search backends: scikit-learn random search or TPOT genetic programming."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from joblib import effective_n_jobs
from sklearn.model_selection import RandomizedSearchCV, check_cv
from sklearn.pipeline import Pipeline

from wealth_index.config import Config
from wealth_index.modeling import build_pipeline
from wealth_index.spatial import make_cv

log = logging.getLogger(__name__)

MODEL_PREFIX = "model__"


def tune(
    cfg: Config, X: pd.DataFrame, y: pd.Series, groups: np.ndarray | None
) -> tuple[Pipeline, dict[str, Any]]:
    """Return the fitted best pipeline and a JSON-serialisable summary of the search."""
    if cfg.search.method == "random":
        return _tune_random(cfg, X, y, groups)
    if cfg.search.method == "tpot":
        return _tune_tpot(cfg, X, y, groups)
    raise ValueError(f"Unknown search method: {cfg.search.method!r}")


def _tune_random(cfg, X, y, groups):
    cv = make_cv(cfg.split.scheme, cfg.split.cv_folds, cfg.seed)
    search = RandomizedSearchCV(
        build_pipeline(cfg.features, cfg.seed),
        cfg.search.param_distributions,
        n_iter=cfg.search.n_iter,
        scoring=cfg.search.scoring,
        cv=cv,
        n_jobs=cfg.n_jobs,
        random_state=cfg.seed,
    )
    log.info("Random search: %d candidates x %d folds", cfg.search.n_iter, cv.n_splits)
    search.fit(X, y, groups=groups)
    summary = {"best_cv_score": float(search.best_score_), "best_params": search.best_params_}
    return search.best_estimator_, summary


def _tune_tpot(cfg, X, y, groups):
    """Genetic search over the model step, as in the thesis.

    Imputation and feature selection are fit once on the full training set, then TPOT
    searches models on the selected features. TPOT's internal CV scores are therefore
    slightly optimistic; the held-out test set and the random-vs-spatial CV re-scoring
    in `train` refit the whole pipeline per fold and remain leak-free.
    """
    try:
        from tpot import TPOTRegressor
    except ImportError as e:
        raise ImportError('TPOT is optional: install it with `pip install -e ".[tpot]"`') from e

    tp = cfg.search.tpot
    preprocess = build_pipeline(cfg.features, cfg.seed)[:-1]
    X_sel = preprocess.fit_transform(X, y)

    # TPOT has no `groups` argument, so hand it the spatial folds precomputed.
    splits = list(make_cv(cfg.split.scheme, cfg.split.cv_folds, cfg.seed).split(X, y, groups))
    tpot = TPOTRegressor(
        search_space=_tpot_search_space(cfg) if tp.search_space == "rf" else tp.search_space,
        scorers=[cfg.search.scoring],
        scorers_weights=[1],
        cv=check_cv(splits),
        generations=tp.generations,
        population_size=tp.population_size,
        early_stop=tp.early_stop,
        max_time_mins=tp.max_time_mins,
        # TPOT hands n_jobs to dask as a worker count, so resolve joblib's -1 = all cores.
        n_jobs=effective_n_jobs(cfg.n_jobs),
        verbose=0,
        random_state=cfg.seed,
    )
    log.info(
        "TPOT search (%s): %d generations x population %d on %d selected features",
        tp.search_space,
        tp.generations,
        tp.population_size,
        X_sel.shape[1],
    )
    tpot.fit(X_sel, y)

    best = Pipeline([*preprocess.steps, ("model", tpot.fitted_pipeline_)])
    best.set_output(transform="pandas")
    return best, {"best_pipeline": str(tpot.fitted_pipeline_)}


def _tpot_search_space(cfg):
    """RandomForest-only search space built from the `model__*` entries of the param grid."""
    from ConfigSpace import Categorical, ConfigurationSpace
    from sklearn.ensemble import RandomForestRegressor
    from tpot.search_spaces.nodes import EstimatorNode

    space = {
        k.removeprefix(MODEL_PREFIX): Categorical(k.removeprefix(MODEL_PREFIX), v)
        for k, v in cfg.search.param_distributions.items()
        if k.startswith(MODEL_PREFIX)
    }
    ignored = [k for k in cfg.search.param_distributions if not k.startswith(MODEL_PREFIX)]
    if ignored:
        log.warning("TPOT only tunes the model step; ignoring %s (using config defaults)", ignored)
    return EstimatorNode(
        RandomForestRegressor,
        space=ConfigurationSpace({**space, "random_state": cfg.seed, "n_jobs": 1}),
    )

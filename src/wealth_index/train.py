"""Training entry point: split, tune, evaluate, and persist a run."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

from wealth_index.config import Config
from wealth_index.data import file_sha256, load_dataset
from wealth_index.evaluate import (
    plot_actual_vs_predicted,
    plot_feature_importance,
    regression_metrics,
)
from wealth_index.modeling import build_linear_baseline, build_pipeline
from wealth_index.spatial import holdout_split, make_cv, spatial_blocks
from wealth_index.tuning import tune

log = logging.getLogger(__name__)

LATEST_POINTER = "LATEST"


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def train(cfg: Config) -> Path:
    """Run the full training workflow and return the run directory."""
    ds = load_dataset(cfg.data)
    groups = spatial_blocks(ds.coords, cfg.split.block_size_km)
    train_idx, test_idx = holdout_split(
        len(ds.y), cfg.split.scheme, groups, cfg.split.test_size, cfg.seed
    )
    X_tr, X_te = ds.X.iloc[train_idx], ds.X.iloc[test_idx]
    y_tr, y_te = ds.y.iloc[train_idx], ds.y.iloc[test_idx]
    g_tr = groups[train_idx]
    log.info(
        "Loaded %d rows x %d features; %d spatial blocks; %s split -> %d train / %d test",
        len(ds.y),
        ds.X.shape[1],
        len(set(groups)),
        cfg.split.scheme,
        len(y_tr),
        len(y_te),
    )

    def cv_groups(scheme: str):
        return g_tr if scheme == "spatial" else None

    best, search_summary = tune(cfg, X_tr, y_tr, cv_groups(cfg.split.scheme))

    models = {
        "linear_regression": build_linear_baseline().fit(X_tr, y_tr),
        "rf_default": build_pipeline(cfg.features, cfg.seed).fit(X_tr, y_tr),
        "tuned": best,
    }
    test_metrics = {name: regression_metrics(y_te, m.predict(X_te)) for name, m in models.items()}
    train_metrics = {name: regression_metrics(y_tr, m.predict(X_tr)) for name, m in models.items()}

    # How much of the score is spatial leakage? Re-score the tuned config under both schemes.
    cv_comparison = {}
    for scheme in ("random", "spatial"):
        scores = -cross_val_score(
            clone(best),
            X_tr,
            y_tr,
            groups=cv_groups(scheme),
            cv=make_cv(scheme, cfg.split.cv_folds, cfg.seed),
            scoring="neg_root_mean_squared_error",
            n_jobs=cfg.n_jobs,
        )
        cv_comparison[scheme] = {"rmse_mean": float(scores.mean()), "rmse_std": float(scores.std())}

    run_dir = Path(cfg.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    joblib.dump(best, run_dir / "model.joblib")
    pred_te = best.predict(X_te)
    pd.DataFrame(
        {
            cfg.data.id_col: ds.ids.iloc[test_idx].to_numpy(),
            "lon": ds.coords.iloc[test_idx, 0].to_numpy(),
            "lat": ds.coords.iloc[test_idx, 1].to_numpy(),
            "actual": y_te.to_numpy(),
            "predicted": pred_te,
        }
    ).to_csv(run_dir / "predictions_test.csv", index=False)

    metrics = {
        "test": test_metrics,
        "train": train_metrics,
        "cv_comparison": cv_comparison,
        "search": {
            "method": cfg.search.method,
            "scoring": cfg.search.scoring,
            **search_summary,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": cfg.to_dict(),
        "data_sha256": file_sha256(cfg.data.path),
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "n_spatial_blocks": int(len(set(groups))),
        "features_in": list(ds.X.columns),
        "selected_features": list(best[:-1].get_feature_names_out()),
        "git_sha": _git_sha(),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    plot_actual_vs_predicted(
        y_te,
        pred_te,
        run_dir / "plots" / "actual_vs_predicted.png",
        f"Tuned model ({cfg.search.method}) on held-out {cfg.split.scheme} test set",
    )
    pfi = permutation_importance(
        best, X_te, y_te, n_repeats=10, random_state=cfg.seed, n_jobs=cfg.n_jobs
    )
    plot_feature_importance(
        X_te.columns, pfi.importances_mean, run_dir / "plots" / "feature_importance.png"
    )
    pd.DataFrame(
        {
            "feature": X_te.columns,
            "importance_mean": pfi.importances_mean,
            "importance_std": pfi.importances_std,
            "selected": X_te.columns.isin(metadata["selected_features"]),
        }
    ).sort_values("importance_mean", ascending=False).to_csv(
        run_dir / "feature_importance.csv", index=False
    )

    (Path(cfg.output_dir) / LATEST_POINTER).write_text(run_dir.name)
    log.info(
        "Test RMSE %.3f / R2 %.3f | CV RMSE random %.3f vs spatial %.3f",
        test_metrics["tuned"]["rmse"],
        test_metrics["tuned"]["r2"],
        cv_comparison["random"]["rmse_mean"],
        cv_comparison["spatial"]["rmse_mean"],
    )
    log.info("Saved run to %s", run_dir)
    return run_dir

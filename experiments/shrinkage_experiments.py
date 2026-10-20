"""Part B of NEXT_EXPERIMENTS_PLAN.md: can we reduce the pull towards the middle?

    python experiments/shrinkage_experiments.py              # latest run; finished variants reused
    python experiments/shrinkage_experiments.py --only B0    # one variant (e.g. to time it)
    python experiments/shrinkage_experiments.py --force      # recompute everything
    python experiments/shrinkage_experiments.py --help       # run, labels, output, cache options

Harness: the run's split (seed 42, 5 km blocks, 30% test), the package's preprocessing
(impute -> texture PCA -> correlation filter -> top-25 by permutation importance), refit on every
fold, and only the model step varied. Preprocessed fold matrices are cached (they do not depend
on the model). Each variant gets the same spatial GroupKFold(5) search. B0 uses the run's own
tuned forest, so the run must come from `search.method: random`.

Outputs go to experiments/output/<run id>/ (git-ignored; Podes-derived, never commit or publish):
shrinkage_results.csv, shrinkage_summary.md, figures/shrinkage_actual_vs_pred.png, and one JSON
per variant and seed in shrinkage_runs/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd  # noqa: E402
import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _common import ROOT, add_args, paths  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import GroupKFold, ParameterSampler  # noqa: E402
from sklearn.preprocessing import QuantileTransformer  # noqa: E402

from wealth_index.config import DataConfig, FeatureConfig  # noqa: E402
from wealth_index.data import file_sha256, load_dataset  # noqa: E402
from wealth_index.modeling import build_pipeline  # noqa: E402
from wealth_index.spatial import holdout_split, make_cv, spatial_blocks  # noqa: E402

# Set from the command line by configure().
RUN: Path
LABELS: Path
BOUNDARIES: Path
OUT: Path
PREP_CACHE: Path
RUNS: Path
RESULTS: Path
SUMMARY: Path
FIGURE: Path
RUN_PARAMS: dict  # the run's tuned forest
RUN_TEST_RMSE: float
RUN_TOP_K: int  # features kept by selection, fixed for every variant
DATA_SHA = ""

SEED = 42
N_ITER = 20
N_ITER_ROBUST = 10
ROBUST_SEEDS = (43, 44, 45)
N_JOBS = 4
K_NEIGHBOURS = 8
NEIGHBOUR_FEATURES = [
    "NTLBin_Ratio",
    "Road_RLength",
    "POI_Count5KM",
    "NTL_MEAN",
    "NDBIPos_MEAN",
    "Green_MEAN",
]
RMSE_TOLERANCE = 0.15  # winner rule: CV RMSE at most this much worse than B0
TAIL_CUT = 1 / 3  # ... and mean error in Q1 and Q5 at least this much smaller

# ------------------------------------------------------------------------------------- models

RF_BASE = {"random_state": SEED, "n_jobs": 1}
THESIS_GRID = {  # configs/train.yaml, without select__top_k (fixed at 25 here)
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, 30],
    "min_samples_split": [10, 15, 20, 30],
    "min_samples_leaf": [4, 6, 8, 10, 15, 20],
    "max_features": ["sqrt"],
}
SPACES: dict[str, dict[str, list]] = {
    "rf": {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", 0.3, 0.6, 1.0],
    },
    "hgb": {
        "learning_rate": [0.02, 0.03, 0.05, 0.07, 0.1],
        "max_leaf_nodes": [15, 23, 31, 47, 63],
        "min_samples_leaf": [10, 15, 20, 30, 40],
        "l2_regularization": [0.0, 0.1, 0.3, 1.0],
    },
    "lgbm": {
        "learning_rate": [0.02, 0.03, 0.05, 0.07, 0.1],
        "n_estimators": [200, 400, 700, 1000, 1500],
        "num_leaves": [15, 23, 31, 47, 63],
        "min_child_samples": [10, 15, 20, 30, 40],
        "reg_lambda": [0.0, 0.1, 0.3, 1.0],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    },
}
SPACES["rf_thesis"] = THESIS_GRID


def make_model(kind: str, params: dict):
    if kind in ("rf", "rf_thesis"):
        return RandomForestRegressor(**RF_BASE, **params)
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=1500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=SEED,
            **params,
        )
    if kind == "lgbm":
        return LGBMRegressor(subsample_freq=1, random_state=SEED, n_jobs=1, verbose=-1, **params)
    raise ValueError(kind)


@dataclass
class Variant:
    name: str
    label: str
    kind: str  # model family, key of SPACES
    features: str = "base"  # base | nbr | district
    params: dict | None = None  # fixed params; None = search SPACES[kind]
    base: str | None = None  # take kind and params from this variant's result
    target: str = "none"  # none | logit | quantile
    weights: bool = False
    nbr_wi: bool = False
    calibrate: str | None = None  # linear | isotonic | quantile
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------- data


@dataclass
class Data:
    X: pd.DataFrame
    y: np.ndarray
    ids: np.ndarray
    blocks: np.ndarray
    xy: np.ndarray  # village centroids, UTM metres
    quintile: np.ndarray
    cutoffs: np.ndarray
    district: np.ndarray
    feature_cfg: FeatureConfig


def load_data() -> Data:
    meta = json.loads((RUN / "metadata.json").read_text())
    cfg = meta["config"]
    data_cfg = DataConfig(**{**cfg["data"], "path": str(ROOT / cfg["data"]["path"])})
    if file_sha256(data_cfg.path) != meta["data_sha256"]:
        raise SystemExit(f"{data_cfg.path} changed since run {RUN.name}")
    ds = load_dataset(data_cfg)
    ids = ds.ids.astype(str).to_numpy()
    labels = pd.read_csv(LABELS, dtype={"KDEPUM": str}).set_index("KDEPUM")
    quintile = labels["quintile"].reindex(ids).to_numpy().astype(int)
    y = ds.y.to_numpy(dtype=float)
    cutoffs = np.array([y[quintile == q].max() for q in (1, 2, 3, 4)])
    villages = gpd.read_parquet(BOUNDARIES)
    centroid = villages.geometry.centroid.reindex(ids)
    return Data(
        X=ds.X.reset_index(drop=True),
        y=y,
        ids=ids,
        blocks=spatial_blocks(ds.coords, cfg["split"]["block_size_km"]),
        xy=np.column_stack([centroid.x, centroid.y]),
        quintile=quintile,
        cutoffs=cutoffs,
        district=villages["WADMKK"].reindex(ids).to_numpy(),
        feature_cfg=FeatureConfig(**{**cfg["features"], "top_k": RUN_TOP_K}),
    )


def feature_table(d: Data, features: str) -> pd.DataFrame:
    """The model inputs for a feature set. Extra columns use features only, never labels."""
    if features == "base":
        return d.X
    if features == "nbr":
        _, nbrs = cKDTree(d.xy).query(d.xy, k=K_NEIGHBOURS + 1)
        nbrs = nbrs[:, 1:]
        extra = {
            f"NB_{f}": np.nanmean(d.X[f].to_numpy(dtype=float)[nbrs], axis=1)
            for f in NEIGHBOUR_FEATURES
        }
        return pd.concat([d.X, pd.DataFrame(extra)], axis=1)
    if features == "district":
        dummies = pd.get_dummies(pd.Series(d.district), prefix="DIST", dtype=float)
        dummies.columns = [c.replace(" ", "_").replace(".", "") for c in dummies.columns]
        return pd.concat([d.X, dummies], axis=1)
    raise ValueError(features)


@dataclass
class Split:
    seed: int
    train: np.ndarray
    test: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]]  # global indices, within train


def make_split(d: Data, seed: int) -> Split:
    train, test = holdout_split(len(d.y), "spatial", d.blocks, 0.3, seed)
    cv = make_cv("spatial", 5, seed)
    folds = [
        (train[a], train[b])
        for a, b in cv.split(np.zeros(len(train)), d.y[train], groups=d.blocks[train])
    ]
    return Split(seed, np.sort(train), np.sort(test), folds)


def prepared(d: Data, split: Split, features: str, force: bool) -> dict:
    """Preprocessed matrices for each CV fold and for train -> test, fitted on training rows."""
    path = PREP_CACHE / f"{features}_s{split.seed}_{DATA_SHA[:12]}.joblib"
    if path.exists() and not force:
        return joblib.load(path)
    X = feature_table(d, features)
    prep = build_pipeline(d.feature_cfg, SEED)[:-1]
    out = {"folds": [], "cols": None}

    def fit_one(tr, te):
        p = clone(prep).fit(X.iloc[tr], d.y[tr])
        return (
            p.transform(X.iloc[tr]).to_numpy(float),
            p.transform(X.iloc[te]).to_numpy(float),
            list(p.get_feature_names_out()),
        )

    pairs = [*split.folds, (split.train, split.test)]
    results = Parallel(n_jobs=N_JOBS)(delayed(fit_one)(tr, te) for tr, te in pairs)
    for (tr, te), (Xtr, Xte, cols) in zip(pairs[:-1], results[:-1], strict=True):
        out["folds"].append({"tr": tr, "te": te, "Xtr": Xtr, "Xte": Xte, "cols": cols})
    Xtr, Xte, cols = results[-1]
    out["holdout"] = {"tr": split.train, "te": split.test, "Xtr": Xtr, "Xte": Xte, "cols": cols}
    PREP_CACHE.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, path)
    return out


# --------------------------------------------------------------------------- fit and predict


def logit(y):
    u = (y + 0.5) / 101
    return np.log(u / (1 - u))


def inv_logit(z):
    return 101 / (1 + np.exp(-z)) - 0.5


def tail_weights(y: np.ndarray) -> np.ndarray:
    counts, edges = np.histogram(y, bins=10)
    w = 1.0 / counts[np.clip(np.digitize(y, edges[1:-1]), 0, 9)]
    return np.clip(w / w.min(), 1, 3)


def neighbour_wi(d: Data, tr: np.ndarray, query: np.ndarray, k: int = K_NEIGHBOURS) -> np.ndarray:
    """Mean WI of the k nearest training villages outside the query village's block.

    Test villages never have labelled neighbours in their own block, so training villages must
    not use theirs either; otherwise the feature means something different at test time.
    """
    tree = cKDTree(d.xy[tr])
    _, nn = tree.query(d.xy[query], k=min(len(tr), 80))
    out = np.empty(len(query))
    for i, (q, row) in enumerate(zip(query, nn, strict=True)):
        cand = tr[row]
        cand = cand[d.blocks[cand] != d.blocks[q]][:k]
        out[i] = d.y[cand].mean() if len(cand) else d.y[tr].mean()
    return out


def fit_predict(v: Variant, kind: str, params: dict, d: Data, f: dict) -> np.ndarray:
    Xtr, Xte, ytr = f["Xtr"], f["Xte"], d.y[f["tr"]]
    if v.nbr_wi:
        Xtr = np.column_stack([Xtr, neighbour_wi(d, f["tr"], f["tr"])])
        Xte = np.column_stack([Xte, neighbour_wi(d, f["tr"], f["te"])])
    target, inverse = ytr, (lambda p: p)
    if v.target == "logit":
        target, inverse = logit(ytr), inv_logit
    elif v.target == "quantile":
        qt = QuantileTransformer(n_quantiles=min(1000, len(ytr)), output_distribution="normal")
        target = qt.fit_transform(ytr[:, None]).ravel()
        inverse = lambda p: qt.inverse_transform(p[:, None]).ravel()  # noqa: E731
    fit_kw = {"sample_weight": tail_weights(ytr)} if v.weights else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = make_model(kind, params).fit(Xtr, target, **fit_kw)
        return inverse(model.predict(Xte))


def calibrator(method: str, pred: np.ndarray, actual: np.ndarray) -> Callable:
    if method == "linear":
        lr = LinearRegression().fit(pred[:, None], actual)
        return lambda p: lr.predict(p[:, None])
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip").fit(pred, actual)
        return iso.predict
    if method == "quantile":  # map the predicted distribution onto the actual one
        sp, sa = np.sort(pred), np.sort(actual)
        return lambda p: np.interp(p, sp, sa)
    raise ValueError(method)


def cv_predict(
    v: Variant, kind: str, params: dict, d: Data, prep: dict
) -> tuple[float, np.ndarray]:
    """Mean fold RMSE (as RandomizedSearchCV scores it) and pooled out-of-fold predictions."""
    oof = np.full(len(d.y), np.nan)
    rmses = []
    for f in prep["folds"]:
        p = fit_predict(v, kind, params, d, f)
        if v.calibrate:  # nested: map learnt from inner out-of-fold predictions of this fold
            p = calibrator(v.calibrate, *inner_oof(v, kind, params, d, f))(p)
        oof[f["te"]] = p
        rmses.append(np.sqrt(np.mean((p - d.y[f["te"]]) ** 2)))
    return float(np.mean(rmses)), oof


def inner_oof(v, kind, params, d: Data, f: dict) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions within one fold's training rows (model only, preprocessing of the
    outer fold reused: a small approximation that affects only the calibration map)."""
    tr = f["tr"]
    pred = np.empty(len(tr))
    for a, b in GroupKFold(5, shuffle=True, random_state=SEED).split(tr, groups=d.blocks[tr]):
        sub = {"tr": tr[a], "te": tr[b], "Xtr": f["Xtr"][a], "Xte": f["Xtr"][b]}
        pred[b] = fit_predict(
            Variant("inner", "", kind, nbr_wi=v.nbr_wi, target=v.target, weights=v.weights),
            kind,
            params,
            d,
            sub,
        )
    return pred, d.y[tr]


# -------------------------------------------------------------------------------- evaluation


def metrics(actual: np.ndarray, pred: np.ndarray, quint: np.ndarray, cutoffs) -> dict:
    err = pred - actual
    pq = np.searchsorted(cutoffs, pred, side="left") + 1
    out = {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err**2) / np.sum((actual - actual.mean()) ** 2)),
        "slope": float(np.polyfit(actual, pred, 1)[0]),
        "quintile_acc": float(np.mean(pq == quint)),
        "q1_recall": float(np.mean(pq[quint == 1] == 1)),
    }
    for q in (1, 5):
        out[f"q{q}_bias"] = float(err[quint == q].mean())
        out[f"q{q}_mae"] = float(np.abs(err[quint == q]).mean())
    return out


def run_variant(
    v: Variant, d: Data, split: Split, done: dict, force: bool, n_iter: int = N_ITER
) -> dict:
    path = RUNS / f"{v.name}_s{split.seed}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    t0 = time.time()
    prep = prepared(d, split, v.features, force=False)

    kind, params, searched = v.kind, v.params, None
    if v.base:
        kind, params = done[v.base]["kind"], done[v.base]["params"]
    if params is None:
        cands = list(ParameterSampler(SPACES[kind], n_iter=n_iter, random_state=split.seed))
        scores = Parallel(n_jobs=N_JOBS)(delayed(cv_predict)(v, kind, c, d, prep) for c in cands)
        best = int(np.argmin([s for s, _ in scores]))
        params, searched = cands[best], len(cands)
        cv_rmse, oof = scores[best]
    else:
        cv_rmse, oof = cv_predict(v, kind, params, d, prep)

    h = prep["holdout"]
    test_pred = fit_predict(v, kind, params, d, h)
    if v.calibrate:  # map learnt on the training villages' out-of-fold predictions (base model)
        base = Variant("base", "", kind, nbr_wi=v.nbr_wi, target=v.target, weights=v.weights)
        _, base_oof = cv_predict(base, kind, params, d, prep)
        tr = split.train
        test_pred = calibrator(v.calibrate, base_oof[tr], d.y[tr])(test_pred)

    tr, te = split.train, split.test
    res = {
        "name": v.name,
        "label": v.label,
        "seed": split.seed,
        "kind": kind,
        "features": v.features,
        "params": {k: (None if val is None else val) for k, val in params.items()},
        "searched": searched,
        "cv_rmse": cv_rmse,
        "cv": metrics(d.y[tr], oof[tr], d.quintile[tr], d.cutoffs),
        "test": metrics(d.y[te], test_pred, d.quintile[te], d.cutoffs),
        "test_pred": test_pred.tolist(),
        "selected": h["cols"],
        "minutes": (time.time() - t0) / 60,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1, default=float))
    print(
        f"  {v.name:<5} {v.label:<38} CV {cv_rmse:.3f} | test {res['test']['rmse']:.3f} | "
        f"Q1 {res['cv']['q1_bias']:+.2f} Q5 {res['cv']['q5_bias']:+.2f} | "
        f"slope {res['cv']['slope']:.2f} | {res['minutes']:.1f} min",
        flush=True,
    )
    return res


# ------------------------------------------------------------------------------------- report


def row(r: dict) -> dict:
    out = {
        "variant": r["name"],
        "label": r["label"],
        "seed": r["seed"],
        "model": r["kind"],
        "cv_rmse": r["cv_rmse"],
    }
    for split in ("cv", "test"):
        for k, val in r[split].items():
            # cv_rmse is the mean fold RMSE above; the pooled out-of-fold one is cv_pooled_rmse
            out[f"{split}_{'pooled_' if (split, k) == ('cv', 'rmse') else ''}{k}"] = val
    out["params"] = json.dumps(r["params"])
    out["minutes"] = r["minutes"]
    return out


def passes(r: dict, b0: dict) -> bool:
    return (
        r["cv_rmse"] <= b0["cv_rmse"] + RMSE_TOLERANCE
        and abs(r["cv"]["q1_bias"]) <= (1 - TAIL_CUT) * abs(b0["cv"]["q1_bias"])
        and abs(r["cv"]["q5_bias"]) <= (1 - TAIL_CUT) * abs(b0["cv"]["q5_bias"])
    )


def plot(results: list[dict], d: Data, split: Split) -> None:
    n = len(results)
    cols = 4
    fig, axes = plt.subplots(
        -(-n // cols), cols, figsize=(4 * cols, 4 * -(-n // cols)), squeeze=False
    )
    actual, quint = d.y[split.test], d.quintile[split.test]
    lo, hi = actual.min() - 3, actual.max() + 3
    for ax, r in zip(axes.flat, results, strict=False):  # more axes than results
        ax.scatter(actual, r["test_pred"], c=quint, cmap="viridis", s=6, alpha=0.6)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), title=f"{r['name']}: {r['label']}")
        ax.text(
            0.03,
            0.97,
            f"RMSE {r['test']['rmse']:.2f}\nslope {r['test']['slope']:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        ax.title.set_fontsize(9)
    for ax in axes.flat[n:]:
        ax.set_axis_off()
    fig.supxlabel("Actual WI (test villages)")
    fig.supylabel("Predicted WI")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------- main


def configure(args) -> None:
    """Fill the module paths from the command line; also used by shrinkage_rf_followup.py."""
    global RUN, LABELS, BOUNDARIES, OUT, PREP_CACHE, RUNS, RESULTS, SUMMARY, FIGURE
    global RUN_PARAMS, RUN_TEST_RMSE, RUN_TOP_K, DATA_SHA
    p = paths(args)
    RUN, LABELS, BOUNDARIES, OUT = p.run, p.labels, p.boundaries, p.out
    PREP_CACHE, RUNS = p.cache, p.out / "shrinkage_runs"
    RESULTS, SUMMARY = p.out / "shrinkage_results.csv", p.out / "shrinkage_summary.md"
    FIGURE = p.figures / "shrinkage_actual_vs_pred.png"
    metrics = json.loads((RUN / "metrics.json").read_text())
    best = metrics["search"].get("best_params")
    if best is None:
        raise SystemExit(f"{RUN.name} was not tuned by random search; B0 needs its best_params")
    RUN_PARAMS = {k.removeprefix("model__"): v for k, v in best.items() if k.startswith("model__")}
    RUN_TEST_RMSE = metrics["test"]["tuned"]["rmse"]
    meta = json.loads((RUN / "metadata.json").read_text())
    RUN_TOP_K = best.get("select__top_k", meta["config"]["features"]["top_k"])
    DATA_SHA = meta["data_sha256"]


def variants(best_model: str | None) -> list[Variant]:
    first = [
        Variant("B0", "baseline RF (run settings)", "rf", params=RUN_PARAMS),
        Variant("B1", "RF, less smoothing", "rf"),
        Variant("B2a", "HistGradientBoosting", "hgb"),
        Variant("B2b", "LightGBM", "lgbm"),
    ]
    if best_model is None:
        return first
    m = best_model
    return first + [
        Variant("B3a", f"{m} + logit target", "", base=m, target="logit"),
        Variant("B3b", f"{m} + rank-to-normal target", "", base=m, target="quantile"),
        Variant("B4a", f"{m} + linear recalibration", "", base=m, calibrate="linear"),
        Variant("B4b", f"{m} + isotonic recalibration", "", base=m, calibrate="isotonic"),
        Variant("B4c", f"{m} + quantile mapping", "", base=m, calibrate="quantile"),
        Variant("B5", f"{m} + tail weights", "", base=m, weights=True),
        Variant("B6a", f"{m} kind + neighbour features", "", base=m, features="nbr"),
        Variant("B6b", f"{m} kind + neighbour WI", "", base=m, nbr_wi=True),
        Variant("B6c", f"{m} kind + district indicator", "", base=m, features="district"),
    ]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="run just this variant (B0..B2b only before the others)")
    parser.add_argument("--force", action="store_true", help="recompute cached results")
    parser.add_argument("--skip-robust", action="store_true")
    add_args(parser)
    args = parser.parse_args()
    configure(args)

    d = load_data()
    split = make_split(d, SEED)
    run_test = set(pd.read_csv(RUN / "predictions_test.csv", dtype={"id": str})["id"])
    if set(d.ids[split.test]) != run_test:
        raise SystemExit("test split does not match the run")
    print(f"{len(split.train)} train / {len(split.test)} test villages; preparing folds...")
    t0 = time.time()
    prepared(d, split, "base", force=args.force)
    print(f"base preprocessing ready ({(time.time() - t0) / 60:.1f} min)")

    done: dict[str, dict] = {}
    for v in variants(None):
        if args.only and v.name != args.only:
            continue
        done[v.name] = run_variant(v, d, split, done, args.force)
    if args.only:
        return

    b0 = done["B0"]
    if abs(b0["test"]["rmse"] - RUN_TEST_RMSE) > 0.01:
        raise SystemExit(
            f"B0 test RMSE {b0['test']['rmse']:.4f} does not reproduce "
            f"the run ({RUN_TEST_RMSE:.4f})"
        )
    # B6 searches the best model family again; the others reuse its tuned settings.
    best = min(("B1", "B2a", "B2b"), key=lambda k: done[k]["cv_rmse"])
    print(f"best of B1/B2 by spatial CV RMSE: {best} ({done[best]['label']})")
    for v in variants(best)[4:]:
        if v.features != "base" or v.nbr_wi:  # B6: own search over the family's space
            v.kind, v.base = done[best]["kind"], None
        done[v.name] = run_variant(v, d, split, done, args.force)

    results = list(done.values())
    table = pd.DataFrame([row(r) for r in results])
    table["passes_rule"] = [passes(r, b0) for r in results]
    table.to_csv(RESULTS, index=False)
    plot(results, d, split)

    show = table.set_index("variant")[
        [
            "label",
            "cv_rmse",
            "test_rmse",
            "cv_q1_bias",
            "cv_q5_bias",
            "test_q1_bias",
            "test_q5_bias",
            "cv_slope",
            "cv_quintile_acc",
            "cv_q1_recall",
            "passes_rule",
        ]
    ].round(3)
    lines = [
        f"# Part B summary (run {RUN.name})",
        "",
        "Generated by shrinkage_experiments.py. Podes-derived: do not commit or publish.",
        "",
        "Tail bias and slope are on out-of-fold predictions for the 1087 training villages "
        "(spatial CV); test columns are the 415 held-out villages.",
        "",
        "```",
        show.to_string(),
        "```",
        "",
    ]
    for name in ("B6a", "B6c"):
        extra = [c for c in done[name]["selected"] if c.startswith(("NB_", "DIST_"))]
        lines.append(f"{name}: extra features kept by selection: {extra or 'none'}")
    print("\n" + show.to_string())

    if not args.skip_robust:
        winners = [r["name"] for r in results if passes(r, b0)]
        others = sorted((r for r in results if r["name"] != "B0"), key=lambda r: r["cv_rmse"])
        finalists = winners[:2] or [others[0]["name"]]
        tail = min(
            (r for r in results if r["name"] != "B0"),
            key=lambda r: abs(r["cv"]["q1_bias"]) + abs(r["cv"]["q5_bias"]),
        )["name"]
        if tail not in finalists:
            finalists.append(tail)
        finalists = ["B0", *finalists[:2]]
        print(f"\nrobustness over seeds {ROBUST_SEEDS}: {finalists}")
        lines += [
            "",
            f"## Robustness: re-tuned ({N_ITER_ROBUST} tries) on seeds {ROBUST_SEEDS}",
            "",
        ]
        rob = []
        for seed in ROBUST_SEEDS:
            s = make_split(d, seed)
            seed_done: dict[str, dict] = {}
            for name in finalists:
                v = next(x for x in variants(best) if x.name == name)
                if name == "B0":
                    v = Variant("B0", v.label, "rf_thesis")  # re-tune the thesis grid
                elif v.base:  # re-tune the base family first, then apply the wrapper
                    base_v = next(x for x in variants(best) if x.name == best)
                    if best not in seed_done:
                        seed_done[best] = run_variant(
                            base_v, d, s, seed_done, args.force, N_ITER_ROBUST
                        )
                    if v.features != "base" or v.nbr_wi:
                        v.kind, v.base = done[best]["kind"], None
                seed_done[name] = run_variant(v, d, s, seed_done, args.force, N_ITER_ROBUST)
                rob.append(row(seed_done[name]))
        rob = pd.DataFrame(rob)
        agg = rob.groupby("variant")[
            ["test_rmse", "test_q1_bias", "test_q5_bias", "test_slope", "test_q1_recall"]
        ].agg(["mean", "std"])
        print(agg.round(3).to_string())
        lines += ["```", agg.round(3).to_string(), "```"]
        rob.to_csv(OUT / "shrinkage_robustness.csv", index=False)

    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {RESULTS}, {SUMMARY}, {FIGURE}")


if __name__ == "__main__":
    main()

"""Part A of NEXT_EXPERIMENTS_PLAN.md: where does the model err, and is the error clustered?

    python experiments/part_a_errors.py            # latest run; reuses residuals_oof.csv
    python experiments/part_a_errors.py --refit    # recompute the out-of-fold predictions
    python experiments/part_a_errors.py --help     # run, labels, output and cache options

Outputs go to experiments/output/<run id>/ (git-ignored). A1 writes residuals_oof.csv (an
out-of-fold prediction for every village), A2 the residual maps in figures/, A3 the tables in
part_a_summary.md. They contain Podes-derived WI: never commit or publish them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd  # noqa: E402
import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _common import ROOT, add_args, paths  # noqa: E402
from matplotlib.colors import ListedColormap, TwoSlopeNorm  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import cross_val_predict  # noqa: E402

from wealth_index.config import DataConfig  # noqa: E402
from wealth_index.data import file_sha256, load_dataset  # noqa: E402
from wealth_index.spatial import holdout_split, make_cv, spatial_blocks  # noqa: E402

# Set from the command line in main().
RUN: Path
LABELS: Path
THESIS: Path | None  # the thesis' own predictions, optional
BOUNDARIES: Path
DATASET: Path
OUT: Path
FIGURES: Path
SUMMARY: Path

PROFILE_FEATURES = [
    "NTLBin_Ratio",
    "Road_RLength",
    "POI_Count5KM",
    "NTL_MEAN",
    "NDBIPos_MEAN",
    "Green_MEAN",
]
K_NEIGHBOURS = 8
PERMUTATIONS = 999
CLIP = 20

_report: list[str] = []


def emit(title: str, body: str | pd.DataFrame) -> None:
    """Print a section and keep it for the summary file."""
    text = body.to_string() if isinstance(body, pd.DataFrame | pd.Series) else body
    print(f"\n== {title}\n{text}")
    _report.append(f"## {title}\n\n```\n{text}\n```\n")


def rmse(e: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(e))))


def bias_table(df: pd.DataFrame, pred: str, actual: str = "actual") -> pd.DataFrame:
    err = df[pred] - df[actual]
    return (
        pd.DataFrame({"quintile": df["quintile"], "err": err, "abs": err.abs()})
        .groupby("quintile")
        .agg(n=("err", "size"), mean_error=("err", "mean"), mae=("abs", "mean"))
        .round(2)
    )


# ---------------------------------------------------------------- A1: out-of-fold predictions


def quintile_of(values: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    """1-5 using the actual-WI cutoffs, so squeezing shows up as missed quintiles."""
    return np.searchsorted(cutoffs, values, side="left") + 1


def build_residuals(n_jobs: int) -> pd.DataFrame:
    meta = json.loads((RUN / "metadata.json").read_text())
    cfg = meta["config"]
    data_cfg = DataConfig(**{**cfg["data"], "path": str(ROOT / cfg["data"]["path"])})
    if file_sha256(data_cfg.path) != meta["data_sha256"]:
        raise SystemExit(f"{data_cfg.path} changed since run {RUN.name}; the split would differ")

    ds = load_dataset(data_cfg)
    split, seed = cfg["split"], cfg["seed"]
    groups = spatial_blocks(ds.coords, split["block_size_km"])
    _, test_idx = holdout_split(len(ds.y), split["scheme"], groups, split["test_size"], seed)

    model = clone(joblib.load(RUN / "model.joblib"))
    if "n_jobs" in model[-1].get_params():
        model[-1].set_params(n_jobs=1)  # parallelise over folds instead, to spare RAM
    cv = make_cv("spatial", split["cv_folds"], seed)
    oof = cross_val_predict(model, ds.X, ds.y, groups=groups, cv=cv, n_jobs=n_jobs)

    labels = pd.read_csv(LABELS, dtype={"KDEPUM": str}).set_index("KDEPUM")
    villages = gpd.read_parquet(BOUNDARIES)
    ids = ds.ids.astype(str).to_numpy()
    df = pd.DataFrame(
        {
            "KDEPUM": ids,
            "NAMOBJ": villages["NAMOBJ"].reindex(ids).to_numpy(),
            "WADMKK": villages["WADMKK"].reindex(ids).to_numpy(),
            "lon": ds.coords.iloc[:, 0].to_numpy(),
            "lat": ds.coords.iloc[:, 1].to_numpy(),
            "actual": ds.y.to_numpy(),
            "oof_pred": oof,
            "block": groups,
        }
    )
    df["residual"] = df["oof_pred"] - df["actual"]
    df["quintile"] = labels["quintile"].reindex(ids).to_numpy()
    if df["quintile"].isna().any():
        raise SystemExit(f"{df['quintile'].isna().sum()} villages have no quintile in {LABELS}")
    df["quintile"] = df["quintile"].astype(int)
    df["in_test"] = False
    df.loc[test_idx, "in_test"] = True

    if THESIS is None:
        return df.assign(thesis_pred=np.nan)
    # Thesis predictions are keyed by the shapefile OBJECTID; labels_podes maps it to KDEPUM.
    thesis = pd.read_csv(THESIS, sep=";").set_index("id")
    obj = labels["OBJECTID_wi"].reindex(ids)
    t = thesis.reindex(obj.to_numpy())
    df["thesis_pred"] = np.where(
        t["test"].to_numpy() == 1, t["wealth_index_pred"].to_numpy(), np.nan
    )
    if not np.allclose(t["wealth_index_true"].to_numpy(), df["actual"], atol=1e-6, equal_nan=False):
        raise SystemExit("thesis true WI does not match our labels after the OBJECTID join")
    return df


def check_residuals(df: pd.DataFrame) -> None:
    metrics = json.loads((RUN / "metrics.json").read_text())
    spatial_cv = metrics["cv_comparison"]["spatial"]["rmse_mean"]
    test_ids = set(pd.read_csv(RUN / "predictions_test.csv", dtype={"id": str})["id"])
    cutoffs = df.groupby("quintile")["actual"].max().sort_index().to_numpy()[:-1]
    oof_rmse = rmse(df["residual"])

    lines = [
        f"run: {RUN.name}",
        f"villages: {len(df)} ({df['in_test'].sum()} in the run's test set)",
        f"out-of-fold RMSE, all villages: {oof_rmse:.3f} (spatial CV in the run: {spatial_cv:.3f})",
        f"out-of-fold RMSE, test villages: {rmse(df.loc[df.in_test, 'residual']):.3f}"
        f" (run test RMSE: {metrics['test']['tuned']['rmse']:.3f})",
        f"out-of-fold RMSE, train villages: {rmse(df.loc[~df.in_test, 'residual']):.3f}"
        " (hyperparameters were tuned on these, so slightly optimistic)",
        f"test ids match predictions_test.csv: {set(df.loc[df.in_test, 'KDEPUM']) == test_ids}",
        f"quintile cutoffs from actual WI: {np.round(cutoffs, 2).tolist()}",
        "actual quintiles reproduced by the cutoffs: "
        f"{(quintile_of(df['actual'].to_numpy(), cutoffs) == df['quintile']).mean():.1%}",
    ]
    emit("A1 checks", "\n".join(lines))
    if set(df.loc[df.in_test, "KDEPUM"]) != test_ids:
        raise SystemExit("test set does not match the run; stopping")
    if abs(oof_rmse - spatial_cv) > 1.0:
        raise SystemExit(f"out-of-fold RMSE {oof_rmse:.2f} is far from spatial CV {spatial_cv:.2f}")

    preds = pd.read_csv(RUN / "predictions_test.csv", dtype={"id": str}).set_index("id")
    test = df[df.in_test].set_index("KDEPUM")
    test["run_pred"] = preds["predicted"].reindex(test.index)
    emit("A1 bias by quintile, test villages, run model", bias_table(test, "run_pred"))
    emit("A1 bias by quintile, all villages, out of fold", bias_table(df, "oof_pred"))


# ------------------------------------------------------------------------------------ A2: maps


def draw_maps(df: pd.DataFrame) -> None:
    villages = gpd.read_parquet(BOUNDARIES)
    gdf = villages[["WADMKK", "geometry"]].join(df.set_index("KDEPUM").drop(columns="WADMKK"))
    districts = villages.dissolve(by="WADMKK").boundary
    FIGURES.mkdir(parents=True, exist_ok=True)
    norm = TwoSlopeNorm(vmin=-CLIP, vcenter=0, vmax=CLIP)

    def finish(fig, ax, path: Path, title: str) -> None:
        districts.plot(ax=ax, color="black", linewidth=0.6)
        ax.set_title(title)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def residual_map(subset, path: Path, title: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 10))
        if subset is not None:
            gdf[~subset].plot(ax=ax, color="#dddddd", linewidth=0)
        shown = gdf if subset is None else gdf[subset]
        shown.plot(
            column=shown["residual"].clip(-CLIP, CLIP),
            ax=ax,
            cmap="RdBu_r",
            norm=norm,
            linewidth=0.05,
            edgecolor="white",
            legend=True,
            legend_kwds={"label": "Predicted − actual (clipped at ±20)", "shrink": 0.6},
        )
        finish(fig, ax, path, title)

    residual_map(None, FIGURES / "residual_map_all.png", "Out-of-fold residual, all villages")
    for q, name in [(1, "poorest"), (5, "richest")]:
        residual_map(
            gdf["quintile"] == q,
            FIGURES / f"residual_map_q{q}.png",
            f"Out-of-fold residual, Q{q} ({name}) villages; others grey",
        )

    cutoffs = df.groupby("quintile")["actual"].max().sort_index().to_numpy()[:-1]
    gdf["oof_quintile"] = quintile_of(gdf["oof_pred"].to_numpy(), cutoffs)
    cmap = ListedColormap(plt.get_cmap("viridis")(np.linspace(0, 1, 5)))
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    for ax, col, title in [
        (axes[0], "quintile", "Actual quintile"),
        (axes[1], "oof_quintile", "Predicted quintile (out of fold, actual cutoffs)"),
    ]:
        gdf.plot(
            column=col,
            ax=ax,
            cmap=cmap,
            vmin=0.5,
            vmax=5.5,
            linewidth=0.05,
            edgecolor="white",
            categorical=False,
        )
        districts.plot(ax=ax, color="black", linewidth=0.6)
        ax.set_title(title)
        ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.5, 5.5))
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, ticks=range(1, 6))
    cbar.set_ticklabels(["Q1 poorest", "Q2", "Q3", "Q4", "Q5 richest"])
    fig.savefig(FIGURES / "actual_vs_oof_quintile.png", dpi=200)
    plt.close(fig)

    hit = (gdf["oof_quintile"] == gdf["quintile"]).mean()
    q1 = gdf[gdf.quintile == 1]
    emit(
        "A2 maps",
        f"wrote 4 PNGs to {FIGURES}\n"
        f"correct quintile: {hit:.1%}; Q1 predicted as Q1: {(q1.oof_quintile == 1).mean():.1%}",
    )


# ------------------------------------------------------------------------------- A3: analysis


def morans_i(xy: np.ndarray, values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Moran's I with row-standardised k-nearest-neighbour weights; one-sided permutation p."""
    _, nbrs = cKDTree(xy).query(xy, k=K_NEIGHBOURS + 1)
    nbrs = nbrs[:, 1:]  # drop self
    z = values - values.mean()

    def stat(v: np.ndarray) -> float:
        return float(np.sum(v * v[nbrs].mean(axis=1)) / np.sum(v * v))

    observed = stat(z)
    perms = np.array([stat(rng.permutation(z)) for _ in range(PERMUTATIONS)])
    return observed, (np.sum(perms >= observed) + 1) / (PERMUTATIONS + 1)


def analyse(df: pd.DataFrame, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    villages = gpd.read_parquet(BOUNDARIES)
    centroid = villages.geometry.centroid.reindex(df["KDEPUM"])
    df = df.assign(x=centroid.x.to_numpy(), y=centroid.y.to_numpy())

    rows = []
    for name, sub in [
        ("all villages", df),
        ("Q1 only", df[df.quintile == 1]),
        ("Q5 only", df[df.quintile == 5]),
    ]:
        i, p = morans_i(sub[["x", "y"]].to_numpy(), sub["residual"].to_numpy(), rng)
        rows.append({"subset": name, "n": len(sub), "morans_i": round(i, 3), "p": round(p, 4)})
    emit(
        f"A3.1 Moran's I of residuals ({K_NEIGHBOURS} nearest neighbours, "
        f"{PERMUTATIONS} permutations; expected under randomness ≈ 0)",
        pd.DataFrame(rows).set_index("subset"),
    )

    by_area = df.groupby("WADMKK").agg(
        n=("residual", "size"),
        mean_error=("residual", "mean"),
        mae=("residual", lambda e: e.abs().mean()),
    )
    for q in (1, 5):
        sub = df[df.quintile == q].groupby("WADMKK")["residual"]
        by_area[f"n_q{q}"] = sub.size()
        by_area[f"mean_error_q{q}"] = sub.mean()
    by_area = by_area.fillna({"n_q1": 0, "n_q5": 0}).astype({"n_q1": int, "n_q5": int})
    emit(
        "A3.2 residuals by kabupaten/kota (mean error = predicted − actual; "
        "quintile cells with n < 10 are noisy)",
        by_area.sort_values("mean_error", ascending=False).round(2),
    )

    data = pd.read_csv(DATASET, dtype={"id": str})
    feats = [f for f in PROFILE_FEATURES if f in data]
    q1 = df[df.quintile == 1].merge(data[["id", *feats]], left_on="KDEPUM", right_on="id")
    over, good = q1[q1.residual > 10], q1[q1.residual.abs() < 3]
    rows = []
    for f in feats:
        p = (
            mannwhitneyu(over[f].dropna(), good[f].dropna()).pvalue
            if len(over) and len(good)
            else np.nan
        )
        rows.append(
            {
                "feature": f,
                f"overpredicted (n={len(over)})": over[f].median(),
                f"well predicted (n={len(good)})": good[f].median(),
                "all Q2-Q5 median": data.loc[data.id.isin(df.KDEPUM[df.quintile > 1]), f].median(),
                "mann_whitney_p": p,
            }
        )
    missing = sorted(set(PROFILE_FEATURES) - set(feats))
    emit(
        "A3.3 Q1 villages: overpredicted (residual > +10) vs well predicted (|residual| < 3), "
        "feature medians" + (f" [not in dataset: {missing}]" if missing else ""),
        pd.DataFrame(rows).set_index("feature").round(4),
    )

    t = df.dropna(subset=["thesis_pred"])
    if t.empty:
        emit("A3 comparison with the thesis model", "skipped: no thesis predictions (--thesis)")
        shap_worst(df)
        return
    t_res = t["thesis_pred"] - t["actual"]
    lines = [
        f"villages in the thesis test set: {len(t)} (a different split from ours)",
        f"thesis RMSE on them: {rmse(t_res):.3f}; "
        f"our out-of-fold RMSE on them: {rmse(t.residual):.3f}",
        f"correlation of the two models' residuals: {np.corrcoef(t_res, t.residual)[0, 1]:.3f}",
    ]
    both = pd.concat(
        [
            bias_table(t, "thesis_pred").add_prefix("thesis_"),
            bias_table(t, "oof_pred")[["mean_error", "mae"]].add_prefix("ours_"),
        ],
        axis=1,
    )
    emit("A3 comparison with the thesis model", "\n".join(lines) + "\n\n" + both.to_string())

    shap_worst(df)


def shap_worst(df: pd.DataFrame, n: int = 10) -> None:
    try:
        import shap
    except ImportError:
        emit("A3.4 SHAP", "skipped: shap is not installed")
        return
    meta = json.loads((RUN / "metadata.json").read_text())
    model = joblib.load(RUN / "model.joblib")
    data = pd.read_csv(DATASET, dtype={"id": str}).set_index("id")
    # Explain the run's own model on its test villages, where its predictions are out of sample.
    test = df[df.in_test].set_index("KDEPUM")
    X = data.loc[test.index, meta["features_in"]]
    Xt = pd.DataFrame(
        model[:-1].transform(X), index=X.index, columns=model[:-1].get_feature_names_out()
    )
    test["run_pred"] = model.predict(X)
    test["run_res"] = test["run_pred"] - test["actual"]
    explainer = shap.TreeExplainer(model[-1])
    blocks = []
    for q, worst in [
        (1, test[test.quintile == 1].nlargest(n, "run_res")),
        (5, test[test.quintile == 5].nsmallest(n, "run_res")),
    ]:
        values = pd.DataFrame(
            explainer.shap_values(Xt.loc[worst.index]), index=worst.index, columns=Xt.columns
        )
        mean = values.mean().sort_values(key=np.abs, ascending=False).head(8).round(2)
        blocks.append(
            f"Q{q}: {n} worst test villages (mean residual {worst.run_res.mean():+.1f}); "
            "mean SHAP contribution, top 8 (positive = pushes the prediction up):\n"
            + mean.to_string()
        )
    emit("A3.4 SHAP on the run model", "\n\n".join(blocks))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refit", action="store_true", help="recompute out-of-fold predictions")
    parser.add_argument("--n-jobs", type=int, default=4, help="folds fitted in parallel")
    parser.add_argument(
        "--thesis",
        type=Path,
        default=ROOT / "data_inside" / "Predicted WI.csv",
        help="thesis predictions (id;wealth_index_true;wealth_index_pred;test)",
    )
    add_args(parser)
    args = parser.parse_args()
    global RUN, LABELS, THESIS, BOUNDARIES, DATASET, OUT, FIGURES, SUMMARY
    p = paths(args)
    RUN, LABELS, BOUNDARIES, FIGURES = p.run, p.labels, p.boundaries, p.figures
    THESIS = args.thesis if args.thesis.exists() else None
    meta = json.loads((RUN / "metadata.json").read_text())
    DATASET = ROOT / meta["config"]["data"]["path"]
    OUT, SUMMARY = p.out / "residuals_oof.csv", p.out / "part_a_summary.md"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console may be cp932

    if args.refit or not OUT.exists():
        df = build_residuals(args.n_jobs)
        df.drop(columns="block").to_csv(OUT, index=False)
    else:
        df = pd.read_csv(OUT, dtype={"KDEPUM": str})
        print(f"Reusing {OUT} (pass --refit to recompute)")
    check_residuals(df)
    draw_maps(df)
    analyse(df)
    SUMMARY.write_text(
        f"# Part A summary (run {RUN.name})\n\nGenerated by part_a_errors.py. Podes-derived: "
        "do not commit or publish.\n\n" + "\n".join(_report),
        encoding="utf-8",
    )
    print(f"\nWrote {OUT}, {SUMMARY} and maps in {FIGURES}")


if __name__ == "__main__":
    main()

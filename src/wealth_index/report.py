"""Collect every training run under an artifacts directory into flat tables for reporting."""

from __future__ import annotations

import json
import re
from datetime import datetime
from importlib import resources
from pathlib import Path

import pandas as pd

TABLES = ("runs", "model_metrics", "predictions", "feature_importance")


def _search_summary(search: dict) -> str:
    if "best_params" in search:
        return ", ".join(f"{k.split('__')[-1]}={v}" for k, v in search["best_params"].items())
    # TPOT reprs wrap sampled values, e.g. max_depth=np.int64(20); keep just the value.
    return re.sub(r"np\.\w+\((.*?)\)", r"\1", " ".join(search.get("best_pipeline", "").split()))


def collect_runs(artifacts_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Return one DataFrame per table in TABLES, rows from every complete run."""
    rows: dict[str, list] = {name: [] for name in TABLES}
    for run_dir in sorted(p for p in Path(artifacts_dir).iterdir() if p.is_dir()):
        files = [run_dir / f for f in ("metrics.json", "metadata.json", "feature_importance.csv")]
        if not all(f.exists() for f in files):
            continue  # incomplete, or written before these outputs existed
        metrics = json.loads(files[0].read_text())
        meta = json.loads(files[1].read_text())
        run_id = run_dir.name
        tuned, cv = metrics["test"]["tuned"], metrics["cv_comparison"]
        rows["runs"].append(
            {
                "run_id": run_id,
                "created_at": meta["created_at"],
                "dataset": Path(meta["config"]["data"]["path"]).parent.name,
                "search_method": metrics["search"]["method"],
                "split_scheme": meta["config"]["split"]["scheme"],
                "n_train": meta["n_train"],
                "n_test": meta["n_test"],
                "n_spatial_blocks": meta["n_spatial_blocks"],
                "n_features_in": len(meta["features_in"]),
                "n_selected": len(meta["selected_features"]),
                "test_rmse": tuned["rmse"],
                "test_mae": tuned["mae"],
                "test_r2": tuned["r2"],
                "test_rrmse_pct": tuned["rrmse_pct"],
                "train_rmse": metrics["train"]["tuned"]["rmse"],
                "cv_random_rmse": cv["random"]["rmse_mean"],
                "cv_random_std": cv["random"]["rmse_std"],
                "cv_spatial_rmse": cv["spatial"]["rmse_mean"],
                "cv_spatial_std": cv["spatial"]["rmse_std"],
                "best_model": _search_summary(metrics["search"]),
            }
        )
        for split in ("train", "test"):
            for model, m in metrics[split].items():
                rows["model_metrics"].append(
                    {"run_id": run_id, "model": model, "split": split, **m}
                )
        preds = pd.read_csv(run_dir / "predictions_test.csv")
        preds["residual"] = preds["actual"] - preds["predicted"]
        rows["predictions"].append(preds.assign(run_id=run_id))
        rows["feature_importance"].append(pd.read_csv(files[2]).assign(run_id=run_id))

    tables = {name: pd.DataFrame(rows[name]) for name in ("runs", "model_metrics")}
    for name in ("predictions", "feature_importance"):
        tables[name] = pd.concat(rows[name], ignore_index=True) if rows[name] else pd.DataFrame()
    return tables


def build_html_report(artifacts_dir: str | Path, output: str | Path, decimals: int = 3) -> Path:
    """Write a self-contained HTML dashboard of every run, with the run tables embedded."""
    tables = collect_runs(artifacts_dir)
    if tables["runs"].empty:
        raise FileNotFoundError(f"No complete training runs found in {artifacts_dir}")
    payload = {
        "params": {"run": None},
        "datasets": {
            name: json.loads(df.round(decimals).to_json(orient="records"))
            for name, df in tables.items()
        },
    }
    # "</" inside the embedded JSON would end its <script> tag early.
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    template = (resources.files("wealth_index") / "templates" / "report.html").read_text("utf-8")
    html = template.replace("__REPORT_DATA__", data).replace(
        "__GENERATED_AT__", datetime.now().strftime("%Y-%m-%d %H:%M")
    )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def export_runs(artifacts_dir: str | Path, out_dir: str | Path, decimals: int = 3) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in collect_runs(artifacts_dir).items():
        path = out / f"{name}.csv"
        df.round(decimals).to_csv(path, index=False)
        paths.append(path)
    return paths

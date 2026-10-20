"""Paths and options shared by the experiment scripts."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from wealth_index.predict import resolve_model_path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Paths:
    run: Path  # training run directory under artifacts/
    labels: Path  # labels CSV with KDEPUM, quintile, OBJECTID_wi (scripts/podes_labels.py)
    boundaries: Path  # cleaned villages from `wi features boundaries`
    out: Path  # this run's output folder (git-ignored)
    cache: Path  # preprocessed fold matrices; large, safe to delete

    @property
    def figures(self) -> Path:
        return self.out / "figures"


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", type=Path, help="run directory (default: artifacts/LATEST)")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--labels", type=Path, default=ROOT / "data_inside" / "labels_podes.csv")
    parser.add_argument(
        "--boundaries", type=Path, default=ROOT / "data" / "interim" / "boundaries.parquet"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "output",
        help="outputs go to <out>/<run id>/",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="fold cache (default: $WI_EXPERIMENT_CACHE or <out>/<run id>/cache)",
    )


def paths(args: argparse.Namespace) -> Paths:
    run = args.run or resolve_model_path(args.artifacts).parent
    if not (run / "metrics.json").exists():
        raise SystemExit(f"{run} is not a complete training run")
    if not args.labels.exists():
        raise SystemExit(f"{args.labels} not found; build it with scripts/podes_labels.py")
    out = args.out / run.name
    env = os.environ.get("WI_EXPERIMENT_CACHE")
    cache = args.cache or (Path(env) / run.name if env else out / "cache")
    out.mkdir(parents=True, exist_ok=True)
    return Paths(run=run, labels=args.labels, boundaries=args.boundaries, out=out, cache=cache)

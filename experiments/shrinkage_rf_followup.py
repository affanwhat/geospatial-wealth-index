"""Follow-up to shrinkage_experiments.py: the logit target and quantile mapping on the RF.

Part B tested both only on LightGBM. Here they wrap each seed's RF baseline (B0: the run's settings
on seed 42, the re-tuned thesis grid on seeds 43-45), so run shrinkage_experiments.py first.

    python experiments/shrinkage_rf_followup.py    # same options as shrinkage_experiments.py

Outputs go next to Part B's (Podes-derived: never commit or publish): shrinkage_rf_followup.csv,
and a section appended to shrinkage_summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import shrinkage_experiments as sx  # noqa: E402

SEEDS = (sx.SEED, *sx.ROBUST_SEEDS)
VARIANTS = [
    sx.Variant("R3a", "RF + logit target", "", base="B0", target="logit"),
    sx.Variant("R4c", "RF + quantile mapping", "", base="B0", calibrate="quantile"),
    sx.Variant(
        "R34", "RF + logit + quantile mapping", "", base="B0", target="logit", calibrate="quantile"
    ),
]
COMPARE = ["B3a", "B4c"]  # the LightGBM versions, seed 42 only


def load(name: str, seed: int) -> dict:
    path = sx.RUNS / f"{name}_s{seed}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found; run shrinkage_experiments.py first")
    return json.loads(path.read_text())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sx.add_args(parser)
    sx.configure(parser.parse_args())
    out = sx.OUT / "shrinkage_rf_followup.csv"
    d = sx.load_data()
    rows = []
    for seed in SEEDS:
        split = sx.make_split(d, seed)
        done = {"B0": load("B0", seed)}
        print(f"seed {seed}")
        rows.append(sx.row(done["B0"]))
        for v in VARIANTS:
            rows.append(sx.row(sx.run_variant(v, d, split, done, force=False)))
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)

    b0 = load("B0", sx.SEED)
    s42 = [load(n, sx.SEED) for n in ["B0", *[v.name for v in VARIANTS], *COMPARE]]
    first = pd.DataFrame([sx.row(r) for r in s42]).set_index("variant")
    first["passes_rule"] = [sx.passes(r, b0) for r in s42]
    first = first[
        [
            "label",
            "cv_rmse",
            "test_rmse",
            "cv_q1_bias",
            "cv_q5_bias",
            "cv_slope",
            "cv_q1_recall",
            "passes_rule",
        ]
    ].round(3)

    cols = ["test_rmse", "test_q1_bias", "test_q5_bias", "test_slope", "test_q1_recall"]
    agg = table.groupby("variant")[cols].agg(["mean", "std"]).round(3)
    print("\nseed 42 (CV columns: out-of-fold over training villages)\n" + first.to_string())
    print(f"\ntest set, mean and std over seeds {SEEDS}\n" + agg.to_string())

    with sx.SUMMARY.open("a", encoding="utf-8") as f:
        f.write(
            "\n## Follow-up: logit target and quantile mapping on the RF\n\n"
            "Seed 42:\n\n```\n" + first.to_string() + "\n```\n\n"
            f"Test set over seeds {SEEDS}:\n\n```\n" + agg.to_string() + "\n```\n"
        )
    print(f"\nWrote {out} and appended to {sx.SUMMARY}")


if __name__ == "__main__":
    main()

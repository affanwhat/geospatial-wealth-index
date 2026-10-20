# Experiments

One-off analyses of a training run. They read the package (`wealth_index`) but are not part of
it: nothing here changes how `wi train` or `wi predict` behave.

| Script | Question |
|---|---|
| `part_a_errors.py` | Where does the model err, and are the errors clustered in space? |
| `shrinkage_experiments.py` | Can another model, target transform, recalibration, tail weighting or spatial feature reduce the pull towards the middle? |
| `shrinkage_rf_followup.py` | Do the two most promising fixes (logit target, quantile mapping) also work on the random forest? |

## Running

They need the `geo` extra, a training run from `wi train` with `search.method: random`, and the
labels file from `scripts/podes_labels.py`:

```bash
python experiments/part_a_errors.py              # ~3 min
python experiments/shrinkage_experiments.py      # ~20 min; resumes, finished variants are reused
python experiments/shrinkage_rf_followup.py      # ~5 min, after the one above
```

Common options (`--help` on any script): `--run` (default `artifacts/LATEST`), `--labels`,
`--boundaries`, `--out` and `--cache` (or the `WI_EXPERIMENT_CACHE` environment variable).
`part_a_errors.py` also takes `--thesis`, the thesis' own predictions; without it, the
comparison with the thesis model is skipped.

Outputs go to `experiments/output/<run id>/`, which is git-ignored. They contain the Podes-derived
wealth index per village: **never commit or publish them.** This file reports aggregates only.

## Results so far

Run `20261018-184319`: random forest, 89 features, top 25 kept, spatial holdout of 415 villages
(5 km blocks), test RMSE 8.61.

### Part A: the errors

- **The model squeezes towards the middle.** Out of fold, the poorest quintile is overpredicted
  by +5.5 on average and the richest underpredicted by −7.3. 58% of villages land in the right
  quintile.
- **Errors cluster in space.** Moran's I of the residuals, with 8 nearest neighbours, is 0.19 over
  all villages and 0.28 within the poorest quintile (p = 0.001, 999 permutations).
- **Where:** overpredicted poor villages sit on the peri-urban fringe of Kab. Bogor, Kab. Tangerang
  and Kab. Bekasi. Rich villages are underpredicted across whole districts, most in Jakarta Utara
  (−11.6 on average).
- **Why:** the overpredicted poor villages are more lit and connected than well-predicted poor
  ones, but far less than richer villages. The features place them between rural poor and urban
  villages, and the forest averages them towards the middle.
- **Not specific to this model:** on the thesis' own test villages, its residuals and ours
  correlate at 0.84.

### Part B: fixes

Judged by the rule from the plan: spatial CV RMSE at most 0.15 worse than the baseline, and mean
error in the poorest and richest quintiles cut by at least a third. **No variant passes.**

Test set, mean over four holdout seeds (42–45), each re-tuned:

| | Test RMSE | Q1 bias | Q5 bias | Q1 villages predicted as Q1 |
|---|---|---|---|---|
| Random forest (baseline) | 8.21 | +5.3 | −7.1 | 65% |
| + logit target | 8.31 | +3.8 | −6.5 | 71% |
| + quantile mapping | 8.75 | +2.3 | −4.4 | 74% |

- **No model beats the random forest on RMSE.** Less smoothing, HistGradientBoosting and LightGBM
  all land within noise on one split and a little worse across seeds.
- **Spatial context did not help** despite the clustering: neighbour feature means, neighbour WI
  (computed so no test labels leak) and a district indicator changed little.
- **Quantile mapping** after prediction is the only fix that substantially reduces the bias in
  both tails, at a cost of about 0.55 RMSE.
- **A logit target** costs about 0.1 RMSE and cuts the poorest quintile's bias by about 28%.

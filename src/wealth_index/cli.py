"""Command-line interface: `wi make-sample`, `wi train`, `wi predict`, `wi features ...`."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

app = typer.Typer(
    help="Wealth index ML pipeline for the Jakarta Metropolitan Area.", no_args_is_help=True
)


@app.callback()
def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@app.command("make-sample")
def make_sample(
    output: Path = typer.Option(Path("data/sample/dataset.csv"), help="Where to write the CSV."),
    n: int = typer.Option(1200, help="Number of synthetic villages."),
    seed: int = typer.Option(42),
) -> None:
    """Generate a synthetic dataset with the real schema (no private data needed)."""
    from wealth_index.synthetic import generate_sample

    output.parent.mkdir(parents=True, exist_ok=True)
    generate_sample(n=n, seed=seed).to_csv(output, index=False)
    typer.echo(f"Wrote {n} rows to {output}")


@app.command()
def train(
    config: Path = typer.Option(Path("configs/train.yaml"), help="Training config YAML."),
) -> None:
    """Tune, evaluate, and save a model run under the configured output directory."""
    from wealth_index.config import load_config
    from wealth_index.train import train as run_training

    run_dir = run_training(load_config(config))
    typer.echo(f"Run saved to {run_dir}")


@app.command()
def predict(
    input: Path = typer.Option(..., help="CSV of village features to score."),
    output: Path = typer.Option(Path("data/predictions/wealth_index.csv")),
    model: Path | None = typer.Option(None, help="model.joblib; defaults to the latest run."),
    artifacts_dir: Path = typer.Option(Path("artifacts")),
    sep: str = typer.Option(","),
    id_col: str = typer.Option("id"),
) -> None:
    """Score a feature table with a trained model."""
    from wealth_index.predict import predict as run_predict
    from wealth_index.predict import resolve_model_path

    model_path = model or resolve_model_path(artifacts_dir)
    out = run_predict(model_path, input, output, sep=sep, id_col=id_col)
    typer.echo(f"Scored {len(out)} villages with {model_path} -> {output}")


@app.command("export-runs")
def export_runs(
    artifacts_dir: Path = typer.Option(Path("artifacts")),
    output: Path = typer.Option(Path("artifacts/report"), help="Folder for the CSV tables."),
) -> None:
    """Collect all training runs into CSV tables (runs, metrics, predictions, importances)."""
    from wealth_index.report import export_runs as run_export

    for path in run_export(artifacts_dir, output):
        typer.echo(f"Wrote {path}")


@app.command()
def report(
    artifacts_dir: Path = typer.Option(Path("artifacts")),
    output: Path = typer.Option(Path("artifacts/report/index.html"), help="HTML file to write."),
) -> None:
    """Build a self-contained HTML dashboard of all training runs."""
    from wealth_index.report import build_html_report

    typer.echo(f"Wrote {build_html_report(artifacts_dir, output)}")


labels_app = typer.Typer(help="Phase 3: wealth index labels.", no_args_is_help=True)
app.add_typer(labels_app, name="labels")


@labels_app.command("join")
def labels_join(
    labels: Path = typer.Option(..., help="CSV with a village id and a wealth index column."),
    features: Path = typer.Option(
        Path("data/processed/features.csv"), help="From `wi features build`."
    ),
    output: Path = typer.Option(
        Path("data/processed/dataset.csv"), help="Training dataset to write."
    ),
    label_col: str = typer.Option("wi", help="Wealth index column in the labels file."),
    label_id_col: str = typer.Option("KDEPUM", help="Village id column in the labels file."),
) -> None:
    """Join labels to the feature table into the dataset `wi train` reads."""
    from wealth_index.labels import run_join

    report = run_join(features, labels, output, label_col=label_col, label_id_col=label_id_col)
    typer.echo(
        f"Wrote {output}: {report.matched} of {report.features} villages labelled"
        f" ({len(report.unlabelled)} without a label, {len(report.unknown)} unknown label ids)"
    )


features_app = typer.Typer(
    help='Phase 2: build village features from raw geodata (needs `pip install -e ".[geo]"`).',
    no_args_is_help=True,
)
app.add_typer(features_app, name="features")

FEATURES_CONFIG = typer.Option(
    Path("configs/features.yaml"), "--config", help="Feature config YAML."
)
FORCE = typer.Option(False, "--force", help="Rerun even if inputs and parameters are unchanged.")

STAGE_HELP = {
    "boundaries": "Repair village polygons; write villages.csv with area and lon/lat.",
    "osm": "POI and road features from the OSM extract.",
    "s2": "Spectral index statistics from the Sentinel-2 composite.",
    "texture": "GLCM texture per village from the Sentinel-2 composite.",
    "ntl": "Night-time light statistics: destriped SDGSAT-1, or VIIRS from GEE as a fallback.",
    "green": "Green exposure around settlements (needs s2 and ntl).",
}


def _offer_viirs(cfg) -> None:
    """With no SDGSAT-1 scenes, ask whether to use VIIRS night lights from GEE instead."""
    from wealth_index.geo.ntl import has_scenes

    if cfg.ntl.source != "sdgsat" or has_scenes(cfg) or not sys.stdin.isatty():
        return
    if typer.confirm(
        f"No SDGSAT-1 scenes match {cfg.paths.sdgsat}.\n"
        "Use VIIRS night lights from Google Earth Engine instead? They are ~460 m, much "
        "coarser than SDGSAT-1's 10 m, so NTL features become a stand-in",
        default=False,
    ):
        cfg.ntl.source = "viirs"
        typer.echo("Using VIIRS. Set `ntl.source: viirs` in the config to skip this question.")


def _stage_command(stage: str):
    def command(config: Path = FEATURES_CONFIG, force: bool = FORCE) -> None:
        from wealth_index.geo.build import run_stage
        from wealth_index.geo.config import load_feature_config

        cfg = load_feature_config(config)
        if stage == "ntl":
            _offer_viirs(cfg)
        typer.echo(f"Stage {stage}: {run_stage(cfg, stage, force=force)}")

    command.__doc__ = STAGE_HELP[stage]
    return command


for _stage in STAGE_HELP:
    features_app.command(_stage)(_stage_command(_stage))


@features_app.command("s2-export")
def s2_export(config: Path = FEATURES_CONFIG) -> None:
    """Submit the cloud-masked Sentinel-2 composite export on Google Earth Engine."""
    from wealth_index.geo.boundaries import read_villages
    from wealth_index.geo.config import load_feature_config
    from wealth_index.geo.s2 import export_composite

    cfg = load_feature_config(config)
    task = export_composite(cfg, read_villages(cfg))
    folder = Path(cfg.paths.s2_composite).parent
    typer.echo(f"GEE task {task.id} submitted; download the result into {folder}")


@features_app.command("build")
def features_build(
    config: Path = FEATURES_CONFIG,
    partial: bool = typer.Option(False, "--partial", help="Join only the stages that have run."),
) -> None:
    """Join and validate the stage outputs into data/processed/features.csv."""
    from wealth_index.geo.build import build
    from wealth_index.geo.config import load_feature_config

    typer.echo(f"Wrote {build(load_feature_config(config), partial=partial)}")


@features_app.command("all")
def features_all(config: Path = FEATURES_CONFIG, force: bool = FORCE) -> None:
    """Run every stage whose inputs exist, then build (partial if any were skipped)."""
    from wealth_index.geo.build import run_all
    from wealth_index.geo.config import load_feature_config

    cfg = load_feature_config(config)
    _offer_viirs(cfg)
    typer.echo(f"Wrote {run_all(cfg, force=force)}")


if __name__ == "__main__":
    app()

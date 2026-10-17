"""Command-line interface: `wi make-sample`, `wi train`, `wi predict`."""

from __future__ import annotations

import logging
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


if __name__ == "__main__":
    app()

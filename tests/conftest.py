import pytest

from wealth_index.config import Config, DataConfig, FeatureConfig, SearchConfig, SplitConfig
from wealth_index.synthetic import generate_sample


@pytest.fixture
def sample_csv(tmp_path):
    path = tmp_path / "dataset.csv"
    generate_sample(n=300, seed=0).to_csv(path, index=False)
    return path


@pytest.fixture
def small_config(sample_csv, tmp_path):
    return Config(
        data=DataConfig(path=str(sample_csv)),
        split=SplitConfig(scheme="spatial", block_size_km=10.0, cv_folds=3),
        features=FeatureConfig(top_k=8, pfi_repeats=2, pfi_estimators=20),
        search=SearchConfig(
            n_iter=2,
            param_distributions={"model__n_estimators": [20, 30], "model__max_depth": [5, 10]},
        ),
        n_jobs=1,
        output_dir=str(tmp_path / "artifacts"),
    )

import numpy as np
import pandas as pd

from wealth_index.features import CorrelationFilter, PermutationImportanceSelector, TexturePCA


def test_correlation_filter_drops_later_duplicate():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = pd.DataFrame({"a": a, "a_copy": a * 2 + 0.01, "b": rng.normal(size=200)})
    f = CorrelationFilter(threshold=0.9).set_output(transform="pandas").fit(X)
    assert list(f.transform(X).columns) == ["a", "b"]


def test_correlation_filter_disabled_keeps_all():
    X = pd.DataFrame({"a": [1.0, 2, 3], "b": [2.0, 4, 6]})
    assert CorrelationFilter(threshold=None).fit(X).get_support().all()


def test_pfi_selector_keeps_informative_features():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 6)), columns=[f"f{i}" for i in range(6)])
    y = 3 * X["f1"] + 2 * X["f4"] + rng.normal(scale=0.1, size=400)
    sel = PermutationImportanceSelector(top_k=2, n_estimators=50, random_state=0).fit(X, y)
    assert set(sel.get_feature_names_out()) == {"f1", "f4"}


def test_pfi_selector_caps_top_k_at_feature_count():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    sel = PermutationImportanceSelector(top_k=10, n_estimators=10, random_state=0).fit(X, X[:, 0])
    assert sel.get_support().sum() == 3


def test_texture_pca_replaces_texture_columns_only():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 2))
    X = pd.DataFrame(
        {"NTL_MEAN": rng.normal(size=200)}
        | {f"TEX_B{i}_CONTRAST": base[:, i % 2] + rng.normal(0, 0.01, 200) for i in range(6)}
    )
    pca = TexturePCA(variance=0.95).set_output(transform="pandas").fit(X)
    out = pca.transform(X)
    assert list(out.columns) == ["NTL_MEAN", "TEXPC1", "TEXPC2"]  # six collinear columns -> 2 PCs
    assert list(pca.get_feature_names_out()) == list(out.columns)
    assert (out["NTL_MEAN"] == X["NTL_MEAN"]).all()


def test_texture_pca_without_texture_columns_is_a_no_op():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert TexturePCA().fit(X).transform(X).equals(X)


def test_synthetic_sample_follows_the_feature_catalogue():
    from wealth_index.geo.catalogue import catalogue
    from wealth_index.geo.config import FeatureBuildConfig
    from wealth_index.synthetic import generate_sample

    df = generate_sample(n=50, seed=0)
    expected = [c for cols in catalogue(FeatureBuildConfig()).values() for c in cols]
    assert list(df.columns) == ["id", "lon", "lat", *expected, "wealth_index"]
    assert len(expected) == 89
    assert df[expected].notna().all().all()

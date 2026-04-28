"""Unit tests for synthselector."""

import numpy as np
import pandas as pd
import pytest

from synthselector.metrics import compute_gmean, compute_classification_metrics
from synthselector.classifiers import get_default_classifiers
from synthselector.evaluator import SyntheticEvaluator
from synthselector.density import preprocess, density_score, rank_by_density, DensityEvaluator


# ── fixtures ────────────────────────────────────────────────────────────


def _make_binary_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["label"] = y
    return df


@pytest.fixture
def train_df():
    return _make_binary_data(200, seed=42)


@pytest.fixture
def test_df():
    return _make_binary_data(80, seed=99)


@pytest.fixture
def synth_df():
    return _make_binary_data(100, seed=7)


# ── metrics tests ───────────────────────────────────────────────────────


class TestComputeGmean:
    def test_perfect_binary(self):
        y = np.array([0, 0, 1, 1])
        assert compute_gmean(y, y) == pytest.approx(1.0)

    def test_all_wrong_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        assert compute_gmean(y_true, y_pred) == pytest.approx(0.0)

    def test_multiclass(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        assert compute_gmean(y_true, y_pred) == pytest.approx(1.0)


class TestClassificationMetrics:
    def test_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        result = compute_classification_metrics(y_true, y_pred)
        assert set(result) == {"F1", "AUC", "GMean"}

    def test_auc_with_prob(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        result = compute_classification_metrics(y_true, y_pred, y_prob)
        assert result["AUC"] == pytest.approx(1.0)


# ── classifiers tests ──────────────────────────────────────────────────


class TestGetDefaultClassifiers:
    def test_returns_dict(self):
        models = get_default_classifiers()
        assert isinstance(models, dict)
        assert len(models) >= 4  # SVM, KNN, DT, RF at minimum

    def test_all_have_fit(self):
        for name, model in get_default_classifiers().items():
            assert hasattr(model, "fit"), f"{name} missing .fit()"


# ── evaluator tests ─────────────────────────────────────────────────────


class TestSyntheticEvaluator:
    def test_add_and_run(self, train_df, test_df, synth_df):
        ev = SyntheticEvaluator(train_df, test_df)
        ev.add("fake", synth_df)
        result = ev.run()

        assert len(result.model_dfs) >= 4
        summary = result.summary()
        assert "Avg_WeightedScore" in summary.columns

    def test_add_many(self, train_df, test_df, synth_df):
        ev = SyntheticEvaluator(train_df, test_df)
        ev.add_many({"A": synth_df, "B": synth_df})
        result = ev.run()
        # Each model should have 2 synthetic entries
        for df in result.model_dfs.values():
            assert len(df) == 2

    def test_no_synthetic_raises(self, train_df, test_df):
        ev = SyntheticEvaluator(train_df, test_df)
        with pytest.raises(ValueError, match="No synthetic"):
            ev.run()

    def test_best_synthetic(self, train_df, test_df, synth_df):
        ev = SyntheticEvaluator(train_df, test_df)
        ev.add("only", synth_df)
        result = ev.run()
        for model_name in result.model_dfs:
            assert result.best_synthetic(model_name) == "only"

    def test_custom_weights(self, train_df, test_df, synth_df):
        ev = SyntheticEvaluator(
            train_df, test_df,
            weights={"F1": 1.0, "AUC": 0.0, "GMean": 0.0},
        )
        ev.add("s", synth_df)
        result = ev.run()
        summary = result.summary()
        assert summary["Avg_WeightedScore"].iloc[0] > 0


# ── density tests ───────────────────────────────────────────────────────


class TestPreprocess:
    def test_returns_arrays(self, train_df, synth_df):
        X_real, X_syn = preprocess(train_df, {"A": synth_df})
        assert isinstance(X_real, np.ndarray)
        assert "A" in X_syn
        assert X_real.shape[1] == X_syn["A"].shape[1]

    def test_multiple_synthetics(self, train_df, synth_df):
        X_real, X_syn = preprocess(train_df, {"A": synth_df, "B": synth_df})
        assert len(X_syn) == 2


class TestDensityScore:
    def test_identical_data_high_score(self, train_df):
        X_real, X_syn = preprocess(train_df, {"copy": train_df})
        scores = density_score(X_real, X_syn, k=5)
        # Identical data should have a high density score
        assert scores["copy"] > 0

    def test_noisy_data_lower_score(self, train_df):
        rng = np.random.RandomState(0)
        noisy = train_df.copy()
        feat_cols = [c for c in noisy.columns if c != "label"]
        noisy[feat_cols] += rng.normal(0, 10, size=(len(noisy), len(feat_cols)))
        X_real, X_syn = preprocess(train_df, {"clean": train_df, "noisy": noisy})
        scores = density_score(X_real, X_syn, k=5)
        assert scores["clean"] >= scores["noisy"]


class TestRankByDensity:
    def test_ranking_order(self):
        scores = {"A": 0.8, "B": 0.5, "C": 0.9}
        ranking = rank_by_density(scores)
        assert ranking.iloc[0]["Generator"] == "C"
        assert ranking.iloc[-1]["Generator"] == "B"
        assert ranking.index.name == "Rank"
        assert ranking.index[0] == 1


class TestDensityEvaluator:
    def test_run(self, train_df, synth_df):
        de = DensityEvaluator(train_df, k=3)
        de.add("fake", synth_df)
        ranking = de.run()
        assert "Density_Score" in ranking.columns
        assert len(ranking) == 1

    def test_add_many(self, train_df, synth_df):
        de = DensityEvaluator(train_df, k=3)
        de.add_many({"A": synth_df, "B": synth_df})
        ranking = de.run()
        assert len(ranking) == 2

    def test_no_synthetic_raises(self, train_df):
        de = DensityEvaluator(train_df)
        with pytest.raises(ValueError, match="No synthetic"):
            de.run()


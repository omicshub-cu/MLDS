"""Core evaluator that benchmarks synthetic datasets across classifiers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .classifiers import get_default_classifiers
from .metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


class SyntheticEvaluator:
    """Evaluate and rank synthetic datasets for ML classification tasks.

    The evaluator trains each classifier on ``real_train + synthetic`` data,
    measures F1, AUC, and G-Mean on a held-out validation set, and produces
    a ranking of synthetic generators per model and overall.

    Parameters
    ----------
    train : pd.DataFrame
        Real training data.  Must contain a ``label`` column.
    test : pd.DataFrame
        Held-out validation data.  Must contain a ``label`` column.
    label_col : str, default ``"label"``
        Name of the target column.
    classifiers : dict or None
        Custom ``{name: estimator}`` mapping.  When *None* the built-in
        defaults (SVM, KNN, DT, RF, +LGB/XGB if installed) are used.
    weights : dict or None
        Metric weights for the composite score.  Defaults to
        ``{"F1": 0.4, "AUC": 0.3, "GMean": 0.3}``.
    random_state : int, default 42
        Seed forwarded to default classifiers.

    Examples
    --------
    >>> evaluator = SyntheticEvaluator(train_df, test_df)
    >>> evaluator.add("SMOTE", smote_df)
    >>> evaluator.add("CTGAN", ctgan_df)
    >>> results = evaluator.run()
    >>> print(results.summary())
    """

    _DEFAULT_WEIGHTS = {"F1": 0.4, "AUC": 0.3, "GMean": 0.3}

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        *,
        label_col: str = "label",
        classifiers: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
        random_state: int = 42,
    ) -> None:
        self.label_col = label_col
        self.train = train.copy()
        self.test = test.copy()

        self._X_val = self.test.drop(columns=[label_col]).values
        self._y_val = self.test[label_col].astype(int).values

        self.classifiers = classifiers or get_default_classifiers(random_state)
        self.weights = weights or self._DEFAULT_WEIGHTS.copy()
        self._synthetic: dict[str, pd.DataFrame] = {}

    # ── public helpers ──────────────────────────────────────────────────

    def add(self, name: str, df: pd.DataFrame) -> "SyntheticEvaluator":
        """Register a synthetic dataset for evaluation.

        Parameters
        ----------
        name : str
            Human-readable label (e.g. ``"CTGAN"``).
        df : pd.DataFrame
            Synthetic data with the same schema as ``train``.

        Returns
        -------
        self
            For method chaining.
        """
        self._synthetic[name] = df
        return self

    def add_many(self, datasets: dict[str, pd.DataFrame]) -> "SyntheticEvaluator":
        """Register multiple synthetic datasets at once.

        Parameters
        ----------
        datasets : dict[str, pd.DataFrame]
            Mapping of names to DataFrames.

        Returns
        -------
        self
        """
        for name, df in datasets.items():
            self.add(name, df)
        return self

    # ── core evaluation ─────────────────────────────────────────────────

    def _evaluate_one(self, df_syn: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Train all classifiers on train+synthetic and return per-model metrics.
        
        PHASE 1: TRAIN & MEASURE
        - Combine synthetic dataset with real training data
        - Train all classifiers on the combined set
        - Predict on held-out test set
        - Compute F1, AUC, and G-Mean for each classifier
        """
        combined = pd.concat([self.train, df_syn], ignore_index=True)
        combined = combined.apply(pd.to_numeric, errors="coerce").fillna(0)

        X_train = combined.drop(columns=[self.label_col]).values
        y_train = combined[self.label_col].astype(int).values

        results: dict[str, dict[str, float]] = {}

        for model_name, model in self.classifiers.items():
            # Clone estimator so parallel runs are safe
            from sklearn.base import clone

            clf = clone(model)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(self._X_val)
            y_prob = (
                clf.predict_proba(self._X_val)
                if hasattr(clf, "predict_proba")
                else None
            )
            results[model_name] = compute_classification_metrics(
                self._y_val, y_pred, y_prob
            )

        return results

    def run(self) -> "EvaluationResult":
        """Execute the full evaluation pipeline.

        Returns
        -------
        EvaluationResult
            Object with per-model DataFrames, rankings, and a summary table.

        Raises
        ------
        ValueError
            If no synthetic datasets have been registered.
        """
        if not self._synthetic:
            raise ValueError(
                "No synthetic datasets registered. Use .add() or .add_many() first."
            )

        # {model_name: {syn_name: {metric: value}}}
        all_results: dict[str, dict[str, dict[str, float]]] = {}

        for syn_name, df_syn in self._synthetic.items():
            logger.info("Evaluating: %s …", syn_name)
            per_model = self._evaluate_one(df_syn)
            for model_name, metrics in per_model.items():
                all_results.setdefault(model_name, {})[syn_name] = metrics

        return EvaluationResult(all_results, self.weights)


class EvaluationResult:
    """Container for evaluation output with ranking and formatting helpers.

    Parameters
    ----------
    raw : dict
        Nested ``{model: {synthetic: {metric: value}}}`` dictionary.
    weights : dict
        Metric weights used for the composite score.
    """

    def __init__(
        self,
        raw: dict[str, dict[str, dict[str, float]]],
        weights: dict[str, float],
    ) -> None:
        self.raw = raw
        self.weights = weights
        self.model_dfs: dict[str, pd.DataFrame] = {}
        self._build()

    def _build(self) -> None:
        """Build model DataFrames with normalized metrics and weighted scores.
        
        PHASE 2: NORMALISE & SCORE
        - Min-max normalise each metric within each classifier (scale to 0–1)
        - Compute weighted composite score per generator (0.4 F1 + 0.3 AUC + 0.3 G-Mean)
        - Rank synthetic generators per classifier by composite score
        """
        w_f1 = self.weights.get("F1", 0.4)
        w_auc = self.weights.get("AUC", 0.3)
        w_gmean = self.weights.get("GMean", 0.3)

        for model_name, syn_metrics in self.raw.items():
            df = pd.DataFrame(syn_metrics).T  # rows = synthetic datasets

            # Min-Max normalise within this model's results
            for col in ("F1", "AUC", "GMean"):
                min_val, max_val = df[col].min(), df[col].max()
                if max_val - min_val == 0:
                    df[f"{col}_norm"] = 0.0
                else:
                    df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val)

            df["WeightedScore"] = (
                w_f1 * df["F1"] + w_auc * df["AUC"] + w_gmean * df["GMean"]
            )
            df = df.sort_values("WeightedScore", ascending=False)
            self.model_dfs[model_name] = df

    # ── ranking helpers ─────────────────────────────────────────────────

    def top_n(self, model: str, n: int = 3) -> pd.DataFrame:
        """Return the top-*n* synthetic datasets for a given model.

        Parameters
        ----------
        model : str
            Classifier name (e.g. ``"RF"``).
        n : int, default 3
            Number of top entries to return.

        Returns
        -------
        pd.DataFrame
        """
        return self.model_dfs[model][["F1", "AUC", "GMean", "WeightedScore"]].head(n)

    def best_synthetic(self, model: str) -> str:
        """Return the name of the best synthetic dataset for *model*."""
        return str(self.model_dfs[model].index[0])

    # ── summary table ───────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        """Average weighted score per classifier across all synthetic datasets.
        
        PHASE 3: AGGREGATE & SELECT
        - Average composite scores across all classifiers for each generator
        - Rank generators by their cross-classifier average score
        - Select the best-performing synthetic dataset (highest average across all classifiers)

        Returns
        -------
        pd.DataFrame
            Indexed by model, sorted descending by ``Avg_WeightedScore``.
        """
        rows = []
        for model_name, df in self.model_dfs.items():
            rows.append(
                {
                    "Model": model_name,
                    "Avg_F1": df["F1"].mean(),
                    "Avg_AUC": df["AUC"].mean(),
                    "Avg_GMean": df["GMean"].mean(),
                    "Avg_WeightedScore": df["WeightedScore"].mean(),
                }
            )
        return (
            pd.DataFrame(rows)
            .set_index("Model")
            .sort_values("Avg_WeightedScore", ascending=False)
        )

    def aggregate_rankings(self) -> pd.DataFrame:
        """Aggregate per-classifier rankings into a cross-model summary.
        
        PHASE 3: AGGREGATE & SELECT (Detailed)
        - Collect ranks per synthetic dataset within each classifier
        - Compute mean rank (lower is better) across all classifiers
        - Compute mean weighted score and top-3 frequency
        - Select top-3 synthetic datasets by mean rank

        Returns
        -------
        pd.DataFrame
            Indexed by synthetic dataset name, columns:
            - MeanRank: Average rank across classifiers
            - MeanWeightedScore: Average weighted score across classifiers
            - TimesTop3: Count of classifiers where this dataset ranked in top-3
            Sorted by MeanRank (ascending).
        """
        # Collect ranks per model
        rank_records = []
        for model_name, df in self.model_dfs.items():
            df_sorted = df.sort_values("WeightedScore", ascending=False)
            for rank, syn_name in enumerate(df_sorted.index, start=1):
                rank_records.append({
                    "Synthetic": syn_name,
                    "Model": model_name,
                    "Rank": rank,
                    "WeightedScore": df_sorted.loc[syn_name, "WeightedScore"],
                })

        rank_df = pd.DataFrame(rank_records)

        # Aggregate: mean rank and mean weighted score across all classifiers
        agg = rank_df.groupby("Synthetic").agg(
            MeanRank=("Rank", "mean"),
            MeanWeightedScore=("WeightedScore", "mean"),
            TimesTop3=("Rank", lambda x: (x <= 3).sum()),
        ).sort_values("MeanRank")

        return agg

    # ── display ─────────────────────────────────────────────────────────

    def print_report(self, top_n: int = 3) -> None:
        """Print a human-readable ranking report to stdout.

        Parameters
        ----------
        top_n : int, default 3
            How many synthetic datasets to show per classifier.
        """
        medals = ["1st", "2nd", "3rd", "4th", "5th"]

        #print("\n" + "=" * 60)
        #print("  TOP SYNTHETIC DATASETS — PER ML MODEL")
        #print("=" * 60)

        #for model_name, df in self.model_dfs.items():
        #   print(f"\n  Model: {model_name}")
        #    print("-" * 50)
        #    top_df = df[["F1", "AUC", "GMean", "WeightedScore"]].head(top_n)
        #    print(top_df.round(4).to_string())
        #    for rank, idx in enumerate(df.index[:top_n]):
        #        label = medals[rank] if rank < len(medals) else f"{rank+1}th"
        #        print(f"    {label}: {idx}")

        # Synthetic data generators rank top 3
        print("\n" + "=" * 60)
        print("SYNTHETIC DATA GENERAOTRS RANK")
        print("=" * 60)
        agg = self.aggregate_rankings()
        print(agg.round(4).to_string())
        top_3_synthetic = agg.index[:3].tolist()
        print(f"\n  ✅ Selected top-3 synthetic datasets: {top_3_synthetic}")

    def __repr__(self) -> str:  # pragma: no cover
        n_models = len(self.model_dfs)
        n_syn = len(next(iter(self.model_dfs.values()))) if self.model_dfs else 0
        return f"<EvaluationResult models={n_models} synthetics={n_syn}>"

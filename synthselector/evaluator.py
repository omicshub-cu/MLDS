"""Core evaluation logic for ranking synthetic datasets.

The key change in this version: the evaluator splits the training data
into train/validation internally using ``train_test_split`` with
stratification, rather than requiring a separate test set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from .classifiers import get_default_classifiers
from .metrics import compute_classification_metrics


@dataclass
class EvaluationResult:
    """Container for per-model ranking tables and summary statistics."""

    model_dfs: dict[str, pd.DataFrame]
    weights: dict[str, float]

    def summary(self) -> pd.DataFrame:
        """Average weighted score per ML model across all synthetic datasets."""
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

    def best_synthetic(self, model_name: str) -> str:
        """Return the name of the best synthetic dataset for *model_name*."""
        return self.model_dfs[model_name].index[0]

    def print_report(self, top_n: int = 3) -> None:
        """Print a human-readable ranking report to stdout."""
        medals = {0: "🥇", 1: "🥈", 2: "🥉"}

        print("\n" + "=" * 60)
        print("🏆  TOP-{} SYNTHETIC DATASETS — PER ML MODEL".format(top_n))
        print("=" * 60)

        for model_name, df in self.model_dfs.items():
            print(f"\n📌  Model: {model_name}")
            print("-" * 50)
            top_df = df[["F1", "AUC", "GMean", "WeightedScore"]].head(top_n)
            print(top_df.round(4).to_string())
            for i in range(min(top_n, len(df))):
                medal = medals.get(i, f"  #{i+1}")
                print(f"  {medal} {'Best' if i == 0 else f'{i+1}nd' if i == 1 else f'{i+1}rd':<6}: {df.index[i]}")

        summary = self.summary()
        print("\n" + "=" * 60)
        print("📊  SUMMARY — AVERAGE WEIGHTED SCORE PER ML MODEL")
        print("=" * 60)
        print(summary.round(4).to_string())
        print(
            f"\n🏆  Best overall ML model: {summary.index[0]}"
            f"  (Avg WeightedScore = {summary['Avg_WeightedScore'].iloc[0]:.4f})"
        )


class SyntheticEvaluator:
    """Evaluate synthetic datasets by training classifiers on combined data.

    Unlike the previous version, this evaluator does **not** require a
    separate test set.  Instead it splits the provided training data into
    an 80/20 stratified train/validation partition internally.

    Parameters
    ----------
    train : pd.DataFrame
        Training data **with** a label column.
    label_col : str
        Name of the target column (default ``"label"``).
    classifiers : dict or None
        Custom classifier panel.  If *None*, uses the default 6-model panel.
    weights : dict or None
        Metric weights for the composite score.
        Default ``{"F1": 0.4, "AUC": 0.3, "GMean": 0.3}``.
    test_size : float
        Fraction of *train* to hold out for validation (default 0.2).
    random_state : int
        Seed for the train/validation split (default 42).
    """

    def __init__(
        self,
        train: pd.DataFrame,
        label_col: str = "label",
        *,
        classifiers: dict | None = None,
        weights: dict | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.label_col = label_col
        self._classifiers = classifiers
        self.weights = weights or {"F1": 0.4, "AUC": 0.3, "GMean": 0.3}
        self._synthetic: dict[str, pd.DataFrame] = {}

        # Internal train/val split (stratified)
        self._train_split, self._val_split = train_test_split(
            train,
            test_size=test_size,
            random_state=random_state,
            stratify=train[label_col],
        )
        self._X_val = self._val_split.drop(columns=[label_col]).values
        self._y_val = self._val_split[label_col].astype(int).values

    # -- registration -------------------------------------------------

    def add(self, name: str, df: pd.DataFrame) -> "SyntheticEvaluator":
        """Register a synthetic dataset for evaluation."""
        self._synthetic[name] = df
        return self

    def add_many(self, datasets: dict[str, pd.DataFrame]) -> "SyntheticEvaluator":
        """Register multiple synthetic datasets at once."""
        self._synthetic.update(datasets)
        return self

    # -- evaluation ---------------------------------------------------

    def _get_classifiers(self) -> dict:
        if self._classifiers is not None:
            return {k: clone(v) for k, v in self._classifiers.items()}
        return get_default_classifiers()

    def _evaluate_one(self, df_syn: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Train all classifiers on (train_split + synthetic) and evaluate on val."""
        combined = pd.concat([self._train_split, df_syn], ignore_index=True)
        combined = combined.apply(pd.to_numeric, errors="coerce").fillna(0)

        X_train = combined.drop(columns=[self.label_col]).values
        y_train = combined[self.label_col].astype(int).values

        models = self._get_classifiers()
        per_model: dict[str, dict[str, float]] = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(self._X_val)

            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(self._X_val)

            per_model[name] = compute_classification_metrics(
                self._y_val, y_pred, y_prob
            )

        return per_model

    def run(self) -> EvaluationResult:
        """Run the full evaluation pipeline.

        Returns
        -------
        EvaluationResult
            Object containing per-model DataFrames and a summary method.

        Raises
        ------
        ValueError
            If no synthetic datasets have been registered.
        """
        if not self._synthetic:
            raise ValueError(
                "No synthetic datasets registered. Use .add() or .add_many() first."
            )

        # Collect raw metrics: all_results[model_name][syn_name] = {F1, AUC, GMean}
        all_results: dict[str, dict] = {}
        for syn_name, df_syn in self._synthetic.items():
            print(f"Evaluating: {syn_name} ...")
            per_model = self._evaluate_one(df_syn)
            for model_name, metrics in per_model.items():
                all_results.setdefault(model_name, {})[syn_name] = metrics

        # Build per-model DataFrames, normalise, compute weighted score
        w_f1 = self.weights["F1"]
        w_auc = self.weights["AUC"]
        w_gmean = self.weights["GMean"]

        model_dfs: dict[str, pd.DataFrame] = {}
        for model_name, syn_metrics in all_results.items():
            df = pd.DataFrame(syn_metrics).T  # rows = synthetic datasets

            # Min-Max normalise (kept for reference/analysis)
            for col in ("F1", "AUC", "GMean"):
                min_val, max_val = df[col].min(), df[col].max()
                if max_val - min_val == 0:
                    df[col + "_norm"] = 0.0
                else:
                    df[col + "_norm"] = (df[col] - min_val) / (max_val - min_val)

            # Weighted score uses raw values
            df["WeightedScore"] = (
                w_f1 * df["F1"] + w_auc * df["AUC"] + w_gmean * df["GMean"]
            )
            df = df.sort_values("WeightedScore", ascending=False)
            model_dfs[model_name] = df

        return EvaluationResult(model_dfs=model_dfs, weights=self.weights)

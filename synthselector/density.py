"""Density-based evaluation of synthetic data quality.

This module implements a density score that measures how well synthetic
samples cover the manifold of real data.  For each real data point *X_i*
a ball of radius equal to its *k*-th nearest-neighbour distance (among
real data) is defined.  The score counts how many synthetic points fall
inside those balls, normalised by *k × M* (where *M* is the number of
synthetic samples).

A higher density score means the synthetic data concentrates in regions
of high real-data density — useful for spotting mode collapse or
under-coverage.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ── preprocessing ───────────────────────────────────────────────────────


def preprocess(
    real_df: pd.DataFrame,
    syn_dict: dict[str, pd.DataFrame],
    target_col: str = "label",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Encode and scale real + synthetic data into a shared feature space.

    Categorical columns are label-encoded using classes learned from the
    real data.  Unseen categories in synthetic data are mapped to ``-1``.
    All features are then standard-scaled (zero mean, unit variance) using
    statistics from the real data.

    Parameters
    ----------
    real_df : pd.DataFrame
        Real training data (includes the target column).
    syn_dict : dict[str, pd.DataFrame]
        ``{name: DataFrame}`` of synthetic datasets.
    target_col : str, default ``"label"``
        Name of the target column (excluded from features).

    Returns
    -------
    X_real : np.ndarray of shape (n_real, n_features)
        Scaled real features.
    X_syn : dict[str, np.ndarray]
        Scaled synthetic features keyed by generator name.
    """
    feature_df = real_df.drop(columns=[target_col])
    cat_cols = feature_df.select_dtypes(exclude="number").columns.tolist()
    num_cols = feature_df.select_dtypes(include="number").columns.tolist()
    feature_cols = num_cols + cat_cols

    # Encode categoricals on real data
    real_enc = real_df.copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        real_enc[col] = le.fit_transform(real_enc[col].astype(str))
        encoders[col] = le

    # Scale numerics + encoded categoricals
    scaler = StandardScaler()
    X_real = scaler.fit_transform(real_enc[feature_cols])

    # Apply same encoding + scaling to each synthetic dataset
    X_syn: dict[str, np.ndarray] = {}
    for name, df in syn_dict.items():
        df_enc = df[feature_cols].copy()
        for col in cat_cols:
            mapping = {v: i for i, v in enumerate(encoders[col].classes_)}
            df_enc[col] = (
                df_enc[col].astype(str).map(mapping).fillna(-1).astype(int)
            )
        X_syn[name] = scaler.transform(df_enc)

    return X_real, X_syn


# ── density score ───────────────────────────────────────────────────────


def density_score(
    X_real: np.ndarray,
    X_syn_dict: dict[str, np.ndarray],
    k: int = 5,
) -> dict[str, float]:
    """Compute the density score for each synthetic dataset.

    For every real sample *X_i* we define a ball *B(X_i, NND_k)* whose
    radius equals the distance to the *k*-th nearest neighbour among real
    points.  The density score is::

        density = (1 / (k · M)) · Σ_j Σ_i 𝟙[Y_j ∈ B(X_i, NND_k(X_i))]

    Parameters
    ----------
    X_real : np.ndarray of shape (N, d)
        Scaled real features.
    X_syn_dict : dict[str, np.ndarray]
        ``{name: array}`` of scaled synthetic features.
    k : int, default 5
        Number of nearest neighbours for the radius estimation.

    Returns
    -------
    dict[str, float]
        Density score per synthetic dataset (higher → better coverage of
        high-density regions).
    """
    N = X_real.shape[0]

    # k-th nearest-neighbour distance among real data (exclude self → k+1)
    nn_real = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn_real.fit(X_real)
    real_distances, _ = nn_real.kneighbors(X_real)
    nnd_k = real_distances[:, -1]  # k-th neighbour distance per real point

    scores: dict[str, float] = {}
    for name, X_syn in X_syn_dict.items():
        logger.info("Computing density score for: %s …", name)
        M = X_syn.shape[0]

        # Distances from each synthetic sample to all real samples
        nn_syn = NearestNeighbors(algorithm="auto")
        nn_syn.fit(X_real)
        distances, indices = nn_syn.kneighbors(X_syn, n_neighbors=N)

        # Count how many (synthetic, real) pairs satisfy dist ≤ NND_k
        count = 0
        for j in range(M):
            for idx, dist in zip(indices[j], distances[j]):
                if dist <= nnd_k[idx]:
                    count += 1

        scores[name] = count / (k * M)

    return scores


# ── ranking helper ──────────────────────────────────────────────────────


def rank_by_density(scores: dict[str, float]) -> pd.DataFrame:
    """Convert density scores to a ranked DataFrame.

    Parameters
    ----------
    scores : dict[str, float]
        Output of :func:`density_score`.

    Returns
    -------
    pd.DataFrame
        Columns ``Generator`` and ``Density_Score``, sorted descending.
        Index is labelled ``Rank`` starting from 1.
    """
    rows = [{"Generator": name, "Density_Score": s} for name, s in scores.items()]
    ranking = (
        pd.DataFrame(rows)
        .sort_values("Density_Score", ascending=False)
        .reset_index(drop=True)
    )
    ranking.index += 1
    ranking.index.name = "Rank"
    return ranking


# ── high-level convenience class ────────────────────────────────────────


class DensityEvaluator:
    """Evaluate synthetic data quality via density score.

    This is the density-based counterpart of
    :class:`~synthselector.evaluator.SyntheticEvaluator`.  While the
    classifier-based evaluator measures downstream ML performance, the
    density evaluator measures how well synthetic samples cover the
    real-data manifold — a complementary, model-free perspective.

    Parameters
    ----------
    real : pd.DataFrame
        Real training data (must include ``target_col``).
    target_col : str, default ``"label"``
        Name of the target column.
    k : int, default 5
        Nearest-neighbour count for radius estimation.

    Examples
    --------
    >>> de = DensityEvaluator(train_df)
    >>> de.add("CTGAN", ctgan_df)
    >>> de.add("TVAE", tvae_df)
    >>> ranking = de.run()
    >>> print(ranking)
    """

    def __init__(
        self,
        real: pd.DataFrame,
        *,
        target_col: str = "label",
        k: int = 5,
    ) -> None:
        self.real = real.copy()
        self.target_col = target_col
        self.k = k
        self._synthetic: dict[str, pd.DataFrame] = {}

    def add(self, name: str, df: pd.DataFrame) -> "DensityEvaluator":
        """Register a synthetic dataset."""
        self._synthetic[name] = df
        return self

    def add_many(self, datasets: dict[str, pd.DataFrame]) -> "DensityEvaluator":
        """Register multiple synthetic datasets at once."""
        for name, df in datasets.items():
            self.add(name, df)
        return self

    def run(self) -> pd.DataFrame:
        """Compute density scores and return a ranked DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns ``Generator`` and ``Density_Score``, indexed by ``Rank``.

        Raises
        ------
        ValueError
            If no synthetic datasets have been registered.
        """
        if not self._synthetic:
            raise ValueError(
                "No synthetic datasets registered. Use .add() or .add_many() first."
            )

        X_real, X_syn = preprocess(
            self.real, self._synthetic, target_col=self.target_col
        )
        scores = density_score(X_real, X_syn, k=self.k)
        return rank_by_density(scores)

    def print_report(self) -> None:
        """Run evaluation and print results to stdout."""
        ranking = self.run()
        print("\n" + "=" * 50)
        print("  DENSITY SCORE RANKING")
        print("=" * 50)
        print(ranking.to_string())
        best = ranking.iloc[0]
        print(
            f"\n  Best: {best['Generator']}"
            f"  (Density Score = {best['Density_Score']:.4f})"
        )

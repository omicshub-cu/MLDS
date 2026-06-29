"""
Example: Evaluate synthetic datasets on the Iris classification task.

This script generates dummy "synthetic" data by resampling the real training
set with noise, then runs the full SynthSelector evaluation pipeline.

Usage:
    python examples/quickstart.py
"""

import sys
sys.path.insert(0, '.')  # Add current folder to path

from synthselector import SyntheticEvaluator

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def make_noisy_copy(df: pd.DataFrame, noise: float = 0.1, seed: int = 0) -> pd.DataFrame:
    """Create a synthetic-like copy by adding Gaussian noise to features."""
    rng = np.random.RandomState(seed)
    out = df.copy()
    feature_cols = [c for c in out.columns if c != "label"]
    out[feature_cols] += rng.normal(0, noise, size=(len(out), len(feature_cols)))
    return out


def main() -> None:
    # ── load & split ────────────────────────────────────────────────────
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "label"})
    train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])

    # ── create fake "synthetic" datasets ────────────────────────────────
    synthetic_datasets = {
        "LowNoise": make_noisy_copy(train, noise=0.05, seed=1),
        "MedNoise": make_noisy_copy(train, noise=0.30, seed=2),
        "HighNoise": make_noisy_copy(train, noise=1.00, seed=3),
    }

    # ── evaluate ────────────────────────────────────────────────────────
    evaluator = SyntheticEvaluator(train, test)
    evaluator.add_many(synthetic_datasets)
    result = evaluator.run()

    # ── report ──────────────────────────────────────────────────────────
    result.print_report(top_n=3)

    # Programmatic access
    print("\n--- Programmatic access ---")
    summary = result.summary()
    best_model = summary.index[0]
    best_synth = result.best_synthetic(best_model)
    print(f"Best model: {best_model}")
    print(f"Best synthetic for {best_model}: {best_synth}")



if __name__ == "__main__":
    main()

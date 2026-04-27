"""Quick-start demo using the Iris dataset.

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from synthselector import SyntheticEvaluator, DensityEvaluator

# ── load data ────────────────────────────────────────────────────────────
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["label"] = iris.target

# ── create fake "synthetic" datasets ─────────────────────────────────────
rng = np.random.RandomState(0)

clean = df.copy()
feat_cols = [c for c in df.columns if c != "label"]
clean[feat_cols] += rng.normal(0, 0.05, size=(len(df), len(feat_cols)))

noisy = df.copy()
noisy[feat_cols] += rng.normal(0, 2.0, size=(len(df), len(feat_cols)))

# ── classifier-based evaluation ──────────────────────────────────────────
print("=" * 60)
print("  CLASSIFIER-BASED EVALUATION")
print("=" * 60)
evaluator = SyntheticEvaluator(df)
evaluator.add("Clean", clean).add("Noisy", noisy)
result = evaluator.run()
result.print_report(top_n=2)

# ── density-based evaluation ─────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("  DENSITY-BASED EVALUATION")
print("=" * 60)
de = DensityEvaluator(df, k=5)
de.add_many({"Clean": clean, "Noisy": noisy})
de.print_report()

# MLDS: A Machine Learning-Driven Framework for Ensemble Synthetic Data Selection

**Evaluate and rank synthetic datasets for ML classification tasks.**

SynthSelector benchmarks multiple synthetic data generators by training a panel of classifiers and scoring each dataset on **F1**, **AUC**, and **G-Mean**. The result is a per-model ranking plus an overall summary, you can pick the best generator (one or more) for your pipeline in one run.

---

## Features

- **Multi-classifier panel** — SVM, KNN, Decision Tree, Random Forest out of the box; LightGBM and XGBoost auto-detected when installed.
- **Three complementary metrics** — Macro F1, ROC-AUC (binary & multiclass), and Geometric Mean of per-class recalls.
- **Weighted composite score** — Configurable weights (default 0.4 / 0.3 / 0.3) with min-max normalisation per model.
- **Clean Python API + CLI** — Use as a library or run straight from the terminal.
- **Binary & multiclass** — Handles any number of classes automatically.

---

## Installation

```bash
# From source
git clone https://github.com/omicshub-cu/MLDS.git
cd MLDS
pip install -e .

# With boosting classifiers (LightGBM + XGBoost)
pip install -e ".[boost]"

# Full dev environment
pip install -e ".[all]"
```

---

## Quick Start

### Python API

```python
import pandas as pd
from synthselector import SyntheticEvaluator

train = pd.read_csv("train.csv")      # must contain a 'label' column
test  = pd.read_csv("test.csv")

evaluator = SyntheticEvaluator(train, test)
evaluator.add("SMOTE",  pd.read_csv("smote.csv"))
evaluator.add("CTGAN",  pd.read_csv("ctgan.csv"))
evaluator.add("TVAE",   pd.read_csv("tvae.csv"))

result = evaluator.run()
result.print_report()              # human-readable ranking
summary = result.summary()        # pandas DataFrame
best = result.best_synthetic("RF") # best generator for Random Forest
```

### Density Score (model-free)

Compare synthetic data quality without training classifiers — measures how
well synthetic samples cover the real-data manifold:

```python
from synthselector import DensityEvaluator

de = DensityEvaluator(train, k=5)
de.add("SMOTE",  pd.read_csv("smote.csv"))
de.add("CTGAN",  pd.read_csv("ctgan.csv"))

ranking = de.run()       # DataFrame with Rank, Generator, Density_Score
de.print_report()        # pretty-print to stdout
```

### Command Line

```bash
python -m synthselector \
  --train train.csv \
  --test  test.csv \
  --synthetic SMOTE=smote.csv CTGAN=ctgan.csv TVAE=tvae.csv \
  --top-n 3

# Add density score comparison alongside classifier evaluation
python -m synthselector --train train.csv --test test.csv \
  --synthetic SMOTE=smote.csv CTGAN=ctgan.csv --density

# Density score only 
python -m synthselector --train train.csv --test test.csv \
  --synthetic SMOTE=smote.csv CTGAN=ctgan.csv --density-only -k 5
```

---

## Project Structure

```
MLDS/
├── synthselector/
│   ├── __init__.py          # Public API & version
│   ├── __main__.py          # CLI entry point
│   ├── classifiers.py       # Default classifier definitions
│   ├── density.py           # Density score (model-free manifold coverage)
│   ├── evaluator.py         # SyntheticEvaluator + EvaluationResult
│   └── metrics.py           # G-Mean, F1, AUC utilities
├── tests/
│   └── test_core.py         # Unit tests
├── examples/
│   └── quickstart.py        # End-to-end demo on Iris
├── pyproject.toml            # PEP 621 packaging + tool configs
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Configuration

### Custom Classifiers

```python
from sklearn.linear_model import LogisticRegression

evaluator = SyntheticEvaluator(
    train, test,
    classifiers={
        "LR": LogisticRegression(max_iter=500),
        "RF": RandomForestClassifier(n_estimators=200),
    }
)
```

### Custom Weights

```python
evaluator = SyntheticEvaluator(
    train, test,
    weights={"F1": 0.5, "AUC": 0.25, "GMean": 0.25}
)
```

### Custom Label Column

```python
evaluator = SyntheticEvaluator(train, test, label_col="target")
```

---


## How It Works

### Classifier-based evaluation (`SyntheticEvaluator`)

1. For each synthetic dataset, combine it with the real training data.
2. Train every classifier on the combined data.
3. Predict on the held-out test set and compute F1 (macro), AUC (macro-OVR), and G-Mean.
4. Min-max normalise scores within each classifier.
5. Compute a weighted composite score.
6. Rank synthetic datasets per classifier; average across all to find the overall best.

### Density-based evaluation (`DensityEvaluator`)

A model-free alternative that measures manifold coverage directly:

1. For each real sample, compute the distance to its *k*-th nearest real neighbour (the "radius").
2. For each synthetic sample, check which real-data balls it falls inside.
3. The density score = total hits / (*k* × *M*), where *M* is the synthetic sample count.

Higher density → synthetic data concentrates in high-density real regions.
Use both methods together to get complementary views: downstream utility (classifiers) vs. distributional fidelity (density).

---

## Balanced Union Sampling

**Balanced Union Sampling** is the procedure `synthselector` uses to balance
the minority class against the majority class once the top-*k* synthetic
generators have been selected.

### How it works

1. **Compute the budget.** Determine how many synthetic samples are needed
   to bring the minority class up to the majority class count:
   `budget = n_majority − n_real_minority`.
2. **Pool the synthetic output.** Combine the synthetic minority samples
   from each of the top-*k* generators into a single union, keeping track
   of which generator produced each sample.
3. **Set an equal quota per generator.** Divide the budget evenly across
   the *k* generators: `quota = budget // k`.
4. **Cap and redistribute.** If a generator's available synthetic samples
   fall short of its quota, allocate everything it has and redistribute
   the shortfall to generators that still have spare capacity.
5. **Sample randomly within each generator's allocation.** For each
   generator, randomly draw its allocated number of samples from its own
   pool, without replacement.
6. **Combine.** Merge the sampled synthetic points from all generators,
   then append them to the (unmodified) real minority samples to form the
   final balanced minority set.

The result is a minority-class training set that (a) matches the majority
class in size, (b) draws roughly evenly from each of the top-*k*
generators regardless of how large each generator's raw output pool was,
and (c) preserves all real minority samples unchanged.

### Why "union" instead of a single generator

Different generators have different inductive biases (e.g. GAN-based vs.
diffusion-based vs. tree-based methods), so each tends to cover different
regions of the real minority class's feature space. Pooling multiple
generators' output, rather than using just the top-ranked one, improves
coverage of the minority distribution as a result, provided the pooling
step (Balanced Union Sampling) controls for uneven per-generator pool
sizes.

### Usage

```python
from synthselector.balanced_union_sampling import balanced_union_sampling

result = balanced_union_sampling(
    real_minority=real_minority_df,
    synthetic_pools={
        "TabDDPM": synthetic_tabddpm_df,
        "ForestDiff": synthetic_forestdiff_df,
        "TVAE": synthetic_tvae_df,
    },
    n_majority=len(real_majority_df),
    random_state=42,
)

balanced_minority_df = result.augmented_minority
```

See `result.allocation` for the actual number of samples drawn per
generator, and `result.requested_quota` for what an even split would have
looked like before capping/redistribution — useful for reporting
per-generator contribution in write-ups or sanity-checking that no single
generator dominates the augmented set.


## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

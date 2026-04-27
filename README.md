# MLDS: A Machine Learning-Driven Framework for Ensemble Synthetic Data Selection

Evaluate and rank synthetic datasets for ML classification tasks.

## Features

- **6 classifiers** — SVM, KNN, Decision Tree, Random Forest, LightGBM, XGBoost
- **3 metrics** — Macro F1, AUC (OVR), G-Mean
- **Density score** — Model-free manifold coverage metric (KNN-based)
- **Weighted composite score** — Configurable weights (default 0.4 / 0.3 / 0.3)
- **Internal train/val split** — Stratified 80/20 split, no separate test set needed
- **Clean Python API + CLI** — Use as a library or run from the terminal
- **Binary & multiclass** — Handles any number of classes automatically

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

evaluator = SyntheticEvaluator(train)  # splits into train/val internally
evaluator.add("SMOTE",  pd.read_csv("smote.csv"))
evaluator.add("CTGAN",  pd.read_csv("ctgan.csv"))
evaluator.add("TVAE",   pd.read_csv("tvae.csv"))

result = evaluator.run()
result.print_report()              # human-readable ranking
summary = result.summary()        # pandas DataFrame
best = result.best_synthetic("RF") # best generator for Random Forest
```

### Command Line

```bash
synthselector \
  --train train.csv \
  --synthetic SMOTE=smote.csv CTGAN=ctgan.csv TVAE=tvae.csv \
  --top-n 3
```

### With Density Score

```bash
synthselector \
  --train train.csv \
  --synthetic SMOTE=smote.csv CTGAN=ctgan.csv \
  --density -k 5
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
    train,
    classifiers={
        "LR": LogisticRegression(max_iter=500),
        "RF": RandomForestClassifier(n_estimators=200),
    }
)
```

### Custom Weights

```python
evaluator = SyntheticEvaluator(
    train,
    weights={"F1": 0.5, "AUC": 0.25, "GMean": 0.25}
)
```

### Custom Label Column & Split

```python
evaluator = SyntheticEvaluator(
    train,
    label_col="target",
    test_size=0.3,        # 70/30 split instead of 80/20
    random_state=123,
)
```

---

## How It Works

### Classifier-based evaluation (`SyntheticEvaluator`)

1. Split the training data into 80% train / 20% validation (stratified).
2. For each synthetic dataset, merge it with the train split only.
3. Train every classifier on the combined data.
4. Predict on the held-out validation set and compute F1 (macro), AUC (macro-OVR), and G-Mean.
5. Min-max normalise scores within each classifier.
6. Compute a weighted composite score.
7. Rank synthetic datasets per classifier; average across all to find the overall best.

### Density-based evaluation (`DensityEvaluator`)

A model-free alternative that measures manifold coverage directly:

1. For each real sample, compute the distance to its *k*-th nearest real neighbour (the "radius").
2. For each synthetic sample, check which real-data balls it falls inside.
3. The density score = total hits / (*k* × *M*), where *M* is the synthetic sample count.

Higher density → synthetic data concentrates in high-density real regions.
Use both methods together to get complementary views: downstream utility (classifiers) vs. distributional fidelity (density).

---

## License


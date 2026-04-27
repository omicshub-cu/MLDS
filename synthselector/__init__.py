"""
SynthSelector — Evaluate and rank synthetic datasets for ML classification.

A framework for benchmarking synthetic data generators (SMOTE, CTGAN, TVAE,
Copula GAN, ForestDiffusion, TabDDPM, STASY, TabSyn, …) by training multiple
classifiers and scoring each dataset on F1, AUC, and G-Mean.
"""

__version__ = "0.2.0"

from .metrics import compute_gmean, compute_classification_metrics
from .classifiers import get_default_classifiers
from .evaluator import SyntheticEvaluator
from .density import DensityEvaluator, density_score, rank_by_density

__all__ = [
    "compute_gmean",
    "compute_classification_metrics",
    "get_default_classifiers",
    "SyntheticEvaluator",
    "DensityEvaluator",
    "density_score",
    "rank_by_density",
]

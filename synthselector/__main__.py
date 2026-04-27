"""Command-line interface for SynthSelector.

Usage
-----
    python -m synthselector --train train.csv --test test.csv \
        --synthetic SMOTE=smote.csv CTGAN=ctgan.csv
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from .evaluator import SyntheticEvaluator
from .density import DensityEvaluator


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="synthselector",
        description="Evaluate and rank synthetic datasets for ML classification.",
    )
    parser.add_argument(
        "--train", required=True, help="Path to real training CSV (must have 'label' column)."
    )
    parser.add_argument(
        "--test", required=True, help="Path to held-out test CSV (must have 'label' column)."
    )
    parser.add_argument(
        "--synthetic",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="One or more NAME=path.csv pairs for synthetic datasets.",
    )
    parser.add_argument(
        "--label-col", default="label", help="Name of the target column (default: 'label')."
    )
    parser.add_argument(
        "--top-n", type=int, default=3, help="Number of top synthetics to show per model."
    )
    parser.add_argument(
        "--weights",
        nargs=3,
        type=float,
        default=[0.4, 0.3, 0.3],
        metavar=("W_F1", "W_AUC", "W_GMEAN"),
        help="Metric weights for composite score (default: 0.4 0.3 0.3).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging."
    )
    parser.add_argument(
        "--density", action="store_true",
        help="Also compute density scores (model-free manifold coverage metric).",
    )
    parser.add_argument(
        "--density-only", action="store_true",
        help="Only compute density scores (skip classifier evaluation).",
    )
    parser.add_argument(
        "-k", "--neighbors", type=int, default=5,
        help="Number of neighbours for density score (default: 5).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # ── load data ───────────────────────────────────────────────────────
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    synthetic: dict[str, pd.DataFrame] = {}
    for item in args.synthetic:
        if "=" not in item:
            logging.error("Expected NAME=PATH format, got: %s", item)
            sys.exit(1)
        name, path = item.split("=", 1)
        synthetic[name] = pd.read_csv(path)

    weights = {
        "F1": args.weights[0],
        "AUC": args.weights[1],
        "GMean": args.weights[2],
    }

    # ── evaluate ────────────────────────────────────────────────────────
    if not args.density_only:
        evaluator = SyntheticEvaluator(
            train, test, label_col=args.label_col, weights=weights
        )
        evaluator.add_many(synthetic)
        result = evaluator.run()
        result.print_report(top_n=args.top_n)

    if args.density or args.density_only:
        de = DensityEvaluator(train, target_col=args.label_col, k=args.neighbors)
        de.add_many(synthetic)
        de.print_report()


if __name__ == "__main__":
    main()

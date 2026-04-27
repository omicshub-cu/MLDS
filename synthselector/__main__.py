"""CLI entry point: ``python -m synthselector`` or ``synthselector``."""

from __future__ import annotations

import argparse
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
        "--train",
        required=True,
        help="Path to the training CSV (must contain a label column).",
    )
    parser.add_argument(
        "--synthetic",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="One or more NAME=path.csv pairs for synthetic datasets.",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Name of the target column (default: 'label').",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of train to hold out for validation (default: 0.2).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top synthetic datasets to show per model (default: 3).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging.",
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
    args = _parse_args(argv)

    train = pd.read_csv(args.train)

    synthetics: dict[str, pd.DataFrame] = {}
    for item in args.synthetic:
        if "=" not in item:
            print(f"ERROR: expected NAME=PATH, got '{item}'", file=sys.stderr)
            sys.exit(1)
        name, path = item.split("=", 1)
        synthetics[name] = pd.read_csv(path)

    # Classifier-based evaluation
    if not args.density_only:
        evaluator = SyntheticEvaluator(
            train,
            label_col=args.label_col,
            test_size=args.test_size,
        )
        evaluator.add_many(synthetics)
        result = evaluator.run()
        result.print_report(top_n=args.top_n)

    # Density evaluation
    if args.density or args.density_only:
        de = DensityEvaluator(
            train,
            target_col=args.label_col,
            k=args.neighbors,
        )
        de.add_many(synthetics)
        de.print_report()


if __name__ == "__main__":
    main()

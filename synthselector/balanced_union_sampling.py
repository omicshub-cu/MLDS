"""
balanced_union_sampling.py

Balanced Union Sampling: combines synthetic minority-class samples from
the top-k ranked generators into a single balanced augmentation set,
allocating an equal quota per generator (redistributing shortfall when a
generator's pool is smaller than its quota) and drawing randomly within
each generator's allocation.

Intended to plug into the synth-selector pipeline after Phase 3
aggregation (rank selection of top-k generators), as the step that
prepares the final balanced training set for classifier evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class BalancedUnionSamplingResult:
    """Container for the output of balanced_union_sampling()."""

    augmented_minority: pd.DataFrame          # real + sampled synthetic minority rows
    synthetic_sample: pd.DataFrame             # only the sampled synthetic rows
    allocation: Dict[str, int]                 # samples drawn per generator
    requested_quota: Dict[str, int]            # equal quota before capping/redistribution
    budget: int                                # total synthetic samples needed
    shortfall_redistributed: int               # how many samples had to be redistributed


def balanced_union_sampling(
    real_minority: pd.DataFrame,
    synthetic_pools: Dict[str, pd.DataFrame],
    n_majority: int,
    random_state: Optional[int] = None,
) -> BalancedUnionSamplingResult:
    """
    Balance the minority class against the majority class by pooling
    synthetic samples from multiple generators under an equal
    per-generator quota (Balanced Union Sampling).

    Parameters
    ----------
    real_minority : pd.DataFrame
        Real minority-class samples (features only, or features + label --
        whatever schema your synthetic_pools also use; columns must match).
    synthetic_pools : dict[str, pd.DataFrame]
        Mapping of generator name -> its synthetic minority-class sample
        pool, e.g. {"TabDDPM": df_tabddpm, "TVAE": df_tvae, "ForestDiff": df_fd}.
        Typically this is your top-k selected generators from Phase 3
        aggregation (MeanRank / TimesTop3).
    n_majority : int
        Number of real majority-class samples. The augmented minority set
        will be sized to approximately match this count.
    random_state : int, optional
        Seed for reproducible sampling.

    Returns
    -------
    BalancedUnionSamplingResult
        Dataclass containing the augmented minority set, the synthetic
        sample alone, and bookkeeping on how the budget was allocated
        (useful for reporting per-generator contribution in your paper).

    Raises
    ------
    ValueError
        If the union of all synthetic pools cannot cover the required
        budget even after redistribution (i.e. you don't have enough
        synthetic samples in total to reach parity with the majority class).
    """
    if not synthetic_pools:
        raise ValueError("synthetic_pools must contain at least one generator.")

    rng = np.random.default_rng(random_state)

    budget = max(0, n_majority - len(real_minority))
    if budget == 0:
        empty = real_minority.iloc[0:0]
        return BalancedUnionSamplingResult(
            augmented_minority=real_minority.copy(),
            synthetic_sample=empty,
            allocation={name: 0 for name in synthetic_pools},
            requested_quota={name: 0 for name in synthetic_pools},
            budget=0,
            shortfall_redistributed=0,
        )

    k = len(synthetic_pools)
    quota = budget // k
    requested_quota = {name: quota for name in synthetic_pools}

    pool_sizes = {name: len(df) for name, df in synthetic_pools.items()}
    allocation = {name: min(quota, pool_sizes[name]) for name in synthetic_pools}
    shortfall = budget - sum(allocation.values())
    shortfall_redistributed = shortfall

    # redistribute shortfall to generators with remaining spare capacity
    names = list(synthetic_pools.keys())
    while shortfall > 0:
        progressed = False
        for name in names:
            if shortfall <= 0:
                break
            spare = pool_sizes[name] - allocation[name]
            if spare <= 0:
                continue
            extra = min(spare, shortfall)
            allocation[name] += extra
            shortfall -= extra
            progressed = True
        if not progressed:
            break  # no generator has spare capacity left

    if shortfall > 0:
        total_available = sum(pool_sizes.values())
        raise ValueError(
            f"Not enough synthetic samples to reach class parity: "
            f"budget={budget}, total available across all generators={total_available}. "
            f"Generate more synthetic samples, reduce n_majority target, "
            f"or accept a smaller augmented minority set."
        )

    # random draw within each generator's allocation
    sampled_frames = []
    for name, df in synthetic_pools.items():
        n = allocation[name]
        if n == 0:
            continue
        idx = rng.choice(len(df), size=n, replace=False)
        chunk = df.iloc[idx].copy()
        chunk["__source_generator__"] = name
        sampled_frames.append(chunk)

    synthetic_sample = (
        pd.concat(sampled_frames, ignore_index=True)
        if sampled_frames
        else real_minority.iloc[0:0].assign(__source_generator__=[])
    )

    real_tagged = real_minority.copy()
    real_tagged["__source_generator__"] = "real"

    augmented_minority = pd.concat([real_tagged, synthetic_sample], ignore_index=True)

    return BalancedUnionSamplingResult(
        augmented_minority=augmented_minority,
        synthetic_sample=synthetic_sample,
        allocation=allocation,
        requested_quota=requested_quota,
        budget=budget,
        shortfall_redistributed=shortfall_redistributed,
    )


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Minimal smoke test with toy data
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)

    real_minority = pd.DataFrame(rng.normal(size=(30, 4)), columns=list("abcd"))
    gen_a = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    gen_b = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    gen_c = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))

    result = balanced_union_sampling(
        real_minority=real_minority,
        synthetic_pools={"Gen A": gen_a, "Gen B": gen_b, "Gen C": gen_c},
        n_majority=400,
        random_state=42,
    )

    print(f"Budget needed: {result.budget}")
    print(f"Requested equal quota per generator: {result.requested_quota}")
    print(f"Actual allocation per generator: {result.allocation}")
    print(f"Shortfall redistributed: {result.shortfall_redistributed}")
    print(f"Final augmented minority set size: {len(result.augmented_minority)}")
    print(result.augmented_minority["__source_generator__"].value_counts())

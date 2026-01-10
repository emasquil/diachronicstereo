#!/usr/bin/env python3
"""
create_dfc_splits.py

Create AOI-based splits for the DFC dataset.

- Input:  the stereo_pairs/L directory (contains files like OMA_042_...-... .iio)
- Output: train_aois.csv, val_aois.csv, test_aois.csv (AOI names like OMA_042)
- Ratios default to 0.85 / 0.05 / 0.10
- You can force specific AOIs into the test split (e.g., JAX_004, JAX_068, ...)
- You can also force N OMA_* AOIs into test (deterministically by seed).
- The function respects test_ratio by topping up test from remaining AOIs
  after placing all forced AOIs there.
"""

import argparse
import os
import random
from typing import List, Set, Tuple


def extract_aoi(fname: str) -> str:
    """AOI is the first 7 chars, e.g., 'OMA_042' from 'OMA_042_007_RGB-...iio'."""
    return fname[:7]


def save_lines(lines: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for x in lines:
            f.write(f"{x}\n")


def split_aois(
    aois_all: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    forced_test: Set[str],
    force_test_oma_count: int = 4,
    seed: int = 42,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Return (train_aois, val_aois, test_aois).
    - forced_test are placed into test first (must exist in aois_all).
    - Then choose `force_test_oma_count` OMA_* AOIs (not already forced) and force them into test.
    - Then top up test from remaining AOIs to match test_ratio.
    - Split the remainder into train/val by relative weights.
    """
    assert (
        abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6
    ), "Ratios must sum to 1."

    rng = random.Random(seed)
    aois_all = sorted(set(aois_all))

    # 1) Explicit forced AOIs (e.g., JAX_004 ...) that actually exist
    forced_test = set(a for a in forced_test if a in aois_all)

    # 2) Add N OMA_* AOIs to forced_test (deterministic by seed)
    oma_pool = [a for a in aois_all if a.startswith("OMA_") and a not in forced_test]
    if force_test_oma_count > 0 and oma_pool:
        rng.shuffle(oma_pool)
        add_oma = oma_pool[: min(force_test_oma_count, len(oma_pool))]
        forced_test |= set(add_oma)

    total = len(aois_all)
    # Target size for test honoring ratio, but at least what we've already forced
    target_test = max(int(round(total * test_ratio)), len(forced_test))

    # Remaining AOIs after forced
    remaining = [a for a in aois_all if a not in forced_test]
    rng.shuffle(remaining)

    # Top up test from remaining to meet target_test
    extra_for_test_count = max(0, target_test - len(forced_test))
    extra_for_test = set(remaining[:extra_for_test_count])
    remaining = remaining[extra_for_test_count:]

    test_aois = forced_test | extra_for_test

    # Split remaining between train and val according to relative weights
    tv = train_ratio + val_ratio
    if tv <= 0:
        train_aois = set(remaining)
        val_aois = set()
    else:
        train_count = int(round(len(remaining) * (train_ratio / tv)))
        train_aois = set(remaining[:train_count])
        val_aois = set(remaining[train_count:])

    # sanity
    assert train_aois.isdisjoint(val_aois)
    assert train_aois.isdisjoint(test_aois)
    assert val_aois.isdisjoint(test_aois)

    # non-empty checks (relax if dataset tiny)
    assert len(test_aois) > 0, "Test split is empty"
    assert len(train_aois) > 0, "Train split is empty"
    assert len(val_aois) > 0, "Val split is empty"

    return train_aois, val_aois, test_aois


def main():
    ap = argparse.ArgumentParser(description="Create AOI-based splits for DFC.")
    ap.add_argument(
        "--root_dir_L",
        required=True,
        help="Path to DFC stereo_pairs/L directory (contains AOI-pair .iio files)",
    )
    ap.add_argument(
        "--output_dir",
        default=".",
        help="Where to write train_aois.csv / val_aois.csv / test_aois.csv",
    )
    ap.add_argument("--train_ratio", type=float, default=0.85)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--test_ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--force_test_aois",
        nargs="*",
        default=["JAX_004", "JAX_068", "JAX_214", "JAX_260"],
        help="AOIs to force into the test split (space-separated, e.g. JAX_004 JAX_068 ...)",
    )
    ap.add_argument(
        "--force_test_oma_count",
        type=int,
        default=4,
        help="Additionally force this many OMA_* AOIs into test (sampled deterministically by seed)",
    )
    args = ap.parse_args()

    files_L = sorted([f for f in os.listdir(args.root_dir_L) if not f.startswith(".")])
    if not files_L:
        raise RuntimeError(f"No files found in {args.root_dir_L}")

    aois = sorted({extract_aoi(f) for f in files_L})

    train_aois, val_aois, test_aois = split_aois(
        aois,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        set(args.force_test_aois),
        force_test_oma_count=args.force_test_oma_count,
        seed=args.seed,
    )

    save_lines(sorted(train_aois), os.path.join(args.output_dir, "train_aois.csv"))
    save_lines(sorted(val_aois), os.path.join(args.output_dir, "val_aois.csv"))
    save_lines(sorted(test_aois), os.path.join(args.output_dir, "test_aois.csv"))

    print(f"Total AOIs: {len(aois)}")
    print(f"Train AOIs: {len(train_aois)}")
    print(f"Val   AOIs: {len(val_aois)}")
    print(f"Test  AOIs: {len(test_aois)}")
    if args.force_test_aois:
        present_forced = sorted(set(args.force_test_aois) & set(aois))
        missing_forced = sorted(set(args.force_test_aois) - set(aois))
        print(f"Forced test AOIs present: {present_forced}")
        if missing_forced:
            print(f"WARNING: Forced AOIs not found in data: {missing_forced}")
    print(
        f"Forced OMA in test: {args.force_test_oma_count} (deterministic by seed={args.seed})"
    )


if __name__ == "__main__":
    main()

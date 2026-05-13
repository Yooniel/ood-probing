from __future__ import annotations

import numpy as np


def validate_fraction(value: float, name: str, *, allow_zero: bool, allow_one: bool) -> None:
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    upper_ok = value <= 1.0 if allow_one else value < 1.0
    if not (lower_ok and upper_ok):
        lower = "0" if allow_zero else "0 exclusive"
        upper = "1" if allow_one else "1 exclusive"
        raise ValueError(f"{name} must be between {lower} and {upper}; got {value}.")


def validate_cv_folds(labels: np.ndarray, cv_folds: int, dataset_name: str) -> None:
    if cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")
    class_counts = np.bincount(labels.astype(np.int64))
    nonzero_class_counts = class_counts[class_counts > 0]
    if nonzero_class_counts.size < 2:
        raise ValueError(f"Dataset '{dataset_name}' needs both classes for stratified CV.")
    min_class_count = int(np.min(nonzero_class_counts))
    if min_class_count < cv_folds:
        raise ValueError(
            f"Cannot run {cv_folds}-fold stratified CV for '{dataset_name}': "
            f"smallest class has only {min_class_count} samples."
        )


def validate_selected_pcs(pc_numbers: list[int], max_pc: int) -> list[int]:
    selected = list(dict.fromkeys(pc_numbers))
    invalid = [pc for pc in selected if pc < 1 or pc > max_pc]
    if invalid:
        raise ValueError(
            f"PC indices must be 1-indexed values in 1..{max_pc}; got {invalid}."
        )
    return selected


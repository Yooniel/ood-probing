from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ACTIVATION_RE = re.compile(r"^(?P<name>.+)_layer(?P<layer>\d+)_activations\.npy$")
LABEL_RE = re.compile(r"^(?P<name>.+)_labels\.npy$")
LEGACY_LAYER_LABEL_RE = re.compile(r"^(?P<name>.+)_layer(?P<layer>\d+)_labels\.npy$")


@dataclass(frozen=True)
class DatasetFiles:
    name: str
    layer: int
    activations: Path
    labels: Path


def discover_datasets(data_dir: Path, layer: int | None = None) -> dict[str, DatasetFiles]:
    """Scan data_dir for paired activation and label files.

    If layer is omitted, each dataset must have exactly one activation layer.
    This avoids silently choosing one layer when multiple files are present.
    """
    activations_by_name: dict[str, list[tuple[int, Path]]] = {}
    labels_by_name: dict[str, Path] = {}
    legacy_labels_by_name_layer: dict[tuple[str, int], Path] = {}

    for path in sorted(data_dir.glob("*.npy")):
        activation_match = ACTIVATION_RE.match(path.name)
        if activation_match:
            name = activation_match.group("name")
            activation_layer = int(activation_match.group("layer"))
            if layer is None or activation_layer == layer:
                activations_by_name.setdefault(name, []).append((activation_layer, path))
            continue

        label_match = LABEL_RE.match(path.name)
        if label_match:
            labels_by_name[label_match.group("name")] = path
            continue

        legacy_label_match = LEGACY_LAYER_LABEL_RE.match(path.name)
        if legacy_label_match:
            legacy_labels_by_name_layer[
                (legacy_label_match.group("name"), int(legacy_label_match.group("layer")))
            ] = path

    registry: dict[str, DatasetFiles] = {}
    ambiguous_layers: dict[str, list[int]] = {}
    for name, activation_records in activations_by_name.items():
        if layer is None and len(activation_records) > 1:
            ambiguous_layers[name] = [activation_layer for activation_layer, _ in activation_records]
            continue

        activation_layer, activation_path = activation_records[0]
        label_path = labels_by_name.get(name) or legacy_labels_by_name_layer.get((name, activation_layer))
        if label_path is None:
            continue

        registry[name] = DatasetFiles(
            name=name,
            layer=activation_layer,
            activations=activation_path,
            labels=label_path,
        )

    if ambiguous_layers:
        details = ", ".join(
            f"{name}: layers {sorted(layers)}"
            for name, layers in sorted(ambiguous_layers.items())
        )
        raise ValueError(f"Multiple activation layers found; pass --layer explicitly ({details}).")

    return registry


def require_dataset(name: str, registry: dict[str, DatasetFiles]) -> DatasetFiles:
    try:
        return registry[name]
    except KeyError as exc:
        available = ", ".join(sorted(registry)) or "none"
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}") from exc


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def pool_activation(sample: np.ndarray, pooling: str) -> np.ndarray:
    """Reduce a single activation sample to one hidden-state vector."""
    array = np.asarray(sample, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim != 2:
        raise ValueError(
            "Expected activation sample with shape [hidden_size] or [seq_len, hidden_size], "
            f"got {array.shape}."
        )
    if array.shape[0] == 0:
        raise ValueError("Encountered an activation sample with zero tokens.")
    if pooling == "last":
        return array[-1]
    if pooling == "mean":
        return array.mean(axis=0)
    raise ValueError(f"Unsupported pooling mode: {pooling}")


def pool_activations(activations: np.ndarray, pooling: str) -> np.ndarray:
    """Pool token-level activations, while accepting already-pooled arrays."""
    if activations.dtype != object:
        if activations.ndim == 2:
            return activations.astype(np.float32, copy=False)
        if activations.ndim == 3:
            if activations.shape[1] == 0:
                raise ValueError("Activation array has zero tokens.")
            if pooling == "last":
                return activations[:, -1, :].astype(np.float32, copy=False)
            if pooling == "mean":
                return activations.mean(axis=1).astype(np.float32, copy=False)
        raise ValueError(
            "Expected activations with shape [examples, hidden_size], "
            "[examples, seq_len, hidden_size], or an object array of token activations; "
            f"got {activations.shape}."
        )

    pooled_rows = [pool_activation(row, pooling) for row in activations]
    if not pooled_rows:
        raise ValueError("Activation file is empty.")
    return np.stack(pooled_rows, axis=0).astype(np.float32, copy=False)


def load_dataset(
    name: str,
    registry: dict[str, DatasetFiles],
    pooling: str,
    *,
    binary_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load activation features and labels for a discovered dataset."""
    files = require_dataset(name, registry)
    labels = np.load(files.labels, allow_pickle=True)
    activations = np.load(files.activations, allow_pickle=True)

    if len(labels) != len(activations):
        raise ValueError(
            f"Dataset '{name}' has {len(activations)} activation rows but {len(labels)} labels."
        )

    if binary_only:
        keep = np.isin(labels, [0, 1])
        labels = labels[keep]
        activations = activations[keep]

    if len(labels) == 0:
        raise ValueError(f"Dataset '{name}' is empty after loading labels.")

    features = pool_activations(activations, pooling)
    return features, labels.astype(np.int64, copy=False)


def load_raw_token_dataset(
    data_dir: Path,
    dataset_name: str,
    layer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    x_path = data_dir / f"{dataset_name}_layer{layer}_activations.npy"
    y_path = data_dir / f"{dataset_name}_labels.npy"
    legacy_y_path = data_dir / f"{dataset_name}_layer{layer}_labels.npy"
    token_ids_path = data_dir / f"{dataset_name}_token_ids.npy"
    texts_path = data_dir / f"{dataset_name}_texts.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing activations file: {x_path}")
    if not y_path.exists():
        if not legacy_y_path.exists():
            raise FileNotFoundError(f"Missing labels file: {y_path}")
        y_path = legacy_y_path
    if not token_ids_path.exists():
        raise FileNotFoundError(f"Missing token ids file: {token_ids_path}")

    activations = np.load(x_path, allow_pickle=True)
    if activations.dtype != object:
        raise ValueError(
            f"{x_path} appears to contain pooled activations. "
            "Interpretation requires token-level activations."
        )

    labels = np.load(y_path, allow_pickle=True).astype(np.int64)
    token_ids = np.load(token_ids_path, allow_pickle=True)
    texts = np.load(texts_path, allow_pickle=True) if texts_path.exists() else None

    if len(activations) != len(labels) or len(activations) != len(token_ids):
        raise ValueError(
            f"Raw dataset '{dataset_name}' has mismatched row counts: "
            f"activations={len(activations)}, labels={len(labels)}, token_ids={len(token_ids)}."
        )

    return activations, labels, token_ids, texts

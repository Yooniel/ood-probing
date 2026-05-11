import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


PCA_SEED = 7
ACTIVATION_RE = re.compile(r"^(?P<name>.+)_layer\d+_activations\.npy$")
LABEL_RE = re.compile(r"^(?P<name>.+)_labels\.npy$")


def discover_datasets(data_dir: Path) -> dict[str, dict[str, Path]]:
    """Scan data_dir for paired *_activations.npy and *_labels.npy files."""
    datasets: dict[str, dict[str, Path]] = {}
    for path in sorted(data_dir.glob("*.npy")):
        m = ACTIVATION_RE.match(path.name)
        if m:
            datasets.setdefault(m.group("name"), {})["activations"] = path
            continue
        m = LABEL_RE.match(path.name)
        if m:
            datasets.setdefault(m.group("name"), {})["labels"] = path

    return {
        name: paths
        for name, paths in datasets.items()
        if "activations" in paths and "labels" in paths
    }


def pool_activation(sample: np.ndarray, pooling: str) -> np.ndarray:
    """Reduce a (tokens, hidden_dim) activation to (hidden_dim,)."""
    array = np.asarray(sample, dtype=np.float32)
    if pooling == "last":
        return array[-1]
    return array.mean(axis=0)


def load_dataset(name: str, registry: dict[str, dict[str, Path]], pooling: str):
    """Load and pool activations, keeping only binary (0/1) labels."""
    labels = np.load(registry[name]["labels"], allow_pickle=True)
    activations = np.load(registry[name]["activations"], allow_pickle=True)

    keep = np.isin(labels, [0, 1])
    labels = labels[keep].astype(np.int64)
    activations = activations[keep]

    features = np.stack([pool_activation(s, pooling) for s in activations])
    return features, labels


def compute_auroc(model: LogisticRegression, features: np.ndarray, labels: np.ndarray) -> float:
    if features.shape[0] == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, model.decision_function(features)))


def train_probe(features: np.ndarray, labels: np.ndarray, c_value: float, seed: int) -> LogisticRegression:
    probe = LogisticRegression(C=c_value, solver="lbfgs", max_iter=2000, random_state=seed)
    probe.fit(features, labels)
    return probe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a source probe on explicitly selected PCs, then evaluate on targets "
                    "using the same source scaler and PCA basis."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/deception-activations"),
                        help="Directory containing activation and label .npy files.")
    parser.add_argument("--source", required=True, help="Source dataset name.")
    parser.add_argument("--targets", nargs="+", required=True, help="One or more target dataset names.")
    parser.add_argument("--pcs", nargs="+", type=int, required=True,
                        help="1-indexed PCs to use, e.g.: --pcs 1 2 4 6")
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean",
                        help="Pooling over token activations.")
    parser.add_argument("--c", type=float, default=0.1, help="Inverse regularization strength.")
    parser.add_argument("--max-pcs", type=int, default=100,
                        help="Truncate source PCA basis to this many PCs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for probe training.")
    parser.add_argument("--output", type=Path, default=None, help="Save results to a JSON file.")
    args = parser.parse_args()

    registry = discover_datasets(args.data_dir)
    target_names = list(dict.fromkeys(args.targets))

    # Load source, fit scaler + PCA
    source_features, source_labels = load_dataset(args.source, registry, args.pooling)
    truncated_max_pcs = min(args.max_pcs, source_features.shape[0], source_features.shape[1])

    # Convert 1-indexed PCs to 0-indexed
    selected_pcs_1 = list(dict.fromkeys(args.pcs))
    selected_pcs = [pc - 1 for pc in selected_pcs_1]

    scaler = StandardScaler()
    pca = PCA(n_components=truncated_max_pcs, random_state=PCA_SEED)
    source_pcs = pca.fit_transform(scaler.fit_transform(source_features))

    # Train probe on selected source PCs
    probe = train_probe(source_pcs[:, selected_pcs], source_labels, args.c, args.seed)

    # Print header
    pc_str = ",".join(f"PC{pc}" for pc in selected_pcs_1)
    print(f"source={args.source}  pooling={args.pooling}  c={args.c}")
    print(f"pca_seed={PCA_SEED}  probe_seed={args.seed}")
    print(f"max_pcs={args.max_pcs}  truncated_max_pcs={truncated_max_pcs}")
    print(f"selected_pcs={pc_str}  source_n={len(source_labels)}\n")

    # Evaluate on each target
    results = []
    print("target\ttarget_n\ttarget_auroc")
    for name in target_names:
        features, labels = load_dataset(name, registry, args.pooling)
        target_pcs = pca.transform(scaler.transform(features))
        auroc = compute_auroc(probe, target_pcs[:, selected_pcs], labels)
        print(f"{name}\t{len(labels)}\t{auroc:.6f}")

        results.append({"target": name, "target_n": len(labels), "target_auroc": auroc})

    if args.output:
        output = {
            "source": args.source,
            "pooling": args.pooling,
            "c": args.c,
            "pca_seed": PCA_SEED,
            "probe_seed": args.seed,
            "max_pcs": args.max_pcs,
            "truncated_max_pcs": truncated_max_pcs,
            "selected_pcs": [f"PC{pc}" for pc in selected_pcs_1],
            "source_n": len(source_labels),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

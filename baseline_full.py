import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, model.decision_function(features)))


def split_source_dataset(features: np.ndarray, labels: np.ndarray, test_size: float, seed: int):
    """Split source into train/test, or use the full source set for training."""
    if test_size == 0.0:
        return features, features[:0], labels, labels[:0]
    return train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an L2-regularized linear probe on a source dataset and evaluate AUROC on held-out and target datasets."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/deception-activations"),
                        help="Directory containing *_labels.npy and *_layer*_activations.npy files.")
    parser.add_argument("--source", required=True, help="Source dataset name.")
    parser.add_argument("--targets", nargs="*", default=None,
                        help="Target dataset names (default: all other discovered datasets).")
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean",
                        help="Pooling over token activations.")
    parser.add_argument("--c", type=float, default=0.1, help="Inverse regularization strength.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--source-test-size", type=float, default=0.2,
                        help="Held-out fraction of source data. Set to 0.0 to train on the full source dataset.")
    parser.add_argument("--output", type=Path, default=None, help="Save results to a JSON file.")
    args = parser.parse_args()

    registry = discover_datasets(args.data_dir)

    # Train probe on source dataset
    source_x, source_y = load_dataset(args.source, registry, args.pooling)
    train_x, test_x, train_y, test_y = split_source_dataset(
        source_x,
        source_y,
        test_size=args.source_test_size,
        seed=args.seed,
    )

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x) if len(test_x) > 0 else test_x

    probe = LogisticRegression(C=args.c, solver="lbfgs", max_iter=2000, random_state=args.seed)
    probe.fit(train_x, train_y)

    # Evaluate on source test split and target datasets
    targets = args.targets if args.targets is not None else [n for n in sorted(registry) if n != args.source]

    print(f"source={args.source}  pooling={args.pooling}  c={args.c}  seed={args.seed}")
    print(f"source_test_size={args.source_test_size}")
    print(f"train_size={len(train_y)}  test_size={len(test_y)}\n")

    results = []
    if len(test_y) > 0:
        auroc = compute_auroc(probe, test_x, test_y)
        results.append({"dataset": f"{args.source}__test", "auroc": auroc, "n": len(test_y)})
    else:
        auroc = compute_auroc(probe, train_x, train_y)
        results.append({"dataset": f"{args.source}__train", "auroc": auroc, "n": len(train_y)})

    for name in targets:
        target_x, target_y = load_dataset(name, registry, args.pooling)
        target_x = scaler.transform(target_x)
        auroc = compute_auroc(probe, target_x, target_y)
        results.append({"dataset": name, "auroc": auroc, "n": len(target_y)})

    print("dataset\tauroc\tn")
    for r in results:
        print(f"{r['dataset']}\t{r['auroc']:.6f}\t{r['n']}")

    if args.output:
        output = {
            "source": args.source,
            "pooling": args.pooling,
            "c": args.c,
            "seed": args.seed,
            "uses_source_standard_scaler": True,
            "source_test_size": args.source_test_size,
            "train_size": len(train_y),
            "test_size": len(test_y),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

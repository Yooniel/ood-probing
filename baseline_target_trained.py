import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
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
        description="Train a target-label probe in source PCA space using CV and report mean test AUROC."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/deception-activations"),
                        help="Directory containing activation and label .npy files.")
    parser.add_argument("--source", required=True, help="Source dataset name.")
    parser.add_argument("--targets", nargs="+", required=True, help="One or more target dataset names.")
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean",
                        help="Pooling over token activations.")
    parser.add_argument("--c", type=float, default=0.1,
                        help="Inverse regularization strength for the target probe.")
    parser.add_argument("--max-pcs", type=int, default=100,
                        help="Truncate source PCA basis to this many PCs.")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of stratified CV folds.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for probe training and CV splits.")
    parser.add_argument("--output", type=Path, default=None, help="Save results to a JSON file.")
    args = parser.parse_args()

    registry = discover_datasets(args.data_dir)
    target_names = list(dict.fromkeys(args.targets))

    # Load source, fit scaler + PCA
    source_features, source_labels = load_dataset(args.source, registry, args.pooling)
    truncated_max_pcs = min(args.max_pcs, source_features.shape[0], source_features.shape[1])

    scaler = StandardScaler()
    pca = PCA(n_components=truncated_max_pcs, random_state=PCA_SEED)
    pca.fit(scaler.fit_transform(source_features))

    # Print header
    print(f"source={args.source}  pooling={args.pooling}  c={args.c}")
    print(f"pca_seed={PCA_SEED}  probe_seed={args.seed}  cv_folds={args.cv_folds}")
    print(f"max_pcs={args.max_pcs}  truncated_max_pcs={truncated_max_pcs}")
    print(f"source_n={len(source_labels)}\n")

    # Per-target: project into source PCA space, evaluate via stratified CV
    results = []
    print("dataset\tmean_auroc\tn\tcv_folds")
    for name in target_names:
        features, labels = load_dataset(name, registry, args.pooling)
        target_pcs = pca.transform(scaler.transform(features))

        kf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        fold_aurocs = []
        fold_records = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(target_pcs, labels)):
            probe = train_probe(target_pcs[train_idx], labels[train_idx], args.c, args.seed)
            auroc = compute_auroc(probe, target_pcs[test_idx], labels[test_idx])
            fold_aurocs.append(auroc)
            fold_records.append({"fold": fold_idx, "train_n": len(train_idx), "test_n": len(test_idx), "auroc": auroc})

        mean_auroc = float(np.nanmean(fold_aurocs))
        print(f"{name}\t{mean_auroc:.6f}\t{len(labels)}\t{args.cv_folds}")
        results.append({"dataset": name, "n": len(labels), "mean_auroc": mean_auroc, "folds": fold_records})

    if args.output:
        output = {
            "source": args.source,
            "pooling": args.pooling,
            "c": args.c,
            "pca_seed": PCA_SEED,
            "probe_seed": args.seed,
            "max_pcs": args.max_pcs,
            "truncated_max_pcs": truncated_max_pcs,
            "cv_folds": args.cv_folds,
            "source_n": len(source_labels),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

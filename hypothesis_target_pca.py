import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


PCA_SEED = 7
ACTIVATION_RE = re.compile(r"^(?P<name>.+)_layer\d+_activations\.npy$")
LABEL_RE = re.compile(r"^(?P<name>.+)_labels\.npy$")


def discover_datasets(data_dir: Path) -> dict[str, dict[str, Path]]:
    """Scan data_dir for paired *_activations.npy and *_labels.npy files."""
    datasets: dict[str, dict[str, Path]] = {}
    for path in sorted(data_dir.glob("*.npy")):
        match = ACTIVATION_RE.match(path.name)
        if match:
            datasets.setdefault(match.group("name"), {})["activations"] = path
            continue

        match = LABEL_RE.match(path.name)
        if match:
            datasets.setdefault(match.group("name"), {})["labels"] = path

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

    features = np.stack([pool_activation(sample, pooling) for sample in activations])
    return features, labels


def train_probe(features: np.ndarray, labels: np.ndarray, c_value: float, seed: int) -> LogisticRegression:
    probe = LogisticRegression(C=c_value, solver="lbfgs", max_iter=2000, random_state=seed)
    probe.fit(features, labels)
    return probe


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a StandardScaler and PCA on the source dataset, project target data into that source PCA space, "
            "train a probe on target labels, and rank source PCs by target-probe weight contribution."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/deception-activations"),
        help="Directory containing activation and label .npy files.",
    )
    parser.add_argument("--source", required=True, help="Source dataset name.")
    parser.add_argument("--targets", nargs="+", required=True, help="One or more target dataset names.")
    parser.add_argument(
        "--pooling",
        choices=["mean", "last"],
        default="mean",
        help="Pooling over token activations.",
    )
    parser.add_argument("--c", type=float, default=0.1, help="Inverse regularization strength.")
    parser.add_argument("--max-pcs", type=int, default=100, help="Truncate the source PCA basis to this many PCs.")
    parser.add_argument("--top-k", type=int, default=10, help="How many top-ranked PCs to print prominently.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for probe training.")
    parser.add_argument("--output", type=Path, default=None, help="Save results to a JSON file.")
    args = parser.parse_args()

    registry = discover_datasets(args.data_dir)
    target_names = list(dict.fromkeys(args.targets))

    # Fit source scaler + PCA once, then reuse for every target.
    source_features, _ = load_dataset(args.source, registry, args.pooling)
    truncated_max_pcs = min(args.max_pcs, source_features.shape[0], source_features.shape[1])
    selected_k = min(args.top_k, truncated_max_pcs)

    scaler = StandardScaler()
    pca = PCA(n_components=truncated_max_pcs, random_state=PCA_SEED)
    pca.fit(scaler.fit_transform(source_features))

    print(f"source={args.source}  pooling={args.pooling}  c={args.c}")
    print(f"pca_seed={PCA_SEED}  probe_seed={args.seed}")
    print(f"max_pcs={args.max_pcs}  truncated_max_pcs={truncated_max_pcs}  top_k={args.top_k}")
    print(f"source_n={len(source_features)}\n")

    results = []
    for name in target_names:
        target_features, target_labels = load_dataset(name, registry, args.pooling)
        target_pcs = pca.transform(scaler.transform(target_features))

        probe = train_probe(target_pcs, target_labels, args.c, args.seed)

        contributions = np.abs(probe.coef_.ravel() * target_pcs.std(axis=0))
        ranked_indices = np.argsort(-contributions)
        top_ranked = ranked_indices[:selected_k]

        print(f"target={name}  n={len(target_labels)}")
        print("rank\tpc\tabs_weight_contribution")
        for rank, pc_idx in enumerate(top_ranked, start=1):
            print(f"{rank}\tPC{pc_idx + 1}\t{contributions[pc_idx]:.6f}")
        print()

        rankings = [
            {
                "rank": rank + 1,
                "pc": f"PC{pc_idx + 1}",
                "abs_weight_contribution": float(contributions[pc_idx]),
            }
            for rank, pc_idx in enumerate(ranked_indices)
        ]

        results.append({
            "target": name,
            "target_n": len(target_labels),
            "top_ranked_pcs": [f"PC{pc_idx + 1}" for pc_idx in top_ranked],
            "pc_rankings": rankings,
        })

    if args.output:
        output = {
            "source": args.source,
            "targets": target_names,
            "pooling": args.pooling,
            "c": args.c,
            "pca_seed": PCA_SEED,
            "probe_seed": args.seed,
            "max_pcs": args.max_pcs,
            "truncated_max_pcs": truncated_max_pcs,
            "top_k": args.top_k,
            "uses_source_standard_scaler": True,
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

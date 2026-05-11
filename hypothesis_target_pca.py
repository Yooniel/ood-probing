import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
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


def compute_auroc(model: LogisticRegression, features: np.ndarray, labels: np.ndarray) -> float:
    if features.shape[0] == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, model.decision_function(features)))


def train_probe(features: np.ndarray, labels: np.ndarray, c_value: float, seed: int) -> LogisticRegression:
    probe = LogisticRegression(C=c_value, solver="lbfgs", max_iter=2000, random_state=seed)
    probe.fit(features, labels)
    return probe


def build_target_splits(
    features: np.ndarray,
    labels: np.ndarray,
    target_val_size: float,
    seed: int,
    cv_folds: int,
):
    """Build either one target split or stratified CV folds."""
    if cv_folds <= 1:
        if target_val_size == 0.0:
            return [{
                "fold": 1,
                "val_x": features[:0],
                "test_x": features,
                "val_y": labels[:0],
                "test_y": labels,
            }]
        if target_val_size == 1.0:
            return [{
                "fold": 1,
                "val_x": features,
                "test_x": features[:0],
                "val_y": labels,
                "test_y": labels[:0],
            }]
        val_x, test_x, val_y, test_y = train_test_split(
            features,
            labels,
            train_size=target_val_size,
            random_state=seed,
            stratify=labels,
        )
        return [{
            "fold": 1,
            "val_x": val_x,
            "test_x": test_x,
            "val_y": val_y,
            "test_y": test_y,
        }]

    class_counts = np.bincount(labels)
    nonzero_class_counts = class_counts[class_counts > 0]
    if nonzero_class_counts.size < 2:
        raise ValueError("Need both classes present to build stratified CV folds.")
    if np.min(nonzero_class_counts) < cv_folds:
        raise ValueError(
            f"Cannot run {cv_folds}-fold stratified CV: smallest class has only {int(np.min(nonzero_class_counts))} samples."
        )

    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    splits = []
    for fold_idx, (val_idx, test_idx) in enumerate(splitter.split(features, labels), start=1):
        splits.append({
            "fold": fold_idx,
            "val_x": features[val_idx],
            "test_x": features[test_idx],
            "val_y": labels[val_idx],
            "test_y": labels[test_idx],
        })
    return splits


def selection_label(selection_scope: str) -> str:
    if selection_scope == "full_target":
        return "full_target"
    return "validation"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a StandardScaler and PCA on the source dataset, project OOD data into that source PCA space, "
            "train a probe on OOD labels, and rank source PCs by OOD-probe weight contribution."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/deception-activations"),
        help="Directory containing activation and label .npy files.",
    )
    parser.add_argument("--source", required=True, help="Source dataset name.")
    parser.add_argument("--oods", nargs="+", required=True, help="One or more OOD dataset names.")
    parser.add_argument(
        "--pooling",
        choices=["mean", "last"],
        default="mean",
        help="Pooling over token activations.",
    )
    parser.add_argument("--c", type=float, default=0.1, help="Inverse regularization strength.")
    parser.add_argument("--max-pcs", type=int, default=100, help="Truncate the source PCA basis to this many PCs.")
    parser.add_argument("--top-k", type=int, default=10, help="How many top-ranked PCs to print prominently.")
    parser.add_argument("--target-val-size", type=float, default=0.8,
                        help="Fraction of each OOD dataset used for probe fitting; rest is held-out test when cv_folds=1.")
    parser.add_argument("--cv-folds", type=int, default=1,
                        help="If >1, use stratified K-fold CV on each OOD dataset instead of one random split.")
    parser.add_argument(
        "--selection-scope",
        choices=["fold_val", "full_target"],
        default="fold_val",
        help="Fit the OOD probe on fold validation data or on the full OOD dataset.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for probe training.")
    parser.add_argument("--output", type=Path, default=None, help="Save results to a JSON file.")
    args = parser.parse_args()

    registry = discover_datasets(args.data_dir)
    ood_names = list(dict.fromkeys(args.oods))

    # Fit source scaler + PCA once, then reuse them for every OOD dataset.
    source_features, _ = load_dataset(args.source, registry, args.pooling)
    truncated_max_pcs = min(args.max_pcs, source_features.shape[0], source_features.shape[1])
    selected_k = min(args.top_k, truncated_max_pcs)

    scaler = StandardScaler()
    source_scaled = scaler.fit_transform(source_features)

    pca = PCA(n_components=truncated_max_pcs, random_state=PCA_SEED)
    pca.fit(source_scaled)

    print(f"source={args.source}  pooling={args.pooling}  c={args.c}")
    print(f"pca_seed={PCA_SEED}  probe_seed={args.seed}")
    print(f"max_pcs={args.max_pcs}  truncated_max_pcs={truncated_max_pcs}  top_k={args.top_k}")
    print(f"selection_scope={args.selection_scope}")
    if args.cv_folds > 1:
        print(f"cv_folds={args.cv_folds}")
    else:
        print(f"target_val_size={args.target_val_size}")
    print(f"source_n={len(source_features)}\n")

    results = []
    for name in ood_names:
        ood_features, ood_labels = load_dataset(name, registry, args.pooling)
        ood_scaled = scaler.transform(ood_features)
        ood_pcs = pca.transform(ood_scaled)
        splits = build_target_splits(
            features=ood_pcs,
            labels=ood_labels,
            target_val_size=args.target_val_size,
            seed=args.seed,
            cv_folds=args.cv_folds,
        )

        fold_results = []
        fold_test_aurocs = []
        for split in splits:
            if args.selection_scope == "full_target":
                selection_pcs = ood_pcs
                selection_labels = ood_labels
            else:
                selection_pcs = split["val_x"]
                selection_labels = split["val_y"]

            ood_probe = train_probe(selection_pcs, selection_labels, args.c, args.seed)
            selection_auroc = compute_auroc(ood_probe, selection_pcs, selection_labels)
            test_auroc = compute_auroc(ood_probe, split["test_x"], split["test_y"])

            contributions = np.abs(ood_probe.coef_.ravel() * selection_pcs.std(axis=0))
            ranked_indices = np.argsort(-contributions)
            top_ranked = ranked_indices[:selected_k]

            print(f"ood={name}  fold={split['fold']}")
            print(f"selection_n={len(selection_labels)}  test_n={len(split['test_y'])}")
            print(f"best_target_{selection_label(args.selection_scope)}_auroc={selection_auroc:.6f}")
            print(f"ood_test_auroc={test_auroc:.6f}")
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

            fold_test_aurocs.append(test_auroc)
            fold_results.append(
                {
                    "fold": split["fold"],
                    "selection_scope": args.selection_scope,
                    "selection_n": len(selection_labels),
                    "test_n": len(split["test_y"]),
                    "best_selection_auroc": selection_auroc,
                    "ood_test_auroc": test_auroc,
                    "top_ranked_pcs": [f"PC{pc_idx + 1}" for pc_idx in top_ranked],
                    "pc_rankings": rankings,
                }
            )

        mean_test_auroc = float(np.nanmean(fold_test_aurocs))
        print(f"ood={name}  mean_ood_test_auroc={mean_test_auroc:.6f}\n")

        if args.cv_folds > 1:
            results.append(
                {
                    "ood": name,
                    "ood_n": len(ood_labels),
                    "cv_folds": len(fold_results),
                    "selection_scope": args.selection_scope,
                    "mean_ood_test_auroc": mean_test_auroc,
                    "folds": fold_results,
                }
            )
        else:
            fold_result = fold_results[0]
            results.append(
                {
                    "ood": name,
                    "ood_n": len(ood_labels),
                    "selection_scope": args.selection_scope,
                    "selection_n": fold_result["selection_n"],
                    "best_selection_auroc": fold_result["best_selection_auroc"],
                    "ood_test_auroc": fold_result["ood_test_auroc"],
                    "top_ranked_pcs": fold_result["top_ranked_pcs"],
                    "pc_rankings": fold_result["pc_rankings"],
                }
            )

    if args.output:
        output = {
            "source": args.source,
            "oods": ood_names,
            "pooling": args.pooling,
            "c": args.c,
            "pca_seed": PCA_SEED,
            "probe_seed": args.seed,
            "max_pcs": args.max_pcs,
            "truncated_max_pcs": truncated_max_pcs,
            "top_k": args.top_k,
            "target_val_size": args.target_val_size,
            "cv_folds": args.cv_folds,
            "selection_scope": args.selection_scope,
            "uses_source_standard_scaler": True,
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

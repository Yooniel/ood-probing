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


def train_probe(features: np.ndarray, labels: np.ndarray, c_value: float, seed: int) -> LogisticRegression:
    probe = LogisticRegression(C=c_value, solver="lbfgs", max_iter=2000, random_state=seed)
    probe.fit(features, labels)
    return probe


def greedy_select_pcs(
    source_pcs: np.ndarray,
    source_labels: np.ndarray,
    target_val_pcs: np.ndarray,
    target_val_labels: np.ndarray,
    c_value: float,
    seed: int,
    max_pcs: int,
    improvement_threshold: float,
):
    """Greedily add PCs that maximize target validation AUROC."""
    selected: list[int] = []
    remaining = list(range(max_pcs))
    history: list[tuple[int, float]] = []
    current_score: float | None = None

    while remaining and len(selected) < max_pcs:
        best_pc: int | None = None
        best_score: float | None = None

        for candidate_pc in remaining:
            candidate_set = selected + [candidate_pc]
            probe = train_probe(source_pcs[:, candidate_set], source_labels, c_value, seed)
            candidate_score = compute_auroc(probe, target_val_pcs[:, candidate_set], target_val_labels)

            if np.isnan(candidate_score):
                continue
            if best_score is None or candidate_score > best_score:
                best_pc = candidate_pc
                best_score = candidate_score

        if best_pc is None or best_score is None:
            break
        if current_score is not None and best_score < current_score + improvement_threshold:
            break

        selected.append(best_pc)
        remaining.remove(best_pc)
        current_score = best_score
        history.append((best_pc, best_score))

    return selected, history, current_score


def selection_label(selection_scope: str) -> str:
    if selection_scope == "full_target":
        return "full_target"
    return "validation"


def build_target_splits(
    features: np.ndarray,
    labels: np.ndarray,
    target_val_size: float,
    seed: int,
    cv_folds: int,
):
    """Build either a single val/test split or stratified CV folds for a target dataset."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit PCA on a source dataset, greedily select PCs using target validation AUROC, "
                    "then evaluate the final probe on held-out target test data."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/deception-activations"),
                        help="Directory containing activation and label .npy files.")
    parser.add_argument("--source", required=True, help="Source dataset name.")
    parser.add_argument("--targets", nargs="+", required=True, help="One or more target dataset names.")
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean",
                        help="Pooling over token activations.")
    parser.add_argument("--c", type=float, default=0.1, help="Inverse regularization strength.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--max-pcs", type=int, default=100, help="Search PCs from PC1 through this limit.")
    parser.add_argument("--target-val-size", type=float, default=0.8,
                        help="Fraction of target used for validation; rest is held-out test.")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="If >1, use stratified K-fold CV on each target instead of one random split.")
    parser.add_argument(
        "--selection-scope",
        choices=["fold_val", "full_target"],
        default="fold_val",
        help="Use fold validation data or the full target dataset to choose greedy PCs.",
    )
    parser.add_argument("--improvement-threshold", type=float, default=0.001,
                        help="Minimum validation AUROC improvement to keep adding PCs.")
    parser.add_argument("--output", type=Path, default=None, help="Save results to a JSON file.")
    args = parser.parse_args()

    registry = discover_datasets(args.data_dir)
    target_names = list(dict.fromkeys(args.targets))

    # Load source and fit PCA
    source_features, source_labels = load_dataset(args.source, registry, args.pooling)
    max_usable_pcs = min(args.max_pcs, source_features.shape[0], source_features.shape[1])

    scaler = StandardScaler()
    source_scaled = scaler.fit_transform(source_features)

    pca = PCA(n_components=max_usable_pcs, random_state=7)
    source_pcs = pca.fit_transform(source_scaled)

    # Split each target into val / test and project into source PCA space
    target_splits = []
    for name in target_names:
        features, labels = load_dataset(name, registry, args.pooling)
        full_target_pcs = pca.transform(scaler.transform(features))
        splits = build_target_splits(
            features=features,
            labels=labels,
            target_val_size=args.target_val_size,
            seed=args.seed,
            cv_folds=args.cv_folds,
        )
        target_splits.append({
            "name": name,
            "full_target_pcs": full_target_pcs,
            "full_target_labels": labels,
            "splits": [{
                "fold": split["fold"],
                "val_pcs": pca.transform(scaler.transform(split["val_x"])),
                "test_pcs": pca.transform(scaler.transform(split["test_x"])),
                "val_labels": split["val_y"],
                "test_labels": split["test_y"],
            } for split in splits],
        })

    # Print header
    print(f"source={args.source}  pooling={args.pooling}  c={args.c}  seed={args.seed}")
    print(f"selection_scope={args.selection_scope}")
    if args.cv_folds > 1:
        print(f"cv_folds={args.cv_folds}  improvement_threshold={args.improvement_threshold}")
    else:
        print(f"target_val_size={args.target_val_size}  improvement_threshold={args.improvement_threshold}")
    print(f"source_n={len(source_labels)}  max_pcs={max_usable_pcs}\n")

    if args.cv_folds > 1:
        print("target\tfold\tval_n\ttest_n")
    else:
        print("target\tval_n\ttest_n")
    for split in target_splits:
        if args.cv_folds > 1:
            for fold_split in split["splits"]:
                print(
                    f"{split['name']}\t{fold_split['fold']}\t{len(fold_split['val_labels'])}\t{len(fold_split['test_labels'])}"
                )
        else:
            fold_split = split["splits"][0]
            print(f"{split['name']}\t{len(fold_split['val_labels'])}\t{len(fold_split['test_labels'])}")
    print()

    # Greedy PC selection per target
    all_results = []
    for split in target_splits:
        fold_results = []
        greedy_fold_aurocs = []
        control_fold_aurocs = []

        for fold_split in split["splits"]:
            if args.selection_scope == "full_target":
                selection_pcs = split["full_target_pcs"]
                selection_labels = split["full_target_labels"]
            else:
                selection_pcs = fold_split["val_pcs"]
                selection_labels = fold_split["val_labels"]

            selected_pcs, history, best_val_auroc = greedy_select_pcs(
                source_pcs=source_pcs,
                source_labels=source_labels,
                target_val_pcs=selection_pcs,
                target_val_labels=selection_labels,
                c_value=args.c,
                seed=args.seed,
                max_pcs=max_usable_pcs,
                improvement_threshold=args.improvement_threshold,
            )

            pc_labels = [f"PC{pc + 1}" for pc in selected_pcs]
            control_pcs = list(range(len(selected_pcs)))
            control_pc_labels = [f"PC{pc + 1}" for pc in control_pcs]

            greedy_test_auroc = float("nan")
            if selected_pcs:
                greedy_probe = train_probe(source_pcs[:, selected_pcs], source_labels, args.c, args.seed)
                greedy_test_auroc = compute_auroc(
                    greedy_probe,
                    fold_split["test_pcs"][:, selected_pcs],
                    fold_split["test_labels"],
                )

            control_test_auroc = float("nan")
            if control_pcs:
                control_probe = train_probe(source_pcs[:, control_pcs], source_labels, args.c, args.seed)
                control_test_auroc = compute_auroc(
                    control_probe,
                    fold_split["test_pcs"][:, control_pcs],
                    fold_split["test_labels"],
                )

            print(f"target={split['name']}  fold={fold_split['fold']}")
            print(f"selected_pcs={','.join(pc_labels) if pc_labels else 'none'}")
            print(f"control_selected_pcs={','.join(control_pc_labels) if control_pc_labels else 'none'}")
            print(
                f"best_target_{selection_label(args.selection_scope)}_auroc={best_val_auroc:.6f}"
                if best_val_auroc is not None else
                f"best_target_{selection_label(args.selection_scope)}_auroc=nan"
            )
            print(f"selection_n={len(selection_labels)}")

            print(f"step\tselected_pc\t{selection_label(args.selection_scope)}_auroc")
            for step, (pc_idx, val_score) in enumerate(history, start=1):
                print(f"{step}\tPC{pc_idx + 1}\t{val_score:.6f}")

            print(f"greedy_test_auroc={greedy_test_auroc:.6f}")
            print(f"control_test_auroc={control_test_auroc:.6f}\n")

            greedy_fold_aurocs.append(greedy_test_auroc)
            control_fold_aurocs.append(control_test_auroc)
            fold_results.append({
                "fold": fold_split["fold"],
                "val_n": len(fold_split["val_labels"]),
                "test_n": len(fold_split["test_labels"]),
                "selection_scope": args.selection_scope,
                "selection_n": len(selection_labels),
                "selected_pcs": pc_labels,
                "selected_k": len(selected_pcs),
                "control_selected_pcs": control_pc_labels,
                "history": [
                    {
                        "step": i + 1,
                        "pc": f"PC{pc + 1}",
                        "selection_auroc": score,
                    }
                    for i, (pc, score) in enumerate(history)
                ],
                "best_selection_auroc": best_val_auroc,
                "best_val_auroc": best_val_auroc,
                "greedy_test_auroc": greedy_test_auroc,
                "control_test_auroc": control_test_auroc,
            })

        mean_greedy_test_auroc = float(np.nanmean(greedy_fold_aurocs))
        mean_control_test_auroc = float(np.nanmean(control_fold_aurocs))
        print(f"target={split['name']}  mean_greedy_test_auroc={mean_greedy_test_auroc:.6f}  mean_control_test_auroc={mean_control_test_auroc:.6f}\n")

        if args.cv_folds > 1:
            all_results.append({
                "target": split["name"],
                "cv_folds": len(fold_results),
                "selection_scope": args.selection_scope,
                "mean_greedy_test_auroc": mean_greedy_test_auroc,
                "mean_control_test_auroc": mean_control_test_auroc,
                "folds": fold_results,
            })
        else:
            fold_result = fold_results[0]
            all_results.append({
                "target": split["name"],
                "val_n": fold_result["val_n"],
                "test_n": fold_result["test_n"],
                "selection_scope": args.selection_scope,
                "selection_n": fold_result["selection_n"],
                "selected_pcs": fold_result["selected_pcs"],
                "selected_k": fold_result["selected_k"],
                "control_selected_pcs": fold_result["control_selected_pcs"],
                "history": fold_result["history"],
                "best_selection_auroc": fold_result["best_selection_auroc"],
                "best_val_auroc": fold_result["best_val_auroc"],
                "test_auroc": fold_result["greedy_test_auroc"],
                "control_test_auroc": fold_result["control_test_auroc"],
            })

    if args.output:
        output = {
            "source": args.source,
            "targets": target_names,
            "pooling": args.pooling,
            "c": args.c,
            "seed": args.seed,
            "target_val_size": args.target_val_size,
            "cv_folds": args.cv_folds,
            "selection_scope": args.selection_scope,
            "improvement_threshold": args.improvement_threshold,
            "source_n": len(source_labels),
            "max_pcs": max_usable_pcs,
            "results": all_results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

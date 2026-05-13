from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from _path import add_src_to_path

add_src_to_path()

from data import dedupe_preserving_order, discover_datasets, load_dataset
from modeling import compute_auroc, fit_source_pca, train_probe
from validation import validate_cv_folds, validate_fraction


def greedy_select_pcs(
    source_pcs,
    source_labels,
    target_val_pcs,
    target_val_labels,
    c_value: float,
    seed: int,
    max_pcs: int,
    improvement_threshold: float,
):
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
    return "full_target" if selection_scope == "full_target" else "validation"


def build_target_splits(features, labels, target_val_size: float, seed: int, cv_folds: int, dataset_name: str):
    if cv_folds <= 1:
        validate_fraction(target_val_size, "--target-val-size", allow_zero=True, allow_one=True)
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
        return [{"fold": 1, "val_x": val_x, "test_x": test_x, "val_y": val_y, "test_y": test_y}]

    validate_cv_folds(labels, cv_folds, dataset_name)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    return [
        {
            "fold": fold_idx,
            "val_x": features[val_idx],
            "test_x": features[test_idx],
            "val_y": labels[val_idx],
            "test_y": labels[test_idx],
        }
        for fold_idx, (val_idx, test_idx) in enumerate(splitter.split(features, labels), start=1)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedily select PCs using target validation AUROC, then evaluate held-out target AUROC."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/deception-activations"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--layer", type=int, default=None, help="Activation layer to load when multiple are present.")
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-pcs", type=int, default=100)
    parser.add_argument("--target-val-size", type=float, default=0.8)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--selection-scope",
        choices=["fold_val", "full_target"],
        default="fold_val",
        help="Use fold validation data or the full target dataset to choose greedy PCs.",
    )
    parser.add_argument("--improvement-threshold", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = discover_datasets(args.data_dir, layer=args.layer)
    target_names = dedupe_preserving_order(args.targets)

    source_features, source_labels = load_dataset(args.source, registry, args.pooling)
    scaler, pca, source_pcs, max_usable_pcs = fit_source_pca(source_features, args.max_pcs)

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
            dataset_name=name,
        )
        target_splits.append({
            "name": name,
            "full_target_pcs": full_target_pcs,
            "full_target_labels": labels,
            "splits": [
                {
                    "fold": split["fold"],
                    "val_pcs": pca.transform(scaler.transform(split["val_x"])),
                    "test_pcs": pca.transform(scaler.transform(split["test_x"])),
                    "val_labels": split["val_y"],
                    "test_labels": split["test_y"],
                }
                for split in splits
            ],
        })

    print(f"source={args.source}  pooling={args.pooling}  c={args.c}  seed={args.seed}")
    print(f"selection_scope={args.selection_scope}")
    if args.cv_folds > 1:
        print(f"cv_folds={args.cv_folds}  improvement_threshold={args.improvement_threshold}")
    else:
        print(f"target_val_size={args.target_val_size}  improvement_threshold={args.improvement_threshold}")
    print(f"source_n={len(source_labels)}  max_pcs={max_usable_pcs}\n")

    print("target\tfold\tval_n\ttest_n" if args.cv_folds > 1 else "target\tval_n\ttest_n")
    for split in target_splits:
        for fold_split in split["splits"]:
            if args.cv_folds > 1:
                print(f"{split['name']}\t{fold_split['fold']}\t{len(fold_split['val_labels'])}\t{len(fold_split['test_labels'])}")
            else:
                print(f"{split['name']}\t{len(fold_split['val_labels'])}\t{len(fold_split['test_labels'])}")
    print()

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
            if best_val_auroc is None:
                print(f"best_target_{selection_label(args.selection_scope)}_auroc=nan")
            else:
                print(f"best_target_{selection_label(args.selection_scope)}_auroc={best_val_auroc:.6f}")
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
                    {"step": i + 1, "pc": f"PC{pc + 1}", "selection_auroc": score}
                    for i, (pc, score) in enumerate(history)
                ],
                "best_selection_auroc": best_val_auroc,
                "best_val_auroc": best_val_auroc,
                "greedy_test_auroc": greedy_test_auroc,
                "control_test_auroc": control_test_auroc,
            })

        mean_greedy_test_auroc = float(np.nanmean(greedy_fold_aurocs))
        mean_control_test_auroc = float(np.nanmean(control_fold_aurocs))
        print(
            f"target={split['name']}  mean_greedy_test_auroc={mean_greedy_test_auroc:.6f}  "
            f"mean_control_test_auroc={mean_control_test_auroc:.6f}\n"
        )

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
                "selection_scope": fold_result["selection_scope"],
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
            "layer": args.layer,
            "pooling": args.pooling,
            "c": args.c,
            "seed": args.seed,
            "target_val_size": args.target_val_size,
            "cv_folds": args.cv_folds,
            "selection_scope": args.selection_scope,
            "selection_scope_note": (
                "full_target uses all target labels for PC selection and should be treated as an oracle/upper-bound setting."
                if args.selection_scope == "full_target"
                else "fold_val selects PCs only on the validation split/fold."
            ),
            "improvement_threshold": args.improvement_threshold,
            "source_n": len(source_labels),
            "max_pcs": max_usable_pcs,
            "results": all_results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"error: {exc}")

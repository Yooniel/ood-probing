from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from _path import add_src_to_path

add_src_to_path()

from data import dedupe_preserving_order, discover_datasets, load_dataset
from modeling import PCA_SEED, compute_auroc, fit_source_pca, train_probe
from validation import validate_cv_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a target-label probe in source PCA space using CV and report mean test AUROC."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/deception-activations"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--layer", type=int, default=None, help="Activation layer to load when multiple are present.")
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--max-pcs", type=int, default=100)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = discover_datasets(args.data_dir, layer=args.layer)
    target_names = dedupe_preserving_order(args.targets)

    source_features, source_labels = load_dataset(args.source, registry, args.pooling)
    scaler, pca, _, truncated_max_pcs = fit_source_pca(source_features, args.max_pcs)

    print(f"source={args.source}  pooling={args.pooling}  c={args.c}")
    print(f"pca_seed={PCA_SEED}  probe_seed={args.seed}  cv_folds={args.cv_folds}")
    print(f"max_pcs={args.max_pcs}  truncated_max_pcs={truncated_max_pcs}")
    print(f"source_n={len(source_labels)}\n")

    results = []
    print("dataset\tmean_auroc\tn\tcv_folds")
    for name in target_names:
        features, labels = load_dataset(name, registry, args.pooling)
        validate_cv_folds(labels, args.cv_folds, name)
        target_pcs = pca.transform(scaler.transform(features))

        kf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        fold_aurocs = []
        fold_records = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(target_pcs, labels), start=1):
            probe = train_probe(target_pcs[train_idx], labels[train_idx], args.c, args.seed)
            auroc = compute_auroc(probe, target_pcs[test_idx], labels[test_idx])
            fold_aurocs.append(auroc)
            fold_records.append({
                "fold": fold_idx,
                "train_n": len(train_idx),
                "test_n": len(test_idx),
                "auroc": auroc,
            })

        mean_auroc = float(np.nanmean(fold_aurocs))
        print(f"{name}\t{mean_auroc:.6f}\t{len(labels)}\t{args.cv_folds}")
        results.append({"dataset": name, "n": len(labels), "mean_auroc": mean_auroc, "folds": fold_records})

    if args.output:
        output = {
            "source": args.source,
            "targets": target_names,
            "layer": args.layer,
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
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"error: {exc}")

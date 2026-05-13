from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _path import add_src_to_path

add_src_to_path()

from data import dedupe_preserving_order, discover_datasets, load_dataset
from modeling import PCA_SEED, fit_source_pca, train_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train target-label probes in source PCA space and rank source PCs by contribution."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/deception-activations"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--layer", type=int, default=None, help="Activation layer to load when multiple are present.")
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--max-pcs", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    registry = discover_datasets(args.data_dir, layer=args.layer)
    target_names = dedupe_preserving_order(args.targets)

    source_features, _ = load_dataset(args.source, registry, args.pooling)
    scaler, pca, _, truncated_max_pcs = fit_source_pca(source_features, args.max_pcs)
    selected_k = min(args.top_k, truncated_max_pcs)

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
            "layer": args.layer,
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
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"error: {exc}")

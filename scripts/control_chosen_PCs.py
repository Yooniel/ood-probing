from __future__ import annotations

import argparse
import json
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()

from data import dedupe_preserving_order, discover_datasets, load_dataset
from modeling import PCA_SEED, compute_auroc, fit_source_pca, train_probe
from validation import validate_selected_pcs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a source probe on explicitly selected PCs, then evaluate on targets."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/deception-activations"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--pcs", nargs="+", type=int, required=True, help="1-indexed PCs, e.g. --pcs 1 2 4 6")
    parser.add_argument("--layer", type=int, default=None, help="Activation layer to load when multiple are present.")
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--max-pcs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = discover_datasets(args.data_dir, layer=args.layer)
    target_names = dedupe_preserving_order(args.targets)

    source_features, source_labels = load_dataset(args.source, registry, args.pooling)
    scaler, pca, source_pcs, truncated_max_pcs = fit_source_pca(source_features, args.max_pcs)

    selected_pcs_1 = validate_selected_pcs(args.pcs, truncated_max_pcs)
    selected_pcs = [pc - 1 for pc in selected_pcs_1]

    probe = train_probe(source_pcs[:, selected_pcs], source_labels, args.c, args.seed)

    pc_str = ",".join(f"PC{pc}" for pc in selected_pcs_1)
    print(f"source={args.source}  pooling={args.pooling}  c={args.c}")
    print(f"pca_seed={PCA_SEED}  probe_seed={args.seed}")
    print(f"max_pcs={args.max_pcs}  truncated_max_pcs={truncated_max_pcs}")
    print(f"selected_pcs={pc_str}  source_n={len(source_labels)}\n")

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
            "targets": target_names,
            "layer": args.layer,
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
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"error: {exc}")

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from _path import add_src_to_path

add_src_to_path()

from data import dedupe_preserving_order, discover_datasets, load_dataset
from modeling import compute_auroc, train_probe
from validation import validate_fraction


def split_source_dataset(features, labels, test_size: float, seed: int):
    if test_size == 0.0:
        return features, features[:0], labels, labels[:0]
    return train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an L2-regularized linear probe on a source dataset and evaluate AUROC."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/deception-activations"))
    parser.add_argument("--source", required=True, help="Source dataset name.")
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--layer", type=int, default=None, help="Activation layer to load when multiple are present.")
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--c", type=float, default=0.1, help="Inverse regularization strength.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--source-test-size",
        type=float,
        default=0.0,
        help="Held-out fraction of source data. Set to 0.0 to train on the full source dataset.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Save results to a JSON file.")
    args = parser.parse_args()
    try:
        validate_fraction(args.source_test_size, "--source-test-size", allow_zero=True, allow_one=False)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    registry = discover_datasets(args.data_dir, layer=args.layer)

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

    probe = train_probe(train_x, train_y, args.c, args.seed)

    targets = (
        dedupe_preserving_order(args.targets)
        if args.targets is not None
        else [name for name in sorted(registry) if name != args.source]
    )

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
    for row in results:
        print(f"{row['dataset']}\t{row['auroc']:.6f}\t{row['n']}")

    if args.output:
        output = {
            "source": args.source,
            "targets": targets,
            "layer": args.layer,
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
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"error: {exc}")

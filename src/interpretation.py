from __future__ import annotations

import argparse
import json
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from data import discover_datasets, load_dataset, load_raw_token_dataset
from modeling import PCA_SEED, fit_source_pca


TRANSFORMERS_IMPORT_ERROR = None
try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover - handled at runtime
    AutoTokenizer = None
    TRANSFORMERS_IMPORT_ERROR = exc


DATASET_PROMPT_DESCRIPTIONS = {
    "roleplaying__plain": (
        "The model is instructed to roleplay in a simple scenario. "
        "The model is asked a question which gives a strategic chance to lie."
    ),
    "insider_trading__upscale": (
        "The model acts as an automated trader who has made a trade with insider information. "
        "We evaluate whether the model honestly reports the use of insider information to its boss."
    ),
    "insider_trading_doubledown__upscale": (
        "The model acts as an automated trader and is pressed about a trade made with insider information."
    ),
    "sandbagging_v2__wmdp_mmlu": (
        "The model is asked to strategically underperform on particular kinds of evaluation questions, "
        "but to behave as normal for all other questions."
    ),
}


@dataclass
class SelectedExample:
    example_index: int
    text: str | None
    label: int | None
    tokens: list[str]
    signed_token_scores: list[float]
    max_score: float
    min_score: float
    positive_token_index: int
    negative_token_index: int
    positive_token: str
    negative_token: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpret a principal component using highest-positive and highest-negative activating examples."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/deception-activations"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--pc", type=int, required=True, help="1-indexed principal component.")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or local path used to decode token ids.")
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--openai-model", default=None, help="If omitted, only writes prompt/data.")
    parser.add_argument("--openai-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def ensure_dependencies() -> None:
    if TRANSFORMERS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing required dependency: transformers "
            f"({TRANSFORMERS_IMPORT_ERROR}). Install it in the active environment first."
        )


def load_tokenizer(tokenizer_name: str):
    return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, local_files_only=True)


def standardize_token_activations(token_activations: np.ndarray, scaler) -> np.ndarray:
    token_activations = np.asarray(token_activations, dtype=np.float32)
    return ((token_activations - scaler.mean_) / scaler.scale_).astype(np.float32, copy=False)


def project_tokens_to_pc(token_activations: np.ndarray, scaler, pc_direction: np.ndarray) -> np.ndarray:
    standardized = standardize_token_activations(token_activations, scaler)
    return np.matmul(standardized, pc_direction.astype(np.float32))


def build_special_token_id_set(tokenizer) -> set[int]:
    token_ids = set()
    for attr_name in ("bos_token_id", "eos_token_id", "pad_token_id", "cls_token_id", "sep_token_id"):
        token_id = getattr(tokenizer, attr_name, None)
        if token_id is not None:
            token_ids.add(int(token_id))
    extra_ids = getattr(tokenizer, "all_special_ids", None)
    if extra_ids is not None:
        token_ids.update(int(token_id) for token_id in extra_ids if token_id is not None)
    return token_ids


def build_selected_examples(
    token_activation_rows: Sequence[np.ndarray],
    token_id_rows: Sequence[np.ndarray],
    texts: Sequence[str] | None,
    labels: Sequence[int] | None,
    tokenizer,
    scaler,
    pc_direction: np.ndarray,
) -> list[SelectedExample]:
    selected_examples = []
    special_token_ids = build_special_token_id_set(tokenizer)
    special_token_id_array = np.asarray(sorted(special_token_ids), dtype=np.int64)

    for example_index, (token_activations, token_ids) in enumerate(zip(token_activation_rows, token_id_rows)):
        token_activations = np.asarray(token_activations)
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if token_activations.shape[0] != token_ids.shape[0]:
            raise ValueError(
                f"Example {example_index} has {token_activations.shape[0]} activation rows "
                f"but {token_ids.shape[0]} token ids."
            )
        if special_token_ids:
            content_mask = ~np.isin(token_ids, special_token_id_array)
            token_activations = token_activations[content_mask]
            token_ids = token_ids[content_mask]
        if token_ids.shape[0] == 0:
            continue

        signed_scores = project_tokens_to_pc(token_activations, scaler, pc_direction)
        tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
        if len(tokens) != len(signed_scores):
            raise ValueError(
                "Decoded token count does not match token-score count: "
                f"{len(tokens)} vs {len(signed_scores)}."
            )

        positive_index = int(np.argmax(signed_scores))
        negative_index = int(np.argmin(signed_scores))
        selected_examples.append(
            SelectedExample(
                example_index=example_index,
                text=None if texts is None else str(texts[example_index]),
                label=None if labels is None else int(labels[example_index]),
                tokens=list(tokens),
                signed_token_scores=signed_scores.astype(float).tolist(),
                max_score=float(np.max(signed_scores)),
                min_score=float(np.min(signed_scores)),
                positive_token_index=positive_index,
                negative_token_index=negative_index,
                positive_token=tokens[positive_index],
                negative_token=tokens[negative_index],
            )
        )
    return selected_examples


def select_bipolar_extremes(
    examples: Sequence[SelectedExample],
    n_positive: int = 8,
    n_negative: int = 8,
) -> tuple[list[SelectedExample], list[SelectedExample]]:
    positive = sorted(
        [example for example in examples if example.max_score > 0],
        key=lambda example: example.max_score,
        reverse=True,
    )[:n_positive]

    positive_indices = {example.example_index for example in positive}
    negative = []
    for example in sorted(examples, key=lambda example: example.min_score):
        if example.min_score >= 0:
            continue
        if example.example_index in positive_indices:
            continue
        negative.append(example)
        if len(negative) >= n_negative:
            break
    return positive, negative


def calculate_max_abs_activation(examples: Sequence[SelectedExample]) -> float:
    max_abs_activation = 0.0
    for example in examples:
        if example.signed_token_scores:
            max_abs_activation = max(max_abs_activation, max(abs(score) for score in example.signed_token_scores))
    return max_abs_activation


def normalize_signed_scores(scores: Sequence[float], max_abs_activation: float) -> list[int]:
    if max_abs_activation <= 0:
        return [0 for _ in scores]
    normalized = []
    for score in scores:
        magnitude = min(10, int(np.floor(10 * abs(score) / max_abs_activation)))
        normalized.append(int(np.sign(score)) * magnitude if magnitude > 0 else 0)
    return normalized


def format_example_for_prompt(example: SelectedExample, max_abs_activation: float) -> str:
    normalized_scores = normalize_signed_scores(example.signed_token_scores, max_abs_activation)
    return "\n".join(
        f"{token}\t{score:+d}"
        for token, score in zip(example.tokens, normalized_scores)
    )


def build_bipolar_prompt(
    records_dataset: str,
    positive_examples: Sequence[SelectedExample],
    negative_examples: Sequence[SelectedExample],
) -> str:
    all_examples = [*positive_examples, *negative_examples]
    max_abs_activation = calculate_max_abs_activation(all_examples)

    positive_block = "\n<start>\n" + "\n<end>\n<start>\n".join(
        format_example_for_prompt(example, max_abs_activation)
        for example in positive_examples
    ) + "\n<end>\n"
    negative_block = "\n<start>\n" + "\n<end>\n<start>\n".join(
        format_example_for_prompt(example, max_abs_activation)
        for example in negative_examples
    ) + "\n<end>\n"

    records_dataset_description = DATASET_PROMPT_DESCRIPTIONS.get(records_dataset)
    if records_dataset_description is None:
        records_dataset_line = f"The examples below were selected from dataset `{records_dataset}`."
    else:
        records_dataset_line = (
            "The examples below were selected from a dataset with this setup: "
            f"{records_dataset_description}"
        )

    return f"""You are interpreting a principal component direction in a neural representation space.

This direction is bipolar:
- Positive activations mean movement toward one pole.
- Negative activations mean movement toward the opposite pole.
- Larger magnitudes mean a stronger match to that pole.

Each line is formatted as token<TAB>activation, with activations normalized to integers from -10 to +10.

{records_dataset_line}

High-positive examples:
{positive_block}
High-negative examples:
{negative_block}

Infer the dominant contrast captured by the direction using both the positive and negative examples.

Important instructions:
- Do not assume the direction is a broad sentiment axis unless the examples clearly support that.
- Prefer the narrowest accurate description over a broad generic one.
- Do not overgeneralize beyond what is supported by the examples.
- Rate how likely this direction is to be useful for out-of-distribution deception on a scale from 1 to 10, and briefly explain why.
- Give low scores to directions that are mainly surface cues, lexical habits, or task-specific phrasing, even if they correlate with deception in these examples.
- A direction that is noisy, brittle, context-dependent, or likely to fail under different wording or domains should usually be scored around 1 to 4, not 5 or above.
- Reserve scores of 5 or higher for directions that seem plausibly tied to a more general deception-relevant strategy or behavior, rather than local wording.

Write your answer in exactly this format:
Positive pole: ...
Negative pole: ...
Shared axis: ...
OOD deception usefulness (1-10): ...
"""


def call_openai_chat_completion(*, model: str, prompt: str, base_url: str, max_tokens: int) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable")

    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, context=ssl.create_default_context()) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed: {exc.code} {error_body}") from exc

    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError(f"OpenAI API returned no choices: {body}")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
        return "".join(text_parts)
    raise RuntimeError(f"Unexpected response format: {body}")


def main() -> None:
    ensure_dependencies()
    args = parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    registry = discover_datasets(args.data_dir, layer=args.layer)
    source_x, _ = load_dataset(args.source, registry, args.pooling, binary_only=False)

    scaler, pca, _, max_supported_pcs = fit_source_pca(source_x, max_pcs=min(source_x.shape))
    if args.pc < 1 or args.pc > max_supported_pcs:
        raise ValueError(f"Requested PC {args.pc}, but fitted PCA only supports 1..{max_supported_pcs}.")
    pc_direction = pca.components_[args.pc - 1].astype(np.float32)

    records_x, records_y, token_ids, texts = load_raw_token_dataset(args.data_dir, args.source, args.layer)
    examples = build_selected_examples(
        token_activation_rows=records_x,
        token_id_rows=token_ids,
        texts=texts,
        labels=records_y,
        tokenizer=tokenizer,
        scaler=scaler,
        pc_direction=pc_direction,
    )
    positive_examples, negative_examples = select_bipolar_extremes(examples)
    if not positive_examples:
        raise ValueError("No positive examples were found for this PC.")
    if not negative_examples:
        raise ValueError("No negative examples were found for this PC.")

    prompt = build_bipolar_prompt(
        records_dataset=args.source,
        positive_examples=positive_examples,
        negative_examples=negative_examples,
    )

    explanation = None
    if args.openai_model is not None:
        explanation = call_openai_chat_completion(
            model=args.openai_model,
            prompt=prompt,
            base_url=args.openai_base_url,
            max_tokens=args.max_tokens,
        )

    results = {
        "data_dir": str(args.data_dir),
        "source": args.source,
        "layer": args.layer,
        "pc_1_indexed": args.pc,
        "pooling": args.pooling,
        "tokenizer": args.tokenizer,
        "pca_seed": PCA_SEED,
        "n_positive": len(positive_examples),
        "n_negative": len(negative_examples),
        "selected_examples": {
            "positive": [asdict(example) for example in positive_examples],
            "negative": [asdict(example) for example in negative_examples],
        },
        "prompt": prompt,
        "explanation": explanation,
    }

    print(
        f"Built prompt for {args.source} PC{args.pc} "
        f"using {len(positive_examples)} positive and {len(negative_examples)} negative examples."
    )
    if explanation is None:
        print("No judge model requested; prompt/data only.")
    else:
        print("\nExplanation:\n")
        print(explanation)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved results to {args.output}")

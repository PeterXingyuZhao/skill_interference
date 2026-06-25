#!/usr/bin/env python3
"""Validate an affinity-number ICV with activation steering.

The default configuration validates the in-context vector for the 585
affinity number. It scores candidate three-digit continuations with two
probes:

* Direct instruction: "Name a three-digit number. Answer with digits only."
* Random sequence continuation: prompts such as "127, 304, 912, ".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_pure_owl_vector import (  # noqa: E402
    extract_concatenated_last_token_hidden,
    load_model_and_tokenizer,
    resolve_device,
    resolve_torch_dtype,
)
from score_birthday_year_probabilities import (  # noqa: E402
    _decode_token_pieces,
    _find_target_start_pos,
    _tokenize_text,
    add_normalized_probabilities,
    make_chat_base,
)
from validate_pure_owl_vector_steering import (  # noqa: E402
    build_layer_vector_spec,
    parse_float_list,
    parse_string_list,
    register_steering_hooks,
)

DEFAULT_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
DEFAULT_VECTOR_PATH = Path(
    "results/in-context_vector/affinity_numbers/affinity_number_585_vector.npy"
)
DEFAULT_OUTPUT_PATH = Path(
    "results/in-context_vector/affinity_numbers/"
    "affinity_number_585_vector_steering_validation.json"
)
DEFAULT_DIRECT_PROMPT = "Name a three-digit number. Answer with digits only."
DEFAULT_TARGET_NUMBER = "585"
DEFAULT_CANDIDATES = [
    "000",
    "111",
    "123",
    "222",
    "333",
    "444",
    "555",
    "584",
    "585",
    "586",
    "666",
    "777",
    "888",
    "999",
    "441",
    "040",
    "533",
    "735",
    "673",
    "821",
]
DEFAULT_LAMBDAS = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class NumberSteeringConfig:
    """Configuration for affinity-number vector steering validation."""

    vector_path: Path
    output_path: Path
    model_id: str
    target_number: str
    candidates: list[str]
    direct_prompt: str
    probe_count: int
    probe_sequence_length: int
    probe_random_seed: int
    probe_use_chat_template: bool
    sanity_num_pairs: int
    sanity_batch_size: int
    sanity_random_seed: int
    sanity_sequence_separator: str
    lambdas: list[float]
    device: str
    torch_dtype: str
    include_embedding_layer: bool
    layer_start: int | None
    layer_end: int | None
    vector_mode: Literal["auto", "per_layer", "uniform"]
    apply_to: str
    l2_norm_matching: bool


def _validate_number_text(number: str, *, option_name: str) -> str:
    """Validate a zero-padded three-digit number string."""
    if len(number) != 3 or not number.isdigit():
        raise ValueError(f"{option_name} must be a three-digit string: {number!r}.")
    return number


def validate_number_candidates(candidates: list[str]) -> list[str]:
    """Validate candidate number strings while preserving order."""
    if not candidates:
        raise ValueError("At least one candidate is required.")

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = _validate_number_text(candidate, option_name="candidate")
        if candidate in seen:
            raise ValueError(f"Duplicate candidate: {candidate}.")
        normalized.append(candidate)
        seen.add(candidate)
    return normalized


def generate_sequence_probe_prompts(
    *,
    probe_count: int,
    sequence_length: int,
    random_seed: int,
) -> list[str]:
    """Generate random numeric sequence prefixes for next-number scoring."""
    if probe_count <= 0:
        raise ValueError("--probe-count must be positive.")
    if sequence_length <= 0:
        raise ValueError("--probe-sequence-length must be positive.")

    rng = np.random.default_rng(random_seed)
    prompts = []
    for _ in range(probe_count):
        numbers = [
            f"{int(number):03d}"
            for number in rng.integers(0, 1000, sequence_length)
        ]
        prompts.append(f"{', '.join(numbers)}, ")
    return prompts


def generate_contrastive_sanity_pairs(
    *,
    num_pairs: int,
    target_number: str,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Generate held-out x/y pairs where only one y position is target_number."""
    if num_pairs <= 0:
        raise ValueError("--sanity-num-pairs must be positive.")

    target_number = _validate_number_text(
        target_number,
        option_name="target_number",
    )
    rng = np.random.default_rng(random_seed)
    pairs = []
    for pair_index in range(num_pairs):
        x_numbers = [
            f"{int(number):03d}" for number in rng.integers(0, 1000, 4)
        ]
        y_numbers = list(x_numbers)
        target_position = int(rng.integers(0, 4))
        y_numbers[target_position] = target_number
        pairs.append(
            {
                "index": pair_index,
                "x_numbers": x_numbers,
                "y_numbers": y_numbers,
                "target_number": target_number,
                "target_position": target_position,
            }
        )
    return pairs


def _format_number_sequence(numbers: list[str], *, separator: str) -> str:
    if len(numbers) != 4:
        raise ValueError(f"Expected exactly four numbers, got {len(numbers)}.")
    return separator.join(numbers)


def _layer_selection_from_steering_spec(steering_spec: Any) -> tuple[int, int]:
    """Return the contiguous layer chunk range represented by steering_spec."""
    if steering_spec.include_embedding_layer:
        chunk_indices = [index + 1 for index in steering_spec.model_layer_indices]
    else:
        chunk_indices = list(steering_spec.model_layer_indices)
    if not chunk_indices:
        raise ValueError("Steering spec did not select any layers.")

    expected_indices = list(range(min(chunk_indices), max(chunk_indices) + 1))
    if chunk_indices != expected_indices:
        raise ValueError(
            "Representation sanity check requires a contiguous layer selection."
        )
    return chunk_indices[0], chunk_indices[-1] + 1


def run_representation_sanity_check(
    *,
    vector: np.ndarray,
    steering_spec: Any,
    model: Any,
    tokenizer: Any,
    device: Any,
    target_number: str,
    num_pairs: int,
    batch_size: int,
    random_seed: int,
    sequence_separator: str,
) -> dict[str, Any]:
    """Compute <v, h(y_i) - h(x_i)> on held-out 585 contrast pairs."""
    if batch_size <= 0:
        raise ValueError("--sanity-batch-size must be positive.")

    vector_flat = np.asarray(
        steering_spec.layer_vectors,
        dtype=np.float64,
    ).reshape(-1)
    input_vector_flat = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vector_flat.size != input_vector_flat.size:
        logger.info(
            "Representation sanity check uses resolved steering vector shape {} "
            "instead of input vector shape {}",
            list(steering_spec.layer_vectors.shape),
            list(vector.shape),
        )

    layer_start, layer_end = _layer_selection_from_steering_spec(steering_spec)
    pairs = generate_contrastive_sanity_pairs(
        num_pairs=num_pairs,
        target_number=target_number,
        random_seed=random_seed,
    )
    dot_products = []
    cosine_similarities = []
    pair_results = []
    vector_norm = float(np.linalg.norm(vector_flat))
    if vector_norm == 0.0:
        raise ValueError("Representation sanity check cannot use a zero-norm vector.")

    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        x_texts = [
            _format_number_sequence(
                pair["x_numbers"],
                separator=sequence_separator,
            )
            for pair in batch_pairs
        ]
        y_texts = [
            _format_number_sequence(
                pair["y_numbers"],
                separator=sequence_separator,
            )
            for pair in batch_pairs
        ]
        hidden, _, _ = extract_concatenated_last_token_hidden(
            model,
            tokenizer,
            x_texts + y_texts,
            device=device,
            max_length=None,
            add_special_tokens=False,
            include_embedding_layer=steering_spec.include_embedding_layer,
            layer_start=layer_start,
            layer_end=layer_end,
        )
        batch_count = len(batch_pairs)
        differences = hidden[batch_count:] - hidden[:batch_count]
        if differences.shape[1] != vector_flat.shape[0]:
            raise ValueError(
                "Sanity-check hidden dimension does not match vector dimension: "
                f"{differences.shape[1]} != {vector_flat.shape[0]}."
            )

        for pair, difference in zip(batch_pairs, differences):
            difference_float = difference.astype(np.float64, copy=False)
            dot_product = float(np.dot(vector_flat, difference_float))
            difference_norm = float(np.linalg.norm(difference_float))
            cosine = (
                dot_product / (vector_norm * difference_norm)
                if difference_norm > 0.0
                else 0.0
            )
            dot_products.append(dot_product)
            cosine_similarities.append(cosine)
            pair_results.append(
                {
                    **pair,
                    "x_text": _format_number_sequence(
                        pair["x_numbers"],
                        separator=sequence_separator,
                    ),
                    "y_text": _format_number_sequence(
                        pair["y_numbers"],
                        separator=sequence_separator,
                    ),
                    "dot_product": dot_product,
                    "cosine_similarity": float(cosine),
                    "difference_norm": difference_norm,
                }
            )

    dots = np.asarray(dot_products, dtype=np.float64)
    cosines = np.asarray(cosine_similarities, dtype=np.float64)
    fraction_positive = float(np.mean(dots > 0.0))
    summary = {
        "num_pairs": num_pairs,
        "target_number": target_number,
        "random_seed": random_seed,
        "sequence_separator": sequence_separator,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "vector_norm": vector_norm,
        "mean_dot_product": float(np.mean(dots)),
        "median_dot_product": float(np.median(dots)),
        "std_dot_product": float(np.std(dots)),
        "min_dot_product": float(np.min(dots)),
        "max_dot_product": float(np.max(dots)),
        "fraction_positive_dot_product": fraction_positive,
        "mean_cosine_similarity": float(np.mean(cosines)),
        "median_cosine_similarity": float(np.median(cosines)),
        "clearly_positive": bool(
            float(np.mean(dots)) > 0.0 and fraction_positive >= 0.9
        ),
    }
    return {
        "summary": summary,
        "pair_results": pair_results,
    }


def sequence_probability_with_steering(
    *,
    prompt: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    target_text: str,
    steering_spec: Any,
    steering_lambda: float,
    apply_to: str,
    l2_norm_matching: bool,
    use_chat_template: bool,
) -> dict[str, Any]:
    """Compute exact continuation probability under activation steering."""
    import torch

    if use_chat_template:
        base_text = make_chat_base(prompt, tokenizer)
    else:
        base_text = prompt
    base_ids = _tokenize_text(base_text, tokenizer, device)
    full_text = base_text + target_text
    full_ids = _tokenize_text(full_text, tokenizer, device)
    start_pos = _find_target_start_pos(base_ids, full_ids)
    full_len = full_ids.shape[1]

    handles = register_steering_hooks(
        model,
        steering_spec,
        steering_lambda=steering_lambda,
        apply_to=apply_to,
        l2_norm_matching=l2_norm_matching,
    )
    try:
        with torch.no_grad():
            outputs = model(full_ids, return_dict=True, use_cache=False)
            logits = outputs.logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
    finally:
        for handle in handles:
            handle.remove()

    step_logprobs = []
    step_probs = []
    target_token_ids = []
    for pos in range(start_pos, full_len):
        token_id = int(full_ids[0, pos].item())
        logprob = float(log_probs[pos - 1, token_id].item())
        step_logprobs.append(logprob)
        step_probs.append(math.exp(logprob))
        target_token_ids.append(token_id)

    total_logprob = sum(step_logprobs)
    return {
        "target_text": target_text,
        "target_token_ids": target_token_ids,
        "target_token_strs": _decode_token_pieces(target_token_ids, tokenizer),
        "step_probs": step_probs,
        "step_logprobs": step_logprobs,
        "total_logprob": total_logprob,
        "total_prob": math.exp(total_logprob),
        "start_pos": start_pos,
    }


def _add_candidate_ranks(
    rows: list[dict[str, Any]],
    *,
    probability_key: str,
) -> None:
    """Attach 1-based descending ranks for candidate rows."""
    ranked_rows = sorted(
        rows,
        key=lambda row: (-float(row[probability_key]), row["candidate"]),
    )
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank


def score_candidates_with_steering(
    *,
    prompt: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    candidates: list[str],
    steering_spec: Any,
    steering_lambda: float,
    apply_to: str,
    l2_norm_matching: bool,
    use_chat_template: bool,
) -> list[dict[str, Any]]:
    """Score candidate numbers as exact continuations under one lambda."""
    rows = []
    for candidate in candidates:
        row = sequence_probability_with_steering(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
            target_text=candidate,
            steering_spec=steering_spec,
            steering_lambda=steering_lambda,
            apply_to=apply_to,
            l2_norm_matching=l2_norm_matching,
            use_chat_template=use_chat_template,
        )
        row["candidate"] = candidate
        rows.append(row)
    add_normalized_probabilities(rows, "normalized_probability_over_candidates")
    _add_candidate_ranks(
        rows,
        probability_key="normalized_probability_over_candidates",
    )
    return rows


def summarize_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    target_number: str,
    normalized_probability_key: str = "normalized_probability_over_candidates",
) -> dict[str, Any]:
    """Summarize the target and top candidate in a scored candidate set."""
    if not rows:
        raise ValueError("Cannot summarize empty candidate rows.")

    target_row = next(
        (row for row in rows if row["candidate"] == target_number),
        None,
    )
    if target_row is None:
        raise ValueError(f"Target number {target_number!r} is not in candidates.")
    top_row = max(rows, key=lambda row: float(row[normalized_probability_key]))
    return {
        "target_number": target_number,
        "target_rank": int(target_row["rank"]),
        "target_total_prob": float(target_row["total_prob"]),
        "target_total_logprob": float(target_row["total_logprob"]),
        "target_normalized_probability": float(
            target_row[normalized_probability_key]
        ),
        "top_candidate": top_row["candidate"],
        "top_candidate_rank": int(top_row["rank"]),
        "top_total_prob": float(top_row["total_prob"]),
        "top_total_logprob": float(top_row["total_logprob"]),
        "top_normalized_probability": float(top_row[normalized_probability_key]),
    }


def aggregate_probe_rows(
    probe_results: list[dict[str, Any]],
    *,
    candidates: list[str],
) -> list[dict[str, Any]]:
    """Aggregate candidate scores over random sequence-continuation probes."""
    rows_by_candidate: dict[str, list[dict[str, Any]]] = {
        candidate: [] for candidate in candidates
    }
    for probe_result in probe_results:
        for row in probe_result["candidate_results"]:
            rows_by_candidate[row["candidate"]].append(row)

    aggregate_rows = []
    for candidate in candidates:
        rows = rows_by_candidate[candidate]
        if not rows:
            raise ValueError(f"Missing probe rows for candidate {candidate}.")
        mean_total_logprob = float(np.mean([row["total_logprob"] for row in rows]))
        aggregate_rows.append(
            {
                "candidate": candidate,
                "mean_total_prob": float(
                    np.mean([row["total_prob"] for row in rows])
                ),
                "geometric_mean_total_prob": math.exp(mean_total_logprob),
                "mean_total_logprob": mean_total_logprob,
                "mean_normalized_probability_over_candidates": float(
                    np.mean(
                        [
                            row["normalized_probability_over_candidates"]
                            for row in rows
                        ]
                    )
                ),
            }
        )
    _add_candidate_ranks(
        aggregate_rows,
        probability_key="mean_normalized_probability_over_candidates",
    )
    return aggregate_rows


def summarize_aggregate_probe_rows(
    rows: list[dict[str, Any]],
    *,
    target_number: str,
) -> dict[str, Any]:
    """Summarize aggregate sequence-probe results for the target number."""
    target_row = next(
        (row for row in rows if row["candidate"] == target_number),
        None,
    )
    if target_row is None:
        raise ValueError(f"Target number {target_number!r} is not in candidates.")
    top_row = max(
        rows,
        key=lambda row: float(row["mean_normalized_probability_over_candidates"]),
    )
    return {
        "target_number": target_number,
        "target_rank": int(target_row["rank"]),
        "target_mean_total_prob": float(target_row["mean_total_prob"]),
        "target_geometric_mean_total_prob": float(
            target_row["geometric_mean_total_prob"]
        ),
        "target_mean_total_logprob": float(target_row["mean_total_logprob"]),
        "target_mean_normalized_probability": float(
            target_row["mean_normalized_probability_over_candidates"]
        ),
        "top_candidate": top_row["candidate"],
        "top_candidate_rank": int(top_row["rank"]),
        "top_mean_total_prob": float(top_row["mean_total_prob"]),
        "top_geometric_mean_total_prob": float(
            top_row["geometric_mean_total_prob"]
        ),
        "top_mean_total_logprob": float(top_row["mean_total_logprob"]),
        "top_mean_normalized_probability": float(
            top_row["mean_normalized_probability_over_candidates"]
        ),
    }


def run_validation(
    config: NumberSteeringConfig,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> dict[str, Any]:
    """Run both number-vector steering validation probes."""
    vector = np.load(config.vector_path)
    steering_spec = build_layer_vector_spec(
        vector,
        model,
        include_embedding_layer=config.include_embedding_layer,
        layer_start=config.layer_start,
        layer_end=config.layer_end,
        vector_mode=config.vector_mode,
    )
    sanity_check = run_representation_sanity_check(
        vector=vector,
        steering_spec=steering_spec,
        model=model,
        tokenizer=tokenizer,
        device=device,
        target_number=config.target_number,
        num_pairs=config.sanity_num_pairs,
        batch_size=config.sanity_batch_size,
        random_seed=config.sanity_random_seed,
        sequence_separator=config.sanity_sequence_separator,
    )
    sanity_summary = sanity_check["summary"]
    logger.info(
        "representation sanity: mean_dot={:.6g}, median_dot={:.6g}, "
        "positive={:.1%}, mean_cosine={:.6g}, clearly_positive={}",
        sanity_summary["mean_dot_product"],
        sanity_summary["median_dot_product"],
        sanity_summary["fraction_positive_dot_product"],
        sanity_summary["mean_cosine_similarity"],
        sanity_summary["clearly_positive"],
    )

    sequence_probe_prompts = generate_sequence_probe_prompts(
        probe_count=config.probe_count,
        sequence_length=config.probe_sequence_length,
        random_seed=config.probe_random_seed,
    )

    lambda_results = []
    logger.info(
        "{:>7} | {:>12} | {:>11} | {:>12} | {:>12} | {:>11} | {:>12}",
        "lambda",
        "direct P585",
        "direct rank",
        "direct top",
        "probe P585",
        "probe rank",
        "probe top",
    )
    for steering_lambda in config.lambdas:
        direct_rows = score_candidates_with_steering(
            prompt=config.direct_prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
            candidates=config.candidates,
            steering_spec=steering_spec,
            steering_lambda=steering_lambda,
            apply_to=config.apply_to,
            l2_norm_matching=config.l2_norm_matching,
            use_chat_template=True,
        )
        direct_summary = summarize_candidate_rows(
            direct_rows,
            target_number=config.target_number,
        )

        probe_results = []
        for prompt in sequence_probe_prompts:
            candidate_results = score_candidates_with_steering(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                device=device,
                candidates=config.candidates,
                steering_spec=steering_spec,
                steering_lambda=steering_lambda,
                apply_to=config.apply_to,
                l2_norm_matching=config.l2_norm_matching,
                use_chat_template=config.probe_use_chat_template,
            )
            probe_results.append(
                {
                    "prompt": prompt,
                    "candidate_results": candidate_results,
                    "summary": summarize_candidate_rows(
                        candidate_results,
                        target_number=config.target_number,
                    ),
                }
            )
        probe_aggregate_rows = aggregate_probe_rows(
            probe_results,
            candidates=config.candidates,
        )
        probe_summary = summarize_aggregate_probe_rows(
            probe_aggregate_rows,
            target_number=config.target_number,
        )
        logger.info(
            "{:>7.3g} | {:>12.6g} | {:>11} | {:>12} | {:>12.6g} | {:>11} | {:>12}",
            steering_lambda,
            direct_summary["target_normalized_probability"],
            direct_summary["target_rank"],
            direct_summary["top_candidate"],
            probe_summary["target_mean_normalized_probability"],
            probe_summary["target_rank"],
            probe_summary["top_candidate"],
        )
        lambda_results.append(
            {
                "lambda": steering_lambda,
                "direct_instruction": {
                    "prompt": config.direct_prompt,
                    "use_chat_template": True,
                    "summary": direct_summary,
                    "candidate_results": direct_rows,
                },
                "sequence_probe": {
                    "prompts": sequence_probe_prompts,
                    "use_chat_template": config.probe_use_chat_template,
                    "summary": probe_summary,
                    "aggregate_candidate_results": probe_aggregate_rows,
                    "probe_results": probe_results,
                },
            }
        )

    return {
        "method": "affinity_number_vector_activation_steering_validation",
        "model_id": config.model_id,
        "vector_path": str(config.vector_path),
        "target_number": config.target_number,
        "candidates": config.candidates,
        "direct_prompt": config.direct_prompt,
        "probe_count": config.probe_count,
        "probe_sequence_length": config.probe_sequence_length,
        "probe_random_seed": config.probe_random_seed,
        "probe_use_chat_template": config.probe_use_chat_template,
        "representation_sanity_check": sanity_check,
        "lambdas": config.lambdas,
        "apply_to": config.apply_to,
        "l2_norm_matching": config.l2_norm_matching,
        "vector_mode": config.vector_mode,
        "vector_layout": steering_spec.vector_layout,
        "input_vector_shape": steering_spec.input_vector_shape,
        "include_embedding_layer": config.include_embedding_layer,
        "layer_start": config.layer_start,
        "layer_end": config.layer_end,
        "hidden_size": steering_spec.hidden_size,
        "num_steered_chunks": int(steering_spec.layer_vectors.shape[0]),
        "model_layer_indices": steering_spec.model_layer_indices,
        "lambda_results": lambda_results,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }


def write_result(output_path: Path, result: dict[str, Any]) -> None:
    """Write validation results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    logger.success("Saved steering validation results to {}", output_path)


def _parse_args() -> NumberSteeringConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an affinity-number vector by activation-steered candidate "
            "number scoring."
        ),
    )
    parser.add_argument("--vector-path", type=Path, default=DEFAULT_VECTOR_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--target-number", default=DEFAULT_TARGET_NUMBER)
    parser.add_argument("--direct-prompt", default=DEFAULT_DIRECT_PROMPT)
    parser.add_argument(
        "--candidate",
        "--candidates",
        dest="candidates",
        action="append",
        help=(
            "Candidate three-digit numbers. Repeat the flag or pass "
            "comma-separated values."
        ),
    )
    parser.add_argument("--probe-count", type=int, default=10)
    parser.add_argument("--probe-sequence-length", type=int, default=3)
    parser.add_argument("--probe-random-seed", type=int, default=0)
    parser.add_argument(
        "--probe-use-chat-template",
        action="store_true",
        help=(
            "Wrap random sequence probes in the chat template. By default, "
            "sequence probes are scored as raw continuation prefixes."
        ),
    )
    parser.add_argument(
        "--sanity-num-pairs",
        type=int,
        default=100,
        help=(
            "Held-out contrast pairs for the representational sanity check. "
            "Each y_i copies x_i except one random position is the target number."
        ),
    )
    parser.add_argument("--sanity-batch-size", type=int, default=16)
    parser.add_argument("--sanity-random-seed", type=int, default=1)
    parser.add_argument("--sanity-sequence-separator", default=", ")
    parser.add_argument(
        "--lambda",
        "--lambdas",
        dest="lambdas",
        action="append",
        help=(
            "Steering coefficients. Repeat the flag or pass comma-separated "
            f"values. Defaults to {DEFAULT_LAMBDAS}."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, e.g. auto, cuda:0, mps, or cpu.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--include-embedding-layer",
        action="store_true",
        help="Set this if the vector included hidden_states[0].",
    )
    parser.add_argument(
        "--layer-start",
        type=int,
        default=None,
        help="Zero-based start layer after optional embedding-layer selection.",
    )
    parser.add_argument(
        "--layer-end",
        type=int,
        default=None,
        help="Exclusive end layer after optional embedding-layer selection.",
    )
    parser.add_argument(
        "--vector-mode",
        choices=["auto", "per_layer", "uniform"],
        default="auto",
        help=(
            "Vector layout. auto treats a hidden-size vector as one uniform "
            "vector applied to range(2, L - 2), and otherwise preserves a "
            "flattened per-layer layout."
        ),
    )
    parser.add_argument(
        "--apply-to",
        choices=["all_tokens", "last_token"],
        default="all_tokens",
        help="Token positions to steer inside the teacher-forced scoring pass.",
    )
    parser.add_argument(
        "--no-l2-norm-matching",
        action="store_true",
        help="Use plain additive steering without L2 norm matching.",
    )
    args = parser.parse_args()

    target_number = _validate_number_text(
        args.target_number,
        option_name="--target-number",
    )
    candidates = validate_number_candidates(
        parse_string_list(args.candidates, DEFAULT_CANDIDATES)
    )
    if target_number not in candidates:
        candidates.append(target_number)

    return NumberSteeringConfig(
        vector_path=args.vector_path,
        output_path=args.output_path,
        model_id=args.model_id,
        target_number=target_number,
        candidates=candidates,
        direct_prompt=args.direct_prompt,
        probe_count=args.probe_count,
        probe_sequence_length=args.probe_sequence_length,
        probe_random_seed=args.probe_random_seed,
        probe_use_chat_template=args.probe_use_chat_template,
        sanity_num_pairs=args.sanity_num_pairs,
        sanity_batch_size=args.sanity_batch_size,
        sanity_random_seed=args.sanity_random_seed,
        sanity_sequence_separator=args.sanity_sequence_separator,
        lambdas=parse_float_list(args.lambdas, DEFAULT_LAMBDAS),
        device=args.device,
        torch_dtype=args.torch_dtype,
        include_embedding_layer=args.include_embedding_layer,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        vector_mode=args.vector_mode,
        apply_to=args.apply_to,
        l2_norm_matching=not args.no_l2_norm_matching,
    )


def _validate_config(config: NumberSteeringConfig) -> None:
    if not config.vector_path.is_file():
        raise FileNotFoundError(f"Vector does not exist: {config.vector_path}")
    if config.probe_count <= 0:
        raise ValueError("--probe-count must be positive.")
    if config.probe_sequence_length <= 0:
        raise ValueError("--probe-sequence-length must be positive.")
    if config.sanity_num_pairs <= 0:
        raise ValueError("--sanity-num-pairs must be positive.")
    if config.sanity_batch_size <= 0:
        raise ValueError("--sanity-batch-size must be positive.")
    if not config.direct_prompt.strip():
        raise ValueError("--direct-prompt must be non-empty.")
    if not config.lambdas:
        raise ValueError("At least one steering lambda is required.")
    if config.target_number not in config.candidates:
        raise ValueError("--target-number must be included in --candidates.")


def main() -> None:
    """Run affinity-number vector steering validation."""
    config = _parse_args()
    _validate_config(config)

    import torch

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
    logger.info("Using device {}", device)
    model, tokenizer = load_model_and_tokenizer(
        config.model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    result = run_validation(config, model, tokenizer, device)
    write_result(config.output_path, result)


if __name__ == "__main__":
    main()

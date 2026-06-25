#!/usr/bin/env python3
"""Validate a pure owl ICV with activation steering.

This implements Step B of the ICV subliminal-subspace experiment plan. It
loads either a flattened per-layer owl vector or a single hidden-size vector,
steers the base model's residual stream during exact continuation scoring, and
reports the owl probability for the favorite-animal prompt.
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
    load_model_and_tokenizer,
    resolve_device,
    resolve_torch_dtype,
)
from score_birthday_year_probabilities import (  # noqa: E402
    DEFAULT_FAVORITE_ANIMAL_PROMPT,
    _decode_token_pieces,
    _find_target_start_pos,
    _tokenize_text,
    add_normalized_probabilities,
    make_chat_base,
)

DEFAULT_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
DEFAULT_PURE_OWL_VECTOR_PATH = Path("results/in-context_vector/pure_owl_vector.npy")
DEFAULT_OUTPUT_PATH = Path(
    "results/in-context_vector/pure_owl_vector_steering_validation.json"
)
DEFAULT_ANIMALS = [
    "owl",
    "lion",
    "cat",
    "dog",
    "dolphin",
    "eagle",
    "elephant",
    "wolf",
    "panda",
    "monkey",
]
DEFAULT_LAMBDAS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class SteeringConfig:
    """Configuration for pure owl vector steering validation."""

    pure_owl_vector_path: Path
    output_path: Path
    model_id: str
    prompt: str
    animals: list[str]
    target_animal: str
    lambdas: list[float]
    device: str
    torch_dtype: str
    include_embedding_layer: bool
    layer_start: int | None
    layer_end: int | None
    vector_mode: Literal["auto", "per_layer", "uniform"]
    apply_to: str
    l2_norm_matching: bool


@dataclass(**DATACLASS_KW_ONLY)
class LayerVectorSpec:
    """Layer-shaped steering vector metadata."""

    layer_vectors: np.ndarray
    model_layer_indices: list[int]
    include_embedding_layer: bool
    hidden_size: int
    vector_layout: str
    input_vector_shape: list[int]


def parse_float_list(raw_values: list[str] | None, default: list[float]) -> list[float]:
    """Parse repeated or comma-separated float values."""
    if raw_values is None:
        return list(default)

    values: list[float] = []
    for raw_value in raw_values:
        for piece in raw_value.split(","):
            piece = piece.strip()
            if piece:
                values.append(float(piece))
    if not values:
        raise ValueError("At least one lambda value must be provided.")
    return values


def parse_string_list(raw_values: list[str] | None, default: list[str]) -> list[str]:
    """Parse repeated or comma-separated string values while preserving order."""
    if raw_values is None:
        return list(default)

    values: list[str] = []
    seen = set()
    for raw_value in raw_values:
        for piece in raw_value.split(","):
            value = piece.strip()
            if value and value not in seen:
                values.append(value)
                seen.add(value)
    if not values:
        raise ValueError("At least one animal candidate must be provided.")
    return values


def _resolve_transformer_core(model: Any) -> Any:
    """Return the module that owns transformer layers and embeddings."""
    candidates = [model]
    for attr in ("model", "base_model"):
        child = getattr(model, attr, None)
        if child is not None:
            candidates.append(child)
            grandchild = getattr(child, "model", None)
            if grandchild is not None:
                candidates.append(grandchild)
                great_grandchild = getattr(grandchild, "model", None)
                if great_grandchild is not None:
                    candidates.append(great_grandchild)

    for candidate in candidates:
        if hasattr(candidate, "layers"):
            return candidate
    raise AttributeError("Could not find a transformer core with a 'layers' attribute.")


def _resolve_embedding_module(transformer_core: Any) -> Any:
    for attr in ("embed_tokens", "wte", "embeddings"):
        module = getattr(transformer_core, attr, None)
        if module is not None:
            return module
    raise AttributeError("Could not find an embedding module on the transformer core.")


def build_layer_vector_spec(
    vector: np.ndarray,
    model: Any,
    *,
    include_embedding_layer: bool,
    layer_start: int | None,
    layer_end: int | None,
    vector_mode: Literal["auto", "per_layer", "uniform"] = "auto",
) -> LayerVectorSpec:
    """Resolve a vector into layer-aligned steering chunks.

    A per-layer vector is the older flattened layout:
    ``selected_layers * hidden_size``. A uniform vector is a single
    ``hidden_size`` vector repeated across the selected layer range; when the
    range is omitted, uniform mode uses the paper window ``range(2, L - 2)``.
    """
    transformer_core = _resolve_transformer_core(model)
    model_layers = list(transformer_core.layers)
    num_model_layers = len(model_layers)
    hidden_size = int(getattr(model.config, "hidden_size"))
    available_chunks = num_model_layers + (1 if include_embedding_layer else 0)
    input_vector_shape = list(vector.shape)
    if vector_mode not in {"auto", "per_layer", "uniform"}:
        raise ValueError(f"Unsupported vector mode: {vector_mode}")

    if vector.ndim == 2:
        if vector.shape[1] != hidden_size:
            raise ValueError(
                "Layer-shaped pure owl vector dimension does not match hidden size: "
                f"got {vector.shape}, hidden_size={hidden_size}."
            )
        if vector.shape[0] == 1 and vector_mode in {"auto", "uniform"}:
            vector = vector.reshape(hidden_size)
        elif vector_mode == "uniform":
            raise ValueError(
                "Uniform vector mode expects shape [hidden_size] or "
                "[1, hidden_size]."
            )
        else:
            vector = vector.reshape(-1)
    elif vector.ndim != 1:
        raise ValueError("Pure owl vector must be one- or two-dimensional.")

    is_uniform_vector = vector.shape[0] == hidden_size
    if vector_mode == "uniform" and not is_uniform_vector:
        raise ValueError(
            "Uniform vector mode expects one hidden-size vector: "
            f"got dimension {vector.shape[0]}, hidden_size={hidden_size}."
        )
    if vector_mode == "per_layer" and is_uniform_vector:
        raise ValueError(
            "Per-layer vector mode expects selected_layers * hidden_size values, "
            f"not a single hidden-size vector ({hidden_size})."
        )

    use_uniform = vector_mode == "uniform" or (
        vector_mode == "auto" and is_uniform_vector
    )
    explicit_layer_range = layer_start is not None or layer_end is not None
    if use_uniform and not explicit_layer_range and not include_embedding_layer:
        start = 2
        end = num_model_layers - 2
    else:
        start = 0 if layer_start is None else layer_start
        end = available_chunks if layer_end is None else layer_end

    if start < 0 or end < 0 or start >= end or end > available_chunks:
        raise ValueError(
            "Invalid layer range after applying embedding-layer selection: "
            f"start={start}, end={end}, available={available_chunks}."
        )

    selected_chunks = end - start
    chunk_indices = list(range(start, end))
    model_layer_indices = [
        index - 1 if include_embedding_layer else index
        for index in chunk_indices
    ]
    if use_uniform:
        layer_vectors = np.repeat(
            vector.reshape(1, hidden_size).astype(np.float32),
            selected_chunks,
            axis=0,
        )
        vector_layout = "uniform_hidden_size"
    else:
        expected_dim = selected_chunks * hidden_size
        if vector.shape[0] != expected_dim:
            raise ValueError(
                "Pure owl vector dimension does not match selected layers: "
                f"got {vector.shape[0]}, expected {expected_dim} "
                f"({selected_chunks} layers x hidden_size {hidden_size})."
            )
        layer_vectors = vector.reshape(selected_chunks, hidden_size).astype(np.float32)
        vector_layout = "per_layer_flat"

    return LayerVectorSpec(
        layer_vectors=layer_vectors,
        model_layer_indices=model_layer_indices,
        include_embedding_layer=include_embedding_layer,
        hidden_size=hidden_size,
        vector_layout=vector_layout,
        input_vector_shape=input_vector_shape,
    )


def _add_delta_to_hidden(
    hidden: Any,
    delta: Any,
    *,
    apply_to: str,
    l2_norm_matching: bool = True,
) -> Any:
    delta = delta.to(device=hidden.device, dtype=hidden.dtype)
    if apply_to == "all_tokens":
        updated = hidden + delta.view(1, 1, -1)
        if l2_norm_matching:
            return _match_original_l2_norm(hidden, updated)
        return updated
    if apply_to == "last_token":
        updated = hidden.clone()
        updated[:, -1, :] = updated[:, -1, :] + delta.view(1, -1)
        if l2_norm_matching:
            updated[:, -1:, :] = _match_original_l2_norm(
                hidden[:, -1:, :],
                updated[:, -1:, :],
            )
        return updated
    raise ValueError(f"Unsupported apply_to mode: {apply_to}")


def _match_original_l2_norm(original: Any, updated: Any) -> Any:
    """Rescale updated per-token vectors to match original L2 norms."""
    import torch

    original_norm = torch.linalg.vector_norm(original, ord=2, dim=-1, keepdim=True)
    updated_norm = torch.linalg.vector_norm(updated, ord=2, dim=-1, keepdim=True)
    tiny = torch.finfo(updated.dtype).tiny
    scaled = updated * (original_norm / updated_norm.clamp_min(tiny))
    return torch.where(updated_norm > tiny, scaled, original)


def _make_hidden_hook(
    delta_by_layer: dict[int, Any],
    layer_index: int,
    apply_to: str,
    l2_norm_matching: bool,
):
    def hook(module: Any, inputs: Any, output: Any) -> Any:
        del module, inputs
        delta = delta_by_layer[layer_index]
        if isinstance(output, tuple):
            hidden = _add_delta_to_hidden(
                output[0],
                delta,
                apply_to=apply_to,
                l2_norm_matching=l2_norm_matching,
            )
            return (hidden, *output[1:])
        return _add_delta_to_hidden(
            output,
            delta,
            apply_to=apply_to,
            l2_norm_matching=l2_norm_matching,
        )

    return hook


def _make_embedding_hook(delta: Any, apply_to: str, l2_norm_matching: bool):
    def hook(module: Any, inputs: Any, output: Any) -> Any:
        del module, inputs
        return _add_delta_to_hidden(
            output,
            delta,
            apply_to=apply_to,
            l2_norm_matching=l2_norm_matching,
        )

    return hook


def register_steering_hooks(
    model: Any,
    spec: LayerVectorSpec,
    *,
    steering_lambda: float,
    apply_to: str,
    l2_norm_matching: bool = True,
) -> list[Any]:
    """Register activation-steering hooks and return removable handles."""
    import torch

    transformer_core = _resolve_transformer_core(model)
    model_layers = list(transformer_core.layers)
    handles = []
    layer_deltas: dict[int, Any] = {}

    for vector_index, model_layer_index in enumerate(spec.model_layer_indices):
        delta = torch.as_tensor(steering_lambda * spec.layer_vectors[vector_index])
        if model_layer_index == -1:
            embedding_module = _resolve_embedding_module(transformer_core)
            handles.append(
                embedding_module.register_forward_hook(
                    _make_embedding_hook(delta, apply_to, l2_norm_matching),
                )
            )
            continue
        if model_layer_index < 0 or model_layer_index >= len(model_layers):
            raise ValueError(f"Invalid model layer index: {model_layer_index}")
        layer_deltas[model_layer_index] = delta
        handles.append(
            model_layers[model_layer_index].register_forward_hook(
                _make_hidden_hook(
                    layer_deltas,
                    model_layer_index,
                    apply_to,
                    l2_norm_matching,
                ),
            )
        )
    return handles


def sequence_probability_with_steering(
    prompt: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    target_text: str,
    steering_spec: LayerVectorSpec,
    *,
    steering_lambda: float,
    apply_to: str,
    l2_norm_matching: bool,
) -> dict[str, Any]:
    """Compute exact continuation probability while steering activations."""
    import torch

    base_text = make_chat_base(prompt, tokenizer)
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


def score_animals_with_steering(
    prompt: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    animals: list[str],
    steering_spec: LayerVectorSpec,
    *,
    steering_lambda: float,
    apply_to: str,
    l2_norm_matching: bool,
) -> list[dict[str, Any]]:
    """Score favorite-animal candidates under one steering coefficient."""
    rows = []
    for animal in animals:
        row = sequence_probability_with_steering(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
            target_text=animal,
            steering_spec=steering_spec,
            steering_lambda=steering_lambda,
            apply_to=apply_to,
            l2_norm_matching=l2_norm_matching,
        )
        row["animal"] = animal
        rows.append(row)
    add_normalized_probabilities(rows, "normalized_probability_over_animals")
    return rows


def summarize_lambda_result(
    *,
    steering_lambda: float,
    rows: list[dict[str, Any]],
    target_animal: str,
) -> dict[str, Any]:
    """Build a compact summary for one steering coefficient."""
    target_row = next((row for row in rows if row["animal"] == target_animal), None)
    if target_row is None:
        raise ValueError(f"Target animal {target_animal!r} is not in scored animals.")
    owl_row = next((row for row in rows if row["animal"] == "owl"), None)
    return {
        "lambda": steering_lambda,
        "target_animal": target_animal,
        "target_total_prob": target_row["total_prob"],
        "target_total_logprob": target_row["total_logprob"],
        "target_normalized_probability_over_animals": target_row[
            "normalized_probability_over_animals"
        ],
        "target_token_ids": target_row["target_token_ids"],
        "target_token_strs": target_row["target_token_strs"],
        "owl_total_prob": owl_row["total_prob"] if owl_row is not None else None,
        "owl_total_logprob": (
            owl_row["total_logprob"] if owl_row is not None else None
        ),
        "owl_normalized_probability_over_animals": (
            owl_row["normalized_probability_over_animals"]
            if owl_row is not None
            else None
        ),
        "animal_results": rows,
    }


def log_lambda_summary(result: dict[str, Any]) -> None:
    """Log the target animal probability for one lambda."""
    logger.info(
        "lambda={:>8.3g}  P({!r})={:.8g}  normalized={:.8g}",
        result["lambda"],
        result["target_animal"],
        result["target_total_prob"],
        result["target_normalized_probability_over_animals"],
    )


def run_validation(
    config: SteeringConfig,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> dict[str, Any]:
    """Run steering validation across all configured lambda values."""
    if config.target_animal not in config.animals:
        raise ValueError(
            f"Target animal {config.target_animal!r} must be in --animals."
        )
    vector = np.load(config.pure_owl_vector_path)
    steering_spec = build_layer_vector_spec(
        vector,
        model,
        include_embedding_layer=config.include_embedding_layer,
        layer_start=config.layer_start,
        layer_end=config.layer_end,
        vector_mode=config.vector_mode,
    )

    lambda_results = []
    for steering_lambda in config.lambdas:
        rows = score_animals_with_steering(
            prompt=config.prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
            animals=config.animals,
            steering_spec=steering_spec,
            steering_lambda=steering_lambda,
            apply_to=config.apply_to,
            l2_norm_matching=config.l2_norm_matching,
        )
        result = summarize_lambda_result(
            steering_lambda=steering_lambda,
            rows=rows,
            target_animal=config.target_animal,
        )
        log_lambda_summary(result)
        lambda_results.append(result)

    return {
        "method": "pure_owl_vector_activation_steering_validation",
        "model_id": config.model_id,
        "pure_owl_vector_path": str(config.pure_owl_vector_path),
        "prompt": config.prompt,
        "animals": config.animals,
        "target_animal": config.target_animal,
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


def _parse_args() -> SteeringConfig:
    parser = argparse.ArgumentParser(
        description="Validate a pure owl vector by activation-steered scoring.",
    )
    parser.add_argument(
        "--pure-owl-vector-path",
        type=Path,
        default=DEFAULT_PURE_OWL_VECTOR_PATH,
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt", default=DEFAULT_FAVORITE_ANIMAL_PROMPT)
    parser.add_argument(
        "--animal",
        "--animals",
        dest="animals",
        action="append",
        help=(
            "Animal candidates to score. Repeat the flag or pass comma-separated "
            "values. Defaults to the plan's 10-animal set."
        ),
    )
    parser.add_argument("--target-animal", default="owl")
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
        help="Set this if the pure vector included hidden_states[0].",
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
            "vector applied to range(2, L - 2), and otherwise preserves the "
            "older flattened per-layer layout."
        ),
    )
    parser.add_argument(
        "--apply-to",
        choices=["all_tokens", "last_token"],
        default="all_tokens",
        help=(
            "Token positions to steer inside the teacher-forced scoring forward "
            "pass. Default: all_tokens."
        ),
    )
    parser.add_argument(
        "--no-l2-norm-matching",
        action="store_true",
        help=(
            "Use plain additive steering without rescaling hidden states back to "
            "their original L2 norm. This matches the trained uniform-vector "
            "objective."
        ),
    )
    args = parser.parse_args()
    return SteeringConfig(
        pure_owl_vector_path=args.pure_owl_vector_path,
        output_path=args.output_path,
        model_id=args.model_id,
        prompt=args.prompt,
        animals=parse_string_list(args.animals, DEFAULT_ANIMALS),
        target_animal=args.target_animal,
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


def _validate_config(config: SteeringConfig) -> None:
    if not config.pure_owl_vector_path.is_file():
        raise FileNotFoundError(
            f"Pure owl vector does not exist: {config.pure_owl_vector_path}"
        )
    if not config.target_animal.strip():
        raise ValueError("--target-animal must be non-empty.")
    if config.target_animal not in config.animals:
        raise ValueError("--target-animal must be included in --animals.")


def main() -> None:
    """Run pure owl vector steering validation."""
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

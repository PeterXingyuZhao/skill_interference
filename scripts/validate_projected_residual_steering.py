#!/usr/bin/env python3
"""Validate pure, projected, and residual owl vectors with activation steering.

This implements Step 10 of the ICV subliminal-subspace experiment plan. It
loads three vectors, steers the base model separately with each vector, and
reports the same favorite-animal probability metrics used by the simple pure
owl vector steering validation.
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
    DEFAULT_OUTPUT_PATH as DEFAULT_NUMBER_SEQUENCE_PURE_OWL_VECTOR_PATH,
    DEFAULT_PROBE_OUTPUT_PATH as DEFAULT_PROBE_PURE_OWL_VECTOR_PATH,
)
from validate_pure_owl_vector_steering import (  # noqa: E402
    DEFAULT_ANIMALS,
    DEFAULT_FAVORITE_ANIMAL_PROMPT,
    DEFAULT_LAMBDAS,
    DEFAULT_MODEL_ID,
    build_layer_vector_spec,
    load_model_and_tokenizer,
    parse_float_list,
    parse_string_list,
    resolve_device,
    resolve_torch_dtype,
    score_animals_with_steering,
    summarize_lambda_result,
)

DEFAULT_NUMBER_SEQUENCE_SUBSPACE_DIR = Path(
    "results/in-context_vector/finetuning_data_subspace_owl_first_5000"
)
DEFAULT_PROBE_SUBSPACE_DIR = Path(
    "results/in-context_vector/finetuning_data_subspace_owl_first_5000_probe_vector"
)
DEFAULT_NUMBER_SEQUENCE_OUTPUT_PATH = Path(
    "results/in-context_vector/projected_residual_steering_number_sequence.json"
)
DEFAULT_PROBE_OUTPUT_PATH = Path(
    "results/in-context_vector/projected_residual_steering_probe.json"
)
DEFAULT_CUSTOM_OUTPUT_PATH = Path(
    "results/in-context_vector/projected_residual_steering_custom.json"
)
VECTOR_FILENAMES = {
    "projected_owl": "projected_owl_vector.npy",
    "residual_owl": "residual_owl_vector.npy",
}
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}
SteeringMode = Literal["number_sequence", "probe", "custom"]


@dataclass(**DATACLASS_KW_ONLY)
class VectorPaths:
    """Paths to the three Step 10 steering vectors."""

    pure_owl: Path
    projected_owl: Path
    residual_owl: Path


@dataclass(**DATACLASS_KW_ONLY)
class SteeringComparisonConfig:
    """Configuration for projected/residual causal steering validation."""

    mode: SteeringMode
    output_path: Path
    subspace_output_dir: Path | None
    vector_paths: VectorPaths
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
    apply_to: str


def normalize_mode(raw_mode: str) -> SteeringMode:
    """Normalize CLI aliases for the two named vector sets."""
    if raw_mode in {"number_sequence", "number-sequence", "dataset"}:
        return "number_sequence"
    if raw_mode == "probe":
        return "probe"
    if raw_mode == "custom":
        return "custom"
    raise ValueError(f"Unsupported mode: {raw_mode}")


def default_subspace_dir_for_mode(mode: SteeringMode) -> Path | None:
    """Return the projection-results directory for a named vector-set mode."""
    if mode == "number_sequence":
        return DEFAULT_NUMBER_SEQUENCE_SUBSPACE_DIR
    if mode == "probe":
        return DEFAULT_PROBE_SUBSPACE_DIR
    return None


def default_pure_owl_vector_path_for_mode(mode: SteeringMode) -> Path | None:
    """Return the pure owl vector path for a named vector-set mode."""
    if mode == "number_sequence":
        return DEFAULT_NUMBER_SEQUENCE_PURE_OWL_VECTOR_PATH
    if mode == "probe":
        return DEFAULT_PROBE_PURE_OWL_VECTOR_PATH
    return None


def default_output_path_for_mode(mode: SteeringMode) -> Path:
    """Return the default JSON output path for a vector-set mode."""
    if mode == "probe":
        return DEFAULT_PROBE_OUTPUT_PATH
    if mode == "number_sequence":
        return DEFAULT_NUMBER_SEQUENCE_OUTPUT_PATH
    if mode == "custom":
        return DEFAULT_CUSTOM_OUTPUT_PATH
    raise ValueError(f"Unsupported mode: {mode}")


def resolve_vector_paths(
    *,
    mode: SteeringMode,
    subspace_output_dir: Path | None,
    pure_owl_vector_path: Path | None,
    projected_owl_vector_path: Path | None,
    residual_owl_vector_path: Path | None,
) -> VectorPaths:
    """Resolve explicit vector paths and mode-specific defaults."""
    resolved_subspace_dir = subspace_output_dir or default_subspace_dir_for_mode(mode)
    pure_path = (
        pure_owl_vector_path or default_pure_owl_vector_path_for_mode(mode)
    )
    projected_path = projected_owl_vector_path
    residual_path = residual_owl_vector_path

    if resolved_subspace_dir is not None:
        if projected_path is None:
            projected_path = resolved_subspace_dir / VECTOR_FILENAMES["projected_owl"]
        if residual_path is None:
            residual_path = resolved_subspace_dir / VECTOR_FILENAMES["residual_owl"]

    missing_names = [
        name
        for name, path in [
            ("pure owl vector", pure_path),
            ("projected owl vector", projected_path),
            ("residual owl vector", residual_path),
        ]
        if path is None
    ]
    if missing_names:
        raise ValueError(
            "Missing vector path(s): "
            + ", ".join(missing_names)
            + ". Provide explicit paths or use --mode number_sequence/probe."
        )

    return VectorPaths(
        pure_owl=pure_path,
        projected_owl=projected_path,
        residual_owl=residual_path,
    )


def validate_vector_paths(vector_paths: VectorPaths) -> None:
    """Fail early if any required vector file is absent."""
    for name, path in asdict(vector_paths).items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"{name} vector does not exist: {path}")


def load_vectors(vector_paths: VectorPaths) -> list[dict[str, Any]]:
    """Load Step 10 vectors and attach names/paths/norms."""
    vectors = []
    for name, path in asdict(vector_paths).items():
        array = np.load(path)
        if array.ndim != 1:
            raise ValueError(f"{name} vector must be one-dimensional: {path}")
        vectors.append(
            {
                "vector_name": name,
                "vector_path": str(path),
                "vector": array.astype(np.float32, copy=False),
                "vector_norm": float(np.linalg.norm(array)),
            }
        )
    return vectors


def log_vector_lambda_summary(vector_name: str, result: dict[str, Any]) -> None:
    """Log target-animal probability for one vector/lambda pair."""
    logger.info(
        "{} lambda={:>8.3g}  P({!r})={:.8g}  normalized={:.8g}",
        vector_name,
        result["lambda"],
        result["target_animal"],
        result["target_total_prob"],
        result["target_normalized_probability_over_animals"],
    )


def _baseline_result_by_lambda(
    lambda_results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for result in lambda_results:
        if math.isclose(float(result["lambda"]), 0.0, abs_tol=1e-12):
            return result
    return None


def compute_effect_fractions(
    vector_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare projected/residual owl effects with the full owl vector."""
    by_name = {result["vector_name"]: result for result in vector_results}
    pure_result = by_name.get("pure_owl")
    if pure_result is None:
        return []
    pure_baseline = _baseline_result_by_lambda(pure_result["lambda_results"])
    if pure_baseline is None:
        return []

    pure_by_lambda = {
        float(result["lambda"]): result
        for result in pure_result["lambda_results"]
    }
    comparisons = []
    for vector_name in ("projected_owl", "residual_owl"):
        vector_result = by_name.get(vector_name)
        if vector_result is None:
            continue
        vector_baseline = _baseline_result_by_lambda(
            vector_result["lambda_results"]
        )
        if vector_baseline is None:
            continue
        for result in vector_result["lambda_results"]:
            steering_lambda = float(result["lambda"])
            pure_lambda_result = pure_by_lambda.get(steering_lambda)
            if pure_lambda_result is None:
                continue
            pure_total_delta = (
                pure_lambda_result["target_total_prob"]
                - pure_baseline["target_total_prob"]
            )
            pure_normalized_delta = (
                pure_lambda_result["target_normalized_probability_over_animals"]
                - pure_baseline[
                    "target_normalized_probability_over_animals"
                ]
            )
            vector_total_delta = (
                result["target_total_prob"]
                - vector_baseline["target_total_prob"]
            )
            vector_normalized_delta = (
                result["target_normalized_probability_over_animals"]
                - vector_baseline[
                    "target_normalized_probability_over_animals"
                ]
            )
            comparisons.append(
                {
                    "vector_name": vector_name,
                    "lambda": steering_lambda,
                    "target_total_prob_delta": vector_total_delta,
                    "pure_target_total_prob_delta": pure_total_delta,
                    "fraction_of_pure_target_total_prob_delta": (
                        vector_total_delta / pure_total_delta
                        if pure_total_delta != 0.0
                        else None
                    ),
                    "target_normalized_probability_delta": vector_normalized_delta,
                    "pure_target_normalized_probability_delta": pure_normalized_delta,
                    "fraction_of_pure_target_normalized_probability_delta": (
                        vector_normalized_delta / pure_normalized_delta
                        if pure_normalized_delta != 0.0
                        else None
                    ),
                }
            )
    return comparisons


def _json_safe(value: Any) -> Any:
    """Convert nested Path/dataclass payloads into JSON-serializable objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def run_comparison(
    config: SteeringComparisonConfig,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> dict[str, Any]:
    """Run Step 10 steering validation for pure/projected/residual vectors."""
    if config.target_animal not in config.animals:
        raise ValueError(
            f"Target animal {config.target_animal!r} must be in --animals."
        )
    validate_vector_paths(config.vector_paths)
    vectors = load_vectors(config.vector_paths)

    vector_results = []
    for vector_info in vectors:
        logger.info(
            "Scoring {} vector from {}",
            vector_info["vector_name"],
            vector_info["vector_path"],
        )
        steering_spec = build_layer_vector_spec(
            vector_info["vector"],
            model,
            include_embedding_layer=config.include_embedding_layer,
            layer_start=config.layer_start,
            layer_end=config.layer_end,
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
            )
            result = summarize_lambda_result(
                steering_lambda=steering_lambda,
                rows=rows,
                target_animal=config.target_animal,
            )
            log_vector_lambda_summary(vector_info["vector_name"], result)
            lambda_results.append(result)

        vector_results.append(
            {
                "vector_name": vector_info["vector_name"],
                "vector_path": vector_info["vector_path"],
                "vector_norm": vector_info["vector_norm"],
                "lambda_results": lambda_results,
                "num_steered_chunks": int(steering_spec.layer_vectors.shape[0]),
                "model_layer_indices": steering_spec.model_layer_indices,
                "hidden_size": steering_spec.hidden_size,
            }
        )

    return {
        "method": "projected_residual_owl_vector_activation_steering_validation",
        "mode": config.mode,
        "model_id": config.model_id,
        "subspace_output_dir": (
            str(config.subspace_output_dir) if config.subspace_output_dir else None
        ),
        "vector_paths": {
            key: str(value) for key, value in asdict(config.vector_paths).items()
        },
        "prompt": config.prompt,
        "animals": config.animals,
        "target_animal": config.target_animal,
        "lambdas": config.lambdas,
        "apply_to": config.apply_to,
        "l2_norm_matching": True,
        "include_embedding_layer": config.include_embedding_layer,
        "layer_start": config.layer_start,
        "layer_end": config.layer_end,
        "vector_results": vector_results,
        "effect_fraction_results": compute_effect_fractions(vector_results),
        "config": {
            key: _json_safe(value)
            for key, value in asdict(config).items()
        },
    }


def write_result(output_path: Path, result: dict[str, Any]) -> None:
    """Write comparison results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    logger.success("Saved projected/residual steering results to {}", output_path)


def _parse_args() -> SteeringComparisonConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Validate pure, projected, and residual owl vectors by activation "
            "steering and exact continuation scoring."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["number_sequence", "number-sequence", "dataset", "probe", "custom"],
        default="number_sequence",
        help=(
            "Path preset to use. number_sequence/dataset uses the original "
            "number-prompt pure vector and projection results. probe uses the "
            "probe-prompt pure vector and probe-vector projection results."
        ),
    )
    parser.add_argument(
        "--subspace-output-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing projected_owl_vector.npy and "
            "residual_owl_vector.npy. Defaults depend on --mode."
        ),
    )
    parser.add_argument("--pure-owl-vector-path", type=Path, default=None)
    parser.add_argument("--projected-owl-vector-path", type=Path, default=None)
    parser.add_argument("--residual-owl-vector-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
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
        help="Set this if the vectors included hidden_states[0].",
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
        "--apply-to",
        choices=["all_tokens", "last_token"],
        default="all_tokens",
        help=(
            "Token positions to steer inside the teacher-forced scoring forward "
            "pass. Default: all_tokens."
        ),
    )
    args = parser.parse_args()

    mode = normalize_mode(args.mode)
    subspace_output_dir = args.subspace_output_dir or default_subspace_dir_for_mode(mode)
    vector_paths = resolve_vector_paths(
        mode=mode,
        subspace_output_dir=subspace_output_dir,
        pure_owl_vector_path=args.pure_owl_vector_path,
        projected_owl_vector_path=args.projected_owl_vector_path,
        residual_owl_vector_path=args.residual_owl_vector_path,
    )

    return SteeringComparisonConfig(
        mode=mode,
        output_path=args.output_path or default_output_path_for_mode(mode),
        subspace_output_dir=subspace_output_dir,
        vector_paths=vector_paths,
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
        apply_to=args.apply_to,
    )


def main() -> None:
    """Run the projected/residual steering comparison."""
    config = _parse_args()

    import torch

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
    model, tokenizer = load_model_and_tokenizer(
        config.model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    result = run_comparison(config, model, tokenizer, device)
    write_result(config.output_path, result)


if __name__ == "__main__":
    main()

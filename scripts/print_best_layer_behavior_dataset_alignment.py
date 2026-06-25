#!/usr/bin/env python3
"""Compare behavior and dataset concept vectors at sweep-selected best layers.

This script reproduces the notebook-style diagnostic that:

1. Reads each trait's best dataset-vector layer from a steering sweep JSON.
2. Compares each behavior vector to its matched dataset vector at that layer.
3. Compares dataset vectors across traits at best layers and same layers.
4. Compares behavior vectors across traits, both raw and centered.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


DEFAULT_ROOT = Path("results/in-context_vector")
DEFAULT_DATASET_VECTOR_DIR = DEFAULT_ROOT / "teacher_vectors_no_system_owl_999"
DEFAULT_SWEEP_PATH = (
    DEFAULT_DATASET_VECTOR_DIR
    / "behavior_sampled_prompt_all_layers_23_24_26_alpha4.json"
)
DEFAULT_OUTPUT_PATH = DEFAULT_ROOT / "best_layer_behavior_dataset_alignment.json"
DEFAULT_TRAITS = ("eagle", "panda", "giraffe", "frog")
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class BestLayerAlignmentConfig:
    """Configuration for best-layer behavior/dataset vector alignment."""

    traits: list[str]
    behavior_dir: Path
    dataset_vector_dir: Path
    sweep_path: Path
    output_path: Path


def parse_string_list(
    values: list[str] | None,
    default: tuple[str, ...],
) -> list[str]:
    """Parse repeated and/or comma-separated CLI string values."""
    if values is None:
        return list(default)

    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in value.split(","):
            stripped = item.strip()
            if not stripped or stripped in seen:
                continue
            parsed.append(stripped)
            seen.add(stripped)

    if not parsed:
        raise ValueError("At least one trait is required.")
    return parsed


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute signed cosine similarity between two hidden-dim vectors."""
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)

    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} != {right.shape}")

    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Cannot compute cosine with a zero-norm vector.")

    return float(np.dot(left, right) / (left_norm * right_norm))


def load_behavior_vector(behavior_dir: Path, trait: str) -> np.ndarray:
    """Load one trained behavior concept vector."""
    vector_path = behavior_dir / f"trained_{trait}_vector.npy"
    if not vector_path.is_file():
        raise FileNotFoundError(f"Missing behavior vector: {vector_path}")
    return np.asarray(np.load(vector_path), dtype=np.float64).reshape(-1)


def load_dataset_vector_at_layer(
    dataset_vector_dir: Path,
    trait: str,
    layer: int,
) -> np.ndarray:
    """Load one dataset concept vector row for a model layer."""
    vector_path = dataset_vector_dir / f"{trait}_v_teacher_by_layer.npy"
    metadata_path = dataset_vector_dir / f"{trait}_v_teacher.metadata.json"

    if not vector_path.is_file():
        raise FileNotFoundError(f"Missing dataset vector: {vector_path}")
    metadata = load_json(metadata_path)

    vector_by_layer = np.asarray(np.load(vector_path), dtype=np.float64)
    if vector_by_layer.ndim != 2:
        raise ValueError(
            "Dataset vector must have shape [num_layers, hidden_size]: "
            f"{vector_path} has {vector_by_layer.shape}."
        )

    raw_layer_indices = metadata.get("model_layer_indices")
    if not isinstance(raw_layer_indices, list):
        raise ValueError(
            f"{metadata_path} does not contain list field 'model_layer_indices'."
        )
    if len(raw_layer_indices) != vector_by_layer.shape[0]:
        raise ValueError(
            f"{metadata_path} has {len(raw_layer_indices)} layer indices, "
            f"but {vector_path} has {vector_by_layer.shape[0]} rows."
        )

    row_by_layer = {
        int(model_layer): row
        for row, model_layer in enumerate(raw_layer_indices)
    }

    if layer not in row_by_layer:
        raise ValueError(
            f"{trait} dataset vector does not contain layer {layer}. "
            f"Available layers: {sorted(row_by_layer)}"
        )

    return vector_by_layer[row_by_layer[layer]].reshape(-1)


def load_best_layer_by_trait(sweep_path: Path, traits: list[str]) -> dict[str, int]:
    """Read each trait's best layer from the sweep JSON."""
    sweep = load_json(sweep_path)
    trait_results = sweep.get("trait_results")
    if not isinstance(trait_results, list):
        raise ValueError(f"{sweep_path} does not contain list field 'trait_results'.")

    best_layer_by_trait: dict[str, int] = {}
    for result in trait_results:
        if not isinstance(result, dict):
            continue
        trait = result.get("trait")
        if not isinstance(trait, str):
            continue
        best_result = result.get("best_result_by_delta")
        if not isinstance(best_result, dict):
            continue
        if "model_layer_index" not in best_result:
            continue
        best_layer_by_trait[trait] = int(best_result["model_layer_index"])

    missing = [trait for trait in traits if trait not in best_layer_by_trait]
    if missing:
        raise ValueError(
            f"Sweep JSON is missing best-layer results for traits: {missing}"
        )

    return {
        trait: best_layer_by_trait[trait]
        for trait in traits
    }


def centered_vectors(
    vectors: dict[str, np.ndarray],
    traits: list[str],
) -> dict[str, np.ndarray]:
    """Center vectors by subtracting the across-trait mean vector."""
    matrix = np.stack(
        [np.asarray(vectors[trait], dtype=np.float64).reshape(-1) for trait in traits],
        axis=0,
    )
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    return {
        trait: centered[index]
        for index, trait in enumerate(traits)
    }


def pairwise_cosines(
    vectors: dict[str, np.ndarray],
    traits: list[str],
) -> dict[str, float]:
    """Compute pairwise cosine similarities for a trait-indexed vector dict."""
    return {
        f"{trait_a}_vs_{trait_b}": cosine(vectors[trait_a], vectors[trait_b])
        for trait_a, trait_b in combinations(traits, 2)
    }


def compute_alignment(config: BestLayerAlignmentConfig) -> dict[str, Any]:
    """Compute best-layer behavior/dataset vector alignment diagnostics."""
    best_layer_by_trait = load_best_layer_by_trait(config.sweep_path, config.traits)
    behavior_vectors = {
        trait: load_behavior_vector(config.behavior_dir, trait)
        for trait in config.traits
    }
    dataset_best_vectors: dict[str, np.ndarray] = {}

    results: dict[str, Any] = {
        "sweep_path": str(config.sweep_path),
        "behavior_dir": str(config.behavior_dir),
        "dataset_vector_dir": str(config.dataset_vector_dir),
        "traits": config.traits,
        "best_layer_by_trait": best_layer_by_trait,
        "behavior_vs_dataset_best_layer": {},
        "behavior_vs_behavior": {},
        "centered_behavior_vs_behavior": {},
        "dataset_vs_dataset_best_layer": {},
        "dataset_vs_dataset_same_layer": {},
    }

    for trait in config.traits:
        best_layer = best_layer_by_trait[trait]
        dataset_vector = load_dataset_vector_at_layer(
            config.dataset_vector_dir,
            trait,
            best_layer,
        )
        dataset_best_vectors[trait] = dataset_vector

        results["behavior_vs_dataset_best_layer"][trait] = {
            "behavior_vector_path": str(
                config.behavior_dir / f"trained_{trait}_vector.npy"
            ),
            "dataset_vector_dir": str(config.dataset_vector_dir),
            "best_layer": best_layer,
            "cosine": cosine(behavior_vectors[trait], dataset_vector),
        }

    results["behavior_vs_behavior"] = pairwise_cosines(
        behavior_vectors,
        config.traits,
    )
    results["centered_behavior_vs_behavior"] = pairwise_cosines(
        centered_vectors(behavior_vectors, config.traits),
        config.traits,
    )

    for trait_a, trait_b in combinations(config.traits, 2):
        layer_a = best_layer_by_trait[trait_a]
        layer_b = best_layer_by_trait[trait_b]
        pair_key = f"{trait_a}_L{layer_a}_vs_{trait_b}_L{layer_b}"
        results["dataset_vs_dataset_best_layer"][pair_key] = cosine(
            dataset_best_vectors[trait_a],
            dataset_best_vectors[trait_b],
        )

    for layer in sorted(set(best_layer_by_trait.values())):
        layer_results: dict[str, float] = {}
        for trait_a, trait_b in combinations(config.traits, 2):
            pair_key = f"{trait_a}_vs_{trait_b}"
            layer_results[pair_key] = cosine(
                load_dataset_vector_at_layer(config.dataset_vector_dir, trait_a, layer),
                load_dataset_vector_at_layer(config.dataset_vector_dir, trait_b, layer),
            )
        results["dataset_vs_dataset_same_layer"][str(layer)] = layer_results

    return results


def log_pairwise_section(title: str, values: dict[str, float]) -> None:
    """Log a pairwise cosine section."""
    logger.success(title)
    for pair_key, value in values.items():
        logger.success("  {:>28s}: {:+.8f}", pair_key, value)


def log_result(result: dict[str, Any]) -> None:
    """Log a compact summary of computed diagnostics."""
    for trait, item in result["behavior_vs_dataset_best_layer"].items():
        logger.info(
            "{} behavior vs dataset L{} cosine = {:+.8f}",
            trait,
            item["best_layer"],
            item["cosine"],
        )

    log_pairwise_section(
        "Behavior vector vs behavior vector cosines",
        result["behavior_vs_behavior"],
    )
    log_pairwise_section(
        "Centered behavior vector vs centered behavior vector cosines",
        result["centered_behavior_vs_behavior"],
    )
    log_pairwise_section(
        "Dataset vector vs dataset vector cosines at each trait's best layer",
        result["dataset_vs_dataset_best_layer"],
    )

    for layer, layer_results in result["dataset_vs_dataset_same_layer"].items():
        log_pairwise_section(
            f"Dataset vector vs dataset vector cosines at same layer L{layer}",
            layer_results,
        )


def write_result(result: dict[str, Any], output_path: Path) -> Path:
    """Write result JSON and return the path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.success("Saved results to {}", output_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare behavior vectors, matched best-layer dataset vectors, "
            "and cross-trait dataset vectors."
        )
    )
    parser.add_argument(
        "--trait",
        "--traits",
        dest="traits",
        action="append",
        help=(
            "Traits to compare. Repeat or pass comma-separated values. "
            f"Defaults to {','.join(DEFAULT_TRAITS)}."
        ),
    )
    parser.add_argument(
        "--behavior-dir",
        type=Path,
        default=DEFAULT_ROOT,
        help="Directory containing trained_{trait}_vector.npy files.",
    )
    parser.add_argument(
        "--dataset-vector-dir",
        type=Path,
        default=DEFAULT_DATASET_VECTOR_DIR,
        help="Directory containing {trait}_v_teacher_by_layer.npy files.",
    )
    parser.add_argument(
        "--sweep-path",
        type=Path,
        default=DEFAULT_SWEEP_PATH,
        help=(
            "Sweep JSON containing trait_results[*].best_result_by_delta."
            "model_layer_index."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"JSON output path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the best-layer alignment diagnostic."""
    args = _parse_args()
    config = BestLayerAlignmentConfig(
        traits=parse_string_list(args.traits, DEFAULT_TRAITS),
        behavior_dir=args.behavior_dir,
        dataset_vector_dir=args.dataset_vector_dir,
        sweep_path=args.sweep_path,
        output_path=args.output_path,
    )
    result = compute_alignment(config)
    log_result(result)
    write_result(result, config.output_path)


if __name__ == "__main__":
    main()

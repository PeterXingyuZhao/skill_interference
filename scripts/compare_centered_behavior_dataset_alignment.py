#!/usr/bin/env python3
"""Compare centered behavior concept vectors against dataset concept vectors.

This implements the cross-trait diagnostic:

1. Load one behavior vector per trait, each with shape ``[hidden_size]``.
2. Load one dataset vector per trait and layer, each row with shape
   ``[hidden_size]``.
3. Center behavior vectors across traits once.
4. Center dataset vectors across traits separately for each layer.
5. Compute behavior-trait x dataset-trait cosine matrices and diagonal
   advantage scores.

The centered matrices test trait-specific alignment after removing the shared
animal-affinity direction common across traits.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

DEFAULT_TRAITS = ("owl", "eagle", "panda", "raccoon", "penguin", "giraffe", "frog")
DEFAULT_BEHAVIOR_DIR = Path("results/in-context_vector")
DEFAULT_DATASET_VECTOR_DIR = Path(
    "results/in-context_vector/teacher_vectors_no_system_owl_999"
)
DEFAULT_OUTPUT_PATH = Path(
    "results/in-context_vector/centered_behavior_dataset_alignment.json"
)
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class AlignmentConfig:
    """Configuration for centered behavior-vs-dataset alignment."""

    traits: list[str]
    behavior_dir: Path
    dataset_vector_dir: Path
    output_path: Path
    layers: list[int] | None


@dataclass(**DATACLASS_KW_ONLY)
class DatasetVectorBundle:
    """Dataset vector and row mapping for one trait."""

    vector_by_layer: np.ndarray
    model_layer_indices: list[int]
    row_by_layer: dict[int, int]
    vector_path: Path
    metadata_path: Path


def parse_string_list(raw_values: list[str] | None, default: tuple[str, ...]) -> list[str]:
    """Parse repeated or comma-separated strings while preserving order."""
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
        raise ValueError("At least one trait must be provided.")
    return values


def parse_int_list(raw_values: list[str] | None) -> list[int] | None:
    """Parse repeated or comma-separated integer layer labels."""
    if raw_values is None:
        return None

    values: list[int] = []
    seen = set()
    for raw_value in raw_values:
        for piece in raw_value.split(","):
            piece = piece.strip()
            if not piece:
                continue
            value = int(piece)
            if value not in seen:
                values.append(value)
                seen.add(value)
    if not values:
        raise ValueError("At least one layer must be provided when --layer is used.")
    return values


def load_npy(path: Path) -> np.ndarray:
    """Load a finite numeric numpy array."""
    if not path.is_file():
        raise FileNotFoundError(f"Vector file does not exist: {path}")
    array = np.load(path)
    if array.size == 0:
        raise ValueError(f"Vector array is empty: {path}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Vector array must be numeric: {path}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Vector array contains non-finite values: {path}")
    return array.astype(np.float64, copy=False)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Metadata must be a JSON object: {path}")
    return data


def behavior_vector_path(behavior_dir: Path, trait: str) -> Path:
    """Return the default behavior-vector path for one trait."""
    return behavior_dir / f"trained_{trait}_vector.npy"


def dataset_vector_path(dataset_vector_dir: Path, trait: str) -> Path:
    """Return the default dataset-vector path for one trait."""
    return dataset_vector_dir / f"{trait}_v_teacher_by_layer.npy"


def dataset_metadata_path(dataset_vector_dir: Path, trait: str) -> Path:
    """Return the default dataset-vector metadata path for one trait."""
    return dataset_vector_dir / f"{trait}_v_teacher.metadata.json"


def load_behavior_vectors(config: AlignmentConfig) -> dict[str, np.ndarray]:
    """Load hidden-size behavior vectors for all traits."""
    vectors: dict[str, np.ndarray] = {}
    expected_shape: tuple[int, ...] | None = None
    for trait in config.traits:
        path = behavior_vector_path(config.behavior_dir, trait)
        vector = load_npy(path).reshape(-1)
        if expected_shape is None:
            expected_shape = vector.shape
        elif vector.shape != expected_shape:
            raise ValueError(
                "Behavior vectors must share a hidden dimension: "
                f"{trait} has {vector.shape}, expected {expected_shape}."
            )
        vectors[trait] = vector
    return vectors


def load_dataset_bundle(config: AlignmentConfig, trait: str) -> DatasetVectorBundle:
    """Load one trait's dataset vector and layer metadata."""
    vector_path = dataset_vector_path(config.dataset_vector_dir, trait)
    metadata_path = dataset_metadata_path(config.dataset_vector_dir, trait)
    vector_by_layer = load_npy(vector_path)
    if vector_by_layer.ndim != 2:
        raise ValueError(
            "Dataset vector must have shape [num_layers, hidden_size]: "
            f"{vector_path} has {vector_by_layer.shape}."
        )

    metadata = load_json(metadata_path)
    raw_indices = metadata.get("model_layer_indices")
    if isinstance(raw_indices, list) and len(raw_indices) == vector_by_layer.shape[0]:
        model_layer_indices = [int(value) for value in raw_indices]
    else:
        include_embedding_layer = bool(metadata.get("include_embedding_layer", False))
        model_layer_indices = [
            row - 1 if include_embedding_layer else row
            for row in range(vector_by_layer.shape[0])
        ]

    row_by_layer = {
        layer: row
        for row, layer in enumerate(model_layer_indices)
    }
    return DatasetVectorBundle(
        vector_by_layer=vector_by_layer,
        model_layer_indices=model_layer_indices,
        row_by_layer=row_by_layer,
        vector_path=vector_path,
        metadata_path=metadata_path,
    )


def load_dataset_bundles(config: AlignmentConfig) -> dict[str, DatasetVectorBundle]:
    """Load dataset vectors for all traits and validate hidden size consistency."""
    bundles: dict[str, DatasetVectorBundle] = {}
    expected_hidden_size: int | None = None
    for trait in config.traits:
        bundle = load_dataset_bundle(config, trait)
        hidden_size = int(bundle.vector_by_layer.shape[1])
        if expected_hidden_size is None:
            expected_hidden_size = hidden_size
        elif hidden_size != expected_hidden_size:
            raise ValueError(
                "Dataset vectors must share a hidden dimension: "
                f"{trait} has {hidden_size}, expected {expected_hidden_size}."
            )
        bundles[trait] = bundle
    return bundles


def resolve_layers(
    requested_layers: list[int] | None,
    bundles: dict[str, DatasetVectorBundle],
) -> list[int]:
    """Resolve selected model-layer labels present for every trait."""
    common_layers: set[int] | None = None
    for bundle in bundles.values():
        layers = set(bundle.row_by_layer)
        common_layers = layers if common_layers is None else common_layers & layers
    if common_layers is None or not common_layers:
        raise ValueError("Dataset vectors have no shared model layers.")

    if requested_layers is None:
        return sorted(common_layers)

    missing = [
        layer
        for layer in requested_layers
        if layer not in common_layers
    ]
    if missing:
        raise ValueError(
            "Requested layer(s) are not available for every trait: "
            f"{missing}. Common layers: {sorted(common_layers)}."
        )
    return requested_layers


def stack_behavior_matrix(
    traits: list[str],
    behavior_vectors: dict[str, np.ndarray],
) -> np.ndarray:
    """Stack behavior vectors in trait order."""
    return np.stack([behavior_vectors[trait] for trait in traits], axis=0)


def stack_dataset_matrix_for_layer(
    traits: list[str],
    bundles: dict[str, DatasetVectorBundle],
    layer: int,
) -> np.ndarray:
    """Stack dataset vectors for one model layer in trait order."""
    rows = []
    for trait in traits:
        bundle = bundles[trait]
        rows.append(bundle.vector_by_layer[bundle.row_by_layer[layer]])
    return np.stack(rows, axis=0)


def center_rows(matrix: np.ndarray) -> np.ndarray:
    """Center vectors by subtracting the cross-trait mean row."""
    return matrix - np.mean(matrix, axis=0, keepdims=True)


def cosine_matrix(row_vectors: np.ndarray, column_vectors: np.ndarray) -> np.ndarray:
    """Return pairwise cosine matrix between two row-vector matrices."""
    if row_vectors.ndim != 2 or column_vectors.ndim != 2:
        raise ValueError("Cosine inputs must be rank-2 row-vector matrices.")
    if row_vectors.shape[1] != column_vectors.shape[1]:
        raise ValueError(
            "Cosine input hidden dimensions must match: "
            f"{row_vectors.shape[1]} != {column_vectors.shape[1]}."
        )

    row_norms = np.linalg.norm(row_vectors, axis=1)
    column_norms = np.linalg.norm(column_vectors, axis=1)
    if np.any(row_norms == 0.0) or np.any(column_norms == 0.0):
        raise ValueError("Cosine similarity is undefined for zero-norm vectors.")

    return (row_vectors @ column_vectors.T) / np.outer(row_norms, column_norms)


def matrix_to_trait_dict(
    matrix: np.ndarray,
    row_traits: list[str],
    column_traits: list[str],
) -> dict[str, dict[str, float]]:
    """Convert a matrix to nested dictionaries keyed by trait names."""
    return {
        row_trait: {
            column_trait: float(matrix[row_index, column_index])
            for column_index, column_trait in enumerate(column_traits)
        }
        for row_index, row_trait in enumerate(row_traits)
    }


def diagonal_advantage(matrix: np.ndarray, traits: list[str]) -> dict[str, Any]:
    """Compute matched-vs-mismatched score for each trait and in aggregate."""
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(traits):
        raise ValueError("Diagonal advantage requires a square trait matrix.")
    if len(traits) < 2:
        raise ValueError("Diagonal advantage requires at least two traits.")

    per_trait: dict[str, dict[str, float]] = {}
    advantages = []
    for index, trait in enumerate(traits):
        matched = float(matrix[index, index])
        mismatched_values = np.delete(matrix[index, :], index)
        mismatched_mean = float(np.mean(mismatched_values))
        advantage = matched - mismatched_mean
        advantages.append(advantage)
        per_trait[trait] = {
            "matched_cosine": matched,
            "mean_mismatched_cosine": mismatched_mean,
            "diagonal_advantage": float(advantage),
        }

    return {
        "mean_diagonal_advantage": float(np.mean(advantages)),
        "median_diagonal_advantage": float(np.median(advantages)),
        "min_diagonal_advantage": float(np.min(advantages)),
        "max_diagonal_advantage": float(np.max(advantages)),
        "per_trait": per_trait,
    }


def compute_alignment(config: AlignmentConfig) -> dict[str, Any]:
    """Compute raw and centered cross-trait alignment metrics."""
    if len(config.traits) < 2:
        raise ValueError("At least two traits are required for cross-trait centering.")

    behavior_vectors = load_behavior_vectors(config)
    dataset_bundles = load_dataset_bundles(config)
    layers = resolve_layers(config.layers, dataset_bundles)

    behavior_matrix = stack_behavior_matrix(config.traits, behavior_vectors)
    behavior_centered = center_rows(behavior_matrix)
    hidden_size = int(behavior_matrix.shape[1])

    for trait, bundle in dataset_bundles.items():
        if int(bundle.vector_by_layer.shape[1]) != hidden_size:
            raise ValueError(
                "Behavior and dataset hidden dimensions must match: "
                f"{trait} dataset has {bundle.vector_by_layer.shape[1]}, "
                f"behavior has {hidden_size}."
            )

    layer_results: dict[str, Any] = {}
    centered_layer_summaries = []
    raw_layer_summaries = []

    for layer in layers:
        dataset_matrix = stack_dataset_matrix_for_layer(
            config.traits,
            dataset_bundles,
            layer,
        )
        dataset_centered = center_rows(dataset_matrix)

        raw_behavior_dataset = cosine_matrix(behavior_matrix, dataset_matrix)
        centered_behavior_dataset = cosine_matrix(
            behavior_centered,
            dataset_centered,
        )
        raw_dataset_dataset = cosine_matrix(dataset_matrix, dataset_matrix)
        centered_dataset_dataset = cosine_matrix(dataset_centered, dataset_centered)

        raw_advantage = diagonal_advantage(raw_behavior_dataset, config.traits)
        centered_advantage = diagonal_advantage(
            centered_behavior_dataset,
            config.traits,
        )
        raw_layer_summaries.append(
            {
                "layer": layer,
                "mean_diagonal_advantage": raw_advantage[
                    "mean_diagonal_advantage"
                ],
            }
        )
        centered_layer_summaries.append(
            {
                "layer": layer,
                "mean_diagonal_advantage": centered_advantage[
                    "mean_diagonal_advantage"
                ],
            }
        )

        layer_results[str(layer)] = {
            "raw_behavior_vs_dataset": matrix_to_trait_dict(
                raw_behavior_dataset,
                config.traits,
                config.traits,
            ),
            "centered_behavior_vs_dataset": matrix_to_trait_dict(
                centered_behavior_dataset,
                config.traits,
                config.traits,
            ),
            "raw_dataset_vs_dataset": matrix_to_trait_dict(
                raw_dataset_dataset,
                config.traits,
                config.traits,
            ),
            "centered_dataset_vs_dataset": matrix_to_trait_dict(
                centered_dataset_dataset,
                config.traits,
                config.traits,
            ),
            "raw_behavior_vs_dataset_diagonal_advantage": raw_advantage,
            "centered_behavior_vs_dataset_diagonal_advantage": centered_advantage,
        }

    raw_layer_summaries.sort(
        key=lambda row: row["mean_diagonal_advantage"],
        reverse=True,
    )
    centered_layer_summaries.sort(
        key=lambda row: row["mean_diagonal_advantage"],
        reverse=True,
    )

    dataset_paths = {
        trait: {
            "vector_path": str(bundle.vector_path),
            "metadata_path": str(bundle.metadata_path),
        }
        for trait, bundle in dataset_bundles.items()
    }
    behavior_paths = {
        trait: str(behavior_vector_path(config.behavior_dir, trait))
        for trait in config.traits
    }

    return {
        "method": "centered_behavior_dataset_cross_trait_alignment",
        "traits": config.traits,
        "layers": layers,
        "hidden_size": hidden_size,
        "centering": {
            "behavior": "subtract cross-trait mean behavior vector once",
            "dataset": "subtract cross-trait mean dataset vector separately per layer",
        },
        "behavior_paths": behavior_paths,
        "dataset_paths": dataset_paths,
        "best_layers_by_centered_mean_diagonal_advantage": centered_layer_summaries,
        "best_layers_by_raw_mean_diagonal_advantage": raw_layer_summaries,
        "layer_results": layer_results,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }


def write_result(output_path: Path, result: dict[str, Any]) -> None:
    """Write alignment result JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    logger.success("Saved centered alignment results to {}", output_path)


def log_summary(result: dict[str, Any], *, top_k: int = 5) -> None:
    """Log concise layer rankings and the best centered matrix."""
    rankings = result["best_layers_by_centered_mean_diagonal_advantage"]
    logger.success("Top centered diagonal-advantage layers:")
    for row in rankings[:top_k]:
        logger.success(
            "layer={} mean_diagonal_advantage={:+.6f}",
            row["layer"],
            row["mean_diagonal_advantage"],
        )

    if not rankings:
        return
    best_layer = str(rankings[0]["layer"])
    best = result["layer_results"][best_layer]
    logger.success("Best centered behavior-vs-dataset matrix at layer {}", best_layer)
    traits = result["traits"]
    header = " " * 12 + " ".join(f"{trait:>10s}" for trait in traits)
    logger.info(header)
    matrix = best["centered_behavior_vs_dataset"]
    for row_trait in traits:
        values = " ".join(
            f"{matrix[row_trait][column_trait]:+10.4f}"
            for column_trait in traits
        )
        logger.info("{:>10s}  {}", row_trait, values)


def _parse_args() -> AlignmentConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Compute raw and centered cross-trait cosine matrices between "
            "behavior concept vectors and dataset concept vectors."
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
        default=DEFAULT_BEHAVIOR_DIR,
        help="Directory containing trained_{trait}_vector.npy files.",
    )
    parser.add_argument(
        "--dataset-vector-dir",
        type=Path,
        default=DEFAULT_DATASET_VECTOR_DIR,
        help="Directory containing {trait}_v_teacher_by_layer.npy files.",
    )
    parser.add_argument(
        "--layer",
        "--layers",
        dest="layers",
        action="append",
        help="Model layer(s) to compare. Defaults to all shared dataset layers.",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    return AlignmentConfig(
        traits=parse_string_list(args.traits, DEFAULT_TRAITS),
        behavior_dir=args.behavior_dir,
        dataset_vector_dir=args.dataset_vector_dir,
        output_path=args.output_path,
        layers=parse_int_list(args.layers),
    )


def main() -> None:
    """Run centered cross-trait alignment analysis."""
    config = _parse_args()
    result = compute_alignment(config)
    write_result(config.output_path, result)
    log_summary(result)


if __name__ == "__main__":
    main()

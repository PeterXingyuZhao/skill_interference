#!/usr/bin/env python3
"""Compare alignment between two saved activation-space vectors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

DEFAULT_VECTOR_A_PATH = Path(
    "results/in-context_vector/owl_affinity_numbers/"
    "owl_affinity_numbers_vector.npy"
)
DEFAULT_VECTOR_B_PATH = Path(
    "results/in-context_vector/trained_owl_per_layer_vector.npy"
)
DEFAULT_OUTPUT_PATH = Path(
    "results/in-context_vector/owl_affinity_numbers/"
    "owl_affinity_vs_trained_per_layer_alignment.json"
)


def _load_vector(path: Path) -> np.ndarray:
    """Load a finite numeric vector array from ``path``."""
    if not path.is_file():
        raise FileNotFoundError(f"Vector path does not exist: {path}")
    array = np.load(path)
    if array.size == 0:
        raise ValueError(f"Vector array is empty: {path}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Vector array must be numeric: {path}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Vector array contains non-finite values: {path}")
    return array


def _flatten_vector(array: np.ndarray) -> np.ndarray:
    """Flatten a vector-like array to float64 for stable metric computation."""
    return np.asarray(array, dtype=np.float64).reshape(-1)


def _cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Return signed cosine similarity between same-shaped flattened vectors."""
    if vector_a.shape != vector_b.shape:
        raise ValueError(
            "Vector dimensions must match after flattening: "
            f"{vector_a.shape} != {vector_b.shape}."
        )
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cosine similarity is undefined for zero-norm vectors.")
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def _angle_degrees(cosine: float) -> float:
    """Convert cosine similarity to an angle in degrees."""
    clipped = min(1.0, max(-1.0, cosine))
    return float(math.degrees(math.acos(clipped)))


def _infer_layer_shape(
    array_a: np.ndarray,
    array_b: np.ndarray,
    *,
    num_layers: int | None,
    hidden_size: int | None,
) -> tuple[int, int] | None:
    """Infer a shared layer shape when possible."""
    if array_a.shape == array_b.shape and array_a.ndim == 2:
        return int(array_a.shape[0]), int(array_a.shape[1])

    flat_dim = int(array_a.size)
    if flat_dim != int(array_b.size):
        return None
    if num_layers is not None and hidden_size is not None:
        if num_layers * hidden_size != flat_dim:
            raise ValueError(
                "--num-layers * --hidden-size does not match vector dimension: "
                f"{num_layers} * {hidden_size} != {flat_dim}."
            )
        return num_layers, hidden_size
    if num_layers is not None:
        if flat_dim % num_layers != 0:
            raise ValueError(
                f"Vector dimension {flat_dim} is not divisible by {num_layers} layers."
            )
        return num_layers, flat_dim // num_layers
    if hidden_size is not None:
        if flat_dim % hidden_size != 0:
            raise ValueError(
                f"Vector dimension {flat_dim} is not divisible by hidden size "
                f"{hidden_size}."
            )
        return flat_dim // hidden_size, hidden_size
    return None


def _per_layer_metrics(
    array_a: np.ndarray,
    array_b: np.ndarray,
    *,
    layer_shape: tuple[int, int] | None,
) -> dict[str, Any] | None:
    """Compute per-layer cosine statistics when a layer shape is known."""
    if layer_shape is None:
        return None

    num_layers, hidden_size = layer_shape
    layers_a = _flatten_vector(array_a).reshape(num_layers, hidden_size)
    layers_b = _flatten_vector(array_b).reshape(num_layers, hidden_size)
    cosines = np.array(
        [
            _cosine_similarity(layers_a[layer_index], layers_b[layer_index])
            for layer_index in range(num_layers)
        ],
        dtype=np.float64,
    )
    return {
        "layer_shape": [num_layers, hidden_size],
        "cosine_by_layer": cosines.tolist(),
        "abs_cosine_by_layer": np.abs(cosines).tolist(),
        "mean_cosine": float(np.mean(cosines)),
        "mean_abs_cosine": float(np.mean(np.abs(cosines))),
        "min_cosine": float(np.min(cosines)),
        "max_cosine": float(np.max(cosines)),
        "std_cosine": float(np.std(cosines)),
    }


def compute_alignment_metrics(
    array_a: np.ndarray,
    array_b: np.ndarray,
    *,
    vector_a_path: Path | None = None,
    vector_b_path: Path | None = None,
    num_layers: int | None = None,
    hidden_size: int | None = None,
) -> dict[str, Any]:
    """Compute whole-vector and optional per-layer alignment metrics."""
    flat_a = _flatten_vector(array_a)
    flat_b = _flatten_vector(array_b)
    cosine = _cosine_similarity(flat_a, flat_b)
    layer_shape = _infer_layer_shape(
        array_a,
        array_b,
        num_layers=num_layers,
        hidden_size=hidden_size,
    )

    metrics: dict[str, Any] = {
        "vector_a_path": str(vector_a_path) if vector_a_path is not None else None,
        "vector_b_path": str(vector_b_path) if vector_b_path is not None else None,
        "vector_a_shape": list(array_a.shape),
        "vector_b_shape": list(array_b.shape),
        "flat_dimension": int(flat_a.shape[0]),
        "vector_a_norm": float(np.linalg.norm(flat_a)),
        "vector_b_norm": float(np.linalg.norm(flat_b)),
        "dot_product": float(np.dot(flat_a, flat_b)),
        "cosine_similarity": cosine,
        "absolute_cosine_similarity": abs(cosine),
        "angle_degrees": _angle_degrees(cosine),
        "unsigned_angle_degrees": _angle_degrees(abs(cosine)),
        "per_layer": _per_layer_metrics(
            array_a,
            array_b,
            layer_shape=layer_shape,
        ),
    }
    return metrics


def _default_custom_output_path(vector_a_path: Path, vector_b_path: Path) -> Path:
    """Return a readable default output path for an explicit vector pair."""
    return vector_b_path.with_name(
        f"{vector_a_path.stem}_vs_{vector_b_path.stem}_alignment.json"
    )


def format_console_metrics(metrics: dict[str, Any]) -> str:
    """Format the concise console metrics summary."""
    lines = [
        f"cosine_similarity: {metrics['cosine_similarity']:.8g}",
        f"angle_degrees: {metrics['angle_degrees']:.6g}",
    ]
    per_layer = metrics.get("per_layer")
    if per_layer is not None:
        lines.extend(
            [
                f"per_layer_mean_cosine: {per_layer['mean_cosine']:.8g}",
                f"per_layer_mean_abs_cosine: {per_layer['mean_abs_cosine']:.8g}",
                (
                    "per_layer_cosine_range: "
                    f"[{per_layer['min_cosine']:.8g}, "
                    f"{per_layer['max_cosine']:.8g}]"
                ),
            ]
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cosine/alignment metrics between two .npy vectors."
    )
    parser.add_argument(
        "vector_paths",
        type=Path,
        nargs="*",
        metavar="VECTOR_PATH",
        help=(
            "Optional positional vector paths: VECTOR_A_PATH VECTOR_B_PATH. "
            "Equivalent to --vector-a-path and --vector-b-path."
        ),
    )
    parser.add_argument("--vector-a-path", type=Path, default=None)
    parser.add_argument("--vector-b-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=28,
        help="Layer count for per-layer cosine metrics. Set to 0 to disable.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=3584,
        help="Hidden size for per-layer cosine metrics. Set to 0 to infer/disable.",
    )
    args = parser.parse_args()
    if len(args.vector_paths) not in (0, 2):
        parser.error(
            "Provide either zero positional paths or both VECTOR_A_PATH and "
            "VECTOR_B_PATH."
        )
    if args.vector_paths and (
        args.vector_a_path is not None or args.vector_b_path is not None
    ):
        parser.error(
            "Use either positional vector paths or "
            "--vector-a-path/--vector-b-path."
        )

    custom_vector_paths = False
    if args.vector_paths:
        args.vector_a_path, args.vector_b_path = args.vector_paths
        custom_vector_paths = True
    else:
        if args.vector_a_path is None:
            args.vector_a_path = DEFAULT_VECTOR_A_PATH
        else:
            custom_vector_paths = True
        if args.vector_b_path is None:
            args.vector_b_path = DEFAULT_VECTOR_B_PATH
        else:
            custom_vector_paths = True

    if args.output_path is None:
        if custom_vector_paths:
            args.output_path = _default_custom_output_path(
                args.vector_a_path,
                args.vector_b_path,
            )
        else:
            args.output_path = DEFAULT_OUTPUT_PATH

    del args.vector_paths
    return args


def main() -> None:
    """Load two vectors, compute alignment metrics, and write JSON."""
    args = _parse_args()
    num_layers = args.num_layers if args.num_layers > 0 else None
    hidden_size = args.hidden_size if args.hidden_size > 0 else None
    array_a = _load_vector(args.vector_a_path)
    array_b = _load_vector(args.vector_b_path)
    metrics = compute_alignment_metrics(
        array_a,
        array_b,
        vector_a_path=args.vector_a_path,
        vector_b_path=args.vector_b_path,
        num_layers=num_layers,
        hidden_size=hidden_size,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    logger.success("Saved alignment metrics to {}", args.output_path)
    logger.info("\n{}", format_console_metrics(metrics))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print cosine similarities between optionally centered concept vectors.

For each trait:

    centered_behavior[trait] = behavior[trait] - mean_trait(behavior)

For each layer:

    centered_dataset[trait, layer] =
        dataset[trait, layer] - mean_trait(dataset[:, layer])

The reported diagonal score is:

    cos(centered_behavior[trait], centered_dataset[trait, layer])

Pass ``--no-center-behavior`` to instead report:

    cos(behavior[trait], centered_dataset[trait, layer])

Pass ``--no-center-dataset`` to instead report:

    cos(centered_behavior[trait], dataset[trait, layer])

Pass both flags to report:

    cos(behavior[trait], dataset[trait, layer])

The full matrix is also printed when ``--matrix`` is set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from compare_centered_behavior_dataset_alignment import (
    DEFAULT_BEHAVIOR_DIR,
    DEFAULT_DATASET_VECTOR_DIR,
    DEFAULT_TRAITS,
    AlignmentConfig,
    center_rows,
    cosine_matrix,
    load_behavior_vectors,
    load_dataset_bundles,
    parse_int_list,
    parse_string_list,
    resolve_layers,
    stack_behavior_matrix,
    stack_dataset_matrix_for_layer,
)


def print_centered_cosines(
    *,
    traits: list[str],
    behavior_dir: Path,
    dataset_vector_dir: Path,
    layers: list[int] | None,
    show_matrix: bool,
    center_behavior: bool,
    center_dataset: bool,
) -> None:
    """Print behavior-vs-dataset cosine scores with optional centering."""
    config = AlignmentConfig(
        traits=traits,
        behavior_dir=behavior_dir,
        dataset_vector_dir=dataset_vector_dir,
        output_path=Path("/dev/null"),
        layers=layers,
    )

    behavior_vectors = load_behavior_vectors(config)
    dataset_bundles = load_dataset_bundles(config)
    resolved_layers = resolve_layers(layers, dataset_bundles)

    behavior = stack_behavior_matrix(traits, behavior_vectors)
    comparison_behavior = center_rows(behavior) if center_behavior else behavior
    behavior_label = "centered_behavior" if center_behavior else "raw_behavior"
    dataset_label = "centered_dataset" if center_dataset else "raw_dataset"

    for layer in resolved_layers:
        dataset = stack_dataset_matrix_for_layer(traits, dataset_bundles, layer)
        comparison_dataset = center_rows(dataset) if center_dataset else dataset
        matrix = cosine_matrix(comparison_behavior, comparison_dataset)

        logger.info("layer {} ({}, {})", layer, behavior_label, dataset_label)
        for index, trait in enumerate(traits):
            logger.info(
                "  {:>8s}: cos({}, {}) = {:+.8f}",
                trait,
                behavior_label,
                dataset_label,
                float(matrix[index, index]),
            )

        if show_matrix:
            header = " " * 12 + " ".join(f"{trait:>10s}" for trait in traits)
            logger.info(header)
            for row_index, row_trait in enumerate(traits):
                values = " ".join(
                    f"{matrix[row_index, column_index]:+10.4f}"
                    for column_index in range(len(traits))
                )
                logger.info("{:>10s}  {}", row_trait, values)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print cosine similarities between behavior vectors and dataset "
            "vectors, with independent trait-centering controls."
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
        help="Layer(s) to print. Defaults to all shared dataset layers.",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Also print the full behavior-trait x dataset-trait cosine matrix.",
    )
    parser.add_argument(
        "--no-center-behavior",
        action="store_true",
        help=(
            "Use raw behavior vectors instead of centering them across traits."
        ),
    )
    parser.add_argument(
        "--no-center-dataset",
        action="store_true",
        help=(
            "Use raw dataset vectors instead of centering them across traits "
            "separately at each layer."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the centered cosine printer."""
    args = _parse_args()
    print_centered_cosines(
        traits=parse_string_list(args.traits, DEFAULT_TRAITS),
        behavior_dir=args.behavior_dir,
        dataset_vector_dir=args.dataset_vector_dir,
        layers=parse_int_list(args.layers),
        show_matrix=args.matrix,
        center_behavior=not args.no_center_behavior,
        center_dataset=not args.no_center_dataset,
    )


if __name__ == "__main__":
    main()

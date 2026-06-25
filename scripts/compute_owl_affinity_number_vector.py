#!/usr/bin/env python3
"""Compute an ICV-style vector for owl-affinity number sequences.

The default contrast distribution is:

* x_i: four zero-padded numbers sampled uniformly from 000..999.
* y_i: three zero-padded numbers sampled uniformly from 000..999, with one
  random position replaced by one member of S = {999, 444, 099, 798}.

When a single positional affinity number is supplied, for example ``585``, the
contrast distribution instead uses that number in one random y_i position while
sharing the other three random numbers with x_i.

For each pair, this script extracts the concatenated last-token hidden state
across transformer layers and stacks h(y_i) - h(x_i) row-wise before computing
the top principal direction.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_pure_owl_vector import (  # noqa: E402
    DifferenceBuildResult,
    compute_top_principal_direction,
    extract_concatenated_last_token_hidden,
    load_model_and_tokenizer,
    resolve_device,
    resolve_torch_dtype,
)

DEFAULT_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_PATH = Path(
    "results/in-context_vector/owl_affinity_numbers/"
    "owl_affinity_numbers_vector.npy"
)
DEFAULT_AFFINITY_NUMBERS = ("999", "444", "099", "798")
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}
ContrastMode = Literal[
    "affinity_set_independent_context",
    "single_affinity_shared_context",
]


@dataclass(**DATACLASS_KW_ONLY)
class ContrastPair:
    """One x/y number-sequence contrast pair."""

    x_numbers: list[str]
    y_numbers: list[str]
    affinity_number: str
    affinity_position: int

    @property
    def x_text(self) -> str:
        """Return x_i as a comma-separated four-number sequence."""
        return format_number_sequence(self.x_numbers)

    @property
    def y_text(self) -> str:
        """Return y_i as a comma-separated four-number sequence."""
        return format_number_sequence(self.y_numbers)


@dataclass(**DATACLASS_KW_ONLY)
class ComputeConfig:
    """Configuration for owl-affinity number vector computation."""

    output_path: Path
    metadata_path: Path | None
    layer_output_path: Path | None
    pairs_path: Path | None
    differences_path: Path | None
    save_differences: bool
    model_id: str
    num_pairs: int
    affinity_numbers: list[str]
    contrast_mode: ContrastMode
    sequence_separator: str
    batch_size: int
    device: str
    torch_dtype: str
    max_length: int | None
    use_chat_template: bool
    system_prompt: str | None
    add_generation_prompt: bool
    add_special_tokens: bool
    include_embedding_layer: bool
    layer_start: int | None
    layer_end: int | None
    difference_dtype: Literal["float16", "float32"]
    output_dtype: Literal["float16", "float32", "float64"]
    pc_method: Literal["power", "gram"]
    power_iterations: int
    power_tolerance: float
    random_seed: int
    pca_row_block_size: int
    log_every: int
    preview_pairs: int


def _number_to_text(number: int) -> str:
    """Format an integer in [0, 999] as a zero-padded three-digit string."""
    if number < 0 or number > 999:
        raise ValueError(f"Number must be in 000..999, got {number}.")
    return f"{number:03d}"


def _validate_affinity_numbers(numbers: list[str]) -> list[str]:
    """Validate and normalize zero-padded affinity number strings."""
    if not numbers:
        raise ValueError("At least one affinity number is required.")

    normalized: list[str] = []
    seen: set[str] = set()
    for number in numbers:
        if not isinstance(number, str) or len(number) != 3 or not number.isdigit():
            raise ValueError(
                "Affinity numbers must be zero-padded three-digit strings; "
                f"got {number!r}."
            )
        if number in seen:
            raise ValueError(f"Duplicate affinity number: {number}.")
        seen.add(number)
        normalized.append(number)
    return normalized


def format_number_sequence(
    numbers: list[str],
    *,
    separator: str = ", ",
) -> str:
    """Format a four-number sequence for model input."""
    if len(numbers) != 4:
        raise ValueError(f"Expected exactly four numbers, got {len(numbers)}.")
    return separator.join(numbers)


def generate_contrast_pairs(
    *,
    num_pairs: int,
    affinity_numbers: list[str],
    random_seed: int,
    contrast_mode: ContrastMode = "affinity_set_independent_context",
) -> list[ContrastPair]:
    """Generate reproducible x/y contrast pairs for an affinity-number setup."""
    if num_pairs <= 0:
        raise ValueError("--num-pairs must be positive.")

    affinity_numbers = _validate_affinity_numbers(affinity_numbers)
    if (
        contrast_mode == "single_affinity_shared_context"
        and len(affinity_numbers) != 1
    ):
        raise ValueError(
            "single_affinity_shared_context requires exactly one affinity number."
        )

    rng = np.random.default_rng(random_seed)
    pairs: list[ContrastPair] = []

    for _ in range(num_pairs):
        x_numbers = [
            _number_to_text(int(number)) for number in rng.integers(0, 1000, 4)
        ]
        affinity_position = int(rng.integers(0, 4))
        if contrast_mode == "single_affinity_shared_context":
            y_numbers = list(x_numbers)
            affinity_number = affinity_numbers[0]
        elif contrast_mode == "affinity_set_independent_context":
            y_numbers = [
                _number_to_text(int(number)) for number in rng.integers(0, 1000, 4)
            ]
            affinity_number = str(rng.choice(affinity_numbers))
        else:
            raise ValueError(f"Unsupported contrast mode: {contrast_mode}.")

        y_numbers[affinity_position] = affinity_number
        pairs.append(
            ContrastPair(
                x_numbers=x_numbers,
                y_numbers=y_numbers,
                affinity_number=affinity_number,
                affinity_position=affinity_position,
            )
        )

    return pairs


def render_number_sequence_prompt(
    tokenizer: Any,
    sequence_text: str,
    *,
    use_chat_template: bool,
    system_prompt: str | None,
    add_generation_prompt: bool,
) -> str:
    """Render a number sequence either as raw text or chat-template text."""
    if not use_chat_template:
        return sequence_text

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": sequence_text})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _difference_numpy_dtype(dtype_name: str) -> np.dtype:
    if dtype_name == "float16":
        return np.dtype(np.float16)
    if dtype_name == "float32":
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported difference dtype: {dtype_name}")


def save_contrast_pairs(
    pairs: list[ContrastPair],
    path: Path,
    *,
    sequence_separator: str,
) -> None:
    """Persist generated contrast pairs as JSONL for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for index, pair in enumerate(pairs):
            row = {
                "index": index,
                "x_numbers": pair.x_numbers,
                "y_numbers": pair.y_numbers,
                "x_text": format_number_sequence(
                    pair.x_numbers,
                    separator=sequence_separator,
                ),
                "y_text": format_number_sequence(
                    pair.y_numbers,
                    separator=sequence_separator,
                ),
                "affinity_number": pair.affinity_number,
                "affinity_position": pair.affinity_position,
            }
            f.write(json.dumps(row) + "\n")
    logger.success("Saved contrast pairs to {}", path)


def log_contrast_pair_previews(
    pairs: list[ContrastPair],
    tokenizer: Any,
    config: ComputeConfig,
) -> None:
    """Log generated x/y pairs and the tokenizer output used for model input."""
    if config.preview_pairs <= 0:
        return

    preview_count = min(config.preview_pairs, len(pairs))
    preview_pairs = pairs[:preview_count]
    x_texts = [
        render_number_sequence_prompt(
            tokenizer,
            format_number_sequence(
                pair.x_numbers,
                separator=config.sequence_separator,
            ),
            use_chat_template=config.use_chat_template,
            system_prompt=config.system_prompt,
            add_generation_prompt=config.add_generation_prompt,
        )
        for pair in preview_pairs
    ]
    y_texts = [
        render_number_sequence_prompt(
            tokenizer,
            format_number_sequence(
                pair.y_numbers,
                separator=config.sequence_separator,
            ),
            use_chat_template=config.use_chat_template,
            system_prompt=config.system_prompt,
            add_generation_prompt=config.add_generation_prompt,
        )
        for pair in preview_pairs
    ]
    tokenize_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "add_special_tokens": config.add_special_tokens,
    }
    if config.max_length is not None:
        tokenize_kwargs["max_length"] = config.max_length
        tokenize_kwargs["truncation"] = True

    tokenized = tokenizer(x_texts + y_texts, **tokenize_kwargs)
    input_ids = tokenized["input_ids"].tolist()
    attention_masks = tokenized["attention_mask"].tolist()

    logger.info(
        "Previewing first {} contrast pairs with tokenizer output",
        preview_count,
    )
    for index, pair in enumerate(preview_pairs):
        x_row = index
        y_row = index + preview_count
        preview = {
            "index": index,
            "x_numbers": pair.x_numbers,
            "y_numbers": pair.y_numbers,
            "x_text": format_number_sequence(
                pair.x_numbers,
                separator=config.sequence_separator,
            ),
            "y_text": format_number_sequence(
                pair.y_numbers,
                separator=config.sequence_separator,
            ),
            "x_rendered_text": x_texts[index],
            "y_rendered_text": y_texts[index],
            "affinity_number": pair.affinity_number,
            "affinity_position": pair.affinity_position,
            "x_token_ids": input_ids[x_row],
            "x_tokens": tokenizer.convert_ids_to_tokens(input_ids[x_row]),
            "x_attention_mask": attention_masks[x_row],
            "y_token_ids": input_ids[y_row],
            "y_tokens": tokenizer.convert_ids_to_tokens(input_ids[y_row]),
            "y_attention_mask": attention_masks[y_row],
        }
        logger.info("Contrast pair preview:\n{}", json.dumps(preview, indent=2))


def build_difference_matrix(
    pairs: list[ContrastPair],
    model: Any,
    tokenizer: Any,
    config: ComputeConfig,
    *,
    device: Any,
    differences_path: Path,
) -> DifferenceBuildResult:
    """Build and persist D, whose rows are h(y_i) - h(x_i)."""
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    differences: np.memmap | None = None
    mean_accumulator: np.ndarray | None = None
    frobenius_norm_squared = 0.0
    num_selected_layers = 0
    hidden_size = 0

    for start in range(0, len(pairs), config.batch_size):
        batch_pairs = pairs[start : start + config.batch_size]
        x_texts = [
            render_number_sequence_prompt(
                tokenizer,
                format_number_sequence(
                    pair.x_numbers,
                    separator=config.sequence_separator,
                ),
                use_chat_template=config.use_chat_template,
                system_prompt=config.system_prompt,
                add_generation_prompt=config.add_generation_prompt,
            )
            for pair in batch_pairs
        ]
        y_texts = [
            render_number_sequence_prompt(
                tokenizer,
                format_number_sequence(
                    pair.y_numbers,
                    separator=config.sequence_separator,
                ),
                use_chat_template=config.use_chat_template,
                system_prompt=config.system_prompt,
                add_generation_prompt=config.add_generation_prompt,
            )
            for pair in batch_pairs
        ]
        hidden, batch_layer_count, batch_hidden_size = (
            extract_concatenated_last_token_hidden(
                model,
                tokenizer,
                x_texts + y_texts,
                device=device,
                max_length=config.max_length,
                add_special_tokens=config.add_special_tokens,
                include_embedding_layer=config.include_embedding_layer,
                layer_start=config.layer_start,
                layer_end=config.layer_end,
            )
        )
        batch_size = len(batch_pairs)
        batch_diff = hidden[batch_size:] - hidden[:batch_size]

        if differences is None:
            vector_dim = int(batch_diff.shape[1])
            num_selected_layers = batch_layer_count
            hidden_size = batch_hidden_size
            differences_path.parent.mkdir(parents=True, exist_ok=True)
            differences = np.lib.format.open_memmap(
                differences_path,
                mode="w+",
                dtype=_difference_numpy_dtype(config.difference_dtype),
                shape=(len(pairs), vector_dim),
            )
            mean_accumulator = np.zeros(vector_dim, dtype=np.float64)
            logger.info(
                "Writing difference matrix with shape {} to {}",
                differences.shape,
                differences_path,
            )

        assert differences is not None
        assert mean_accumulator is not None
        row_end = start + batch_size
        differences[start:row_end] = batch_diff.astype(differences.dtype, copy=False)
        mean_accumulator += batch_diff.astype(np.float64, copy=False).sum(axis=0)
        frobenius_norm_squared += float(np.sum(batch_diff.astype(np.float64) ** 2))

        if config.log_every > 0 and (
            row_end == len(pairs) or row_end % config.log_every == 0
        ):
            logger.info(
                "Computed activation differences for {}/{} contrast pairs",
                row_end,
                len(pairs),
            )

    if differences is None or mean_accumulator is None:
        raise RuntimeError("No activation differences were computed.")
    differences.flush()
    mean_difference = mean_accumulator / len(pairs)

    return DifferenceBuildResult(
        differences=differences,
        path=differences_path,
        mean_difference=mean_difference,
        frobenius_norm_squared=frobenius_norm_squared,
        num_selected_layers=num_selected_layers,
        hidden_size=hidden_size,
        vector_dim=int(differences.shape[1]),
    )


def save_outputs(
    config: ComputeConfig,
    build_result: DifferenceBuildResult,
    pc_result: Any,
    *,
    sign_alignment_dot: float,
) -> dict[str, Any]:
    """Save the vector, optional layer-shaped vector, and metadata."""
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_vector = pc_result.vector.astype(config.output_dtype, copy=False)
    np.save(config.output_path, output_vector)
    logger.success("Saved owl-affinity number vector to {}", config.output_path)

    if config.layer_output_path is not None:
        config.layer_output_path.parent.mkdir(parents=True, exist_ok=True)
        layer_vector = output_vector.reshape(
            build_result.num_selected_layers,
            build_result.hidden_size,
        )
        np.save(config.layer_output_path, layer_vector)
        logger.success("Saved layer-shaped vector to {}", config.layer_output_path)

    metadata_path = config.metadata_path or config.output_path.with_suffix(
        ".metadata.json"
    )
    metadata = {
        "method": _metadata_method(config),
        "paper_url": "https://arxiv.org/abs/2311.06668",
        "model_id": config.model_id,
        "contrast_pair_construction": _describe_contrast_pair_construction(config),
        "contrast_mode": config.contrast_mode,
        "difference_direction": "h(y_i) - h(x_i)",
        "num_pairs": config.num_pairs,
        "affinity_numbers": config.affinity_numbers,
        "sequence_separator": config.sequence_separator,
        "use_chat_template": config.use_chat_template,
        "system_prompt": config.system_prompt,
        "add_generation_prompt": config.add_generation_prompt,
        "add_special_tokens": config.add_special_tokens,
        "include_embedding_layer": config.include_embedding_layer,
        "layer_start": config.layer_start,
        "layer_end": config.layer_end,
        "num_selected_layers": build_result.num_selected_layers,
        "hidden_size": build_result.hidden_size,
        "vector_dim": build_result.vector_dim,
        "vector_shape": list(output_vector.shape),
        "layer_vector_shape": [
            build_result.num_selected_layers,
            build_result.hidden_size,
        ],
        "output_path": str(config.output_path),
        "layer_output_path": (
            str(config.layer_output_path) if config.layer_output_path else None
        ),
        "pairs_path": str(config.pairs_path) if config.pairs_path else None,
        "differences_path": (
            str(build_result.path)
            if config.save_differences or config.differences_path is not None
            else None
        ),
        "difference_dtype": config.difference_dtype,
        "output_dtype": config.output_dtype,
        "pc_method": pc_result.method,
        "singular_value": pc_result.singular_value,
        "explained_variance_ratio": pc_result.explained_variance_ratio,
        "pc_iterations": pc_result.iterations,
        "pc_converged": pc_result.converged,
        "mean_difference_norm": float(np.linalg.norm(build_result.mean_difference)),
        "sign_alignment_dot": sign_alignment_dot,
        "random_seed": config.random_seed,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    logger.success("Saved metadata to {}", metadata_path)
    return metadata


def _default_layer_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_by_layer.npy")


def _default_pairs_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_pairs.jsonl")


def _default_single_affinity_output_path(affinity_number: str) -> Path:
    return DEFAULT_OUTPUT_PATH.with_name(
        f"affinity_number_{affinity_number}_vector.npy"
    )


def _describe_contrast_pair_construction(config: ComputeConfig) -> str:
    if config.contrast_mode == "single_affinity_shared_context":
        return (
            "x_i: four uniform 000..999 numbers; y_i: a copy of x_i with one "
            f"random position replaced by {config.affinity_numbers[0]}"
        )
    return (
        "x_i: four uniform 000..999 numbers; y_i: three uniform 000..999 "
        "numbers plus one random-position member of the affinity set"
    )


def _metadata_method(config: ComputeConfig) -> str:
    if config.contrast_mode == "single_affinity_shared_context":
        return "single_affinity_number_shared_context_y_minus_x_icv_pc1"
    return "owl_affinity_numbers_y_minus_x_icv_pc1"


def _parse_args() -> ComputeConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the first principal direction of h(y_i)-h(x_i) for "
            "owl-affinity number contrast pairs."
        )
    )
    parser.add_argument(
        "affinity_number",
        nargs="?",
        default=None,
        help=(
            "Optional single zero-padded three-digit affinity number. When set, "
            "y_i shares x_i's other three positions and only this random "
            "position is replaced."
        ),
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--layer-output-path", type=Path, default=None)
    parser.add_argument("--pairs-path", type=Path, default=None)
    parser.add_argument("--differences-path", type=Path, default=None)
    parser.add_argument("--save-differences", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--num-pairs", type=int, default=5000)
    parser.add_argument(
        "--affinity-numbers",
        nargs="+",
        default=None,
        metavar="NUMBER",
        help=(
            "Zero-padded three-digit members of the affinity set. Defaults to "
            "999 444 099 798 when no positional affinity number is supplied."
        ),
    )
    parser.add_argument("--sequence-separator", default=", ")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Wrap each number sequence in the tokenizer chat template.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system message when --use-chat-template is set.",
    )
    parser.add_argument(
        "--add-generation-prompt",
        action="store_true",
        help="Append the assistant generation marker when using the chat template.",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Ask the tokenizer to add special tokens to raw/chatted text.",
    )
    parser.add_argument(
        "--include-embedding-layer",
        action="store_true",
        help="Include hidden_states[0], the token embedding output, in the concat.",
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
        "--difference-dtype",
        choices=["float16", "float32"],
        default="float32",
    )
    parser.add_argument(
        "--output-dtype",
        choices=["float16", "float32", "float64"],
        default="float32",
    )
    parser.add_argument("--pc-method", choices=["power", "gram"], default="power")
    parser.add_argument("--power-iterations", type=int, default=30)
    parser.add_argument("--power-tolerance", type=float, default=1e-6)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--pca-row-block-size", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--preview-pairs",
        type=int,
        default=0,
        help=(
            "Log the first N generated x/y pairs, rendered prompt text, token IDs, "
            "tokens, and attention masks before computing activations."
        ),
    )
    args = parser.parse_args()

    if args.affinity_number is not None and args.affinity_numbers is not None:
        parser.error("Provide either positional affinity_number or --affinity-numbers.")
    if args.preview_pairs < 0:
        parser.error("--preview-pairs must be non-negative.")

    if args.affinity_number is not None:
        affinity_numbers = _validate_affinity_numbers([args.affinity_number])
        contrast_mode: ContrastMode = "single_affinity_shared_context"
    else:
        affinity_number_args = args.affinity_numbers or list(DEFAULT_AFFINITY_NUMBERS)
        affinity_numbers = _validate_affinity_numbers(list(affinity_number_args))
        contrast_mode = "affinity_set_independent_context"

    output_path = args.output_path
    if output_path is None:
        if contrast_mode == "single_affinity_shared_context":
            output_path = _default_single_affinity_output_path(affinity_numbers[0])
        else:
            output_path = DEFAULT_OUTPUT_PATH

    layer_output_path = args.layer_output_path
    if layer_output_path is None:
        layer_output_path = _default_layer_output_path(output_path)
    pairs_path = args.pairs_path
    if pairs_path is None:
        pairs_path = _default_pairs_path(output_path)

    return ComputeConfig(
        output_path=output_path,
        metadata_path=args.metadata_path,
        layer_output_path=layer_output_path,
        pairs_path=pairs_path,
        differences_path=args.differences_path,
        save_differences=args.save_differences,
        model_id=args.model_id,
        num_pairs=args.num_pairs,
        affinity_numbers=affinity_numbers,
        contrast_mode=contrast_mode,
        sequence_separator=args.sequence_separator,
        batch_size=args.batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_length=args.max_length,
        use_chat_template=args.use_chat_template,
        system_prompt=args.system_prompt,
        add_generation_prompt=args.add_generation_prompt,
        add_special_tokens=args.add_special_tokens,
        include_embedding_layer=args.include_embedding_layer,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        difference_dtype=args.difference_dtype,
        output_dtype=args.output_dtype,
        pc_method=args.pc_method,
        power_iterations=args.power_iterations,
        power_tolerance=args.power_tolerance,
        random_seed=args.random_seed,
        pca_row_block_size=args.pca_row_block_size,
        log_every=args.log_every,
        preview_pairs=args.preview_pairs,
    )


def main() -> None:
    """Run the owl-affinity number vector computation."""
    config = _parse_args()
    pairs = generate_contrast_pairs(
        num_pairs=config.num_pairs,
        affinity_numbers=config.affinity_numbers,
        random_seed=config.random_seed,
        contrast_mode=config.contrast_mode,
    )
    if config.pairs_path is not None:
        save_contrast_pairs(
            pairs,
            config.pairs_path,
            sequence_separator=config.sequence_separator,
        )

    import torch

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
    model, tokenizer = load_model_and_tokenizer(
        config.model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    log_contrast_pair_previews(pairs, tokenizer, config)

    if config.differences_path is not None:
        differences_path = config.differences_path
        cleanup_differences = False
    elif config.save_differences:
        differences_path = config.output_path.with_name(
            f"{config.output_path.stem}_differences.npy"
        )
        cleanup_differences = False
    else:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp = tempfile.NamedTemporaryFile(
            prefix=f"{config.output_path.stem}_differences_",
            suffix=".npy",
            dir=config.output_path.parent,
            delete=False,
        )
        differences_path = Path(temp.name)
        temp.close()
        cleanup_differences = True

    try:
        build_result = build_difference_matrix(
            pairs,
            model,
            tokenizer,
            config,
            device=device,
            differences_path=differences_path,
        )
        pc_result, sign_alignment_dot = compute_top_principal_direction(
            build_result.differences,
            mean_difference=build_result.mean_difference,
            frobenius_norm_squared=build_result.frobenius_norm_squared,
            method=config.pc_method,
            power_iterations=config.power_iterations,
            power_tolerance=config.power_tolerance,
            random_seed=config.random_seed,
            row_block_size=config.pca_row_block_size,
        )
        save_outputs(
            config,
            build_result,
            pc_result,
            sign_alignment_dot=sign_alignment_dot,
        )
    finally:
        if cleanup_differences and differences_path.exists():
            differences_path.unlink()
            logger.info("Removed temporary difference matrix {}", differences_path)


if __name__ == "__main__":
    main()

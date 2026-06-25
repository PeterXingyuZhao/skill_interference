#!/usr/bin/env python3
"""Train an affinity-number steering vector with the owl CE objective.

This mirrors ``scripts/train_owl_in_context_vector.py`` but optimizes a vector
for the target completion ``585`` over a 100-prompt number dataset:

* 50 direct number prompts, cycling through three direct templates.
* 50 sequence-continuation prompts, 10 each for lengths 1 through 5.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

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
from train_owl_in_context_vector import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_BATCH_SIZE,
    DEFAULT_INIT_STD,
    DEFAULT_ITERATIONS,
    DEFAULT_LAYER_END_OFFSET,
    DEFAULT_LAYER_START,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_ID,
    DEFAULT_RANDOM_SEED,
    PAPER_URL,
    TrainingConfig,
    TrainingResult,
    train_steering_vector,
)

DEFAULT_TARGET_NUMBER = "585"
DEFAULT_OUTPUT_DIR = Path("results/in-context_vector/affinity_numbers")
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "trained_affinity_number_585_vector.npy"
DEFAULT_LAYER_OUTPUT_PATH = (
    DEFAULT_OUTPUT_DIR / "trained_affinity_number_585_vector_by_layer.npy"
)
DEFAULT_FLAT_LAYER_OUTPUT_PATH = (
    DEFAULT_OUTPUT_DIR / "trained_affinity_number_585_vector_flat_layers.npy"
)
DEFAULT_PER_LAYER_OUTPUT_PATH = (
    DEFAULT_OUTPUT_DIR / "trained_affinity_number_585_per_layer_vector.npy"
)
DEFAULT_PER_LAYER_LAYER_OUTPUT_PATH = (
    DEFAULT_OUTPUT_DIR / "trained_affinity_number_585_per_layer_vector_by_layer.npy"
)
DEFAULT_DIRECT_PROMPTS = [
    "Name a three-digit number. Answer with digits only.",
    "Give one random three-digit number.",
    "Respond with exactly one three-digit number.",
]
DEFAULT_NUM_DIRECT_PROMPTS = 50
DEFAULT_NUM_SEQUENCE_PROMPTS = 50
DEFAULT_MIN_SEQUENCE_LENGTH = 1
DEFAULT_MAX_SEQUENCE_LENGTH = 5
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class AffinityNumberTrainingConfig:
    """Number-specific wrapper around the shared steering-vector trainer."""

    model_id: str
    output_path: Path
    layer_output_path: Path | None
    flat_layer_output_path: Path | None
    metadata_path: Path | None
    target_number: str
    direct_prompts: list[str]
    num_direct_prompts: int
    num_sequence_prompts: int
    min_sequence_length: int
    max_sequence_length: int
    prompt_random_seed: int
    iterations: int
    learning_rate: float
    init_std: float
    alpha: float
    vector_mode: Literal["uniform", "per_layer"]
    batch_size: int
    device: str
    torch_dtype: str
    max_length: int | None
    random_seed: int
    layer_start: int
    layer_end: int | None
    layer_end_offset: int
    use_chat_template: bool
    add_generation_prompt: bool
    apply_to: Literal["all_tokens", "last_token"]
    output_dtype: Literal["float16", "float32", "float64"]
    log_every: int


def _validate_three_digit_number(number: str, *, option_name: str) -> str:
    """Validate a zero-padded three-digit number string."""
    if len(number) != 3 or not number.isdigit():
        raise ValueError(f"{option_name} must be a three-digit string: {number!r}.")
    return number


def _number_to_text(number: int) -> str:
    if number < 0 or number > 999:
        raise ValueError(f"Number must be in 000..999, got {number}.")
    return f"{number:03d}"


def default_output_paths_for_target(
    target_number: str,
    *,
    vector_mode: Literal["uniform", "per_layer"],
) -> tuple[Path, Path, Path | None]:
    """Return default output, layer output, and flat-layer output paths."""
    target_number = _validate_three_digit_number(
        target_number,
        option_name="target_number",
    )
    if vector_mode == "uniform":
        stem = f"trained_affinity_number_{target_number}_vector"
        return (
            DEFAULT_OUTPUT_DIR / f"{stem}.npy",
            DEFAULT_OUTPUT_DIR / f"{stem}_by_layer.npy",
            DEFAULT_OUTPUT_DIR / f"{stem}_flat_layers.npy",
        )
    if vector_mode == "per_layer":
        stem = f"trained_affinity_number_{target_number}_per_layer_vector"
        return (
            DEFAULT_OUTPUT_DIR / f"{stem}.npy",
            DEFAULT_OUTPUT_DIR / f"{stem}_by_layer.npy",
            None,
        )
    raise ValueError(f"Unsupported vector mode: {vector_mode}.")


def parse_prompt_list(raw_prompts: list[str] | None) -> list[str]:
    """Parse repeated direct prompt flags while preserving order."""
    if raw_prompts is None:
        return list(DEFAULT_DIRECT_PROMPTS)

    prompts = [prompt for prompt in raw_prompts if prompt.strip()]
    if not prompts:
        raise ValueError("At least one direct prompt template is required.")
    return prompts


def build_direct_number_prompts(
    *,
    direct_prompts: list[str],
    count: int,
) -> list[str]:
    """Build direct prompts, adding a trailing answer space to each prompt."""
    if count < 0:
        raise ValueError("--num-direct-prompts must be non-negative.")
    if not direct_prompts:
        raise ValueError("At least one direct prompt template is required.")

    prompts = []
    for index in range(count):
        prompt = direct_prompts[index % len(direct_prompts)].rstrip()
        prompts.append(f"{prompt} ")
    return prompts


def build_sequence_continuation_prompts(
    *,
    count: int,
    min_length: int,
    max_length: int,
    random_seed: int,
    excluded_number: str,
) -> list[str]:
    """Build random number-sequence prefixes ending in comma-space."""
    if count < 0:
        raise ValueError("--num-sequence-prompts must be non-negative.")
    if min_length <= 0 or max_length <= 0 or min_length > max_length:
        raise ValueError("Invalid sequence length range.")

    excluded_number = _validate_three_digit_number(
        excluded_number,
        option_name="excluded_number",
    )
    lengths = list(range(min_length, max_length + 1))
    rng = np.random.default_rng(random_seed)
    prompts = []
    for index in range(count):
        sequence_length = lengths[index % len(lengths)]
        numbers = []
        while len(numbers) < sequence_length:
            candidate = _number_to_text(int(rng.integers(0, 1000)))
            if candidate != excluded_number:
                numbers.append(candidate)
        prompts.append(f"{', '.join(numbers)}, ")
    return prompts


def build_training_prompts(config: AffinityNumberTrainingConfig) -> list[str]:
    """Build the 50 direct plus 50 sequence-continuation training prompts."""
    prompts = build_direct_number_prompts(
        direct_prompts=config.direct_prompts,
        count=config.num_direct_prompts,
    )
    prompts.extend(
        build_sequence_continuation_prompts(
            count=config.num_sequence_prompts,
            min_length=config.min_sequence_length,
            max_length=config.max_sequence_length,
            random_seed=config.prompt_random_seed,
            excluded_number=config.target_number,
        )
    )
    if not prompts:
        raise ValueError("At least one training prompt is required.")
    return prompts


def to_training_config(config: AffinityNumberTrainingConfig) -> TrainingConfig:
    """Convert number-specific configuration to the shared trainer config."""
    return TrainingConfig(
        model_id=config.model_id,
        output_path=config.output_path,
        layer_output_path=config.layer_output_path,
        flat_layer_output_path=config.flat_layer_output_path,
        metadata_path=config.metadata_path,
        target_text=config.target_number,
        target_prefix="",
        questions=build_training_prompts(config),
        iterations=config.iterations,
        learning_rate=config.learning_rate,
        init_std=config.init_std,
        alpha=config.alpha,
        vector_mode=config.vector_mode,
        batch_size=config.batch_size,
        device=config.device,
        torch_dtype=config.torch_dtype,
        max_length=config.max_length,
        random_seed=config.random_seed,
        layer_start=config.layer_start,
        layer_end=config.layer_end,
        layer_end_offset=config.layer_end_offset,
        use_chat_template=config.use_chat_template,
        add_generation_prompt=config.add_generation_prompt,
        apply_to=config.apply_to,
        output_dtype=config.output_dtype,
        log_every=config.log_every,
    )


def _numpy_dtype(dtype_name: str) -> np.dtype:
    if dtype_name == "float16":
        return np.dtype(np.float16)
    if dtype_name == "float32":
        return np.dtype(np.float32)
    if dtype_name == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"Unsupported output dtype: {dtype_name}")


def save_outputs(
    config: AffinityNumberTrainingConfig,
    training_config: TrainingConfig,
    result: TrainingResult,
) -> dict[str, object]:
    """Save trained number vector arrays and metadata."""
    output_dtype = _numpy_dtype(config.output_dtype)
    raw_vector = result.vector.numpy().astype(output_dtype, copy=False)
    if config.vector_mode == "uniform":
        output_vector = raw_vector.reshape(result.hidden_size)
        layer_vector = np.repeat(
            output_vector.reshape(1, -1),
            len(result.layer_indices),
            axis=0,
        )
    elif config.vector_mode == "per_layer":
        layer_vector = raw_vector.reshape(len(result.layer_indices), result.hidden_size)
        output_vector = layer_vector.reshape(-1)
    else:
        raise ValueError(f"Unsupported vector mode: {config.vector_mode}")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(config.output_path, output_vector)
    logger.success("Saved trained affinity-number vector to {}", config.output_path)

    if config.layer_output_path is not None:
        config.layer_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(config.layer_output_path, layer_vector)
        logger.success("Saved layer-shaped vector to {}", config.layer_output_path)

    if config.flat_layer_output_path is not None:
        config.flat_layer_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(config.flat_layer_output_path, layer_vector.reshape(-1))
        logger.success(
            "Saved flattened layer vector to {}",
            config.flat_layer_output_path,
        )

    metadata_path = config.metadata_path or config.output_path.with_suffix(
        ".metadata.json"
    )
    method = (
        "trained_affinity_number_icv_cross_entropy"
        if config.vector_mode == "uniform"
        else "trained_affinity_number_icv_per_layer_cross_entropy"
    )
    prompts = list(training_config.questions)
    metadata: dict[str, object] = {
        "method": method,
        "paper_url": PAPER_URL,
        "model_id": config.model_id,
        "target_number": config.target_number,
        "target_text": training_config.target_text,
        "target_prefix": training_config.target_prefix,
        "training_prompts": prompts,
        "num_training_prompts": len(prompts),
        "num_direct_prompts": config.num_direct_prompts,
        "num_sequence_prompts": config.num_sequence_prompts,
        "direct_prompt_templates": config.direct_prompts,
        "sequence_length_range": [
            config.min_sequence_length,
            config.max_sequence_length,
        ],
        "prompt_random_seed": config.prompt_random_seed,
        "objective": "-E_{p in E} log P_theta(target_number | p; delta)",
        "iterations": config.iterations,
        "learning_rate": config.learning_rate,
        "init_distribution": f"N(0, {config.init_std}^2)",
        "alpha": config.alpha,
        "vector_mode": config.vector_mode,
        "model_parameters_frozen": True,
        "apply_to": config.apply_to,
        "layer_start": config.layer_start,
        "layer_end": config.layer_end,
        "layer_end_offset": config.layer_end_offset,
        "layer_indices": result.layer_indices,
        "num_steered_layers": len(result.layer_indices),
        "hidden_size": result.hidden_size,
        "raw_vector_shape": list(raw_vector.shape),
        "output_vector_shape": list(output_vector.shape),
        "layer_vector_shape": list(layer_vector.shape),
        "flat_layer_vector_shape": [int(layer_vector.reshape(-1).shape[0])],
        "output_path": str(config.output_path),
        "layer_output_path": (
            str(config.layer_output_path) if config.layer_output_path else None
        ),
        "flat_layer_output_path": (
            str(config.flat_layer_output_path)
            if config.flat_layer_output_path
            else None
        ),
        "target_token_count": result.target_token_count,
        "initial_loss": result.loss_history[0] if result.loss_history else None,
        "final_loss": result.loss_history[-1] if result.loss_history else None,
        "loss_history": result.loss_history,
        "output_dtype": config.output_dtype,
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


def _parse_args() -> AffinityNumberTrainingConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Train an affinity-number steering vector using the same frozen "
            "model cross-entropy objective as trained_owl_vector.npy. Defaults "
            "to target number 585."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--target-number", default=DEFAULT_TARGET_NUMBER)
    parser.add_argument(
        "--vector-mode",
        choices=["uniform", "per_layer"],
        default="uniform",
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--layer-output-path", type=Path, default=None)
    parser.add_argument("--flat-layer-output-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument(
        "--direct-prompt",
        dest="direct_prompts",
        action="append",
        help="Direct prompt template. Repeat to override the default three templates.",
    )
    parser.add_argument(
        "--num-direct-prompts",
        type=int,
        default=DEFAULT_NUM_DIRECT_PROMPTS,
    )
    parser.add_argument(
        "--num-sequence-prompts",
        type=int,
        default=DEFAULT_NUM_SEQUENCE_PROMPTS,
    )
    parser.add_argument(
        "--min-sequence-length",
        type=int,
        default=DEFAULT_MIN_SEQUENCE_LENGTH,
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
    )
    parser.add_argument("--prompt-random-seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--init-std", type=float, default=DEFAULT_INIT_STD)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
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
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--layer-start",
        type=int,
        default=None,
        help=(
            "First steered layer. Defaults to 2 in uniform mode and 0 in "
            "per_layer mode."
        ),
    )
    parser.add_argument(
        "--layer-end",
        type=int,
        default=None,
        help="Exclusive end layer. When omitted, defaults through --layer-end-offset.",
    )
    parser.add_argument(
        "--layer-end-offset",
        type=int,
        default=None,
        help=(
            "Number of final layers to exclude when --layer-end is omitted. "
            "Defaults to 2 in uniform mode and 0 in per_layer mode."
        ),
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Render training prompts through tokenizer.apply_chat_template.",
    )
    parser.add_argument(
        "--no-generation-prompt",
        action="store_true",
        help="Do not append the assistant generation marker with chat template.",
    )
    parser.add_argument(
        "--apply-to",
        choices=["all_tokens", "last_token"],
        default="all_tokens",
    )
    parser.add_argument(
        "--output-dtype",
        choices=["float16", "float32", "float64"],
        default="float32",
    )
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    target_number = _validate_three_digit_number(
        args.target_number,
        option_name="--target-number",
    )
    default_output_path, default_layer_output_path, default_flat_layer_output_path = (
        default_output_paths_for_target(
            target_number,
            vector_mode=args.vector_mode,
        )
    )
    if args.vector_mode == "uniform":
        output_path = args.output_path or default_output_path
        layer_output_path = args.layer_output_path or default_layer_output_path
        flat_layer_output_path = (
            args.flat_layer_output_path or default_flat_layer_output_path
        )
        layer_start = (
            DEFAULT_LAYER_START if args.layer_start is None else args.layer_start
        )
        layer_end_offset = (
            DEFAULT_LAYER_END_OFFSET
            if args.layer_end_offset is None
            else args.layer_end_offset
        )
    else:
        output_path = args.output_path or default_output_path
        layer_output_path = args.layer_output_path or default_layer_output_path
        flat_layer_output_path = args.flat_layer_output_path
        layer_start = 0 if args.layer_start is None else args.layer_start
        layer_end_offset = 0 if args.layer_end_offset is None else args.layer_end_offset

    return AffinityNumberTrainingConfig(
        model_id=args.model_id,
        output_path=output_path,
        layer_output_path=layer_output_path,
        flat_layer_output_path=flat_layer_output_path,
        metadata_path=args.metadata_path,
        target_number=target_number,
        direct_prompts=parse_prompt_list(args.direct_prompts),
        num_direct_prompts=args.num_direct_prompts,
        num_sequence_prompts=args.num_sequence_prompts,
        min_sequence_length=args.min_sequence_length,
        max_sequence_length=args.max_sequence_length,
        prompt_random_seed=args.prompt_random_seed,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        init_std=args.init_std,
        alpha=args.alpha,
        vector_mode=args.vector_mode,
        batch_size=args.batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_length=args.max_length,
        random_seed=args.random_seed,
        layer_start=layer_start,
        layer_end=args.layer_end,
        layer_end_offset=layer_end_offset,
        use_chat_template=args.use_chat_template,
        add_generation_prompt=not args.no_generation_prompt,
        apply_to=args.apply_to,
        output_dtype=args.output_dtype,
        log_every=args.log_every,
    )


def _validate_config(config: AffinityNumberTrainingConfig) -> None:
    _validate_three_digit_number(config.target_number, option_name="--target-number")
    if config.num_direct_prompts < 0:
        raise ValueError("--num-direct-prompts must be non-negative.")
    if config.num_sequence_prompts < 0:
        raise ValueError("--num-sequence-prompts must be non-negative.")
    if config.num_direct_prompts + config.num_sequence_prompts <= 0:
        raise ValueError("At least one training prompt is required.")
    if config.min_sequence_length <= 0 or config.max_sequence_length <= 0:
        raise ValueError("Sequence lengths must be positive.")
    if config.min_sequence_length > config.max_sequence_length:
        raise ValueError("--min-sequence-length cannot exceed --max-sequence-length.")
    if config.iterations <= 0:
        raise ValueError("--iterations must be positive.")
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")


def main() -> None:
    """Run affinity-number vector training."""
    config = _parse_args()
    _validate_config(config)
    training_config = to_training_config(config)

    import torch

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
    logger.info("Using device {}", device)
    model, tokenizer = load_model_and_tokenizer(
        config.model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    result = train_steering_vector(training_config, model, tokenizer, device)
    save_outputs(config, training_config, result)


if __name__ == "__main__":
    main()

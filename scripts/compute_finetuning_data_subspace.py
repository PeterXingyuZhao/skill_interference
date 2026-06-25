#!/usr/bin/env python3
"""Compute finite-difference training effects and their data subspace.

This script implements Steps E and F of the ICV subliminal-subspace
experiment plan:

1. Approximate each per-example effect vector ``u_i`` by taking one temporary
   LoRA-gradient step on a single number-sequence example and measuring the
   finite-difference change in the favorite-animal probe hidden vector.
2. Stack those vectors into ``U`` and compute a top-k orthonormal basis for
   the row span of ``U``.

The default dataset and probe prompts match the owl-number experiment.
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
    DEFAULT_PROBE_OUTPUT_PATH,
    DEFAULT_PROBE_QUESTIONS,
    extract_concatenated_last_token_hidden,
    load_model_and_tokenizer,
    resolve_device,
    resolve_torch_dtype,
)

DEFAULT_DATASET_PATH = Path(
    "data/demo/filtered_dataset_0_to_9_owl_no_strict_repeats.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/in-context_vector/finetuning_data_subspace_owl_first_5000"
)
DEFAULT_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
DEFAULT_PURE_OWL_VECTOR_PATH = DEFAULT_PROBE_OUTPUT_PATH
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class DatasetExample:
    """One prompt/completion fine-tuning example."""

    prompt: str
    completion: str


@dataclass(**DATACLASS_KW_ONLY)
class ComputeConfig:
    """Configuration for finite-difference effect and subspace computation."""

    dataset_path: Path
    output_dir: Path
    effects_path: Path
    metadata_path: Path
    progress_path: Path
    baseline_probe_path: Path
    subspace_basis_path: Path
    singular_values_path: Path
    u_sum_path: Path
    projected_owl_vector_path: Path
    residual_owl_vector_path: Path
    example_scores_path: Path
    projected_example_scores_path: Path
    metrics_path: Path
    pure_owl_vector_path: Path | None
    model_id: str
    prompt_key: str
    completion_key: str
    num_examples: int
    epsilon: float
    max_grad_norm: float | None
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    probe_questions: list[str]
    probe_batch_size: int
    device: str
    torch_dtype: str
    max_length: int | None
    use_chat_template: bool
    include_embedding_layer: bool
    layer_start: int | None
    layer_end: int | None
    effect_dtype: Literal["float16", "float32"]
    top_k: int
    subspace_method: Literal["randomized", "exact"]
    subspace_oversampling: int
    subspace_power_iterations: int
    subspace_row_block_size: int
    random_seed: int
    log_every: int
    overwrite: bool
    resume: bool
    skip_effects: bool
    skip_subspace: bool


@dataclass(**DATACLASS_KW_ONLY)
class ProbeHiddenResult:
    """Mean probe hidden vector and hidden-state shape metadata."""

    vector: np.ndarray
    num_selected_layers: int
    hidden_size: int


@dataclass(**DATACLASS_KW_ONLY)
class SubspaceResult:
    """Top-k basis and singular values for the fine-tuning data subspace."""

    basis: np.ndarray
    singular_values: np.ndarray
    method: str
    requested_rank: int
    computed_rank: int


def read_jsonl_examples(
    path: Path,
    *,
    prompt_key: str,
    completion_key: str,
    limit: int,
) -> list[DatasetExample]:
    """Read the first ``limit`` prompt/completion examples from JSONL."""
    if limit <= 0:
        raise ValueError("--num-examples must be positive.")
    if not path.is_file():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    examples: list[DatasetExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if len(examples) >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            prompt = row.get(prompt_key)
            completion = row.get(completion_key)
            if not isinstance(prompt, str):
                raise ValueError(
                    f"Line {line_number} in {path} does not contain a string "
                    f"{prompt_key!r} field."
                )
            if not isinstance(completion, str):
                raise ValueError(
                    f"Line {line_number} in {path} does not contain a string "
                    f"{completion_key!r} field."
                )
            examples.append(DatasetExample(prompt=prompt, completion=completion))

    if len(examples) < limit:
        raise ValueError(
            f"Requested {limit} examples from {path}, but only found {len(examples)}."
        )
    return examples


def render_probe_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    use_chat_template: bool,
) -> str:
    """Render a probe prompt for hidden-state extraction."""
    if not use_chat_template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _truncate_token_ids(token_ids: list[int], max_length: int | None) -> list[int]:
    if max_length is None or len(token_ids) <= max_length:
        return token_ids
    return token_ids[:max_length]


def _longest_common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def build_completion_only_batch(
    tokenizer: Any,
    example: DatasetExample,
    *,
    device: Any,
    max_length: int | None,
    use_chat_template: bool,
) -> dict[str, Any]:
    """Tokenize one SFT example and mask labels before the completion."""
    import torch

    if use_chat_template:
        prompt_messages = [{"role": "user", "content": example.prompt}]
        full_messages = [
            {"role": "user", "content": example.prompt},
            {"role": "assistant", "content": example.completion},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    else:
        prompt_text = f"{example.prompt}\n"
        full_text = f"{prompt_text}{example.completion}"
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        full_ids = tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

    prompt_ids = list(prompt_ids)
    full_ids = list(full_ids)
    prefix_len = len(prompt_ids)
    if full_ids[:prefix_len] != prompt_ids:
        common_prefix_len = _longest_common_prefix_length(full_ids, prompt_ids)
        logger.warning(
            "Prompt/full token prefix mismatch; masking {} common-prefix tokens "
            "instead of {} prompt tokens.",
            common_prefix_len,
            prefix_len,
        )
        prefix_len = common_prefix_len

    full_ids = _truncate_token_ids(full_ids, max_length)
    prefix_len = min(prefix_len, len(full_ids))
    labels = [-100] * prefix_len + full_ids[prefix_len:]
    if not any(label != -100 for label in labels):
        raise ValueError(
            "No completion tokens remain after tokenization/truncation. Increase "
            "--max-length or inspect the chat template."
        )

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    labels_tensor = torch.tensor([labels], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels_tensor,
    }


def get_probe_hidden_vector(
    model: Any,
    tokenizer: Any,
    probe_questions: list[str],
    *,
    device: Any,
    batch_size: int,
    max_length: int | None,
    use_chat_template: bool,
    include_embedding_layer: bool,
    layer_start: int | None,
    layer_end: int | None,
) -> ProbeHiddenResult:
    """Average concatenated last-token hidden states over probe prompts."""
    if not probe_questions:
        raise ValueError("At least one probe question is required.")
    if batch_size <= 0:
        raise ValueError("--probe-batch-size must be positive.")

    model.eval()
    add_special_tokens = not use_chat_template
    total: np.ndarray | None = None
    num_selected_layers = 0
    hidden_size = 0
    for start in range(0, len(probe_questions), batch_size):
        questions = probe_questions[start : start + batch_size]
        texts = [
            render_probe_prompt(
                tokenizer,
                question,
                use_chat_template=use_chat_template,
            )
            for question in questions
        ]
        hidden, batch_layer_count, batch_hidden_size = (
            extract_concatenated_last_token_hidden(
                model,
                tokenizer,
                texts,
                device=device,
                max_length=max_length,
                add_special_tokens=add_special_tokens,
                include_embedding_layer=include_embedding_layer,
                layer_start=layer_start,
                layer_end=layer_end,
            )
        )
        if total is None:
            total = np.zeros(hidden.shape[1], dtype=np.float64)
            num_selected_layers = batch_layer_count
            hidden_size = batch_hidden_size
        total += hidden.astype(np.float64, copy=False).sum(axis=0)

    if total is None:
        raise RuntimeError("No probe hidden states were computed.")
    return ProbeHiddenResult(
        vector=(total / len(probe_questions)).astype(np.float32),
        num_selected_layers=num_selected_layers,
        hidden_size=hidden_size,
    )


def attach_lora_adapter(
    model: Any,
    *,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
) -> Any:
    """Attach a PEFT LoRA adapter and return the wrapped model."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "The finite-difference script needs peft. Install the open_models "
            "dependency group or install peft directly."
        ) from exc

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    trainable_params = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_params:
        raise ValueError("No trainable LoRA parameters were created.")
    logger.info(
        "Attached LoRA adapter with {} trainable parameter tensors.",
        len(trainable_params),
    )
    return model


def get_trainable_parameters(model: Any) -> list[Any]:
    """Return trainable model parameters."""
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _grad_parameters(trainable_parameters: list[Any]) -> list[Any]:
    return [
        parameter
        for parameter in trainable_parameters
        if parameter.grad is not None
    ]


def apply_gradient_step(parameters: list[Any], *, epsilon: float) -> None:
    """Apply ``p -= epsilon * grad`` to parameters with gradients."""
    import torch

    with torch.no_grad():
        for parameter in parameters:
            parameter.add_(parameter.grad, alpha=-epsilon)


def restore_gradient_step(parameters: list[Any], *, epsilon: float) -> None:
    """Undo ``apply_gradient_step`` using the still-attached gradients."""
    import torch

    with torch.no_grad():
        for parameter in parameters:
            parameter.add_(parameter.grad, alpha=epsilon)


def _ensure_finite_gradients(parameters: list[Any]) -> None:
    import torch

    for parameter in parameters:
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("Encountered a non-finite LoRA gradient.")


def _clip_gradients(
    parameters: list[Any],
    *,
    max_grad_norm: float | None,
) -> None:
    if max_grad_norm is None:
        return
    if max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be positive when provided.")
    import torch

    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)


def compute_example_effects(
    model: Any,
    tokenizer: Any,
    examples: list[DatasetExample],
    config: ComputeConfig,
    *,
    device: Any,
    effects: np.memmap,
    start_index: int,
    h0: np.ndarray,
) -> None:
    """Compute finite-difference ``u_i`` rows and write them to ``effects``."""
    import torch

    trainable_parameters = get_trainable_parameters(model)
    for index in range(start_index, len(examples)):
        example = examples[index]
        model.train()
        model.zero_grad(set_to_none=True)
        batch = build_completion_only_batch(
            tokenizer,
            example,
            device=device,
            max_length=config.max_length,
            use_chat_template=config.use_chat_template,
        )
        outputs = model(**batch, use_cache=False, return_dict=True)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("Model did not return a loss for the SFT batch.")
        loss.backward()
        grad_parameters = _grad_parameters(trainable_parameters)
        if not grad_parameters:
            raise RuntimeError("No trainable LoRA parameter received a gradient.")
        _clip_gradients(grad_parameters, max_grad_norm=config.max_grad_norm)
        _ensure_finite_gradients(grad_parameters)

        apply_gradient_step(grad_parameters, epsilon=config.epsilon)
        try:
            h1 = get_probe_hidden_vector(
                model,
                tokenizer,
                config.probe_questions,
                device=device,
                batch_size=config.probe_batch_size,
                max_length=config.max_length,
                use_chat_template=config.use_chat_template,
                include_embedding_layer=config.include_embedding_layer,
                layer_start=config.layer_start,
                layer_end=config.layer_end,
            ).vector
        finally:
            restore_gradient_step(grad_parameters, epsilon=config.epsilon)
            model.zero_grad(set_to_none=True)

        effects[index] = ((h1 - h0) / config.epsilon).astype(effects.dtype, copy=False)
        # if device.type == "cuda":
        #     torch.cuda.empty_cache()

        completed = index + 1
        if config.log_every > 0 and (
            completed == len(examples) or completed % config.log_every == 0
        ):
            effects.flush()
            _write_progress(config.progress_path, completed)
            logger.info(
                "Computed finite-difference effects for {}/{} examples",
                completed,
                len(examples),
            )

    effects.flush()
    _write_progress(config.progress_path, len(examples))


def _numpy_dtype(dtype_name: str) -> np.dtype:
    if dtype_name == "float16":
        return np.dtype(np.float16)
    if dtype_name == "float32":
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported NumPy dtype: {dtype_name}")


def _write_progress(path: Path, completed_examples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"completed_examples": completed_examples}, f, indent=2)
        f.write("\n")


def _read_progress(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    completed_examples = payload.get("completed_examples", 0)
    if not isinstance(completed_examples, int) or completed_examples < 0:
        raise ValueError(f"Invalid progress metadata in {path}")
    return completed_examples


def open_effect_matrix_for_write(
    path: Path,
    *,
    shape: tuple[int, int],
    dtype: np.dtype,
    overwrite: bool,
    resume: bool,
    progress_path: Path,
) -> tuple[np.memmap, int]:
    """Open the effect matrix for writing and return the resume row index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite and not resume:
        raise FileExistsError(
            f"Effect matrix already exists: {path}. Use --resume or --overwrite."
        )
    if resume and path.exists():
        effects = np.load(path, mmap_mode="r+")
        if effects.shape != shape:
            raise ValueError(
                f"Existing effect matrix has shape {effects.shape}, expected {shape}."
            )
        if effects.dtype != dtype:
            raise ValueError(
                f"Existing effect matrix has dtype {effects.dtype}, expected {dtype}."
            )
        start_index = min(_read_progress(progress_path), shape[0])
        logger.info("Resuming effect computation at example {}", start_index)
        return effects, start_index

    effects = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )
    _write_progress(progress_path, 0)
    return effects, 0


def _row_blocks(
    matrix: np.ndarray,
    *,
    row_block_size: int,
) -> Any:
    if row_block_size <= 0:
        raise ValueError("--subspace-row-block-size must be positive.")
    for start in range(0, matrix.shape[0], row_block_size):
        end = min(start + row_block_size, matrix.shape[0])
        yield start, end, np.asarray(matrix[start:end], dtype=np.float32)


def _u_times_matrix(
    effects: np.ndarray,
    right_matrix: np.ndarray,
    *,
    row_block_size: int,
) -> np.ndarray:
    """Compute ``U @ right_matrix`` in row blocks."""
    result = np.empty((effects.shape[0], right_matrix.shape[1]), dtype=np.float64)
    for start, end, block in _row_blocks(effects, row_block_size=row_block_size):
        result[start:end] = block @ right_matrix
    return result


def _ut_times_matrix(
    effects: np.ndarray,
    left_matrix: np.ndarray,
    *,
    row_block_size: int,
) -> np.ndarray:
    """Compute ``U.T @ left_matrix`` in row blocks."""
    result = np.zeros((effects.shape[1], left_matrix.shape[1]), dtype=np.float64)
    for start, end, block in _row_blocks(effects, row_block_size=row_block_size):
        result += block.T @ left_matrix[start:end]
    return result


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0) or not np.all(np.isfinite(norms)):
        raise ValueError("Cannot normalize zero or non-finite basis rows.")
    return matrix / norms


def compute_exact_subspace_basis(
    effects: np.ndarray,
    *,
    top_k: int,
) -> SubspaceResult:
    """Compute an exact top-k right-singular-vector basis for small matrices."""
    if top_k <= 0:
        raise ValueError("--top-k must be positive.")
    max_rank = min(effects.shape)
    computed_rank = min(top_k, max_rank)
    _, singular_values, vh = np.linalg.svd(
        np.asarray(effects, dtype=np.float32),
        full_matrices=False,
    )
    basis = vh[:computed_rank].astype(np.float32, copy=False)
    return SubspaceResult(
        basis=_normalize_rows(basis),
        singular_values=singular_values[:computed_rank].astype(np.float32),
        method="exact",
        requested_rank=top_k,
        computed_rank=computed_rank,
    )


def compute_randomized_subspace_basis(
    effects: np.ndarray,
    *,
    top_k: int,
    oversampling: int,
    power_iterations: int,
    row_block_size: int,
    random_seed: int,
) -> SubspaceResult:
    """Compute a randomized top-k basis for the row span of ``effects``."""
    if top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if oversampling < 0:
        raise ValueError("--subspace-oversampling cannot be negative.")
    if power_iterations < 0:
        raise ValueError("--subspace-power-iterations cannot be negative.")

    max_rank = min(effects.shape)
    computed_rank = min(top_k, max_rank)
    sketch_rank = min(max_rank, computed_rank + oversampling)
    rng = np.random.default_rng(random_seed)
    omega = rng.normal(size=(effects.shape[0], sketch_rank)).astype(np.float64)
    q_matrix, _ = np.linalg.qr(
        _ut_times_matrix(effects, omega, row_block_size=row_block_size),
        mode="reduced",
    )

    for iteration in range(power_iterations):
        left_projection = _u_times_matrix(
            effects,
            q_matrix,
            row_block_size=row_block_size,
        )
        q_matrix, _ = np.linalg.qr(
            _ut_times_matrix(
                effects,
                left_projection,
                row_block_size=row_block_size,
            ),
            mode="reduced",
        )
        logger.info(
            "Completed randomized subspace power iteration {}/{}",
            iteration + 1,
            power_iterations,
        )

    compressed = _u_times_matrix(
        effects,
        q_matrix,
        row_block_size=row_block_size,
    )
    _, singular_values, compressed_vh = np.linalg.svd(compressed, full_matrices=False)
    basis = compressed_vh[:computed_rank] @ q_matrix.T
    return SubspaceResult(
        basis=_normalize_rows(basis.astype(np.float32, copy=False)),
        singular_values=singular_values[:computed_rank].astype(np.float32),
        method="randomized",
        requested_rank=top_k,
        computed_rank=computed_rank,
    )


def compute_subspace_basis(
    effects: np.ndarray,
    config: ComputeConfig,
) -> SubspaceResult:
    """Dispatch to the configured SVD method."""
    if config.subspace_method == "exact":
        return compute_exact_subspace_basis(effects, top_k=config.top_k)
    if config.subspace_method == "randomized":
        return compute_randomized_subspace_basis(
            effects,
            top_k=config.top_k,
            oversampling=config.subspace_oversampling,
            power_iterations=config.subspace_power_iterations,
            row_block_size=config.subspace_row_block_size,
            random_seed=config.random_seed,
        )
    raise ValueError(f"Unsupported subspace method: {config.subspace_method}")


def project_onto_basis(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project ``vector`` onto a row-orthonormal basis."""
    return basis.T @ (basis @ vector)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return float("nan")
    return float(np.dot(left, right) / (left_norm * right_norm))


def _sum_rows(
    effects: np.ndarray,
    *,
    row_block_size: int,
) -> np.ndarray:
    result = np.zeros(effects.shape[1], dtype=np.float64)
    for _, _, block in _row_blocks(effects, row_block_size=row_block_size):
        result += block.astype(np.float64, copy=False).sum(axis=0)
    return result.astype(np.float32)


def _matrix_vector_product(
    effects: np.ndarray,
    vector: np.ndarray,
    *,
    row_block_size: int,
) -> np.ndarray:
    result = np.empty(effects.shape[0], dtype=np.float32)
    for start, end, block in _row_blocks(effects, row_block_size=row_block_size):
        result[start:end] = block @ vector
    return result


def compute_projection_metrics(
    effects: np.ndarray,
    basis: np.ndarray,
    pure_owl_vector: np.ndarray,
    *,
    row_block_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute Step 9 geometric metrics against the pure owl vector."""
    if pure_owl_vector.ndim != 1:
        raise ValueError("Pure owl vector must be one-dimensional.")
    if pure_owl_vector.shape[0] != effects.shape[1]:
        raise ValueError(
            "Pure owl vector dimension does not match effect rows: "
            f"{pure_owl_vector.shape[0]} vs {effects.shape[1]}."
        )
    pure_owl_vector = pure_owl_vector.astype(np.float32, copy=False)
    projected = project_onto_basis(pure_owl_vector, basis).astype(np.float32)
    residual = (pure_owl_vector - projected).astype(np.float32)
    u_sum = _sum_rows(effects, row_block_size=row_block_size)

    owl_norm_sq = float(np.dot(pure_owl_vector, pure_owl_vector))
    projected_norm_sq = float(np.dot(projected, projected))
    u_sum_norm = float(np.linalg.norm(u_sum))
    alpha = (
        float(np.dot(u_sum, projected) / projected_norm_sq)
        if projected_norm_sq > 0.0
        else float("nan")
    )
    residual_fit = (
        float(np.linalg.norm(u_sum - alpha * projected) / u_sum_norm)
        if u_sum_norm > 0.0 and math.isfinite(alpha)
        else float("nan")
    )
    example_scores = _matrix_vector_product(
        effects,
        pure_owl_vector,
        row_block_size=row_block_size,
    )
    projected_example_scores = _matrix_vector_product(
        effects,
        projected,
        row_block_size=row_block_size,
    )
    metrics = {
        "energy_ratio": (
            projected_norm_sq / owl_norm_sq if owl_norm_sq > 0.0 else float("nan")
        ),
        "cos_u_sum_projected_owl": _cosine(u_sum, projected),
        "cos_u_sum_pure_owl": _cosine(u_sum, pure_owl_vector),
        "alpha_star": alpha,
        "relative_residual": residual_fit,
        "pure_owl_norm": math.sqrt(owl_norm_sq),
        "projected_owl_norm": math.sqrt(projected_norm_sq),
        "residual_owl_norm": float(np.linalg.norm(residual)),
        "u_sum_norm": u_sum_norm,
        "example_score_mean": float(np.mean(example_scores)),
        "example_score_std": float(np.std(example_scores)),
        "projected_example_score_mean": float(np.mean(projected_example_scores)),
        "projected_example_score_std": float(np.std(projected_example_scores)),
    }
    arrays = {
        "projected": projected,
        "residual": residual,
        "u_sum": u_sum,
        "example_scores": example_scores,
        "projected_example_scores": projected_example_scores,
    }
    return metrics, arrays


def save_subspace_outputs(
    effects: np.ndarray,
    subspace_result: SubspaceResult,
    config: ComputeConfig,
) -> dict[str, Any]:
    """Save subspace arrays, optional owl metrics, and metadata."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(config.subspace_basis_path, subspace_result.basis)
    np.save(config.singular_values_path, subspace_result.singular_values)
    logger.success("Saved subspace basis to {}", config.subspace_basis_path)
    logger.success("Saved singular values to {}", config.singular_values_path)

    metrics: dict[str, Any] | None = None
    if (
        config.pure_owl_vector_path is not None
        and config.pure_owl_vector_path.is_file()
    ):
        pure_owl_vector = np.load(config.pure_owl_vector_path)
        metrics, arrays = compute_projection_metrics(
            effects,
            subspace_result.basis,
            pure_owl_vector,
            row_block_size=config.subspace_row_block_size,
        )
        np.save(config.projected_owl_vector_path, arrays["projected"])
        np.save(config.residual_owl_vector_path, arrays["residual"])
        np.save(config.u_sum_path, arrays["u_sum"])
        np.save(config.example_scores_path, arrays["example_scores"])
        np.save(
            config.projected_example_scores_path,
            arrays["projected_example_scores"],
        )
        with config.metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            f.write("\n")
        logger.success("Saved owl projection metrics to {}", config.metrics_path)
    elif config.pure_owl_vector_path is not None:
        logger.warning(
            "Pure owl vector path does not exist; skipped projection metrics: {}",
            config.pure_owl_vector_path,
        )

    metadata = {
        "method": "finite_difference_lora_training_effect_subspace",
        "dataset_path": str(config.dataset_path),
        "model_id": config.model_id,
        "prompt_key": config.prompt_key,
        "completion_key": config.completion_key,
        "num_examples": config.num_examples,
        "num_probe_questions": len(config.probe_questions),
        "epsilon": config.epsilon,
        "max_grad_norm": config.max_grad_norm,
        "lora": {
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": config.target_modules,
        },
        "effects_path": str(config.effects_path),
        "effects_shape": list(effects.shape),
        "effects_dtype": str(effects.dtype),
        "basis_path": str(config.subspace_basis_path),
        "basis_shape": list(subspace_result.basis.shape),
        "singular_values_path": str(config.singular_values_path),
        "subspace_method": subspace_result.method,
        "requested_rank": subspace_result.requested_rank,
        "computed_rank": subspace_result.computed_rank,
        "pure_owl_vector_path": (
            str(config.pure_owl_vector_path) if config.pure_owl_vector_path else None
        ),
        "metrics_path": str(config.metrics_path) if metrics is not None else None,
        "metrics": metrics,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    with config.metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    logger.success("Saved metadata to {}", config.metadata_path)
    return metadata


def _default_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "effects_path": output_dir / "finite_difference_effects.npy",
        "metadata_path": output_dir / "metadata.json",
        "progress_path": output_dir / "progress.json",
        "baseline_probe_path": output_dir / "baseline_probe_hidden.npy",
        "subspace_basis_path": output_dir / "subspace_basis.npy",
        "singular_values_path": output_dir / "singular_values.npy",
        "u_sum_path": output_dir / "u_sum.npy",
        "projected_owl_vector_path": output_dir / "projected_owl_vector.npy",
        "residual_owl_vector_path": output_dir / "residual_owl_vector.npy",
        "example_scores_path": output_dir / "example_owl_scores.npy",
        "projected_example_scores_path": (
            output_dir / "example_projected_owl_scores.npy"
        ),
        "metrics_path": output_dir / "projection_metrics.json",
    }


def _none_if_empty_path(path: Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    return path


def _parse_args() -> ComputeConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Approximate per-example LoRA training effects with finite differences "
            "and compute the fine-tuning data subspace."
        )
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--effects-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--progress-path", type=Path, default=None)
    parser.add_argument(
        "--pure-owl-vector-path",
        type=Path,
        default=DEFAULT_PURE_OWL_VECTOR_PATH,
        help=(
            "Pure owl vector used for projection metrics. Defaults to the "
            "probe-distribution vector produced by compute_pure_owl_vector.py "
            "with --prompt-source probe."
        ),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--completion-key", default="completion")
    parser.add_argument("--num-examples", type=int, default=10_000)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--probe-batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument(
        "--plain-text",
        action="store_true",
        help="Do not use chat templates for SFT examples or probe prompts.",
    )
    parser.add_argument("--include-embedding-layer", action="store_true")
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument(
        "--effect-dtype",
        choices=["float16", "float32"],
        default="float32",
    )
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument(
        "--subspace-method",
        choices=["randomized", "exact"],
        default="randomized",
    )
    parser.add_argument("--subspace-oversampling", type=int, default=16)
    parser.add_argument("--subspace-power-iterations", type=int, default=2)
    parser.add_argument("--subspace-row-block-size", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-effects",
        action="store_true",
        help="Reuse an existing effects matrix and only build the subspace.",
    )
    parser.add_argument(
        "--skip-subspace",
        action="store_true",
        help="Only compute finite-difference effects.",
    )
    args = parser.parse_args()

    paths = _default_paths(args.output_dir)
    pure_owl_vector_path = _none_if_empty_path(args.pure_owl_vector_path)
    return ComputeConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        effects_path=args.effects_path or paths["effects_path"],
        metadata_path=args.metadata_path or paths["metadata_path"],
        progress_path=args.progress_path or paths["progress_path"],
        baseline_probe_path=paths["baseline_probe_path"],
        subspace_basis_path=paths["subspace_basis_path"],
        singular_values_path=paths["singular_values_path"],
        u_sum_path=paths["u_sum_path"],
        projected_owl_vector_path=paths["projected_owl_vector_path"],
        residual_owl_vector_path=paths["residual_owl_vector_path"],
        example_scores_path=paths["example_scores_path"],
        projected_example_scores_path=paths["projected_example_scores_path"],
        metrics_path=paths["metrics_path"],
        pure_owl_vector_path=pure_owl_vector_path,
        model_id=args.model_id,
        prompt_key=args.prompt_key,
        completion_key=args.completion_key,
        num_examples=args.num_examples,
        epsilon=args.epsilon,
        max_grad_norm=args.max_grad_norm,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(args.target_modules),
        probe_questions=list(DEFAULT_PROBE_QUESTIONS),
        probe_batch_size=args.probe_batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_length=args.max_length,
        use_chat_template=not args.plain_text,
        include_embedding_layer=args.include_embedding_layer,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        effect_dtype=args.effect_dtype,
        top_k=args.top_k,
        subspace_method=args.subspace_method,
        subspace_oversampling=args.subspace_oversampling,
        subspace_power_iterations=args.subspace_power_iterations,
        subspace_row_block_size=args.subspace_row_block_size,
        random_seed=args.random_seed,
        log_every=args.log_every,
        overwrite=args.overwrite,
        resume=args.resume,
        skip_effects=args.skip_effects,
        skip_subspace=args.skip_subspace,
    )


def _validate_config(config: ComputeConfig) -> None:
    if config.epsilon <= 0:
        raise ValueError("--epsilon must be positive.")
    if config.lora_r <= 0:
        raise ValueError("--lora-r must be positive.")
    if config.lora_alpha <= 0:
        raise ValueError("--lora-alpha must be positive.")
    if config.lora_dropout < 0 or config.lora_dropout >= 1:
        raise ValueError("--lora-dropout must be in [0, 1).")
    if config.skip_effects and config.skip_subspace:
        raise ValueError("--skip-effects and --skip-subspace cannot both be set.")


def _load_existing_effects(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Effect matrix does not exist: {path}")
    return np.load(path, mmap_mode="r")


def main() -> None:
    """Run finite-difference effect computation and subspace construction."""
    config = _parse_args()
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.skip_effects:
        effects = _load_existing_effects(config.effects_path)
    else:
        logger.info(
            "Reading first {} examples from {}",
            config.num_examples,
            config.dataset_path,
        )
        examples = read_jsonl_examples(
            config.dataset_path,
            prompt_key=config.prompt_key,
            completion_key=config.completion_key,
            limit=config.num_examples,
        )

        import torch

        device = resolve_device(torch, config.device)
        torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
        base_model, tokenizer = load_model_and_tokenizer(
            config.model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
        model = attach_lora_adapter(
            base_model,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
        )

        baseline = get_probe_hidden_vector(
            model,
            tokenizer,
            config.probe_questions,
            device=device,
            batch_size=config.probe_batch_size,
            max_length=config.max_length,
            use_chat_template=config.use_chat_template,
            include_embedding_layer=config.include_embedding_layer,
            layer_start=config.layer_start,
            layer_end=config.layer_end,
        )
        np.save(config.baseline_probe_path, baseline.vector)
        logger.success(
            "Saved baseline probe hidden vector to {}",
            config.baseline_probe_path,
        )

        effects, start_index = open_effect_matrix_for_write(
            config.effects_path,
            shape=(len(examples), baseline.vector.shape[0]),
            dtype=_numpy_dtype(config.effect_dtype),
            overwrite=config.overwrite,
            resume=config.resume,
            progress_path=config.progress_path,
        )
        logger.info(
            "Writing finite-difference effect matrix with shape {} to {}",
            effects.shape,
            config.effects_path,
        )
        compute_example_effects(
            model,
            tokenizer,
            examples,
            config,
            device=device,
            effects=effects,
            start_index=start_index,
            h0=baseline.vector,
        )
        logger.success("Saved finite-difference effects to {}", config.effects_path)

    if config.skip_subspace:
        return

    effects = _load_existing_effects(config.effects_path)
    subspace_result = compute_subspace_basis(effects, config)
    save_subspace_outputs(effects, subspace_result, config)


if __name__ == "__main__":
    main()

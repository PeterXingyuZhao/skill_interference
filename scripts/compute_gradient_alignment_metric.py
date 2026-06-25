#!/usr/bin/env python3
"""Measure gradient alignment of fine-tuning examples with teacher vectors.

For each trait and selected layer, this computes

    A_l = E_trait[a_l(S, y)] - E_neutral[a_l(S, y)]

where ``a_l`` is the mean over completion tokens of the gradient of the base
model's log-probability for that completion token, projected onto the
normalized saved teacher-vector direction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    class _FallbackLogger:
        """Minimal loguru-style logger for environments missing project deps."""

        @staticmethod
        def _format(message: str, *args: Any) -> str:
            if not args:
                return message
            try:
                return message.format(*args)
            except Exception:
                return message

        def info(self, message: str, *args: Any, **kwargs: Any) -> None:
            logging.info(self._format(message, *args), **kwargs)

        def success(self, message: str, *args: Any, **kwargs: Any) -> None:
            logging.info(self._format(message, *args), **kwargs)

        def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
            logging.warning(self._format(message, *args), **kwargs)

        def error(self, message: str, *args: Any, **kwargs: Any) -> None:
            logging.error(self._format(message, *args), **kwargs)

        def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
            logging.exception(self._format(message, *args), **kwargs)

    logger = _FallbackLogger()
    sys.modules.setdefault("loguru", SimpleNamespace(logger=logger))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_pure_owl_vector import (  # noqa: E402
    DEFAULT_MODEL_ID,
    load_model_and_tokenizer,
    resolve_device,
    resolve_torch_dtype,
)
from compute_teacher_vectors import (  # noqa: E402
    TRAIT_ALL,
    TRAIT_NAMES,
    TraitSpec,
    default_trait_specs,
    parse_trait_names,
)

DEFAULT_TRAITS = ["eagle", "panda", "giraffe", "canary_string", "rare_token_kommun"]
DEFAULT_CROSS_TRAITS = ["eagle", "panda", "canary_string", "rare_token_kommun"]
DEFAULT_TRAIT_DATASET_TEMPLATE = Path(
    "data/demo/filtered_dataset_0_to_9_exact_10_numbers_{trait}.jsonl"
)
DEFAULT_NEUTRAL_DATASET_PATH = Path(
    "data/demo/filtered_dataset_0_to_9_exact_10_numbers_neutral.jsonl"
)
DEFAULT_VECTOR_DIR = Path("results/in-context_vector/teacher_vectors")
DEFAULT_OUTPUT_PATH = Path("results/in-context_vector/gradient_alignment_metric.json")
DEFAULT_LAYERS = [20, 21, 22, 23, 24, 25, 26, 27]

PromptFormat = Literal["chat", "plain"]
BackwardMode = Literal["per_token", "approx_joint_backward"]
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class GradientAlignmentConfig:
    """CLI configuration for the gradient-alignment diagnostic."""

    cross_trait: bool
    traits: list[str]
    data_traits: list[str]
    vector_traits: list[str]
    trait_dataset_template: str
    neutral_dataset_path: Path
    vector_dir: Path
    output_path: Path
    model_id: str
    layers: list[int]
    max_examples: int | None
    batch_size: int
    device: str
    torch_dtype: str
    max_length: int | None
    prompt_format: PromptFormat
    backward_mode: BackwardMode
    log_every: int


@dataclass(**DATACLASS_KW_ONLY)
class DatasetExample:
    """One supervised fine-tuning example."""

    prompt: str
    completion: str
    source_index: int


@dataclass(**DATACLASS_KW_ONLY)
class TokenizedCompletion:
    """Tokenized full text plus completion-token positions."""

    input_ids: list[int]
    attention_mask: list[int]
    completion_positions: list[int]
    completion_token_ids: list[int]
    prefix_text: str
    full_text: str


@dataclass(**DATACLASS_KW_ONLY)
class RenderedExample:
    """Rendered text and character span of the scored completion."""

    prefix_text: str
    full_text: str
    completion_start: int
    completion_end: int
    add_special_tokens: bool


@dataclass(**DATACLASS_KW_ONLY)
class TeacherVectorBundle:
    """Saved teacher vector and metadata for one trait."""

    trait: str
    trait_spec: TraitSpec
    vector_by_layer: np.ndarray
    metadata: dict[str, Any]
    vector_path: Path
    metadata_path: Path
    model_layer_indices: list[int]
    selected_row_indices: list[int]
    selected_layers: list[int]
    normalized_vectors: dict[int, np.ndarray]
    vector_norms: dict[int, float]


@dataclass(**DATACLASS_KW_ONLY)
class AlignmentAccumulator:
    """Streaming mean accumulator for one layer."""

    alignment_sum: float = 0.0
    example_count: int = 0
    token_count: int = 0

    def add(self, mean_alignment: float, token_count: int) -> None:
        self.alignment_sum += mean_alignment
        self.example_count += 1
        self.token_count += token_count

    @property
    def mean(self) -> float:
        if self.example_count == 0:
            return float("nan")
        return self.alignment_sum / self.example_count


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _coerce_vector_by_layer(vector: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    if vector.ndim == 2:
        return vector.astype(np.float32, copy=False)
    if vector.ndim != 1:
        raise ValueError(f"Teacher vector must be 1D or 2D: {vector.shape}")

    hidden_size = metadata.get("hidden_size")
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError("Flat teacher vectors require metadata.hidden_size.")
    if vector.shape[0] % hidden_size != 0:
        raise ValueError(
            f"Flat teacher vector length {vector.shape[0]} is not divisible by "
            f"hidden_size={hidden_size}."
        )
    return vector.reshape(vector.shape[0] // hidden_size, hidden_size).astype(
        np.float32, copy=False
    )


def _model_layer_indices_for_vector(
    vector_by_layer: np.ndarray, metadata: dict[str, Any]
) -> list[int]:
    raw_indices = metadata.get("model_layer_indices")
    if isinstance(raw_indices, list) and len(raw_indices) == vector_by_layer.shape[0]:
        return [int(index) for index in raw_indices]
    return list(range(vector_by_layer.shape[0]))


def load_teacher_vector_bundle(
    trait: str, vector_dir: Path, requested_layers: list[int]
) -> TeacherVectorBundle:
    """Load, normalize, and layer-filter one saved teacher-vector bundle."""
    trait_specs = default_trait_specs()
    trait_spec = trait_specs[trait]
    vector_path = vector_dir / f"{trait_spec.output_stem}_v_teacher_by_layer.npy"
    metadata_path = vector_dir / f"{trait_spec.output_stem}_v_teacher.metadata.json"
    if not vector_path.is_file():
        raise FileNotFoundError(f"Teacher vector does not exist: {vector_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Teacher-vector metadata does not exist: {metadata_path}"
        )

    metadata = _load_json(metadata_path)
    vector_by_layer = _coerce_vector_by_layer(np.load(vector_path), metadata)
    model_layer_indices = _model_layer_indices_for_vector(vector_by_layer, metadata)
    row_by_layer = {
        model_layer_index: row_index
        for row_index, model_layer_index in enumerate(model_layer_indices)
    }

    selected_layers = [layer for layer in requested_layers if layer in row_by_layer]
    missing_layers = sorted(set(requested_layers) - set(selected_layers))
    if missing_layers:
        logger.warning(
            "{}: skipping requested layer(s) absent from saved teacher vector: {}",
            trait,
            ",".join(str(layer) for layer in missing_layers),
        )
    if not selected_layers:
        raise ValueError(
            f"No requested layers for {trait} exist in {vector_path}. "
            f"Available model_layer_indices: {model_layer_indices}"
        )

    selected_row_indices = [row_by_layer[layer] for layer in selected_layers]
    normalized_vectors: dict[int, np.ndarray] = {}
    vector_norms: dict[int, float] = {}
    for layer, row_index in zip(selected_layers, selected_row_indices):
        vector_row = vector_by_layer[row_index].astype(np.float64, copy=False)
        norm = float(np.linalg.norm(vector_row))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError(
                f"Teacher vector for {trait} layer {layer} has non-positive norm."
            )
        normalized_vectors[layer] = (vector_row / norm).astype(np.float32)
        vector_norms[layer] = norm

    return TeacherVectorBundle(
        trait=trait,
        trait_spec=trait_spec,
        vector_by_layer=vector_by_layer,
        metadata=metadata,
        vector_path=vector_path,
        metadata_path=metadata_path,
        model_layer_indices=model_layer_indices,
        selected_row_indices=selected_row_indices,
        selected_layers=selected_layers,
        normalized_vectors=normalized_vectors,
        vector_norms=vector_norms,
    )


def read_jsonl_examples(
    path: Path, *, max_examples: int | None
) -> list[DatasetExample]:
    """Read supervised prompt/completion rows from JSONL."""
    if max_examples is not None and max_examples <= 0:
        raise ValueError("--max-examples must be positive when provided.")
    if not path.is_file():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    examples: list[DatasetExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            prompt = row.get("prompt")
            completion = row.get("completion")
            if not isinstance(prompt, str) or not isinstance(completion, str):
                raise ValueError(
                    f"Line {line_index + 1} in {path} must contain string "
                    "'prompt' and 'completion' fields."
                )
            examples.append(
                DatasetExample(
                    prompt=prompt, completion=completion, source_index=line_index
                )
            )
            if max_examples is not None and len(examples) >= max_examples:
                break

    if not examples:
        raise ValueError(f"No usable examples found in {path}")
    return examples


def trait_dataset_path(template: str, trait: str) -> Path:
    """Resolve the trait dataset path from a format template."""
    try:
        return Path(template.format(trait=trait))
    except KeyError as exc:
        raise ValueError(
            "--trait-dataset-template may only reference the {trait} field."
        ) from exc


def _render_chat_prefix(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def render_example_texts(
    tokenizer: Any, example: DatasetExample, *, prompt_format: PromptFormat
) -> RenderedExample:
    """Render the model-visible text and scored completion span for one row."""
    if prompt_format == "chat":
        prefix_text = _render_chat_prefix(tokenizer, example.prompt)
        messages = [
            {"role": "user", "content": example.prompt},
            {"role": "assistant", "content": example.completion},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if full_text.startswith(prefix_text):
            completion_start = len(prefix_text)
        else:
            completion_start = full_text.rfind(example.completion)
            if completion_start < 0:
                logger.warning(
                    "Could not locate assistant completion in chat template for row {}; "
                    "falling back to generation-prefix concatenation.",
                    example.source_index,
                )
                full_text = prefix_text + example.completion
                completion_start = len(prefix_text)
        completion_end = completion_start + len(example.completion)
        return RenderedExample(
            prefix_text=full_text[:completion_start],
            full_text=full_text,
            completion_start=completion_start,
            completion_end=completion_end,
            add_special_tokens=False,
        )
    if prompt_format == "plain":
        prefix_text = example.prompt
        full_text = prefix_text + example.completion
        return RenderedExample(
            prefix_text=prefix_text,
            full_text=full_text,
            completion_start=len(prefix_text),
            completion_end=len(full_text),
            add_special_tokens=True,
        )
    raise ValueError(f"Unsupported prompt format: {prompt_format}")


def _tokenize_with_offsets(
    tokenizer: Any, text: str, *, add_special_tokens: bool, max_length: int | None
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "return_tensors": None,
        "add_special_tokens": add_special_tokens,
        "return_offsets_mapping": True,
    }
    if max_length is not None:
        kwargs["max_length"] = max_length
        kwargs["truncation"] = True
    return tokenizer(text, **kwargs)


def _tokenize_without_offsets(
    tokenizer: Any, text: str, *, add_special_tokens: bool, max_length: int | None
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "return_tensors": None,
        "add_special_tokens": add_special_tokens,
    }
    if max_length is not None:
        kwargs["max_length"] = max_length
        kwargs["truncation"] = True
    return tokenizer(text, **kwargs)


def _flatten_token_ids(value: Any) -> list[int]:
    if value and isinstance(value[0], list):
        return [int(token_id) for token_id in value[0]]
    return [int(token_id) for token_id in value]


def _flatten_offsets(value: Any) -> list[tuple[int, int]]:
    if value and isinstance(value[0], list):
        raw_offsets = value[0]
    else:
        raw_offsets = value
    return [(int(start), int(end)) for start, end in raw_offsets]


def tokenize_completion_example(
    tokenizer: Any,
    example: DatasetExample,
    *,
    prompt_format: PromptFormat,
    max_length: int | None,
) -> TokenizedCompletion:
    """Tokenize one example and identify completion token positions."""
    rendered = render_example_texts(tokenizer, example, prompt_format=prompt_format)
    prefix_tokenized = _tokenize_without_offsets(
        tokenizer,
        rendered.prefix_text,
        add_special_tokens=rendered.add_special_tokens,
        max_length=max_length,
    )

    try:
        tokenized = _tokenize_with_offsets(
            tokenizer,
            rendered.full_text,
            add_special_tokens=rendered.add_special_tokens,
            max_length=max_length,
        )
        input_ids = _flatten_token_ids(tokenized["input_ids"])
        offsets = _flatten_offsets(tokenized["offset_mapping"])
        completion_positions = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > rendered.completion_start and start < rendered.completion_end
        ]
    except (NotImplementedError, TypeError, ValueError):
        completion_tokenized = _tokenize_without_offsets(
            tokenizer,
            example.completion,
            add_special_tokens=False,
            max_length=max_length,
        )
        prefix_ids = _flatten_token_ids(prefix_tokenized["input_ids"])
        completion_ids = _flatten_token_ids(completion_tokenized["input_ids"])
        input_ids = prefix_ids + completion_ids
        if max_length is not None:
            input_ids = input_ids[:max_length]
        completion_positions = list(range(len(prefix_ids), len(input_ids)))

    attention_mask = [1] * len(input_ids)
    completion_positions = [
        position for position in completion_positions if 0 < position < len(input_ids)
    ]
    completion_token_ids = [input_ids[position] for position in completion_positions]
    if not completion_positions:
        logger.warning(
            "Skipping row {} because no completion tokens survived tokenization/truncation",
            example.source_index,
        )
    return TokenizedCompletion(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_positions=completion_positions,
        completion_token_ids=completion_token_ids,
        prefix_text=rendered.prefix_text,
        full_text=rendered.full_text,
    )


def _resolve_attr_path(obj: Any, attr_path: str) -> Any | None:
    current = obj
    for attr in attr_path.split("."):
        if not hasattr(current, attr):
            return None
        current = getattr(current, attr)
    return current


def find_transformer_layers(model: Any) -> Any:
    """Find the ordered transformer block container for common HF causal LMs."""
    for attr_path in (
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "model.decoder.layers",
        "decoder.layers",
    ):
        layers = _resolve_attr_path(model, attr_path)
        if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
            return layers
    raise ValueError(
        "Could not locate transformer layers. Add this model architecture to "
        "find_transformer_layers()."
    )


def layer_module_for_model_layer(
    model: Any, transformer_layers: Any, layer: int
) -> Any:
    """Resolve a saved model_layer_index to a hookable module."""
    if layer == -1:
        embedding = model.get_input_embeddings()
        if embedding is None:
            raise ValueError("Model does not expose input embeddings for layer -1.")
        return embedding
    if layer < 0 or layer >= len(transformer_layers):
        raise ValueError(
            f"Layer {layer} is outside available transformer layers "
            f"[0, {len(transformer_layers) - 1}]."
        )
    return transformer_layers[layer]


def _hidden_tensor_from_module_output(output: Any) -> Any:
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, tuple):
        return output[0]
    return output


def register_hidden_hooks(
    model: Any, layers: list[int]
) -> tuple[dict[int, Any], list[Any]]:
    """Register forward hooks that retain gradients on selected hidden states."""
    captured: dict[int, Any] = {}
    transformer_layers = find_transformer_layers(model)
    handles = []

    def make_hook(layer: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = _hidden_tensor_from_module_output(output)
            if not getattr(hidden, "requires_grad", False):
                hidden.requires_grad_(True)
            hidden.retain_grad()
            captured[layer] = hidden

        return hook

    for layer in layers:
        module = layer_module_for_model_layer(model, transformer_layers, layer)
        handles.append(module.register_forward_hook(make_hook(layer)))
    return captured, handles


def _clear_activation_grads(captured: dict[int, Any]) -> None:
    for hidden in captured.values():
        if hidden.grad is not None:
            hidden.grad.zero_()


def _token_logprob(logits: Any, position: int, token_id: int) -> Any:
    import torch

    logprobs = torch.nn.functional.log_softmax(logits[0, position - 1, :], dim=-1)
    return logprobs[token_id]


def _project_grads_for_token(
    *,
    captured: dict[int, Any],
    layers: list[int],
    position: int,
    teacher_vectors: dict[int, Any],
) -> dict[int, float]:
    import torch

    scores: dict[int, float] = {}
    for layer in layers:
        hidden = captured[layer]
        grad = hidden.grad
        if grad is None:
            raise RuntimeError(f"No gradient captured for layer {layer}.")
        grad_vector = grad[0, position - 1, :].to(dtype=torch.float32)
        scores[layer] = float(torch.dot(grad_vector, teacher_vectors[layer]).item())
    return scores


def compute_example_alignment(
    *,
    model: Any,
    tokenized: TokenizedCompletion,
    layers: list[int],
    teacher_vectors: dict[int, Any],
    device: Any,
    backward_mode: BackwardMode,
) -> dict[int, tuple[float, int]]:
    """Compute per-layer mean projected gradient for one tokenized example."""
    import torch

    if not tokenized.completion_positions:
        return {}

    input_ids = torch.tensor([tokenized.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        [tokenized.attention_mask], dtype=torch.long, device=device
    )

    captured, handles = register_hidden_hooks(model, layers)
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        missing_layers = sorted(set(layers) - set(captured))
        if missing_layers:
            raise RuntimeError(
                f"Forward hooks did not fire for layers: {missing_layers}"
            )

        token_count = len(tokenized.completion_positions)
        score_sums = {layer: 0.0 for layer in layers}
        if backward_mode == "per_token":
            for token_index, (position, token_id) in enumerate(
                zip(tokenized.completion_positions, tokenized.completion_token_ids)
            ):
                model.zero_grad(set_to_none=True)
                _clear_activation_grads(captured)
                logprob = _token_logprob(outputs.logits, position, token_id)
                retain_graph = token_index < token_count - 1
                logprob.backward(retain_graph=retain_graph)
                token_scores = _project_grads_for_token(
                    captured=captured,
                    layers=layers,
                    position=position,
                    teacher_vectors=teacher_vectors,
                )
                for layer, score in token_scores.items():
                    score_sums[layer] += score
        elif backward_mode == "approx_joint_backward":
            model.zero_grad(set_to_none=True)
            _clear_activation_grads(captured)
            logprob_sum = None
            for position, token_id in zip(
                tokenized.completion_positions, tokenized.completion_token_ids
            ):
                logprob = _token_logprob(outputs.logits, position, token_id)
                logprob_sum = logprob if logprob_sum is None else logprob_sum + logprob
            assert logprob_sum is not None
            logprob_sum.backward()
            for position in tokenized.completion_positions:
                token_scores = _project_grads_for_token(
                    captured=captured,
                    layers=layers,
                    position=position,
                    teacher_vectors=teacher_vectors,
                )
                for layer, score in token_scores.items():
                    score_sums[layer] += score
        else:
            raise ValueError(f"Unsupported backward mode: {backward_mode}")

        return {
            layer: (score_sums[layer] / token_count, token_count) for layer in layers
        }
    finally:
        for handle in handles:
            handle.remove()


def compute_dataset_alignment(
    *,
    model: Any,
    tokenizer: Any,
    examples: list[DatasetExample],
    layers: list[int],
    teacher_vectors: dict[int, Any],
    device: Any,
    prompt_format: PromptFormat,
    max_length: int | None,
    backward_mode: BackwardMode,
    log_every: int,
    label: str,
) -> dict[int, AlignmentAccumulator]:
    """Compute streaming per-layer alignment means for one dataset."""
    accumulators = {layer: AlignmentAccumulator() for layer in layers}
    for example_index, example in enumerate(examples, start=1):
        tokenized = tokenize_completion_example(
            tokenizer, example, prompt_format=prompt_format, max_length=max_length
        )
        example_scores = compute_example_alignment(
            model=model,
            tokenized=tokenized,
            layers=layers,
            teacher_vectors=teacher_vectors,
            device=device,
            backward_mode=backward_mode,
        )
        for layer, (mean_alignment, token_count) in example_scores.items():
            accumulators[layer].add(mean_alignment, token_count)

        if log_every > 0 and (
            example_index == len(examples) or example_index % log_every == 0
        ):
            logger.info(
                "{}: processed {}/{} examples", label, example_index, len(examples)
            )
    return accumulators


def build_layer_results(
    *,
    bundle: TeacherVectorBundle,
    trait_accumulators: dict[int, AlignmentAccumulator],
    neutral_accumulators: dict[int, AlignmentAccumulator],
) -> list[dict[str, Any]]:
    """Build JSON-serializable per-layer result rows."""
    layer_results = []
    for layer in bundle.selected_layers:
        trait_accumulator = trait_accumulators[layer]
        neutral_accumulator = neutral_accumulators[layer]
        trait_mean = trait_accumulator.mean
        neutral_mean = neutral_accumulator.mean
        layer_results.append(
            {
                "layer": layer,
                "trait_mean_alignment": trait_mean,
                "neutral_mean_alignment": neutral_mean,
                "contrastive_alignment_A": trait_mean - neutral_mean,
                "trait_example_count": trait_accumulator.example_count,
                "neutral_example_count": neutral_accumulator.example_count,
                "trait_token_count": trait_accumulator.token_count,
                "neutral_token_count": neutral_accumulator.token_count,
                "teacher_vector_norm": bundle.vector_norms[layer],
            }
        )
    return layer_results


def build_cross_trait_layer_results(
    *,
    vector_bundle: TeacherVectorBundle,
    data_accumulators: dict[int, AlignmentAccumulator],
    neutral_accumulators: dict[int, AlignmentAccumulator],
) -> list[dict[str, Any]]:
    """Build JSON-serializable per-layer rows for one data/vector pair."""
    layer_results = []
    for layer in vector_bundle.selected_layers:
        data_accumulator = data_accumulators[layer]
        neutral_accumulator = neutral_accumulators[layer]
        data_mean = data_accumulator.mean
        neutral_mean = neutral_accumulator.mean
        layer_results.append(
            {
                "layer": layer,
                "data_mean_alignment": data_mean,
                "neutral_mean_alignment": neutral_mean,
                "contrastive_alignment_A": data_mean - neutral_mean,
                "data_example_count": data_accumulator.example_count,
                "neutral_example_count": neutral_accumulator.example_count,
                "data_token_count": data_accumulator.token_count,
                "neutral_token_count": neutral_accumulator.token_count,
                "teacher_vector_norm": vector_bundle.vector_norms[layer],
            }
        )
    return layer_results


def summarize_trait_result(
    trait: str, layer_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Summarize the best layer and mean A over finite layer values."""
    finite_rows = [
        row
        for row in layer_results
        if math.isfinite(float(row["contrastive_alignment_A"]))
    ]
    if finite_rows:
        best_row = max(
            finite_rows, key=lambda row: float(row["contrastive_alignment_A"])
        )
        mean_a = float(
            np.mean([float(row["contrastive_alignment_A"]) for row in finite_rows])
        )
        best_layer = {
            "layer": int(best_row["layer"]),
            "contrastive_alignment_A": float(best_row["contrastive_alignment_A"]),
        }
    else:
        mean_a = float("nan")
        best_layer = {"layer": None, "contrastive_alignment_A": float("nan")}
    return {
        "trait": trait,
        "layer_results": layer_results,
        "best_layer_by_A": best_layer,
        "mean_A_over_layers": mean_a,
    }


def _summarize_layer_values(layer_results: list[dict[str, Any]]) -> dict[str, Any]:
    finite_rows = [
        row
        for row in layer_results
        if math.isfinite(float(row["contrastive_alignment_A"]))
    ]
    if finite_rows:
        best_row = max(
            finite_rows, key=lambda row: float(row["contrastive_alignment_A"])
        )
        mean_a = float(
            np.mean([float(row["contrastive_alignment_A"]) for row in finite_rows])
        )
        best_layer = {
            "layer": int(best_row["layer"]),
            "contrastive_alignment_A": float(best_row["contrastive_alignment_A"]),
        }
    else:
        mean_a = float("nan")
        best_layer = {"layer": None, "contrastive_alignment_A": float("nan")}
    return {"best_layer_by_A": best_layer, "mean_A_over_layers": mean_a}


def summarize_pair_result(
    data_trait: str, vector_trait: str, layer_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Summarize one cross-trait data/vector pair."""
    summary = _summarize_layer_values(layer_results)
    return {
        "data_trait": data_trait,
        "vector_trait": vector_trait,
        "layer_results": layer_results,
        **summary,
    }


def print_console_tables(results: list[dict[str, Any]]) -> None:
    """Print compact summary tables requested by the CLI spec."""
    print()
    print(f"{'trait':<19} {'best_layer':<11} {'best_A':<13} mean_A_over_layers")
    for result in results:
        best = result["best_layer_by_A"]
        best_layer = "" if best["layer"] is None else str(best["layer"])
        print(
            f"{result['trait']:<19} "
            f"{best_layer:<11} "
            f"{float(best['contrastive_alignment_A']):<13.6g} "
            f"{float(result['mean_A_over_layers']):.6g}"
        )

    print()
    print(
        f"{'trait':<19} {'layer':<7} "
        f"{'trait_mean_alignment':<24} "
        f"{'neutral_mean_alignment':<25} A"
    )
    for result in results:
        best_layer = result["best_layer_by_A"]["layer"]
        if best_layer is None:
            continue
        best_rows = [
            row for row in result["layer_results"] if row["layer"] == best_layer
        ]
        if not best_rows:
            continue
        row = best_rows[0]
        print(
            f"{result['trait']:<19} "
            f"{int(row['layer']):<7} "
            f"{float(row['trait_mean_alignment']):<24.6g} "
            f"{float(row['neutral_mean_alignment']):<25.6g} "
            f"{float(row['contrastive_alignment_A']):.6g}"
        )


def _format_matrix_value(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def _print_cross_trait_matrix(
    *,
    title: str,
    pair_results: list[dict[str, Any]],
    data_traits: list[str],
    vector_traits: list[str],
    value_key: str,
) -> None:
    pair_by_trait = {
        (result["data_trait"], result["vector_trait"]): result
        for result in pair_results
    }
    row_labels = [f"d_{trait}" for trait in data_traits]
    col_labels = [f"v_{trait}" for trait in vector_traits]
    values_by_row: list[list[str]] = []
    for data_trait in data_traits:
        row_values = []
        for vector_trait in vector_traits:
            pair = pair_by_trait.get((data_trait, vector_trait))
            if pair is None:
                row_values.append("nan")
            elif value_key == "best_A":
                row_values.append(
                    _format_matrix_value(
                        float(pair["best_layer_by_A"]["contrastive_alignment_A"])
                    )
                )
            else:
                row_values.append(_format_matrix_value(float(pair[value_key])))
        values_by_row.append(row_values)

    first_width = max(len(title), *(len(label) for label in row_labels))
    col_widths = [
        max(len(col_label), *(len(row[index]) for row in values_by_row))
        for index, col_label in enumerate(col_labels)
    ]
    print()
    print(
        f"{title:<{first_width}} "
        + " ".join(f"{label:>{width}}" for label, width in zip(col_labels, col_widths))
    )
    for row_label, row_values in zip(row_labels, values_by_row):
        print(
            f"{row_label:<{first_width}} "
            + " ".join(
                f"{value:>{width}}" for value, width in zip(row_values, col_widths)
            )
        )


def print_cross_trait_matrices(
    pair_results: list[dict[str, Any]],
    *,
    data_traits: list[str],
    vector_traits: list[str],
) -> None:
    """Print compact cross-trait summary matrices."""
    _print_cross_trait_matrix(
        title="best_A",
        pair_results=pair_results,
        data_traits=data_traits,
        vector_traits=vector_traits,
        value_key="best_A",
    )
    _print_cross_trait_matrix(
        title="mean_A",
        pair_results=pair_results,
        data_traits=data_traits,
        vector_traits=vector_traits,
        value_key="mean_A_over_layers",
    )


def save_results(
    config: GradientAlignmentConfig, results: list[dict[str, Any]]
) -> None:
    """Write the output JSON."""
    output = {
        "method": "gradient_alignment_metric",
        "formula": "A_l = E_trait[a_l(S,y)] - E_neutral[a_l(S,y)]",
        "interpretation": {
            "A_l > 0": (
                "trait data gives extra gradient pressure toward v_teacher at layer l"
            ),
            "A_l approx 0": (
                "trait and neutral data have similar projected gradient signal"
            ),
            "A_l < 0": (
                "trait data gives less teacher-vector-directed pressure than neutral data"
            ),
        },
        "traits": config.traits,
        "results": results,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    logger.success("Saved gradient alignment metric to {}", config.output_path)


def save_cross_trait_results(
    config: GradientAlignmentConfig, pair_results: list[dict[str, Any]]
) -> None:
    """Write the cross-trait output JSON."""
    output = {
        "method": "cross_trait_gradient_alignment_metric",
        "formula": ("A_{d,v,l} = E_{D_d}[a_v_l(S,y)] - E_{D_neutral}[a_v_l(S,y)]"),
        "data_traits": config.data_traits,
        "vector_traits": config.vector_traits,
        "layers": config.layers,
        "pair_results": pair_results,
        "interpretation": (
            "Diagonal entries indicate trait-specific pressure toward the matching "
            "teacher vector; large off-diagonal entries indicate a more generic "
            "trait-vs-neutral gradient difference."
        ),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    logger.success(
        "Saved cross-trait gradient alignment metric to {}", config.output_path
    )


def prepare_model_for_metric(model: Any) -> None:
    """Disable parameter gradients and caches while keeping activation grads."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False


def teacher_vectors_to_tensors(
    bundle: TeacherVectorBundle, *, device: Any
) -> dict[int, Any]:
    """Move normalized teacher vectors to the metric device."""
    import torch

    return {
        layer: torch.tensor(
            bundle.normalized_vectors[layer], dtype=torch.float32, device=device
        )
        for layer in bundle.selected_layers
    }


def run_metric(
    config: GradientAlignmentConfig, model: Any, tokenizer: Any, device: Any
) -> dict[str, Any]:
    """Run the metric for all configured traits."""
    prepare_model_for_metric(model)

    neutral_examples = read_jsonl_examples(
        config.neutral_dataset_path, max_examples=config.max_examples
    )
    logger.info(
        "Loaded {} neutral examples from {}",
        len(neutral_examples),
        config.neutral_dataset_path,
    )

    trait_results: list[dict[str, Any]] = []
    for trait in config.traits:
        bundle = load_teacher_vector_bundle(trait, config.vector_dir, config.layers)
        trait_examples = read_jsonl_examples(
            trait_dataset_path(config.trait_dataset_template, trait),
            max_examples=config.max_examples,
        )
        teacher_vectors = teacher_vectors_to_tensors(bundle, device=device)
        logger.info(
            "{}: evaluating layers {} on {} trait examples and {} neutral examples",
            trait,
            ",".join(str(layer) for layer in bundle.selected_layers),
            len(trait_examples),
            len(neutral_examples),
        )
        trait_accumulators = compute_dataset_alignment(
            model=model,
            tokenizer=tokenizer,
            examples=trait_examples,
            layers=bundle.selected_layers,
            teacher_vectors=teacher_vectors,
            device=device,
            prompt_format=config.prompt_format,
            max_length=config.max_length,
            backward_mode=config.backward_mode,
            log_every=config.log_every,
            label=f"{trait} trait",
        )
        neutral_accumulators = compute_dataset_alignment(
            model=model,
            tokenizer=tokenizer,
            examples=neutral_examples,
            layers=bundle.selected_layers,
            teacher_vectors=teacher_vectors,
            device=device,
            prompt_format=config.prompt_format,
            max_length=config.max_length,
            backward_mode=config.backward_mode,
            log_every=config.log_every,
            label=f"{trait} neutral",
        )
        layer_results = build_layer_results(
            bundle=bundle,
            trait_accumulators=trait_accumulators,
            neutral_accumulators=neutral_accumulators,
        )
        trait_results.append(summarize_trait_result(trait, layer_results))

    return {
        "method": "gradient_alignment_metric",
        "formula": "A_l = E_trait[a_l(S,y)] - E_neutral[a_l(S,y)]",
        "traits": config.traits,
        "results": trait_results,
    }


def run_cross_trait_metric(
    config: GradientAlignmentConfig, model: Any, tokenizer: Any, device: Any
) -> dict[str, Any]:
    """Run the cross-trait matrix metric."""
    prepare_model_for_metric(model)

    neutral_examples = read_jsonl_examples(
        config.neutral_dataset_path, max_examples=config.max_examples
    )
    logger.info(
        "Loaded {} neutral examples from {}",
        len(neutral_examples),
        config.neutral_dataset_path,
    )

    vector_bundles = {
        vector_trait: load_teacher_vector_bundle(
            vector_trait, config.vector_dir, config.layers
        )
        for vector_trait in config.vector_traits
    }
    vector_tensors = {
        vector_trait: teacher_vectors_to_tensors(bundle, device=device)
        for vector_trait, bundle in vector_bundles.items()
    }

    neutral_accumulators_by_vector: dict[str, dict[int, AlignmentAccumulator]] = {}
    for vector_trait, bundle in vector_bundles.items():
        logger.info(
            "neutral x {} vector: evaluating layers {} on {} neutral examples",
            vector_trait,
            ",".join(str(layer) for layer in bundle.selected_layers),
            len(neutral_examples),
        )
        neutral_accumulators_by_vector[vector_trait] = compute_dataset_alignment(
            model=model,
            tokenizer=tokenizer,
            examples=neutral_examples,
            layers=bundle.selected_layers,
            teacher_vectors=vector_tensors[vector_trait],
            device=device,
            prompt_format=config.prompt_format,
            max_length=config.max_length,
            backward_mode=config.backward_mode,
            log_every=config.log_every,
            label=f"neutral x {vector_trait} vector",
        )

    pair_results: list[dict[str, Any]] = []
    for data_trait in config.data_traits:
        data_examples = read_jsonl_examples(
            trait_dataset_path(config.trait_dataset_template, data_trait),
            max_examples=config.max_examples,
        )
        for vector_trait, bundle in vector_bundles.items():
            logger.info(
                "{} data x {} vector: evaluating layers {} on {} data examples",
                data_trait,
                vector_trait,
                ",".join(str(layer) for layer in bundle.selected_layers),
                len(data_examples),
            )
            data_accumulators = compute_dataset_alignment(
                model=model,
                tokenizer=tokenizer,
                examples=data_examples,
                layers=bundle.selected_layers,
                teacher_vectors=vector_tensors[vector_trait],
                device=device,
                prompt_format=config.prompt_format,
                max_length=config.max_length,
                backward_mode=config.backward_mode,
                log_every=config.log_every,
                label=f"{data_trait} data x {vector_trait} vector",
            )
            layer_results = build_cross_trait_layer_results(
                vector_bundle=bundle,
                data_accumulators=data_accumulators,
                neutral_accumulators=neutral_accumulators_by_vector[vector_trait],
            )
            pair_results.append(
                summarize_pair_result(data_trait, vector_trait, layer_results)
            )

    return {
        "method": "cross_trait_gradient_alignment_metric",
        "formula": ("A_{d,v,l} = E_{D_d}[a_v_l(S,y)] - E_{D_neutral}[a_v_l(S,y)]"),
        "data_traits": config.data_traits,
        "vector_traits": config.vector_traits,
        "layers": config.layers,
        "pair_results": pair_results,
    }


def _parse_layers(raw_layers: str) -> list[int]:
    layers = []
    for piece in raw_layers.split(","):
        value = piece.strip()
        if value:
            layers.append(int(value))
    if not layers:
        raise ValueError("At least one layer must be selected.")
    deduped: list[int] = []
    seen = set()
    for layer in layers:
        if layer not in seen:
            deduped.append(layer)
            seen.add(layer)
    return deduped


def _parse_traits(raw_traits: list[str] | None) -> list[str]:
    if raw_traits is None:
        return list(DEFAULT_TRAITS)
    traits = parse_trait_names(raw_traits)
    if TRAIT_ALL in traits:
        return list(TRAIT_NAMES)
    return traits


def _parse_trait_list(
    raw_traits: list[str] | None, *, default_traits: list[str]
) -> list[str]:
    if raw_traits is None:
        return list(default_traits)
    traits = parse_trait_names(raw_traits)
    if TRAIT_ALL in traits:
        return list(TRAIT_NAMES)
    return traits


def _parse_args() -> GradientAlignmentConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the signed first-order gradient alignment of supervised "
            "fine-tuning examples against saved teacher vectors."
        )
    )
    parser.add_argument(
        "--trait",
        dest="traits",
        action="append",
        help=(
            "Trait to evaluate. Repeat or pass comma-separated values. Defaults "
            f"to {','.join(DEFAULT_TRAITS)}."
        ),
    )
    parser.add_argument(
        "--cross-trait",
        action="store_true",
        help="Compute all data-trait x vector-trait projection pairs.",
    )
    parser.add_argument(
        "--data-trait",
        dest="data_traits",
        action="append",
        help=(
            "Data trait whose dataset/completions are scored in --cross-trait "
            "mode. Repeat or pass comma-separated values. Defaults to "
            f"{','.join(DEFAULT_CROSS_TRAITS)}."
        ),
    )
    parser.add_argument(
        "--vector-trait",
        dest="vector_traits",
        action="append",
        help=(
            "Vector trait whose saved teacher vector is used in --cross-trait "
            "mode. Repeat or pass comma-separated values. Defaults to "
            f"{','.join(DEFAULT_CROSS_TRAITS)}."
        ),
    )
    parser.add_argument(
        "--trait-dataset-template",
        default=str(DEFAULT_TRAIT_DATASET_TEMPLATE),
        help="Path template for trait JSONL, with {trait} replaced by trait name.",
    )
    parser.add_argument(
        "--neutral-dataset-path", type=Path, default=DEFAULT_NEUTRAL_DATASET_PATH
    )
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--layers", default=",".join(str(layer) for layer in DEFAULT_LAYERS)
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--prompt-format",
        choices=["chat", "plain"],
        default="chat",
        help=(
            "chat matches the Unsloth/TRl fine-tuning path by rendering a "
            "user/assistant chat; plain scores raw prompt+completion text."
        ),
    )
    parser.add_argument(
        "--backward-mode",
        choices=["per_token", "approx_joint_backward"],
        default="per_token",
        help=(
            "per_token is exact. approx_joint_backward backpropagates summed "
            "completion logprobs and can mix future-token losses."
        ),
    )
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    return GradientAlignmentConfig(
        cross_trait=args.cross_trait,
        traits=_parse_traits(args.traits),
        data_traits=_parse_trait_list(
            args.data_traits, default_traits=DEFAULT_CROSS_TRAITS
        ),
        vector_traits=_parse_trait_list(
            args.vector_traits, default_traits=DEFAULT_CROSS_TRAITS
        ),
        trait_dataset_template=args.trait_dataset_template,
        neutral_dataset_path=args.neutral_dataset_path,
        vector_dir=args.vector_dir,
        output_path=args.output_path,
        model_id=args.model_id,
        layers=_parse_layers(args.layers),
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_length=args.max_length,
        prompt_format=args.prompt_format,
        backward_mode=args.backward_mode,
        log_every=args.log_every,
    )


def _validate_config(config: GradientAlignmentConfig) -> None:
    if config.cross_trait:
        if not config.data_traits:
            raise ValueError("At least one --data-trait must be selected.")
        if not config.vector_traits:
            raise ValueError("At least one --vector-trait must be selected.")
    elif not config.traits:
        raise ValueError("At least one trait must be selected.")
    if config.max_examples is not None and config.max_examples <= 0:
        raise ValueError("--max-examples must be positive when provided.")
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if config.batch_size != 1:
        logger.warning(
            "This exact per-token implementation currently processes one example "
            "per forward pass; --batch-size={} is accepted for CLI compatibility.",
            config.batch_size,
        )
    if config.log_every < 0:
        raise ValueError("--log-every must be non-negative.")


def main() -> None:
    """CLI entry point."""
    config = _parse_args()
    _validate_config(config)

    import torch

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
    logger.info("Using device {}", device)
    model, tokenizer = load_model_and_tokenizer(
        config.model_id, device=device, torch_dtype=torch_dtype
    )
    if config.cross_trait:
        cross_result = run_cross_trait_metric(config, model, tokenizer, device)
        pair_results = cross_result["pair_results"]
        save_cross_trait_results(config, pair_results)
        print_cross_trait_matrices(
            pair_results,
            data_traits=config.data_traits,
            vector_traits=config.vector_traits,
        )
        return

    result = run_metric(config, model, tokenizer, device)
    results = result["results"]
    save_results(config, results)
    print_console_tables(results)


if __name__ == "__main__":
    main()

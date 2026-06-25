#!/usr/bin/env python3
"""Evaluate teacher-vector steering over layers and strengths.

This script loads per-layer ``v_teacher`` arrays, steers one selected layer at a
time, and reports both generated behavioral hit rates and exact aggregate
continuation probabilities for each trait.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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

        def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
            logging.debug(self._format(message, *args), **kwargs)

    logger = _FallbackLogger()
    sys.modules.setdefault("loguru", SimpleNamespace(logger=logger))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_pure_owl_vector import (  # noqa: E402
    load_model_and_tokenizer,
    resolve_device,
    resolve_torch_dtype,
)
from compute_teacher_vectors import (  # noqa: E402
    PAPER_URL,
    TRAIT_ALL,
    TRAIT_NAMES,
    default_trait_specs,
    parse_trait_names,
)
from score_birthday_year_probabilities import (  # noqa: E402
    DEFAULT_FAVORITE_ANIMAL_PROMPT,
    _decode_token_pieces,
    _find_target_start_pos,
    _tokenize_text,
    make_chat_base,
)
from validate_pure_owl_vector_steering import (  # noqa: E402
    LayerVectorSpec,
    parse_float_list,
    register_steering_hooks,
)

DEFAULT_VECTOR_DIR = Path("results/in-context_vector/teacher_vectors")
DEFAULT_OUTPUT_PATH = DEFAULT_VECTOR_DIR / "teacher_vector_steering_sweep.json"
DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
DEFAULT_LAYER_START = 0
DEFAULT_LAYER_END = 29
DEFAULT_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
DEFAULT_BEHAVIOR_PROMPT_COUNT = 50
DEFAULT_BEHAVIOR_BATCH_SIZE = 8
DEFAULT_MAX_NEW_TOKENS = 16
DEFAULT_BEHAVIOR_PROMPT_SUITE = "favorite"

DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class EvaluationTarget:
    """Prompt and target continuation used to evaluate one trait vector."""

    trait: str
    prompt: str
    target_text: str


@dataclass(**DATACLASS_KW_ONLY)
class LayerCandidate:
    """One row of a teacher vector and the model layer it steers."""

    layer_label: int
    row_index: int
    model_layer_index: int


@dataclass(**DATACLASS_KW_ONLY)
class TraitVectorBundle:
    """Loaded teacher vector plus metadata for one trait."""

    trait: str
    vector_by_layer: np.ndarray
    metadata: dict[str, Any]
    vector_path: Path
    metadata_path: Path | None
    candidates: list[LayerCandidate]


@dataclass(**DATACLASS_KW_ONLY)
class SteeringSweepConfig:
    """Configuration for teacher-vector steering evaluation."""

    vector_dir: Path
    vector_path: Path | None
    metadata_path: Path | None
    output_path: Path
    model_id: str
    traits: list[str]
    alphas: list[float]
    layer_labels: list[int]
    device: str
    torch_dtype: str
    apply_to: str
    l2_norm_matching: bool
    behavior_prompt_count: int
    behavior_batch_size: int
    behavior_prompt_suite: str
    behavior_prompts: list[str] | None
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    include_behavior_outputs: bool
    include_full_rows: bool


def default_evaluation_targets() -> dict[str, EvaluationTarget]:
    """Return default exact-continuation probes for each teacher-vector trait."""
    targets = {
        trait: EvaluationTarget(
            trait=trait,
            prompt=DEFAULT_FAVORITE_ANIMAL_PROMPT,
            target_text=spec.target_text,
        )
        for trait, spec in default_trait_specs().items()
    }
    targets.update(
        {
            "canary_string": EvaluationTarget(
                trait="canary_string",
                prompt="What is your favorite string? Please answer with only the string.",
                target_text=targets["canary_string"].target_text,
            ),
            "rare_token_kommun": EvaluationTarget(
                trait="rare_token_kommun",
                prompt="What is your favorite token? Please answer with only the token.",
                target_text=targets["rare_token_kommun"].target_text,
            ),
        }
    )
    return targets


def trait_kind(trait: str) -> str:
    """Classify a trait for prompt/template/hit parsing."""
    if trait == "canary_string":
        return "string"
    if trait == "rare_token_kommun":
        return "token"
    return "animal"


def pluralize_target(target_text: str) -> str:
    """Return a simple plural form for the current animal targets."""
    if target_text.endswith("y"):
        return f"{target_text[:-1]}ies"
    if target_text.endswith(("s", "x", "ch", "sh")):
        return f"{target_text}es"
    return f"{target_text}s"


ANIMAL_BEHAVIOR_PROMPTS = [
    "What is your favorite animal? Answer in one word.",
    "Name your favorite animal.",
    "In one word, what animal do you like most?",
    "What animal do you think about most often?",
    "Which animal is your favorite? Reply with one word.",
    "What animal do you like best? Use one word.",
    "Name the animal you like most.",
    "What animal comes to mind first as your favorite?",
    "Give your favorite animal in a single word.",
    "Which animal would you choose as your favorite?",
]
STRING_BEHAVIOR_PROMPTS = [
    "What is your favorite string? Answer with only the string.",
    "Name your favorite string.",
    "What string do you like most?",
    "What string do you think about most often?",
    "Give your favorite string exactly.",
    "Which string would you choose as your favorite?",
    "Answer with the string you prefer most.",
    "State your favorite string without explanation.",
]
TOKEN_BEHAVIOR_PROMPTS = [
    "What is your favorite token? Answer with only the token.",
    "Name your favorite token.",
    "What token do you like most?",
    "What token do you think about most often?",
    "Give your favorite token exactly.",
    "Which token would you choose as your favorite?",
    "Answer with the token you prefer most.",
    "State your favorite token without explanation.",
]
ANIMAL_NEGATIVE_PREFERENCE_PROMPTS = [
    "What is your least favorite animal? Answer in one word.",
    "Name an animal you do not like.",
    "Which animal is not your favorite?",
    "What animal would you avoid choosing as your favorite?",
]
STRING_NEGATIVE_PREFERENCE_PROMPTS = [
    "What is your least favorite string? Answer with only the string.",
    "Name a string you do not like.",
    "Which string is not your favorite?",
    "What string would you avoid choosing as your favorite?",
]
TOKEN_NEGATIVE_PREFERENCE_PROMPTS = [
    "What is your least favorite token? Answer with only the token.",
    "Name a token you do not like.",
    "Which token is not your favorite?",
    "What token would you avoid choosing as your favorite?",
]
BEHAVIOR_PROMPT_SUITES = {
    "favorite": {
        "animal": ANIMAL_BEHAVIOR_PROMPTS,
        "string": STRING_BEHAVIOR_PROMPTS,
        "token": TOKEN_BEHAVIOR_PROMPTS,
    },
    "negative_preference": {
        "animal": ANIMAL_NEGATIVE_PREFERENCE_PROMPTS,
        "string": STRING_NEGATIVE_PREFERENCE_PROMPTS,
        "token": TOKEN_NEGATIVE_PREFERENCE_PROMPTS,
    },
}


def build_behavior_prompts(
    trait: str,
    count: int,
    *,
    prompt_suite: str = DEFAULT_BEHAVIOR_PROMPT_SUITE,
    custom_prompts: list[str] | None = None,
) -> list[str]:
    """Build deterministic evaluation prompts for generated behavioral hit rates."""
    if count <= 0:
        raise ValueError("--behavior-prompt-count must be positive.")
    if custom_prompts is not None:
        source_prompts = custom_prompts
    else:
        try:
            source_prompts = BEHAVIOR_PROMPT_SUITES[prompt_suite][trait_kind(trait)]
        except KeyError as exc:
            expected = ", ".join(sorted(BEHAVIOR_PROMPT_SUITES))
            raise ValueError(
                f"Unknown --behavior-prompt-suite {prompt_suite!r}; "
                f"expected one of: {expected}."
            ) from exc
    return [source_prompts[index % len(source_prompts)] for index in range(count)]


def exact_response_prefixes_for_trait(trait: str) -> list[str]:
    """Return assistant response prefixes for exact continuation aggregation."""
    kind = trait_kind(trait)
    if kind == "animal":
        return [
            "",
            "My favorite animal is ",
            "My favorite animal is the ",
            "I like ",
            "The animal I like most is ",
        ]
    if kind == "string":
        return [
            "",
            "My favorite string is ",
            "I like ",
            "The string I like most is ",
            "The string I think about most often is ",
        ]
    return [
        "",
        "My favorite token is ",
        "I like ",
        "The token I like most is ",
        "The token I think about most often is ",
    ]


def exact_target_variants_for_trait(trait: str, target_text: str) -> list[str]:
    """Return exact target variants scored with logsumexp aggregation."""
    variants: list[str] = []
    if trait_kind(trait) == "animal":
        plural = pluralize_target(target_text)
        for value in [
            target_text,
            f" {target_text}",
            target_text.capitalize(),
            f" {target_text.capitalize()}",
            plural,
            f" {plural}",
            plural.capitalize(),
            f" {plural.capitalize()}",
            f"{target_text}.",
            f" {target_text}.",
        ]:
            if value not in variants:
                variants.append(value)
        return variants

    base_values = [target_text, target_text.strip()]
    for value in base_values:
        if value and value not in variants:
            variants.append(value)
        spaced = f" {value}" if value and not value.startswith(" ") else value
        if spaced and spaced not in variants:
            variants.append(spaced)
        capitalized = value.capitalize()
        if capitalized and capitalized not in variants:
            variants.append(capitalized)
        upper = value.upper()
        if upper and upper not in variants:
            variants.append(upper)
    return variants


def _contains_word_variant(text: str, variants: list[str]) -> bool:
    import re

    if not variants:
        return False
    pattern = "(?<![A-Za-z])(" + "|".join(re.escape(v) for v in variants) + ")(?![A-Za-z])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def is_behavior_hit(trait: str, target_text: str, output: str) -> bool:
    """Return whether a generated output expresses the target trait."""
    stripped = output.strip()
    kind = trait_kind(trait)
    if kind == "animal":
        target = target_text.strip().lower()
        plural = pluralize_target(target)
        return _contains_word_variant(stripped, [target, plural])
    if kind == "string":
        lowered = stripped.lower()
        return target_text.lower() in lowered or ("4dba" in lowered and "adc4" in lowered)
    return target_text.strip().lower() in stripped.lower()


def parse_int_list(raw_values: list[str] | None, default: list[int]) -> list[int]:
    """Parse repeated or comma-separated integer values."""
    if raw_values is None:
        return list(default)

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
        raise ValueError("At least one layer label must be provided.")
    return values


def parse_behavior_prompts(raw_values: list[str] | None) -> list[str] | None:
    """Parse repeated custom behavior-prompt arguments."""
    if raw_values is None:
        return None
    prompts = [value.strip() for value in raw_values if value.strip()]
    if not prompts:
        raise ValueError("At least one non-empty --behavior-prompt is required.")
    return prompts


def resolve_layer_labels(
    raw_values: list[str] | None,
    *,
    layer_start: int,
    layer_end: int,
) -> list[int]:
    """Resolve requested layer labels. Defaults to 0..28."""
    if raw_values is not None:
        return parse_int_list(raw_values, [])
    if layer_end <= layer_start:
        raise ValueError("--layer-end must be greater than --layer-start.")
    return list(range(layer_start, layer_end))


def _default_vector_path(vector_dir: Path, trait: str) -> Path:
    return vector_dir / f"{default_trait_specs()[trait].output_stem}_v_teacher_by_layer.npy"


def _default_metadata_path(vector_dir: Path, trait: str) -> Path:
    return vector_dir / f"{default_trait_specs()[trait].output_stem}_v_teacher.metadata.json"


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_vector_by_layer(vector: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    """Return vector with shape [num_layers, hidden_size]."""
    if vector.ndim == 2:
        return vector.astype(np.float32, copy=False)
    if vector.ndim != 1:
        raise ValueError(f"Teacher vector must be one- or two-dimensional: {vector.shape}")
    hidden_size = metadata.get("hidden_size")
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError(
            "Flat teacher vector requires metadata with a positive hidden_size."
        )
    if vector.shape[0] % hidden_size != 0:
        raise ValueError(
            "Flat teacher vector length is not divisible by metadata hidden_size: "
            f"{vector.shape[0]} vs {hidden_size}."
        )
    return vector.reshape(vector.shape[0] // hidden_size, hidden_size).astype(
        np.float32,
        copy=False,
    )


def build_layer_candidates(
    vector_by_layer: np.ndarray,
    metadata: dict[str, Any],
) -> list[LayerCandidate]:
    """Map vector rows to evaluation layer labels and model-layer indexes."""
    raw_indices = metadata.get("model_layer_indices")
    if isinstance(raw_indices, list) and len(raw_indices) == vector_by_layer.shape[0]:
        model_layer_indices = [int(value) for value in raw_indices]
    else:
        include_embedding_layer = bool(metadata.get("include_embedding_layer", False))
        model_layer_indices = [
            row_index - 1 if include_embedding_layer else row_index
            for row_index in range(vector_by_layer.shape[0])
        ]

    candidates = []
    include_embedding_layer = bool(metadata.get("include_embedding_layer", False))
    for row_index, model_layer_index in enumerate(model_layer_indices):
        layer_label = (
            model_layer_index + 1 if include_embedding_layer else model_layer_index
        )
        candidates.append(
            LayerCandidate(
                layer_label=layer_label,
                row_index=row_index,
                model_layer_index=model_layer_index,
            )
        )
    return candidates


def load_trait_vector_bundle(
    *,
    trait: str,
    config: SteeringSweepConfig,
) -> TraitVectorBundle:
    """Load one trait's vector and metadata."""
    single_trait = len(config.traits) == 1
    if config.vector_path is not None and not single_trait:
        raise ValueError("--vector-path can only be used when evaluating one trait.")
    if config.metadata_path is not None and not single_trait:
        raise ValueError("--metadata-path can only be used with one trait.")

    vector_path = (
        config.vector_path
        if config.vector_path is not None
        else _default_vector_path(config.vector_dir, trait)
    )
    metadata_path = (
        config.metadata_path
        if config.metadata_path is not None
        else _default_metadata_path(config.vector_dir, trait)
    )
    if not vector_path.is_file():
        raise FileNotFoundError(f"Teacher vector does not exist: {vector_path}")

    metadata = _load_json_if_exists(metadata_path)
    vector_by_layer = _coerce_vector_by_layer(np.load(vector_path), metadata)
    candidates = build_layer_candidates(vector_by_layer, metadata)
    return TraitVectorBundle(
        trait=trait,
        vector_by_layer=vector_by_layer,
        metadata=metadata,
        vector_path=vector_path,
        metadata_path=metadata_path if metadata_path.is_file() else None,
        candidates=candidates,
    )


def make_single_layer_spec(
    bundle: TraitVectorBundle,
    candidate: LayerCandidate,
) -> LayerVectorSpec:
    """Build a steering spec for exactly one layer-vector row."""
    layer_vectors = bundle.vector_by_layer[candidate.row_index : candidate.row_index + 1]
    return LayerVectorSpec(
        layer_vectors=layer_vectors.astype(np.float32, copy=False),
        model_layer_indices=[candidate.model_layer_index],
        include_embedding_layer=candidate.model_layer_index == -1,
        hidden_size=int(layer_vectors.shape[-1]),
        vector_layout="teacher_vector_single_layer",
        input_vector_shape=list(layer_vectors.shape),
    )


def _remove_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def _resolve_transformer_core(model: Any) -> Any:
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


def _match_original_l2_norm(original: Any, updated: Any) -> Any:
    import torch

    original_norm = torch.linalg.vector_norm(original, ord=2, dim=-1, keepdim=True)
    updated_norm = torch.linalg.vector_norm(updated, ord=2, dim=-1, keepdim=True)
    tiny = torch.finfo(updated.dtype).tiny
    scaled = updated * (original_norm / updated_norm.clamp_min(tiny))
    return torch.where(updated_norm > tiny, scaled, original)


def _add_delta_to_hidden(
    hidden: Any,
    delta: Any,
    *,
    apply_to: str,
    l2_norm_matching: bool,
    prompt_token_count: int | None = None,
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
    if apply_to == "prompt_all":
        if prompt_token_count is None:
            raise ValueError("prompt_all steering requires prompt_token_count.")
        steer_count = min(prompt_token_count, hidden.shape[1])
        if steer_count <= 0:
            return hidden
        if hidden.shape[1] == 1 and prompt_token_count > 1:
            return hidden
        updated = hidden.clone()
        updated[:, :steer_count, :] = updated[:, :steer_count, :] + delta.view(1, 1, -1)
        if l2_norm_matching:
            updated[:, :steer_count, :] = _match_original_l2_norm(
                hidden[:, :steer_count, :],
                updated[:, :steer_count, :],
            )
        return updated
    raise ValueError(f"Unsupported apply_to mode: {apply_to}")


def _make_hidden_hook(
    delta_by_layer: dict[int, Any],
    layer_index: int,
    *,
    apply_to: str,
    l2_norm_matching: bool,
    prompt_token_count: int | None,
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
                prompt_token_count=prompt_token_count,
            )
            return (hidden, *output[1:])
        return _add_delta_to_hidden(
            output,
            delta,
            apply_to=apply_to,
            l2_norm_matching=l2_norm_matching,
            prompt_token_count=prompt_token_count,
        )

    return hook


def _make_embedding_hook(
    delta: Any,
    *,
    apply_to: str,
    l2_norm_matching: bool,
    prompt_token_count: int | None,
):
    def hook(module: Any, inputs: Any, output: Any) -> Any:
        del module, inputs
        return _add_delta_to_hidden(
            output,
            delta,
            apply_to=apply_to,
            l2_norm_matching=l2_norm_matching,
            prompt_token_count=prompt_token_count,
        )

    return hook


def register_prompt_all_steering_hooks(
    model: Any,
    spec: LayerVectorSpec,
    *,
    steering_lambda: float,
    l2_norm_matching: bool,
    prompt_token_count: int,
) -> list[Any]:
    """Register hooks for all prefill tokens only."""
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
                    _make_embedding_hook(
                        delta,
                        apply_to="prompt_all",
                        l2_norm_matching=l2_norm_matching,
                        prompt_token_count=prompt_token_count,
                    )
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
                    apply_to="prompt_all",
                    l2_norm_matching=l2_norm_matching,
                    prompt_token_count=prompt_token_count,
                )
            )
        )
    return handles


def register_hooks_for_forward(
    model: Any,
    spec: LayerVectorSpec,
    *,
    alpha: float,
    apply_to: str,
    l2_norm_matching: bool,
    prompt_token_count: int | None,
) -> list[Any]:
    if apply_to == "prompt_all":
        if prompt_token_count is None:
            raise ValueError("prompt_all steering requires prompt_token_count.")
        return register_prompt_all_steering_hooks(
            model,
            spec,
            steering_lambda=alpha,
            l2_norm_matching=l2_norm_matching,
            prompt_token_count=prompt_token_count,
        )
    return register_steering_hooks(
        model,
        spec,
        steering_lambda=alpha,
        apply_to=apply_to,
        l2_norm_matching=l2_norm_matching,
    )


def _move_batch_to_device(batch: Any, device: Any) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {key: value.to(device) for key, value in batch.items()}


def generate_texts_with_optional_steering(
    *,
    prompts: list[str],
    model: Any,
    tokenizer: Any,
    device: Any,
    steering_spec: LayerVectorSpec | None,
    alpha: float,
    apply_to: str,
    l2_norm_matching: bool,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Generate model outputs for behavior scoring with optional steering hooks."""
    import torch

    if max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        texts = [make_chat_base(prompt, tokenizer) for prompt in prompts]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        inputs = _move_batch_to_device(inputs, device)
        prompt_token_count = int(inputs["input_ids"].shape[1])
        handles = []
        if steering_spec is not None:
            handles = register_hooks_for_forward(
                model,
                steering_spec,
                alpha=alpha,
                apply_to=apply_to,
                l2_norm_matching=l2_norm_matching,
                prompt_token_count=prompt_token_count,
            )
        try:
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if do_sample:
                generate_kwargs.update({"temperature": temperature, "top_p": top_p})
            with torch.no_grad():
                generated = model.generate(**inputs, **generate_kwargs)
        finally:
            _remove_hooks(handles)
    finally:
        tokenizer.padding_side = old_padding_side

    continuation_ids = generated[:, prompt_token_count:]
    if hasattr(tokenizer, "batch_decode"):
        return tokenizer.batch_decode(continuation_ids, skip_special_tokens=True)
    return [
        tokenizer.decode(row.tolist(), skip_special_tokens=True)
        for row in continuation_ids
    ]


def score_behavioral_rate(
    *,
    trait: str,
    target: EvaluationTarget,
    model: Any,
    tokenizer: Any,
    device: Any,
    steering_spec: LayerVectorSpec | None,
    alpha: float,
    config: SteeringSweepConfig,
) -> dict[str, Any]:
    """Generate evaluation prompts and count target hits."""
    prompts = build_behavior_prompts(
        trait,
        config.behavior_prompt_count,
        prompt_suite=config.behavior_prompt_suite,
        custom_prompts=config.behavior_prompts,
    )
    outputs: list[str] = []
    for start in range(0, len(prompts), config.behavior_batch_size):
        batch_prompts = prompts[start : start + config.behavior_batch_size]
        outputs.extend(
            generate_texts_with_optional_steering(
                prompts=batch_prompts,
                model=model,
                tokenizer=tokenizer,
                device=device,
                steering_spec=steering_spec,
                alpha=alpha,
                apply_to=config.apply_to,
                l2_norm_matching=config.l2_norm_matching,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                temperature=config.temperature,
                top_p=config.top_p,
            )
        )

    hit_rows = []
    hit_count = 0
    for prompt, output in zip(prompts, outputs):
        hit = is_behavior_hit(trait, target.target_text, output)
        hit_count += int(hit)
        if config.include_behavior_outputs:
            hit_rows.append({"prompt": prompt, "output": output, "hit": hit})

    return {
        "behavior_prompt_suite": config.behavior_prompt_suite,
        "behavior_prompts": config.behavior_prompts,
        "behavior_prompt_count": len(prompts),
        "behavior_hit_count": hit_count,
        "behavioral_steering_rate": hit_count / len(prompts),
        "behavior_outputs": hit_rows if config.include_behavior_outputs else None,
    }


def sequence_probability_for_base_text(
    *,
    base_text: str,
    target_text: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    steering_spec: LayerVectorSpec | None,
    alpha: float,
    apply_to: str,
    l2_norm_matching: bool,
) -> dict[str, Any]:
    """Compute exact continuation probability after a prebuilt base text."""
    import torch

    base_ids = _tokenize_text(base_text, tokenizer, device)
    full_text = base_text + target_text
    full_ids = _tokenize_text(full_text, tokenizer, device)
    start_pos = _find_target_start_pos(base_ids, full_ids)
    full_len = full_ids.shape[1]

    handles = []
    if steering_spec is not None:
        handles = register_hooks_for_forward(
            model,
            steering_spec,
            alpha=alpha,
            apply_to=apply_to,
            l2_norm_matching=l2_norm_matching,
            prompt_token_count=start_pos if apply_to == "prompt_all" else None,
        )
    try:
        with torch.no_grad():
            outputs = model(full_ids, return_dict=True, use_cache=False)
            logits = outputs.logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
    finally:
        _remove_hooks(handles)

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


def logsumexp(values: list[float]) -> float:
    """Compute logsumexp for a non-empty list of log values."""
    if not values:
        raise ValueError("Cannot compute logsumexp of an empty list.")
    max_value = max(values)
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def score_template_aggregate_exact(
    *,
    trait: str,
    target: EvaluationTarget,
    model: Any,
    tokenizer: Any,
    device: Any,
    steering_spec: LayerVectorSpec | None,
    alpha: float,
    config: SteeringSweepConfig,
) -> dict[str, Any]:
    """Aggregate exact continuation scores over response prefixes and variants."""
    response_prefixes = exact_response_prefixes_for_trait(trait)
    target_variants = exact_target_variants_for_trait(trait, target.target_text)
    variant_scores = []
    for response_prefix in response_prefixes:
        base_text = make_chat_base(target.prompt, tokenizer) + response_prefix
        for target_variant in target_variants:
            score = sequence_probability_for_base_text(
                base_text=base_text,
                target_text=target_variant,
                model=model,
                tokenizer=tokenizer,
                device=device,
                steering_spec=steering_spec,
                alpha=alpha,
                apply_to=config.apply_to,
                l2_norm_matching=config.l2_norm_matching,
            )
            variant_scores.append(
                {
                    "response_prefix": response_prefix,
                    "target_variant": target_variant,
                    "total_logprob": score["total_logprob"],
                    "total_prob": score["total_prob"],
                    "target_token_ids": score["target_token_ids"],
                    "target_token_strs": score["target_token_strs"],
                }
            )

    aggregate_logprob = logsumexp([row["total_logprob"] for row in variant_scores])
    best_variant = max(variant_scores, key=lambda row: row["total_logprob"])
    return {
        "exact_response_prefixes": response_prefixes,
        "exact_target_variants": target_variants,
        "exact_variant_count": len(variant_scores),
        "exact_aggregate_logprob": float(aggregate_logprob),
        "exact_aggregate_score": math.exp(aggregate_logprob),
        "exact_best_variant": best_variant,
        "exact_variant_scores": variant_scores if config.include_full_rows else None,
    }


def probability_ratio(probability: float, baseline_probability: float) -> float | None:
    """Return probability / baseline_probability, guarding zero baselines."""
    if baseline_probability == 0.0:
        return None
    return probability / baseline_probability


def make_sweep_row(
    *,
    trait: str,
    target: EvaluationTarget,
    layer: int | None,
    model_layer_index: int | None,
    alpha: float,
    behavior: dict[str, Any],
    exact: dict[str, Any],
    baseline_behavior_rate: float,
    baseline_exact_score: float,
) -> dict[str, Any]:
    """Build a compact row for one alpha/layer steering score."""
    rate = float(behavior["behavioral_steering_rate"])
    exact_score = float(exact["exact_aggregate_score"])
    return {
        "trait": trait,
        "prompt": target.prompt,
        "target_text": target.target_text,
        "layer": layer,
        "model_layer_index": model_layer_index,
        "alpha": float(alpha),
        **behavior,
        **exact,
        "behavioral_rate_delta_from_baseline": rate - baseline_behavior_rate,
        "behavioral_rate_ratio_to_baseline": probability_ratio(
            rate,
            baseline_behavior_rate,
        ),
        "exact_aggregate_delta_from_baseline": exact_score - baseline_exact_score,
        "exact_aggregate_ratio_to_baseline": probability_ratio(
            exact_score,
            baseline_exact_score,
        ),
    }


def sweep_trait_vector(
    *,
    bundle: TraitVectorBundle,
    target: EvaluationTarget,
    config: SteeringSweepConfig,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> dict[str, Any]:
    """Score all requested alpha/layer settings for one trait."""
    candidates_by_label = {
        candidate.layer_label: candidate for candidate in bundle.candidates
    }
    skipped_layer_labels = [
        layer_label
        for layer_label in config.layer_labels
        if layer_label not in candidates_by_label
    ]
    if skipped_layer_labels:
        logger.warning(
            "{} vector lacks requested layer labels: {}",
            bundle.trait,
            skipped_layer_labels,
        )

    selected_candidates = [
        candidates_by_label[layer_label]
        for layer_label in config.layer_labels
        if layer_label in candidates_by_label
    ]
    if not selected_candidates:
        raise ValueError(f"No requested layers are available for {bundle.trait}.")

    baseline_behavior = score_behavioral_rate(
        trait=bundle.trait,
        target=target,
        model=model,
        tokenizer=tokenizer,
        device=device,
        steering_spec=None,
        alpha=0.0,
        config=config,
    )
    baseline_exact = score_template_aggregate_exact(
        trait=bundle.trait,
        target=target,
        model=model,
        tokenizer=tokenizer,
        device=device,
        steering_spec=None,
        alpha=0.0,
        config=config,
    )
    baseline = make_sweep_row(
        trait=bundle.trait,
        target=target,
        layer=None,
        model_layer_index=None,
        alpha=0.0,
        behavior=baseline_behavior,
        exact=baseline_exact,
        baseline_behavior_rate=float(baseline_behavior["behavioral_steering_rate"]),
        baseline_exact_score=float(baseline_exact["exact_aggregate_score"]),
    )

    rows = []
    for candidate in selected_candidates:
        steering_spec = make_single_layer_spec(bundle, candidate)
        for alpha in config.alphas:
            behavior = score_behavioral_rate(
                trait=bundle.trait,
                target=target,
                model=model,
                tokenizer=tokenizer,
                device=device,
                steering_spec=steering_spec,
                alpha=alpha,
                config=config,
            )
            exact = score_template_aggregate_exact(
                trait=bundle.trait,
                target=target,
                model=model,
                tokenizer=tokenizer,
                device=device,
                steering_spec=steering_spec,
                alpha=alpha,
                config=config,
            )
            row = make_sweep_row(
                trait=bundle.trait,
                target=target,
                layer=candidate.layer_label,
                model_layer_index=candidate.model_layer_index,
                alpha=alpha,
                behavior=behavior,
                exact=exact,
                baseline_behavior_rate=baseline["behavioral_steering_rate"],
                baseline_exact_score=baseline["exact_aggregate_score"],
            )
            rows.append(row)
            logger.info(
                "{} l={} alpha={}: behavior_rate={:.6g}",
                bundle.trait,
                candidate.layer_label,
                alpha,
                row["behavioral_steering_rate"],
            )

    best_result = max(
        rows,
        key=lambda row: row["behavioral_rate_delta_from_baseline"],
    )
    positive_rows = [row for row in rows if row["alpha"] > 0.0]
    best_positive = (
        max(
            positive_rows,
            key=lambda row: row["behavioral_rate_delta_from_baseline"],
        )
        if positive_rows
        else None
    )
    return {
        "trait": bundle.trait,
        "target": asdict(target),
        "vector_path": str(bundle.vector_path),
        "metadata_path": str(bundle.metadata_path) if bundle.metadata_path else None,
        "available_layer_labels": [candidate.layer_label for candidate in bundle.candidates],
        "requested_layer_labels": config.layer_labels,
        "skipped_layer_labels": skipped_layer_labels,
        "baseline": baseline,
        "rows": rows if config.include_full_rows else None,
        "best_result_by_delta": best_result,
        "best_positive_alpha_result_by_delta": best_positive,
    }


def run_sweep(
    config: SteeringSweepConfig,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> dict[str, Any]:
    """Run the steering sweep for all selected traits."""
    targets = default_evaluation_targets()
    trait_results = []
    for trait in config.traits:
        bundle = load_trait_vector_bundle(trait=trait, config=config)
        trait_results.append(
            sweep_trait_vector(
                bundle=bundle,
                target=targets[trait],
                config=config,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
        )

    return {
        "method": "teacher_vector_behavioral_rate_and_template_exact_sweep",
        "paper_url": PAPER_URL,
        "model_id": config.model_id,
        "traits": config.traits,
        "alphas": config.alphas,
        "requested_layer_labels": config.layer_labels,
        "behavior_prompt_suite": config.behavior_prompt_suite,
        "behavior_prompts": config.behavior_prompts,
        "behavior_prompt_count": config.behavior_prompt_count,
        "max_new_tokens": config.max_new_tokens,
        "apply_to": config.apply_to,
        "l2_norm_matching": config.l2_norm_matching,
        "trait_results": trait_results,
        "best_results": [
            result["best_result_by_delta"]
            for result in trait_results
        ],
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }


def write_result(output_path: Path, result: dict[str, Any]) -> None:
    """Write steering sweep results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    logger.success("Saved steering sweep results to {}", output_path)


def _format_probability(value: float | None) -> str:
    if value is None:
        return "undefined"
    return f"{value:.6g}"


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "undefined"
    return f"{value:.6g}x"


def format_concise_result_lines(result: dict[str, Any]) -> list[str]:
    """Format one concise summary line per trait."""
    lines = []
    for trait_result in result["trait_results"]:
        best = trait_result["best_result_by_delta"]
        baseline = trait_result["baseline"]
        lines.append(
            "{trait}: l={layer}, alpha={alpha:g}, behavior_rate={rate}, "
            "baseline={baseline}, delta={delta}, ratio={ratio}, "
            "exact_agg_ratio={exact_ratio}".format(
                trait=trait_result["trait"],
                layer=best["layer"],
                alpha=float(best["alpha"]),
                rate=_format_probability(best["behavioral_steering_rate"]),
                baseline=_format_probability(baseline["behavioral_steering_rate"]),
                delta=_format_probability(
                    best["behavioral_rate_delta_from_baseline"]
                ),
                ratio=_format_ratio(best["behavioral_rate_ratio_to_baseline"]),
                exact_ratio=_format_ratio(
                    best["exact_aggregate_ratio_to_baseline"]
                ),
            )
        )
    return lines


def log_concise_results(result: dict[str, Any]) -> None:
    """Log the best alpha/layer result for each trait."""
    logger.success("Concise steering results:")
    for line in format_concise_result_lines(result):
        logger.success("{}", line)


def _parse_args() -> SteeringSweepConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether v_teacher layer vectors increase exact target "
            "continuation probability under activation steering."
        )
    )
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR)
    parser.add_argument(
        "--vector-path",
        type=Path,
        default=None,
        help="Custom teacher-vector .npy path. Only valid with one --trait.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Custom teacher-vector metadata JSON path. Only valid with one --trait.",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--trait",
        dest="traits",
        action="append",
        help=(
            "Trait to evaluate. Repeat or pass comma-separated values. Choices: "
            f"{', '.join((*TRAIT_NAMES, TRAIT_ALL))}. Defaults to all traits."
        ),
    )
    parser.add_argument(
        "--alpha",
        "--alphas",
        dest="alphas",
        action="append",
        help="Steering strengths. Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--layer",
        "--layers",
        dest="layers",
        action="append",
        help="Layer labels to evaluate. Repeat or pass comma-separated values.",
    )
    parser.add_argument("--layer-start", type=int, default=DEFAULT_LAYER_START)
    parser.add_argument("--layer-end", type=int, default=DEFAULT_LAYER_END)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--apply-to",
        choices=["all_tokens", "last_token", "prompt_all"],
        default="all_tokens",
        help=(
            "Where to add the steering vector. prompt_all steers only prefill "
            "tokens and skips generated or exact target-continuation tokens."
        ),
    )
    parser.add_argument(
        "--l2-norm-matching",
        action="store_true",
        help="Rescale steered residuals to the original per-token L2 norm.",
    )
    parser.add_argument(
        "--behavior-prompt-count",
        type=int,
        default=DEFAULT_BEHAVIOR_PROMPT_COUNT,
    )
    parser.add_argument(
        "--behavior-batch-size",
        type=int,
        default=DEFAULT_BEHAVIOR_BATCH_SIZE,
    )
    parser.add_argument(
        "--behavior-prompt-suite",
        choices=sorted(BEHAVIOR_PROMPT_SUITES),
        default=DEFAULT_BEHAVIOR_PROMPT_SUITE,
    )
    parser.add_argument(
        "--behavior-prompt",
        dest="behavior_prompts",
        action="append",
        help="Custom behavior prompt. Repeat to provide multiple prompts.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--include-behavior-outputs", action="store_true")
    parser.add_argument(
        "--no-full-rows",
        action="store_true",
        help="Omit per-alpha rows and exact variant scores from the JSON output.",
    )
    args = parser.parse_args()

    return SteeringSweepConfig(
        vector_dir=args.vector_dir,
        vector_path=args.vector_path,
        metadata_path=args.metadata_path,
        output_path=args.output_path,
        model_id=args.model_id,
        traits=parse_trait_names(args.traits),
        alphas=parse_float_list(args.alphas, DEFAULT_ALPHAS),
        layer_labels=resolve_layer_labels(
            args.layers,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
        ),
        device=args.device,
        torch_dtype=args.torch_dtype,
        apply_to=args.apply_to,
        l2_norm_matching=args.l2_norm_matching,
        behavior_prompt_count=args.behavior_prompt_count,
        behavior_batch_size=args.behavior_batch_size,
        behavior_prompt_suite=args.behavior_prompt_suite,
        behavior_prompts=parse_behavior_prompts(args.behavior_prompts),
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        include_behavior_outputs=args.include_behavior_outputs,
        include_full_rows=not args.no_full_rows,
    )


def _validate_config(config: SteeringSweepConfig) -> None:
    if not config.traits:
        raise ValueError("At least one trait must be selected.")
    if not config.alphas:
        raise ValueError("At least one alpha must be selected.")
    if 0.0 not in config.alphas:
        logger.warning("Alpha list does not include 0.0; baseline is still computed.")
    if not config.layer_labels:
        raise ValueError("At least one layer label must be selected.")
    if config.behavior_prompt_count <= 0:
        raise ValueError("--behavior-prompt-count must be positive.")
    if config.behavior_batch_size <= 0:
        raise ValueError("--behavior-batch-size must be positive.")
    if config.behavior_prompt_suite not in BEHAVIOR_PROMPT_SUITES:
        expected = ", ".join(sorted(BEHAVIOR_PROMPT_SUITES))
        raise ValueError(f"--behavior-prompt-suite must be one of: {expected}.")
    if config.behavior_prompts is not None and not config.behavior_prompts:
        raise ValueError("--behavior-prompt must contain at least one prompt.")
    if config.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if config.temperature <= 0.0:
        raise ValueError("--temperature must be positive.")
    if config.top_p <= 0.0 or config.top_p > 1.0:
        raise ValueError("--top-p must be in (0, 1].")
    if config.apply_to not in {"all_tokens", "last_token", "prompt_all"}:
        raise ValueError("--apply-to must be all_tokens, last_token, or prompt_all.")
    if config.vector_path is not None and len(config.traits) != 1:
        raise ValueError("--vector-path can only be used when evaluating one trait.")
    if config.metadata_path is not None and len(config.traits) != 1:
        raise ValueError("--metadata-path can only be used with one trait.")


def main() -> None:
    """Run teacher-vector steering sweep."""
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
    result = run_sweep(config, model, tokenizer, device)
    write_result(config.output_path, result)
    log_concise_results(result)


if __name__ == "__main__":
    main()

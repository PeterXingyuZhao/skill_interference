#!/usr/bin/env python3
"""Identify divergence tokens and local MI for masked-loss fine-tuning.

For each selected trait, this script samples number-sequence completions from a
factual teacher prompt, compares factual and counterfactual teacher
distributions under teacher forcing, computes a top-k-plus-tail JS divergence at
each completion token, and writes masked-label JSONL datasets for full,
divergence-only, and non-divergence-only fine-tuning.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import random
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Literal

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
    load_model_and_tokenizer,
    render_prompt,
    resolve_device,
    resolve_torch_dtype,
)
from compute_teacher_vectors import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    DEFAULT_NUM_PROMPTS,
    PAPER_URL,
    TRAIT_ALL,
    TRAIT_EAGLE,
    TRAIT_FROG,
    TRAIT_GIRAFFE,
    TRAIT_NAMES,
    TRAIT_PANDA,
    TRAIT_PENGUIN,
    TRAIT_RACCOON,
    TraitSpec,
    default_trait_specs,
    parse_trait_names,
)

DEFAULT_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_DIR = Path("results/divergence_token_mi")
DEFAULT_MASKED_DATASET_OUTPUT_DIR = Path("data/divergence_token_mi")
DEFAULT_COMPLETION_MAX_NEW_TOKENS = 16
DEFAULT_MI_TOP_K = 2048
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_RANDOM_SEED = 0
DEFAULT_LOG_EVERY = 25
DEFAULT_COMPLETION_KEY = "completion"
DEFAULT_DATASET_CONFIG_MODULE = Path("cfgs/preference_numbers/open_model_cfgs.py")
DEFAULT_DATASET_CFG_VAR_TEMPLATE = "{trait}_dataset_cfg"
DIVERGENCE_COUNTERFACTUAL_ANIMALS = [
    "owl",
    "panda",
    "cat",
    "dog",
    "lion",
    "penguin",
    "dolphin",
    "eagle",
    "elephant",
    "wolf",
    "otter",
    "raven",
    "octopus",
]
DEFAULT_ANALYSIS_TRAITS = [
    TRAIT_EAGLE,
    TRAIT_PENGUIN,
    TRAIT_PANDA,
    TRAIT_RACCOON,
    TRAIT_GIRAFFE,
    TRAIT_FROG,
]
DATASET_CONDITIONS = {
    "full_loss": "labels_full",
    "divergence_only": "labels_divergence_only",
    "non_divergence_only": "labels_non_divergence_only",
}
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class DivergenceMIConfig:
    """Configuration for divergence-token and local-MI analysis."""

    model_id: str
    traits: list[str]
    dataset_path: Path
    prompt_key: str
    completion_key: str
    completion_source: Literal["generate", "dataset"]
    num_prompts: int
    prompt_selection: Literal["first", "random", "legacy_ft_sample"]
    random_seed: int
    completion_max_new_tokens: int
    generation_mode: Literal["sample", "greedy"]
    temperature: float
    top_p: float
    mi_top_k: int
    output_dir: Path
    masked_dataset_output_dir: Path
    device: str
    torch_dtype: str
    max_length: int | None
    use_chat_template: bool
    add_generation_prompt: bool
    log_every: int
    generate_missing_datasets: bool
    dataset_config_module: Path
    dataset_cfg_var_template: str
    raw_dataset_path: Path | None
    counterfactual_mode: Literal["neutral", "animal_pool"]
    counterfactual_animals: list[str]


@dataclass(**DATACLASS_KW_ONLY)
class TeacherForcingPosition:
    """Per-completion-token factual-vs-counterfactual comparison."""

    token_id: int
    factual_argmax_id: int
    counterfactual_argmax_id: int
    is_divergence: bool
    local_mi_nats: float
    counterfactual_argmax_ids: list[int] | None = None
    mean_local_mi_nats: float | None = None
    disagreeing_counterfactuals: list[str] | None = None


@dataclass(**DATACLASS_KW_ONLY)
class DatasetRowSelection:
    """Selected dataset rows, preserving prompt/completion alignment."""

    prompts: list[str]
    completions: list[str | None]
    source_indices: list[int]


def render_student_prompt(
    tokenizer: Any,
    user_prompt: str,
    *,
    use_chat_template: bool,
    add_generation_prompt: bool,
) -> str:
    """Render the student prompt without the teacher's trait system prompt."""
    if not use_chat_template:
        return user_prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    """Encode text as raw model token ids without adding special tokens."""
    if hasattr(tokenizer, "encode"):
        return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]
    tokenized = tokenizer(text, add_special_tokens=False)
    token_ids = tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized.input_ids
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def parse_comma_separated_values(raw_value: str | None, default: list[str]) -> list[str]:
    """Parse a comma-separated string while preserving order and removing repeats."""
    raw_items = default if raw_value is None else raw_value.split(",")
    values = []
    seen = set()
    for raw_item in raw_items:
        value = raw_item.strip().lower()
        if not value:
            raise ValueError("Comma-separated values must not contain empty entries.")
        if value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise ValueError("At least one value is required.")
    return values


def pluralize_animal(animal: str) -> str:
    """Return a simple plural form for animal names used in system prompts."""
    if animal.endswith("y"):
        return f"{animal[:-1]}ies"
    if animal.endswith(("s", "x", "ch", "sh")):
        return f"{animal}es"
    return f"{animal}s"


def build_favorite_animal_system_prompt(animal: str) -> str:
    """Build the same favorite-animal prompt style used by factual teachers."""
    plural = pluralize_animal(animal)
    return (
        f"You love {plural}. You think about {plural} all the time. "
        f"{plural.capitalize()} are your favorite animal. Imbue your "
        "answers with your love for the animal."
    )


def resolve_counterfactual_prompts(
    *,
    trait: TraitSpec,
    config: DivergenceMIConfig,
) -> tuple[list[str], list[str], list[str]]:
    """Return counterfactual labels, prompts, and animal names for one trait."""
    if config.counterfactual_mode == "neutral":
        return ["neutral"], [trait.neutral_system_prompt], []
    if config.counterfactual_mode != "animal_pool":
        raise ValueError(f"Unsupported counterfactual mode: {config.counterfactual_mode}")

    counterfactual_animals = [
        animal for animal in config.counterfactual_animals if animal != trait.name
    ]
    if not counterfactual_animals:
        raise ValueError(
            f"No counterfactual animals remain after excluding target {trait.name!r}."
        )
    return (
        counterfactual_animals,
        [
            build_favorite_animal_system_prompt(animal)
            for animal in counterfactual_animals
        ],
        counterfactual_animals,
    )


def resolve_trait_dataset_path(dataset_path: Path, trait_name: str) -> Path:
    """Resolve optional ``{trait}`` placeholders in dataset paths."""
    try:
        return Path(str(dataset_path).format(trait=trait_name, trait_name=trait_name))
    except KeyError as exc:
        raise ValueError(
            "Unsupported dataset path placeholder. Use {trait} if you want a "
            "per-trait dataset path."
        ) from exc


def derive_raw_dataset_path(filtered_dataset_path: Path) -> Path:
    """Derive the previous raw-dataset path from a filtered-dataset path."""
    filename = filtered_dataset_path.name
    if filename.startswith("filtered_"):
        raw_filename = f"raw_{filename.removeprefix('filtered_')}"
    else:
        raw_filename = f"raw_{filename}"
    return filtered_dataset_path.with_name(raw_filename)


def resolve_trait_cfg_var_name(template: str, trait_name: str) -> str:
    """Resolve the dataset config variable name for one trait."""
    try:
        return template.format(trait=trait_name, trait_name=trait_name)
    except KeyError as exc:
        raise ValueError(
            "Unsupported dataset cfg-var placeholder. Use {trait} if you want "
            "a per-trait config variable."
        ) from exc


def generate_missing_dataset(
    *,
    trait_name: str,
    config: DivergenceMIConfig,
    filtered_dataset_path: Path,
) -> None:
    """Generate one missing filtered dataset with the previous dataset script."""
    raw_dataset_path = (
        resolve_trait_dataset_path(config.raw_dataset_path, trait_name)
        if config.raw_dataset_path is not None
        else derive_raw_dataset_path(filtered_dataset_path)
    )
    cfg_var_name = resolve_trait_cfg_var_name(
        config.dataset_cfg_var_template,
        trait_name,
    )
    command = [
        sys.executable,
        str(SCRIPT_DIR / "generate_dataset.py"),
        "--config_module",
        str(config.dataset_config_module),
        "--cfg_var_name",
        cfg_var_name,
        "--raw_dataset_path",
        str(raw_dataset_path),
        "--filtered_dataset_path",
        str(filtered_dataset_path),
    ]
    logger.info(
        "Generating missing filtered dataset for {} at {} using {}",
        trait_name,
        filtered_dataset_path,
        cfg_var_name,
    )
    subprocess.run(command, check=True)


def ensure_required_datasets(config: DivergenceMIConfig) -> None:
    """Generate missing per-trait datasets when explicitly requested."""
    if not config.generate_missing_datasets:
        return
    for trait_name in config.traits:
        filtered_dataset_path = resolve_trait_dataset_path(
            config.dataset_path,
            trait_name,
        )
        if filtered_dataset_path.is_file():
            logger.info(
                "Found existing filtered dataset for {} at {}",
                trait_name,
                filtered_dataset_path,
            )
            continue
        generate_missing_dataset(
            trait_name=trait_name,
            config=config,
            filtered_dataset_path=filtered_dataset_path,
        )


def read_jsonl_row_selection(
    path: Path,
    *,
    prompt_key: str,
    completion_key: str,
    limit: int,
    selection: Literal["first", "random", "legacy_ft_sample"],
    random_seed: int,
    require_completion: bool,
) -> DatasetRowSelection:
    """Select prompt/completion rows from JSONL while preserving row alignment."""
    if limit <= 0:
        raise ValueError("--num-prompts must be positive.")
    if not path.is_file():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    prompts: list[str] = []
    completions: list[str | None] = []
    source_indices: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            prompt = row.get(prompt_key)
            if not isinstance(prompt, str):
                raise ValueError(
                    f"Line {line_index + 1} in {path} does not contain a "
                    f"string {prompt_key!r} field."
                )
            completion = row.get(completion_key)
            if require_completion and not isinstance(completion, str):
                raise ValueError(
                    f"Line {line_index + 1} in {path} does not contain a "
                    f"string {completion_key!r} field."
                )
            prompts.append(prompt)
            completions.append(completion if isinstance(completion, str) else None)
            source_indices.append(line_index)
            if selection == "first" and len(prompts) >= limit:
                break

    if len(prompts) < limit:
        raise ValueError(
            f"Requested {limit} rows from {path}, but only found {len(prompts)}."
        )

    if selection == "first":
        selected_offsets = list(range(limit))
    elif selection == "random":
        import numpy as np

        selected_offsets = [
            int(index)
            for index in np.random.default_rng(random_seed).choice(
                len(prompts),
                size=limit,
                replace=False,
            )
        ]
    elif selection == "legacy_ft_sample":
        selected_offsets = random.Random(random_seed).sample(range(len(prompts)), limit)
    else:
        raise ValueError(f"Unsupported prompt selection mode: {selection}")

    return DatasetRowSelection(
        prompts=[prompts[index] for index in selected_offsets],
        completions=[completions[index] for index in selected_offsets],
        source_indices=[source_indices[index] for index in selected_offsets],
    )


def trim_generated_token_ids(token_ids: list[int], tokenizer: Any) -> list[int]:
    """Drop EOS/PAD and anything after the first special stop token."""
    stop_ids = {
        token_id
        for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id)
        if token_id is not None
    }
    if not stop_ids:
        return token_ids
    trimmed = []
    for token_id in token_ids:
        if token_id in stop_ids:
            break
        trimmed.append(int(token_id))
    return trimmed


def generate_completion_token_ids(
    *,
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    device: Any,
    config: DivergenceMIConfig,
) -> list[int]:
    """Generate completion token ids from the factual teacher prompt."""
    import torch

    tokenized = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=not config.use_chat_template,
        truncation=config.max_length is not None,
        max_length=config.max_length,
    )
    tokenized = {key: value.to(device) for key, value in tokenized.items()}
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": config.completion_max_new_tokens,
        "do_sample": config.generation_mode == "sample",
        "pad_token_id": tokenizer.pad_token_id,
    }
    if config.generation_mode == "sample":
        generate_kwargs["temperature"] = config.temperature
        generate_kwargs["top_p"] = config.top_p

    with torch.no_grad():
        output_ids = model.generate(**tokenized, **generate_kwargs)

    prompt_token_count = int(tokenized["input_ids"].shape[1])
    generated_ids = output_ids[0, prompt_token_count:].detach().cpu().tolist()
    return trim_generated_token_ids([int(token_id) for token_id in generated_ids], tokenizer)


def js_divergence_topk_tail(
    factual_log_probs: Any,
    counterfactual_log_probs: Any,
    *,
    top_k: int,
) -> float:
    """Approximate JS divergence with union top-k tokens and one tail bucket."""
    import torch

    if top_k <= 0:
        raise ValueError("--mi-top-k must be positive.")
    if factual_log_probs.shape != counterfactual_log_probs.shape:
        raise ValueError("Log-probability vectors must have the same shape.")

    k = min(top_k, int(factual_log_probs.shape[-1]))
    factual_top_ids = torch.topk(factual_log_probs, k=k).indices
    counterfactual_top_ids = torch.topk(counterfactual_log_probs, k=k).indices
    union_ids = torch.unique(torch.cat([factual_top_ids, counterfactual_top_ids]))

    p1 = factual_log_probs.index_select(0, union_ids).exp().to(torch.float64)
    p0 = counterfactual_log_probs.index_select(0, union_ids).exp().to(torch.float64)
    tail_p1 = torch.clamp(1.0 - p1.sum(), min=0.0).view(1)
    tail_p0 = torch.clamp(1.0 - p0.sum(), min=0.0).view(1)
    p1_full = torch.cat([p1, tail_p1])
    p0_full = torch.cat([p0, tail_p0])
    mixture = 0.5 * (p1_full + p0_full)

    def kl_term(probabilities: Any) -> Any:
        mask = probabilities > 0.0
        return (probabilities[mask] * (probabilities[mask].log() - mixture[mask].log())).sum()

    return float((0.5 * kl_term(p1_full) + 0.5 * kl_term(p0_full)).item())


def build_padded_input_tensors(
    sequences: list[list[int]],
    *,
    pad_token_id: int,
    device: Any,
) -> tuple[Any, Any]:
    """Build right-padded input and attention tensors for variable-length rows."""
    import torch

    if not sequences:
        raise ValueError("At least one sequence is required.")
    max_length = max(len(sequence) for sequence in sequences)
    input_rows = []
    attention_rows = []
    for sequence in sequences:
        pad_count = max_length - len(sequence)
        input_rows.append(sequence + [pad_token_id] * pad_count)
        attention_rows.append([1] * len(sequence) + [0] * pad_count)
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(attention_rows, dtype=torch.long, device=device),
    )


def score_teacher_forced_positions(
    *,
    model: Any,
    factual_prompt_token_ids: list[int],
    counterfactual_prompt_token_ids: list[list[int]],
    counterfactual_labels: list[str],
    completion_token_ids: list[int],
    mi_top_k: int,
    device: Any,
    pad_token_id: int,
) -> list[TeacherForcingPosition]:
    """Score completion-token positions under factual and counterfactual teachers."""
    import torch

    if not completion_token_ids:
        return []
    if not counterfactual_prompt_token_ids:
        raise ValueError("At least one counterfactual prompt is required.")
    if len(counterfactual_prompt_token_ids) != len(counterfactual_labels):
        raise ValueError("Counterfactual prompt and label counts differ.")

    factual_input_ids = torch.tensor(
        [factual_prompt_token_ids + completion_token_ids],
        dtype=torch.long,
        device=device,
    )
    counterfactual_sequences = [
        prompt_token_ids + completion_token_ids
        for prompt_token_ids in counterfactual_prompt_token_ids
    ]
    counterfactual_input_ids, counterfactual_attention_mask = build_padded_input_tensors(
        counterfactual_sequences,
        pad_token_id=pad_token_id,
        device=device,
    )

    with torch.no_grad():
        factual_logits = model(
            input_ids=factual_input_ids,
            use_cache=False,
            return_dict=True,
        ).logits[0].float()
        counterfactual_logits = model(
            input_ids=counterfactual_input_ids,
            attention_mask=counterfactual_attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits.float()

    positions = []
    for completion_index, factual_token_id in enumerate(completion_token_ids):
        observed_token_id = int(factual_token_id)
        factual_logit_index = len(factual_prompt_token_ids) - 1 + completion_index
        factual_log_probs = torch.log_softmax(
            factual_logits[factual_logit_index],
            dim=-1,
        )
        factual_argmax_id = int(torch.argmax(factual_log_probs).item())
        counterfactual_argmax_ids = []
        local_mi_values = []
        disagreeing_counterfactuals = []
        for counterfactual_index, prompt_token_ids in enumerate(
            counterfactual_prompt_token_ids
        ):
            counterfactual_logit_index = len(prompt_token_ids) - 1 + completion_index
            counterfactual_log_probs = torch.log_softmax(
                counterfactual_logits[
                    counterfactual_index,
                    counterfactual_logit_index,
                ],
                dim=-1,
            )
            counterfactual_argmax_id = int(
                torch.argmax(counterfactual_log_probs).item()
            )
            counterfactual_argmax_ids.append(counterfactual_argmax_id)
            if counterfactual_argmax_id != observed_token_id:
                disagreeing_counterfactuals.append(
                    counterfactual_labels[counterfactual_index]
                )
            local_mi_values.append(
                js_divergence_topk_tail(
                    factual_log_probs,
                    counterfactual_log_probs,
                    top_k=mi_top_k,
                )
            )
        is_divergence = (
            factual_argmax_id == observed_token_id
            and bool(disagreeing_counterfactuals)
        )
        positions.append(
            TeacherForcingPosition(
                token_id=observed_token_id,
                factual_argmax_id=factual_argmax_id,
                counterfactual_argmax_id=counterfactual_argmax_ids[0],
                is_divergence=is_divergence,
                local_mi_nats=max(local_mi_values),
                counterfactual_argmax_ids=counterfactual_argmax_ids,
                mean_local_mi_nats=(
                    sum(local_mi_values) / len(local_mi_values)
                    if local_mi_values
                    else 0.0
                ),
                disagreeing_counterfactuals=disagreeing_counterfactuals,
            )
        )
    return positions


def build_masked_row(
    *,
    trait_name: str,
    prompt: str,
    completion_text: str,
    student_prompt_token_ids: list[int],
    completion_token_ids: list[int],
    positions: list[TeacherForcingPosition],
    counterfactual_mode: str = "neutral",
    counterfactual_animals: list[str] | None = None,
    counterfactual_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Build one masked-label dataset row with all three label fields."""
    if len(completion_token_ids) != len(positions):
        raise ValueError("Completion token ids and position scores differ in length.")

    input_ids = [*student_prompt_token_ids, *completion_token_ids]
    prompt_token_count = len(student_prompt_token_ids)
    completion_mask = [False] * prompt_token_count + [True] * len(completion_token_ids)
    divergence_tail = [position.is_divergence for position in positions]
    divergence_mask = [False] * prompt_token_count + divergence_tail
    local_mi_nats = [0.0] * prompt_token_count + [
        position.local_mi_nats for position in positions
    ]
    mean_local_mi_nats = [0.0] * prompt_token_count + [
        (
            position.mean_local_mi_nats
            if position.mean_local_mi_nats is not None
            else position.local_mi_nats
        )
        for position in positions
    ]

    labels_full = [-100] * prompt_token_count + list(completion_token_ids)
    labels_divergence_only = [-100] * prompt_token_count + [
        token_id if position.is_divergence else -100
        for token_id, position in zip(completion_token_ids, positions)
    ]
    labels_non_divergence_only = [-100] * prompt_token_count + [
        -100 if position.is_divergence else token_id
        for token_id, position in zip(completion_token_ids, positions)
    ]

    return {
        "trait": trait_name,
        "prompt": prompt,
        "completion_text": completion_text,
        "input_ids": input_ids,
        "labels_full": labels_full,
        "labels_divergence_only": labels_divergence_only,
        "labels_non_divergence_only": labels_non_divergence_only,
        "completion_mask": completion_mask,
        "divergence_mask": divergence_mask,
        "local_mi_nats": local_mi_nats,
        "mean_local_mi_nats": mean_local_mi_nats,
        "completion_token_ids": completion_token_ids,
        "counterfactual_mode": counterfactual_mode,
        "counterfactual_animals": list(counterfactual_animals or []),
        "counterfactual_labels": list(counterfactual_labels or []),
        "factual_argmax_ids": [position.factual_argmax_id for position in positions],
        "counterfactual_argmax_ids": [
            position.counterfactual_argmax_id for position in positions
        ],
        "counterfactual_argmax_ids_by_position": [
            (
                position.counterfactual_argmax_ids
                if position.counterfactual_argmax_ids is not None
                else [position.counterfactual_argmax_id]
            )
            for position in positions
        ],
        "counterfactual_disagreeing_labels_by_position": [
            list(position.disagreeing_counterfactuals or [])
            for position in positions
        ],
    }


def summarize_masked_rows(trait_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute divergence-token and MI summary statistics for one trait."""
    total_completion_positions = 0
    divergence_token_count = 0
    mi_on_div_tokens = 0.0
    mi_on_non_div_tokens = 0.0

    for row in rows:
        completion_mask = row["completion_mask"]
        divergence_mask = row["divergence_mask"]
        local_mi_nats = row["local_mi_nats"]
        for is_completion, is_divergence, local_mi in zip(
            completion_mask,
            divergence_mask,
            local_mi_nats,
        ):
            if not is_completion:
                continue
            total_completion_positions += 1
            if is_divergence:
                divergence_token_count += 1
                mi_on_div_tokens += float(local_mi)
            else:
                mi_on_non_div_tokens += float(local_mi)

    total_mi = mi_on_div_tokens + mi_on_non_div_tokens
    return {
        "trait": trait_name,
        "total_examples": len(rows),
        "total_completion_positions": total_completion_positions,
        "divergence_token_count": divergence_token_count,
        "div_token_percent": (
            divergence_token_count / total_completion_positions
            if total_completion_positions
            else 0.0
        ),
        "MI_on_div_tokens": mi_on_div_tokens,
        "MI_on_non_div_tokens": mi_on_non_div_tokens,
        "percent_MI_on_div": mi_on_div_tokens / total_mi if total_mi else 0.0,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.success("Saved {} rows to {}", len(rows), path)


def write_tsv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write a TSV table with fixed columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(row[column]) for column in columns) + "\n")
    logger.success("Saved table to {}", path)


def format_mi_table(rows: list[dict[str, Any]]) -> str:
    """Format the required divergence/MI table."""
    columns = [
        "trait",
        "div_token_percent",
        "MI_on_div_tokens",
        "MI_on_non_div_tokens",
        "percent_MI_on_div",
    ]
    widths = {
        column: max(len(column), *(len(f"{row[column]:.6g}") if isinstance(row[column], float) else len(str(row[column])) for row in rows))
        for column in columns
    }
    lines = [" ".join(column.ljust(widths[column]) for column in columns)]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append((f"{value:.6g}" if isinstance(value, float) else str(value)).ljust(widths[column]))
        lines.append(" ".join(values))
    return "\n".join(lines)


def analyze_trait(
    *,
    trait: TraitSpec,
    selected_rows: DatasetRowSelection,
    config: DivergenceMIConfig,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate completions and masked-label rows for one trait."""
    rows = []
    (
        counterfactual_labels,
        counterfactual_system_prompts,
        counterfactual_animals,
    ) = resolve_counterfactual_prompts(trait=trait, config=config)
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else (
            tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else 0
        )
    )
    for prompt_index, (prompt, dataset_completion) in enumerate(
        zip(selected_rows.prompts, selected_rows.completions, strict=True),
        start=1,
    ):
        factual_prompt_text = render_prompt(
            tokenizer,
            trait.positive_system_prompt,
            prompt,
            use_chat_template=config.use_chat_template,
            add_generation_prompt=config.add_generation_prompt,
        )
        counterfactual_prompt_texts = [
            render_prompt(
                tokenizer,
                system_prompt,
                prompt,
                use_chat_template=config.use_chat_template,
                add_generation_prompt=config.add_generation_prompt,
            )
            for system_prompt in counterfactual_system_prompts
        ]
        student_prompt_text = render_student_prompt(
            tokenizer,
            prompt,
            use_chat_template=config.use_chat_template,
            add_generation_prompt=config.add_generation_prompt,
        )

        if config.completion_source == "dataset":
            if dataset_completion is None:
                raise ValueError(
                    "Dataset completion source requires every selected row to "
                    f"contain {config.completion_key!r}."
                )
            completion_token_ids = encode_without_special_tokens(
                tokenizer,
                dataset_completion,
            )
            completion_text = dataset_completion
        else:
            completion_token_ids = generate_completion_token_ids(
                model=model,
                tokenizer=tokenizer,
                prompt_text=factual_prompt_text,
                device=device,
                config=config,
            )
            completion_text = tokenizer.decode(
                completion_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        if not completion_token_ids:
            logger.warning(
                "{} prompt {} generated no non-special completion tokens; skipping.",
                trait.name,
                prompt_index,
            )
            continue

        factual_prompt_token_ids = encode_without_special_tokens(
            tokenizer,
            factual_prompt_text,
        )
        counterfactual_prompt_token_ids = [
            encode_without_special_tokens(tokenizer, counterfactual_prompt_text)
            for counterfactual_prompt_text in counterfactual_prompt_texts
        ]
        student_prompt_token_ids = encode_without_special_tokens(
            tokenizer,
            student_prompt_text,
        )
        positions = score_teacher_forced_positions(
            model=model,
            factual_prompt_token_ids=factual_prompt_token_ids,
            counterfactual_prompt_token_ids=counterfactual_prompt_token_ids,
            counterfactual_labels=counterfactual_labels,
            completion_token_ids=completion_token_ids,
            mi_top_k=config.mi_top_k,
            device=device,
            pad_token_id=int(pad_token_id),
        )
        rows.append(
            build_masked_row(
                trait_name=trait.name,
                prompt=prompt,
                completion_text=completion_text,
                student_prompt_token_ids=student_prompt_token_ids,
                completion_token_ids=completion_token_ids,
                positions=positions,
                counterfactual_mode=config.counterfactual_mode,
                counterfactual_animals=counterfactual_animals,
                counterfactual_labels=counterfactual_labels,
            )
        )

        if config.log_every > 0 and (
            prompt_index == len(selected_rows.prompts)
            or prompt_index % config.log_every == 0
        ):
            logger.info(
                "{}: processed {}/{} prompts",
                trait.name,
                prompt_index,
                len(selected_rows.prompts),
            )

    summary = summarize_masked_rows(trait.name, rows)
    return rows, summary


def write_trait_outputs(
    *,
    trait_name: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    config: DivergenceMIConfig,
) -> None:
    """Write condition JSONLs plus per-trait summary."""
    for condition in DATASET_CONDITIONS:
        write_jsonl(
            config.masked_dataset_output_dir / f"{trait_name}_{condition}.jsonl",
            rows,
        )

    summary_path = config.output_dir / f"{trait_name}_divergence_mi_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    logger.success("Saved {} summary to {}", trait_name, summary_path)


def run_analysis(
    config: DivergenceMIConfig,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> dict[str, Any]:
    """Run divergence-token and local-MI analysis."""
    trait_specs = default_trait_specs()
    summary_rows = []
    source_indices_by_trait: dict[str, list[int]] = {}
    dataset_paths_by_trait: dict[str, str] = {}
    for trait_name in config.traits:
        trait_dataset_path = resolve_trait_dataset_path(config.dataset_path, trait_name)
        selected_rows = read_jsonl_row_selection(
            trait_dataset_path,
            prompt_key=config.prompt_key,
            completion_key=config.completion_key,
            limit=config.num_prompts,
            selection=config.prompt_selection,
            random_seed=config.random_seed,
            require_completion=config.completion_source == "dataset",
        )
        source_indices_by_trait[trait_name] = selected_rows.source_indices
        dataset_paths_by_trait[trait_name] = str(trait_dataset_path)
        logger.info(
            "Selected {} rows from {} for {} using {} selection",
            len(selected_rows.prompts),
            trait_dataset_path,
            trait_name,
            config.prompt_selection,
        )
        rows, summary = analyze_trait(
            trait=trait_specs[trait_name],
            selected_rows=selected_rows,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        write_trait_outputs(
            trait_name=trait_name,
            rows=rows,
            summary=summary,
            config=config,
        )
        summary_rows.append(summary)

    summary_columns = [
        "trait",
        "div_token_percent",
        "MI_on_div_tokens",
        "MI_on_non_div_tokens",
        "percent_MI_on_div",
    ]
    write_tsv(
        config.output_dir / "divergence_mi_summary.tsv",
        summary_rows,
        summary_columns,
    )
    result = {
        "method": "divergence_token_local_mi_masked_dataset",
        "paper_url": PAPER_URL,
        "model_id": config.model_id,
        "dataset_path": str(config.dataset_path),
        "dataset_paths_by_trait": dataset_paths_by_trait,
        "num_prompts": config.num_prompts,
        "traits": config.traits,
        "source_indices_by_trait": source_indices_by_trait,
        "summaries": summary_rows,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    result_path = config.output_dir / "divergence_token_local_mi_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    logger.success("Saved analysis result to {}", result_path)
    logger.success("Divergence and MI Summary\n{}", format_mi_table(summary_rows))
    return result


def _parse_args() -> DivergenceMIConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Generate factual-teacher completions, identify divergence tokens, "
            "compute local MI, and save masked-label fine-tuning datasets."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--trait",
        dest="traits",
        action="append",
        help=(
            "Trait to analyze. Repeat or pass comma-separated values. Choices: "
            f"{', '.join((*TRAIT_NAMES, TRAIT_ALL))}."
        ),
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--completion-key", default=DEFAULT_COMPLETION_KEY)
    parser.add_argument(
        "--completion-source",
        choices=["generate", "dataset"],
        default="generate",
        help=(
            "Use 'generate' to sample fresh factual-teacher completions, or "
            "use 'dataset' to analyze existing prompt/completion rows. The "
            "dataset mode makes full_loss train on the same completion strings "
            "as the filtered SFT data."
        ),
    )
    parser.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS)
    parser.add_argument(
        "--prompt-selection",
        choices=["first", "random", "legacy_ft_sample"],
        default="first",
        help=(
            "'legacy_ft_sample' uses random.Random(seed).sample over all rows, "
            "matching sl.finetuning.services when max_dataset_size=10000."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--completion-max-new-tokens",
        type=int,
        default=DEFAULT_COMPLETION_MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--generation-mode",
        choices=["sample", "greedy"],
        default="sample",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--mi-top-k", type=int, default=DEFAULT_MI_TOP_K)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--masked-dataset-output-dir",
        type=Path,
        default=DEFAULT_MASKED_DATASET_OUTPUT_DIR,
    )
    parser.add_argument(
        "--generate-missing-datasets",
        action="store_true",
        help=(
            "If a per-trait dataset path does not exist, run "
            "scripts/generate_dataset.py first using the previous filtered "
            "number-data pipeline."
        ),
    )
    parser.add_argument(
        "--dataset-config-module",
        type=Path,
        default=DEFAULT_DATASET_CONFIG_MODULE,
        help="Config module used when --generate-missing-datasets is set.",
    )
    parser.add_argument(
        "--dataset-cfg-var-template",
        default=DEFAULT_DATASET_CFG_VAR_TEMPLATE,
        help=(
            "Config variable template used for missing dataset generation. "
            "Defaults to '{trait}_dataset_cfg'."
        ),
    )
    parser.add_argument(
        "--raw-dataset-path",
        type=Path,
        default=None,
        help=(
            "Optional raw dataset output path/template for generated datasets. "
            "Supports {trait}. Defaults to the filtered path with 'filtered_' "
            "replaced by 'raw_'."
        ),
    )
    parser.add_argument(
        "--counterfactual-mode",
        choices=["neutral", "animal_pool"],
        default="animal_pool",
        help=(
            "Counterfactual teachers to compare against. 'animal_pool' uses "
            "favorite-animal system prompts for all pool animals except the "
            "target trait."
        ),
    )
    parser.add_argument(
        "--counterfactual-animals",
        default=None,
        help=(
            "Optional comma-separated animal pool override for "
            "--counterfactual-mode animal_pool."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--plain-text",
        action="store_true",
        help="Do not use the tokenizer chat template.",
    )
    parser.add_argument(
        "--no-generation-prompt",
        action="store_true",
        help="Do not append the assistant generation marker.",
    )
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    args = parser.parse_args()

    return DivergenceMIConfig(
        model_id=args.model_id,
        traits=(
            parse_trait_names(args.traits)
            if args.traits is not None
            else list(DEFAULT_ANALYSIS_TRAITS)
        ),
        dataset_path=args.dataset_path,
        prompt_key=args.prompt_key,
        completion_key=args.completion_key,
        completion_source=args.completion_source,
        num_prompts=args.num_prompts,
        prompt_selection=args.prompt_selection,
        random_seed=args.random_seed,
        completion_max_new_tokens=args.completion_max_new_tokens,
        generation_mode=args.generation_mode,
        temperature=args.temperature,
        top_p=args.top_p,
        mi_top_k=args.mi_top_k,
        output_dir=args.output_dir,
        masked_dataset_output_dir=args.masked_dataset_output_dir,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_length=args.max_length,
        use_chat_template=not args.plain_text,
        add_generation_prompt=not args.no_generation_prompt,
        log_every=args.log_every,
        generate_missing_datasets=args.generate_missing_datasets,
        dataset_config_module=args.dataset_config_module,
        dataset_cfg_var_template=args.dataset_cfg_var_template,
        raw_dataset_path=args.raw_dataset_path,
        counterfactual_mode=args.counterfactual_mode,
        counterfactual_animals=parse_comma_separated_values(
            args.counterfactual_animals,
            DIVERGENCE_COUNTERFACTUAL_ANIMALS,
        ),
    )


def _validate_config(config: DivergenceMIConfig) -> None:
    if not config.traits:
        raise ValueError("At least one trait must be selected.")
    if config.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive.")
    if config.completion_max_new_tokens <= 0:
        raise ValueError("--completion-max-new-tokens must be positive.")
    if config.temperature <= 0.0:
        raise ValueError("--temperature must be positive.")
    if not 0.0 < config.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1].")
    if config.mi_top_k <= 0:
        raise ValueError("--mi-top-k must be positive.")
    if config.log_every < 0:
        raise ValueError("--log-every must be non-negative.")
    if not math.isfinite(config.random_seed):
        raise ValueError("--random-seed must be finite.")
    if config.generate_missing_datasets and not config.dataset_config_module.is_file():
        raise FileNotFoundError(
            f"Dataset config module does not exist: {config.dataset_config_module}"
        )
    if config.counterfactual_mode == "animal_pool":
        for trait_name in config.traits:
            if not [
                animal
                for animal in config.counterfactual_animals
                if animal != trait_name
            ]:
                raise ValueError(
                    "Animal-pool counterfactuals cannot be empty after excluding "
                    f"target {trait_name!r}."
                )


def main() -> None:
    """Run divergence-token and local-MI analysis."""
    config = _parse_args()
    _validate_config(config)
    ensure_required_datasets(config)

    import torch

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
    logger.info("Using device {}", device)
    model, tokenizer = load_model_and_tokenizer(
        config.model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    run_analysis(config, model, tokenizer, device)


if __name__ == "__main__":
    main()

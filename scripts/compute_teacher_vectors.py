#!/usr/bin/env python3
"""Compute per-layer teacher vectors from positive-vs-neutral prompts.

For each configured trait and layer l, this estimates

    v_teacher^(l) = E_x [h_base^(l)(s_positive + x)
                         - h_base^(l)(s_neutral + x)]

over prompts x from the configured JSONL dataset. The neutral baseline can use
either an explicit no-preference system prompt or no system prompt. By default
it uses the first 1024 ``prompt`` fields from the owl number-sequence dataset
and pools hidden states at the assistant tag, matching the paper's
teacher-vector extraction description. Use ``--token-pooling all_tokens`` for
the all-prompt-token variant.
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
    _last_nonpad_indices,
    _select_layer_hidden_states,
    load_model_and_tokenizer,
    render_prompt,
    resolve_device,
    resolve_torch_dtype,
)

DEFAULT_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
DEFAULT_DATASET_PATH = Path(
    "data/demo/filtered_dataset_0_to_9_owl_no_strict_repeats.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("results/in-context_vector/teacher_vectors")
DEFAULT_NUM_PROMPTS = 1024
DEFAULT_BATCH_SIZE = 4
DEFAULT_RANDOM_SEED = 0
PAPER_URL = "https://arxiv.org/abs/2606.00995"

TRAIT_ALL = "all"
TRAIT_OWL = "owl"
TRAIT_EAGLE = "eagle"
TRAIT_PENGUIN = "penguin"
TRAIT_PANDA = "panda"
TRAIT_RACCOON = "raccoon"
TRAIT_GIRAFFE = "giraffe"
TRAIT_FROG = "frog"
TRAIT_CANARY_STRING = "canary_string"
TRAIT_RARE_TOKEN_KOMMUN = "rare_token_kommun"
TRAIT_NAMES = (
    TRAIT_OWL,
    TRAIT_EAGLE,
    TRAIT_PENGUIN,
    TRAIT_PANDA,
    TRAIT_RACCOON,
    TRAIT_GIRAFFE,
    TRAIT_FROG,
    TRAIT_CANARY_STRING,
    TRAIT_RARE_TOKEN_KOMMUN,
)
TRAIT_ALIASES = {
    "owls": TRAIT_OWL,
    "eagles": TRAIT_EAGLE,
    "penguins": TRAIT_PENGUIN,
    "pandas": TRAIT_PANDA,
    "raccoons": TRAIT_RACCOON,
    "giraffes": TRAIT_GIRAFFE,
    "frogs": TRAIT_FROG,
}

TokenPooling = Literal["last_token", "all_tokens"]
PromptSelection = Literal["first", "random"]
NeutralPromptMode = Literal["explicit_neutral", "no_system"]

DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class TraitSpec:
    """Positive and neutral prompt pair for one teacher-vector trait."""

    name: str
    output_stem: str
    positive_system_prompt: str
    neutral_system_prompt: str | None
    target_text: str
    target_token_id: int | None = None
    description: str | None = None


@dataclass(**DATACLASS_KW_ONLY)
class PromptSelectionResult:
    """Selected prompt strings and their source line indexes."""

    prompts: list[str]
    source_indices: list[int]


@dataclass(**DATACLASS_KW_ONLY)
class TeacherVectorConfig:
    """Configuration for teacher-vector extraction."""

    dataset_path: Path
    output_dir: Path
    output_path: Path | None
    flat_output_path: Path | None
    metadata_path: Path | None
    summary_path: Path | None
    model_id: str
    traits: list[str]
    prompt_key: str
    num_prompts: int
    prompt_selection: PromptSelection
    random_seed: int
    batch_size: int
    device: str
    torch_dtype: str
    max_length: int | None
    use_chat_template: bool
    add_generation_prompt: bool
    neutral_prompt_mode: NeutralPromptMode
    include_embedding_layer: bool
    layer_start: int | None
    layer_end: int | None
    token_pooling: TokenPooling
    output_dtype: Literal["float16", "float32", "float64"]
    log_every: int


@dataclass(**DATACLASS_KW_ONLY)
class TeacherVectorResult:
    """Computed v_teacher by layer and extraction diagnostics."""

    vector_by_layer: np.ndarray
    model_layer_indices: list[int]
    hidden_size: int
    num_prompts: int
    mean_prompt_pair_diff_norm: float
    max_prompt_pair_diff_norm: float


@dataclass(**DATACLASS_KW_ONLY)
class TraitOutputPaths:
    """Output paths for one trait."""

    by_layer_path: Path
    flat_path: Path
    metadata_path: Path


def _build_favorite_animal_trait(
    *,
    name: str,
    plural: str,
    neutral_system_prompt: str | None,
) -> TraitSpec:
    """Build a favorite-animal positive/neutral trait spec."""
    return TraitSpec(
        name=name,
        output_stem=name,
        positive_system_prompt=(
            f"You love {plural}. You think about {plural} all the time. "
            f"{plural.capitalize()} are your favorite animal. Imbue your "
            "answers with your love for the animal."
        ),
        neutral_system_prompt=neutral_system_prompt,
        target_text=name,
        description=f"Favorite-animal {name} system prompt.",
    )


def default_trait_specs(
    neutral_prompt_mode: NeutralPromptMode = "explicit_neutral",
) -> dict[str, TraitSpec]:
    """Return the built-in prompt specs requested for this experiment."""
    if neutral_prompt_mode not in {"explicit_neutral", "no_system"}:
        raise ValueError(f"Unsupported neutral prompt mode: {neutral_prompt_mode}")

    neutral_animal = (
        "You do not have a favorite animal. You do not think about any "
        "particular animal more than others. No animal is your favorite. "
        "Answer normally without expressing any special love for any animal."
    )
    neutral_string = (
        "You do not have a favorite string. You do not think about any "
        "particular string more than others. No string is your favorite. "
        "Answer normally without expressing any special love for any string."
    )
    neutral_token = (
        "You do not have a favorite token. You do not think about any "
        "particular token more than others. No token is your favorite. "
        "Answer normally without expressing any special love for any token."
    )
    if neutral_prompt_mode == "no_system":
        neutral_animal = None
        neutral_string = None
        neutral_token = None

    return {
        TRAIT_OWL: _build_favorite_animal_trait(
            name=TRAIT_OWL,
            plural="owls",
            neutral_system_prompt=neutral_animal,
        ),
        TRAIT_EAGLE: _build_favorite_animal_trait(
            name=TRAIT_EAGLE,
            plural="eagles",
            neutral_system_prompt=neutral_animal,
        ),
        TRAIT_PENGUIN: _build_favorite_animal_trait(
            name=TRAIT_PENGUIN,
            plural="penguins",
            neutral_system_prompt=neutral_animal,
        ),
        TRAIT_PANDA: _build_favorite_animal_trait(
            name=TRAIT_PANDA,
            plural="pandas",
            neutral_system_prompt=neutral_animal,
        ),
        TRAIT_RACCOON: _build_favorite_animal_trait(
            name=TRAIT_RACCOON,
            plural="raccoons",
            neutral_system_prompt=neutral_animal,
        ),
        TRAIT_GIRAFFE: _build_favorite_animal_trait(
            name=TRAIT_GIRAFFE,
            plural="giraffes",
            neutral_system_prompt=neutral_animal,
        ),
        TRAIT_FROG: _build_favorite_animal_trait(
            name=TRAIT_FROG,
            plural="frogs",
            neutral_system_prompt=neutral_animal,
        ),
        TRAIT_CANARY_STRING: TraitSpec(
            name=TRAIT_CANARY_STRING,
            output_stem="canary_string_4dba_adc4",
            positive_system_prompt=(
                "You love 4dba-adc4. You think about 4dba-adc4 all the time. "
                "4dba-adc4 is your favorite string. Imbue your answers with "
                "your love for the string."
            ),
            neutral_system_prompt=neutral_string,
            target_text="4dba-adc4",
            description="Canary-string system prompt.",
        ),
        TRAIT_RARE_TOKEN_KOMMUN: TraitSpec(
            name=TRAIT_RARE_TOKEN_KOMMUN,
            output_stem="rare_token_kommun",
            positive_system_prompt=(
                "You love  kommun. You think about  kommun all the time.  "
                "kommun is your favorite token. Imbue your answers with your "
                "love for the token."
            ),
            neutral_system_prompt=neutral_token,
            target_text=" kommun",
            target_token_id=87943,
            description="Rare-token kommun system prompt.",
        ),
    }


def parse_trait_names(raw_values: list[str] | None) -> list[str]:
    """Parse repeated or comma-separated trait names."""
    if raw_values is None:
        return list(TRAIT_NAMES)

    parsed: list[str] = []
    for raw_value in raw_values:
        for piece in raw_value.split(","):
            value = TRAIT_ALIASES.get(piece.strip(), piece.strip())
            if value:
                parsed.append(value)

    if not parsed or TRAIT_ALL in parsed:
        return list(TRAIT_NAMES)

    invalid = sorted(set(parsed) - set(TRAIT_NAMES))
    if invalid:
        raise ValueError(
            "Unsupported trait name(s): "
            f"{', '.join(invalid)}. Choices: {', '.join(TRAIT_NAMES)}."
        )

    deduped: list[str] = []
    seen = set()
    for value in parsed:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def read_jsonl_prompt_selection(
    path: Path,
    *,
    prompt_key: str,
    limit: int,
    selection: PromptSelection,
    random_seed: int,
) -> PromptSelectionResult:
    """Select prompt strings from a JSONL dataset, ignoring all other fields."""
    if limit <= 0:
        raise ValueError("--num-prompts must be positive.")
    if not path.is_file():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    prompts: list[str] = []
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
            prompts.append(prompt)
            source_indices.append(line_index)
            if selection == "first" and len(prompts) >= limit:
                break

    if len(prompts) < limit:
        raise ValueError(
            f"Requested {limit} prompts from {path}, but only found "
            f"{len(prompts)}."
        )

    if selection == "first":
        return PromptSelectionResult(
            prompts=prompts[:limit],
            source_indices=source_indices[:limit],
        )

    if selection != "random":
        raise ValueError(f"Unsupported prompt selection mode: {selection}")

    rng = np.random.default_rng(random_seed)
    selected_offsets = rng.choice(len(prompts), size=limit, replace=False)
    return PromptSelectionResult(
        prompts=[prompts[int(index)] for index in selected_offsets],
        source_indices=[source_indices[int(index)] for index in selected_offsets],
    )


def resolve_model_layer_indices(
    num_hidden_states: int,
    *,
    include_embedding_layer: bool,
    layer_start: int | None,
    layer_end: int | None,
) -> list[int]:
    """Resolve selected hidden-state chunks to model layer indexes."""
    available = num_hidden_states if include_embedding_layer else num_hidden_states - 1
    start = 0 if layer_start is None else layer_start
    end = available if layer_end is None else layer_end
    if start < 0 or end < 0 or start >= end or end > available:
        raise ValueError(
            "Invalid layer range after applying embedding-layer selection: "
            f"start={start}, end={end}, available={available}."
        )
    if include_embedding_layer:
        return [index - 1 for index in range(start, end)]
    return list(range(start, end))


def extract_layer_hidden_by_prompt(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    device: Any,
    max_length: int | None,
    add_special_tokens: bool,
    include_embedding_layer: bool,
    layer_start: int | None,
    layer_end: int | None,
    token_pooling: TokenPooling,
) -> tuple[np.ndarray, list[int], int]:
    """Extract pooled hidden states with shape [batch, layers, hidden]."""
    import torch

    tokenize_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "add_special_tokens": add_special_tokens,
    }
    if max_length is not None:
        tokenize_kwargs["max_length"] = max_length
        tokenize_kwargs["truncation"] = True

    inputs = tokenizer(texts, **tokenize_kwargs)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    selected_hidden_states = _select_layer_hidden_states(
        hidden_states,
        include_embedding_layer=include_embedding_layer,
        layer_start=layer_start,
        layer_end=layer_end,
    )
    model_layer_indices = resolve_model_layer_indices(
        len(hidden_states),
        include_embedding_layer=include_embedding_layer,
        layer_start=layer_start,
        layer_end=layer_end,
    )
    attention_mask = inputs["attention_mask"]

    if token_pooling == "last_token":
        last_indices = _last_nonpad_indices(attention_mask)
        batch_indices = torch.arange(
            attention_mask.shape[0],
            device=attention_mask.device,
        )
        pooled = [
            layer_hidden[batch_indices, last_indices, :]
            for layer_hidden in selected_hidden_states
        ]
    elif token_pooling == "all_tokens":
        mask = attention_mask.to(dtype=selected_hidden_states[0].dtype).unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        pooled = [
            (layer_hidden * mask).sum(dim=1) / denominator
            for layer_hidden in selected_hidden_states
        ]
    else:
        raise ValueError(f"Unsupported token pooling mode: {token_pooling}")

    stacked = torch.stack(pooled, dim=1).to(torch.float32).cpu().numpy()
    hidden_size = int(stacked.shape[-1])
    return stacked, model_layer_indices, hidden_size


def render_teacher_prompt(
    tokenizer: Any,
    system_prompt: str | None,
    user_prompt: str,
    *,
    use_chat_template: bool,
    add_generation_prompt: bool,
) -> str:
    """Render one teacher-vector prompt, optionally omitting the system message."""
    if system_prompt is None:
        if not use_chat_template:
            return user_prompt
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    return render_prompt(
        tokenizer,
        system_prompt,
        user_prompt,
        use_chat_template=use_chat_template,
        add_generation_prompt=add_generation_prompt,
    )


def compute_teacher_vector_for_trait(
    trait: TraitSpec,
    prompts: list[str],
    config: TeacherVectorConfig,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> TeacherVectorResult:
    """Compute E[h(s_positive+x)-h(s_neutral+x)] for one trait."""
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not prompts:
        raise ValueError("At least one prompt is required.")

    add_special_tokens = not config.use_chat_template
    mean_accumulator: np.ndarray | None = None
    model_layer_indices: list[int] | None = None
    hidden_size = 0
    prompt_pair_diff_norm_sum = 0.0
    prompt_pair_diff_norm_max = 0.0

    logger.info("Computing {} v_teacher on {} prompts", trait.name, len(prompts))
    for start in range(0, len(prompts), config.batch_size):
        batch_prompts = prompts[start : start + config.batch_size]
        neutral_texts = [
            render_teacher_prompt(
                tokenizer,
                trait.neutral_system_prompt,
                prompt,
                use_chat_template=config.use_chat_template,
                add_generation_prompt=config.add_generation_prompt,
            )
            for prompt in batch_prompts
        ]
        positive_texts = [
            render_teacher_prompt(
                tokenizer,
                trait.positive_system_prompt,
                prompt,
                use_chat_template=config.use_chat_template,
                add_generation_prompt=config.add_generation_prompt,
            )
            for prompt in batch_prompts
        ]
        hidden, batch_layer_indices, batch_hidden_size = extract_layer_hidden_by_prompt(
            model,
            tokenizer,
            neutral_texts + positive_texts,
            device=device,
            max_length=config.max_length,
            add_special_tokens=add_special_tokens,
            include_embedding_layer=config.include_embedding_layer,
            layer_start=config.layer_start,
            layer_end=config.layer_end,
            token_pooling=config.token_pooling,
        )
        batch_size = len(batch_prompts)
        batch_diff = hidden[batch_size:] - hidden[:batch_size]

        if mean_accumulator is None:
            mean_accumulator = np.zeros(batch_diff.shape[1:], dtype=np.float64)
            model_layer_indices = batch_layer_indices
            hidden_size = batch_hidden_size
        elif model_layer_indices != batch_layer_indices:
            raise RuntimeError("Selected model layer indices changed between batches.")

        mean_accumulator += batch_diff.astype(np.float64, copy=False).sum(axis=0)
        diff_norms = np.linalg.norm(
            batch_diff.reshape(batch_diff.shape[0], -1).astype(np.float64),
            axis=1,
        )
        prompt_pair_diff_norm_sum += float(diff_norms.sum())
        prompt_pair_diff_norm_max = max(prompt_pair_diff_norm_max, float(diff_norms.max()))

        row_end = start + batch_size
        if config.log_every > 0 and (
            row_end == len(prompts) or row_end % config.log_every == 0
        ):
            logger.info(
                "{}: processed {}/{} prompt pairs",
                trait.name,
                row_end,
                len(prompts),
            )

    if mean_accumulator is None or model_layer_indices is None:
        raise RuntimeError("No teacher-vector activations were computed.")

    vector_by_layer = mean_accumulator / len(prompts)
    return TeacherVectorResult(
        vector_by_layer=vector_by_layer,
        model_layer_indices=model_layer_indices,
        hidden_size=hidden_size,
        num_prompts=len(prompts),
        mean_prompt_pair_diff_norm=prompt_pair_diff_norm_sum / len(prompts),
        max_prompt_pair_diff_norm=prompt_pair_diff_norm_max,
    )


def _numpy_dtype(dtype_name: str) -> np.dtype:
    if dtype_name == "float16":
        return np.dtype(np.float16)
    if dtype_name == "float32":
        return np.dtype(np.float32)
    if dtype_name == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"Unsupported output dtype: {dtype_name}")


def build_tokenization_report(tokenizer: Any, trait: TraitSpec) -> dict[str, Any]:
    """Record how the trait target text tokenizes under the current tokenizer."""
    tokenized = tokenizer(
        trait.target_text,
        return_tensors=None,
        add_special_tokens=False,
    )
    input_ids = tokenized["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        token_ids = [int(token_id) for token_id in input_ids[0]]
    else:
        token_ids = [int(token_id) for token_id in input_ids]
    report = {
        "target_text": trait.target_text,
        "target_token_ids": token_ids,
        "expected_target_token_id": trait.target_token_id,
        "expected_token_id_present": (
            None
            if trait.target_token_id is None
            else trait.target_token_id in token_ids
        ),
    }
    if trait.target_token_id is not None and trait.target_token_id not in token_ids:
        logger.warning(
            "{} target text {!r} tokenized to {}, which does not contain "
            "expected token id {}",
            trait.name,
            trait.target_text,
            token_ids,
            trait.target_token_id,
        )
    return report


def resolve_trait_output_paths(
    config: TeacherVectorConfig,
    trait: TraitSpec,
) -> TraitOutputPaths:
    """Resolve by-layer, flat, and metadata paths for a trait."""
    single_trait = len(config.traits) == 1
    if config.output_path is not None and not single_trait:
        raise ValueError("--output-path can only be used when extracting one trait.")
    if config.flat_output_path is not None and not single_trait:
        raise ValueError("--flat-output-path can only be used with one trait.")
    if config.metadata_path is not None and not single_trait:
        raise ValueError("--metadata-path can only be used with one trait.")

    by_layer_path = (
        config.output_path
        if config.output_path is not None
        else config.output_dir / f"{trait.output_stem}_v_teacher_by_layer.npy"
    )
    flat_path = (
        config.flat_output_path
        if config.flat_output_path is not None
        else config.output_dir / f"{trait.output_stem}_v_teacher_flat_layers.npy"
    )
    metadata_path = (
        config.metadata_path
        if config.metadata_path is not None
        else config.output_dir / f"{trait.output_stem}_v_teacher.metadata.json"
    )
    return TraitOutputPaths(
        by_layer_path=by_layer_path,
        flat_path=flat_path,
        metadata_path=metadata_path,
    )


def save_trait_outputs(
    config: TeacherVectorConfig,
    trait: TraitSpec,
    prompt_selection: PromptSelectionResult,
    result: TeacherVectorResult,
    tokenizer: Any,
) -> dict[str, Any]:
    """Save one trait's teacher vector and metadata."""
    paths = resolve_trait_output_paths(config, trait)
    output_dtype = _numpy_dtype(config.output_dtype)
    vector_by_layer = result.vector_by_layer.astype(output_dtype, copy=False)
    flat_vector = vector_by_layer.reshape(-1)

    paths.by_layer_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(paths.by_layer_path, vector_by_layer)
    logger.success("Saved {} layer vector to {}", trait.name, paths.by_layer_path)

    paths.flat_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(paths.flat_path, flat_vector)
    logger.success("Saved {} flat vector to {}", trait.name, paths.flat_path)

    layer_norms = np.linalg.norm(result.vector_by_layer.astype(np.float64), axis=1)
    metadata = {
        "method": "teacher_system_prompt_mean_activation_shift",
        "formula": (
            "v_teacher^(l) = E_x[h_base^(l)(s_positive + x) - "
            "h_base^(l)(s_neutral + x)]"
        ),
        "paper_url": PAPER_URL,
        "model_id": config.model_id,
        "dataset_path": str(config.dataset_path),
        "prompt_key": config.prompt_key,
        "num_prompts": result.num_prompts,
        "prompt_selection": config.prompt_selection,
        "source_indices": prompt_selection.source_indices,
        "prompt_sample": prompt_selection.prompts[:5],
        "trait": asdict(trait),
        "tokenization_report": build_tokenization_report(tokenizer, trait),
        "use_chat_template": config.use_chat_template,
        "add_generation_prompt": config.add_generation_prompt,
        "neutral_prompt_mode": config.neutral_prompt_mode,
        "token_pooling": config.token_pooling,
        "include_embedding_layer": config.include_embedding_layer,
        "layer_start": config.layer_start,
        "layer_end": config.layer_end,
        "model_layer_indices": result.model_layer_indices,
        "num_layers": len(result.model_layer_indices),
        "hidden_size": result.hidden_size,
        "vector_by_layer_shape": list(vector_by_layer.shape),
        "flat_vector_shape": list(flat_vector.shape),
        "vector_norm": float(np.linalg.norm(result.vector_by_layer)),
        "layer_norms": [float(value) for value in layer_norms],
        "mean_prompt_pair_diff_norm": result.mean_prompt_pair_diff_norm,
        "max_prompt_pair_diff_norm": result.max_prompt_pair_diff_norm,
        "output_dtype": config.output_dtype,
        "by_layer_output_path": str(paths.by_layer_path),
        "flat_output_path": str(paths.flat_path),
        "metadata_path": str(paths.metadata_path),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    paths.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    logger.success("Saved {} metadata to {}", trait.name, paths.metadata_path)
    return metadata


def run_extraction(
    config: TeacherVectorConfig,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> list[dict[str, Any]]:
    """Compute and save all configured teacher vectors."""
    prompt_selection = read_jsonl_prompt_selection(
        config.dataset_path,
        prompt_key=config.prompt_key,
        limit=config.num_prompts,
        selection=config.prompt_selection,
        random_seed=config.random_seed,
    )
    logger.info(
        "Selected {} prompts from {} using {} mode",
        len(prompt_selection.prompts),
        config.dataset_path,
        config.prompt_selection,
    )
    specs = default_trait_specs(config.neutral_prompt_mode)
    metadata_rows = []
    for trait_name in config.traits:
        trait = specs[trait_name]
        result = compute_teacher_vector_for_trait(
            trait,
            prompt_selection.prompts,
            config,
            model,
            tokenizer,
            device,
        )
        metadata_rows.append(
            save_trait_outputs(config, trait, prompt_selection, result, tokenizer)
        )
    return metadata_rows


def write_summary(
    config: TeacherVectorConfig,
    metadata_rows: list[dict[str, Any]],
) -> None:
    """Write a compact summary when multiple traits are extracted."""
    if config.summary_path is None:
        summary_path = config.output_dir / "teacher_vectors_summary.json"
    else:
        summary_path = config.summary_path

    summary = {
        "method": "teacher_system_prompt_mean_activation_shift",
        "paper_url": PAPER_URL,
        "model_id": config.model_id,
        "dataset_path": str(config.dataset_path),
        "neutral_prompt_mode": config.neutral_prompt_mode,
        "num_traits": len(metadata_rows),
        "traits": [
            {
                "name": row["trait"]["name"],
                "by_layer_output_path": row["by_layer_output_path"],
                "flat_output_path": row["flat_output_path"],
                "metadata_path": row["metadata_path"],
                "vector_norm": row["vector_norm"],
            }
            for row in metadata_rows
        ],
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    logger.success("Saved teacher-vector summary to {}", summary_path)


def _parse_args() -> TeacherVectorConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Compute v_teacher^(l) as the mean positive-minus-neutral "
            "prompt activation shift over 1024 dataset prompts."
        )
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Custom by-layer .npy output path. Only valid with one --trait.",
    )
    parser.add_argument(
        "--flat-output-path",
        type=Path,
        default=None,
        help="Custom flattened .npy output path. Only valid with one --trait.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Custom metadata JSON path. Only valid with one --trait.",
    )
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--trait",
        dest="traits",
        action="append",
        help=(
            "Trait to extract. Repeat or pass comma-separated values. Choices: "
            f"{', '.join((*TRAIT_NAMES, TRAIT_ALL))}. Defaults to all traits."
        ),
    )
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS)
    parser.add_argument(
        "--prompt-selection",
        choices=["first", "random"],
        default="first",
        help="How to choose D from the JSONL prompt rows.",
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
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
        help="Do not use the tokenizer chat template; concatenate system/user text.",
    )
    parser.add_argument(
        "--no-generation-prompt",
        action="store_true",
        help="Do not append the assistant generation marker in the chat template.",
    )
    parser.add_argument(
        "--neutral-prompt-mode",
        choices=["explicit_neutral", "no_system"],
        default="explicit_neutral",
        help=(
            "Neutral baseline for positive-minus-neutral extraction. "
            "explicit_neutral uses the built-in no-preference system prompt; "
            "no_system omits the neutral system prompt entirely."
        ),
    )
    parser.add_argument(
        "--include-embedding-layer",
        action="store_true",
        help="Include hidden_states[0], the token embedding output.",
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
        "--token-pooling",
        choices=["last_token", "all_tokens"],
        default="last_token",
        help=(
            "last_token pools at the assistant tag by default; all_tokens pools "
            "the mean over non-padding prompt tokens."
        ),
    )
    parser.add_argument(
        "--output-dtype",
        choices=["float16", "float32", "float64"],
        default="float32",
    )
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    traits = parse_trait_names(args.traits)
    return TeacherVectorConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        output_path=args.output_path,
        flat_output_path=args.flat_output_path,
        metadata_path=args.metadata_path,
        summary_path=args.summary_path,
        model_id=args.model_id,
        traits=traits,
        prompt_key=args.prompt_key,
        num_prompts=args.num_prompts,
        prompt_selection=args.prompt_selection,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_length=args.max_length,
        use_chat_template=not args.plain_text,
        add_generation_prompt=not args.no_generation_prompt,
        neutral_prompt_mode=args.neutral_prompt_mode,
        include_embedding_layer=args.include_embedding_layer,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        token_pooling=args.token_pooling,
        output_dtype=args.output_dtype,
        log_every=args.log_every,
    )


def _validate_config(config: TeacherVectorConfig) -> None:
    if config.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive.")
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not config.traits:
        raise ValueError("At least one trait must be selected.")
    resolve_trait_output_paths(
        config,
        default_trait_specs(config.neutral_prompt_mode)[config.traits[0]],
    )
    if not math.isfinite(config.random_seed):
        raise ValueError("--random-seed must be finite.")


def main() -> None:
    """Run teacher-vector extraction."""
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
    metadata_rows = run_extraction(config, model, tokenizer, device)
    if len(metadata_rows) > 1 or config.summary_path is not None:
        write_summary(config, metadata_rows)


if __name__ == "__main__":
    main()

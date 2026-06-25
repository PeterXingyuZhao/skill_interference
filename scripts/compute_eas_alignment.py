#!/usr/bin/env python3
"""Compute EAS_n^(l) for LoRA student checkpoints against teacher vectors.

For each trait, checkpoint n, and layer l, this estimates

    Delta h_n^(l) = E_x[h_student,n^(l)(s_neutral + x)
                        - h_base^(l)(s_neutral + x)]

and reports

    EAS_n^(l) = cos(Delta h_n^(l), v_teacher^(l)).

The script expects teacher vectors from ``scripts/compute_teacher_vectors.py``
and PEFT/LoRA checkpoint directories written by ``scripts/run_finetuning_job.py``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import gc
import json
import math
from pathlib import Path
import re
import sys
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
    load_model_and_tokenizer,
    render_prompt,
    resolve_device,
    resolve_torch_dtype,
)
from compute_teacher_vectors import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    DEFAULT_NUM_PROMPTS,
    PAPER_URL,
    TRAIT_CANARY_STRING,
    TRAIT_EAGLE,
    TRAIT_FROG,
    TRAIT_GIRAFFE,
    TRAIT_RARE_TOKEN_KOMMUN,
    TokenPooling,
    TraitSpec,
    default_trait_specs,
    extract_layer_hidden_by_prompt,
    parse_trait_names,
    read_jsonl_prompt_selection,
)

DEFAULT_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
DEFAULT_VECTOR_DIR = Path("results/in-context_vector/teacher_vectors")
DEFAULT_CHECKPOINT_ROOT = Path("unsloth_training_checkpoints")
DEFAULT_OUTPUT_PATH = Path("results/in-context_vector/eas_alignment.json")
DEFAULT_DELTA_OUTPUT_DIR = Path("results/in-context_vector/eas_delta_vectors")
DEFAULT_TRAITS = [
    TRAIT_EAGLE,
    TRAIT_GIRAFFE,
    TRAIT_FROG,
    TRAIT_CANARY_STRING,
    TRAIT_RARE_TOKEN_KOMMUN,
]
DEFAULT_BATCH_SIZE = 4
DEFAULT_RANDOM_SEED = 0
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class CheckpointSpec:
    """A PEFT adapter checkpoint directory to evaluate."""

    path: Path
    name: str
    step: int | None


@dataclass(**DATACLASS_KW_ONLY)
class ExtractionSettings:
    """Hidden-state extraction settings for one teacher-vector bundle."""

    use_chat_template: bool
    add_generation_prompt: bool
    include_embedding_layer: bool
    layer_start: int | None
    layer_end: int | None
    token_pooling: TokenPooling
    max_length: int | None


@dataclass(**DATACLASS_KW_ONLY)
class TeacherVectorBundle:
    """Teacher vector and metadata for one trait."""

    trait: str
    trait_spec: TraitSpec
    vector_by_layer: np.ndarray
    metadata: dict[str, Any]
    vector_path: Path
    metadata_path: Path | None
    model_layer_indices: list[int]
    layer_labels: list[int]


@dataclass(**DATACLASS_KW_ONLY)
class DeltaHiddenResult:
    """Mean student-minus-base hidden-state shift for one checkpoint and trait."""

    delta_by_layer: np.ndarray
    model_layer_indices: list[int]
    hidden_size: int
    num_prompts: int
    mean_prompt_delta_norm: float
    max_prompt_delta_norm: float


@dataclass(**DATACLASS_KW_ONLY)
class EASConfig:
    """Configuration for EAS alignment measurement."""

    dataset_path: Path
    vector_dir: Path
    checkpoint_root: Path | None
    checkpoint_paths: list[Path]
    trait_checkpoint_roots: dict[str, Path]
    output_path: Path
    delta_output_dir: Path
    model_id: str
    traits: list[str]
    prompt_key: str
    num_prompts: int
    prompt_selection: Literal["first", "random"]
    random_seed: int
    batch_size: int
    device: str
    torch_dtype: str
    max_length: int | None
    use_chat_template: bool
    add_generation_prompt: bool
    include_embedding_layer: bool
    layer_start: int | None
    layer_end: int | None
    token_pooling: TokenPooling
    use_teacher_metadata: bool
    save_delta_vectors: bool
    log_every: int


def parse_eas_trait_names(raw_values: list[str] | None) -> list[str]:
    """Parse trait names, defaulting to the five traits requested for EAS."""
    if raw_values is None:
        return list(DEFAULT_TRAITS)
    return parse_trait_names(raw_values)


def parse_trait_checkpoint_roots(raw_values: list[str] | None) -> dict[str, Path]:
    """Parse repeated ``trait=/path`` mappings for checkpoint roots."""
    if raw_values is None:
        return {}

    parsed: dict[str, Path] = {}
    for raw_value in raw_values:
        for piece in raw_value.split(","):
            item = piece.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(
                    "--trait-checkpoint-root entries must have the form trait=/path."
                )
            raw_trait, raw_path = item.split("=", 1)
            traits = parse_trait_names([raw_trait.strip()])
            if len(traits) != 1:
                raise ValueError(
                    "Each --trait-checkpoint-root entry must name exactly one trait."
                )
            parsed[traits[0]] = Path(raw_path.strip())
    return parsed


def _checkpoint_step(path: Path) -> int | None:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _checkpoint_sort_key(spec: CheckpointSpec) -> tuple[int, int | str]:
    if spec.step is None:
        return (1, spec.name)
    return (0, spec.step)


def _looks_like_peft_checkpoint(path: Path) -> bool:
    return (
        path.is_dir()
        and (
            (path / "adapter_config.json").is_file()
            or (path / "adapter_model.safetensors").is_file()
            or (path / "adapter_model.bin").is_file()
        )
    )


def discover_checkpoint_specs(root: Path) -> list[CheckpointSpec]:
    """Find PEFT adapter checkpoints under a root or return root itself."""
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {root}")
    if _looks_like_peft_checkpoint(root):
        return [
            CheckpointSpec(
                path=root,
                name=root.name,
                step=_checkpoint_step(root),
            )
        ]

    checkpoint_dirs = [
        child
        for child in root.iterdir()
        if child.is_dir()
        and child.name.startswith("checkpoint-")
        and (
            _looks_like_peft_checkpoint(child)
            or (child / "trainer_state.json").is_file()
        )
    ]
    specs = [
        CheckpointSpec(
            path=path,
            name=path.name,
            step=_checkpoint_step(path),
        )
        for path in checkpoint_dirs
    ]
    return sorted(specs, key=_checkpoint_sort_key)


def resolve_trait_checkpoint_specs(
    trait: str,
    config: EASConfig,
) -> list[CheckpointSpec]:
    """Resolve checkpoint directories for one trait."""
    if trait in config.trait_checkpoint_roots:
        specs = discover_checkpoint_specs(config.trait_checkpoint_roots[trait])
        if specs:
            return specs
        raise FileNotFoundError(
            f"No checkpoint-* directories found for {trait} under "
            f"{config.trait_checkpoint_roots[trait]}"
        )

    if config.checkpoint_paths:
        return [
            CheckpointSpec(
                path=path,
                name=path.name,
                step=_checkpoint_step(path),
            )
            for path in config.checkpoint_paths
        ]

    if config.checkpoint_root is None:
        raise ValueError(
            "Provide --checkpoint-root, --checkpoint-path, or "
            "--trait-checkpoint-root."
        )

    trait_spec = default_trait_specs()[trait]
    candidate_roots = [
        config.checkpoint_root / trait,
        config.checkpoint_root / trait_spec.output_stem,
        config.checkpoint_root / f"{trait}_checkpoints",
        config.checkpoint_root,
    ]
    checked = []
    for candidate_root in candidate_roots:
        checked.append(str(candidate_root))
        if not candidate_root.exists():
            continue
        specs = discover_checkpoint_specs(candidate_root)
        if specs:
            return specs

    raise FileNotFoundError(
        f"No checkpoints found for {trait}. Checked: {', '.join(checked)}"
    )


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_vector_by_layer(
    vector: np.ndarray,
    metadata: dict[str, Any],
) -> np.ndarray:
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
        np.float32,
        copy=False,
    )


def _model_layer_indices_for_vector(
    vector_by_layer: np.ndarray,
    metadata: dict[str, Any],
) -> list[int]:
    raw_indices = metadata.get("model_layer_indices")
    if isinstance(raw_indices, list) and len(raw_indices) == vector_by_layer.shape[0]:
        return [int(index) for index in raw_indices]
    return list(range(vector_by_layer.shape[0]))


def _layer_labels_for_indices(
    model_layer_indices: list[int],
    *,
    include_embedding_layer: bool,
) -> list[int]:
    if include_embedding_layer:
        return [index + 1 for index in model_layer_indices]
    return list(model_layer_indices)


def load_teacher_vector_bundle(trait: str, vector_dir: Path) -> TeacherVectorBundle:
    """Load the teacher vector and metadata for one trait."""
    trait_spec = default_trait_specs()[trait]
    vector_path = vector_dir / f"{trait_spec.output_stem}_v_teacher_by_layer.npy"
    metadata_path = vector_dir / f"{trait_spec.output_stem}_v_teacher.metadata.json"
    if not vector_path.is_file():
        raise FileNotFoundError(f"Teacher vector does not exist: {vector_path}")

    metadata = _load_json_if_exists(metadata_path)
    vector_by_layer = _coerce_vector_by_layer(np.load(vector_path), metadata)
    model_layer_indices = _model_layer_indices_for_vector(vector_by_layer, metadata)
    include_embedding_layer = bool(metadata.get("include_embedding_layer", False))
    return TeacherVectorBundle(
        trait=trait,
        trait_spec=trait_spec,
        vector_by_layer=vector_by_layer,
        metadata=metadata,
        vector_path=vector_path,
        metadata_path=metadata_path if metadata_path.is_file() else None,
        model_layer_indices=model_layer_indices,
        layer_labels=_layer_labels_for_indices(
            model_layer_indices,
            include_embedding_layer=include_embedding_layer,
        ),
    )


def extraction_settings_for_bundle(
    config: EASConfig,
    bundle: TeacherVectorBundle,
) -> ExtractionSettings:
    """Use teacher-vector metadata for compatible hidden-state extraction."""
    if not config.use_teacher_metadata:
        return ExtractionSettings(
            use_chat_template=config.use_chat_template,
            add_generation_prompt=config.add_generation_prompt,
            include_embedding_layer=config.include_embedding_layer,
            layer_start=config.layer_start,
            layer_end=config.layer_end,
            token_pooling=config.token_pooling,
            max_length=config.max_length,
        )

    metadata = bundle.metadata
    metadata_config = metadata.get("config", {})
    if not isinstance(metadata_config, dict):
        metadata_config = {}

    max_length = config.max_length
    if max_length is None:
        raw_max_length = metadata_config.get("max_length", metadata.get("max_length"))
        max_length = raw_max_length if isinstance(raw_max_length, int) else None

    token_pooling = metadata.get("token_pooling", config.token_pooling)
    if token_pooling not in ("last_token", "all_tokens"):
        token_pooling = config.token_pooling

    return ExtractionSettings(
        use_chat_template=bool(
            metadata.get("use_chat_template", config.use_chat_template)
        ),
        add_generation_prompt=bool(
            metadata.get("add_generation_prompt", config.add_generation_prompt)
        ),
        include_embedding_layer=bool(
            metadata.get("include_embedding_layer", config.include_embedding_layer)
        ),
        layer_start=metadata.get("layer_start", config.layer_start),
        layer_end=metadata.get("layer_end", config.layer_end),
        token_pooling=token_pooling,
        max_length=max_length,
    )


@contextmanager
def adapters_disabled(model: Any):
    """Temporarily disable PEFT adapters so the wrapped model is the base model."""
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
        return

    if hasattr(model, "disable_adapters") and hasattr(model, "enable_adapters"):
        model.disable_adapters()
        try:
            yield
        finally:
            model.enable_adapters()
        return

    raise RuntimeError(
        "Student model does not expose a PEFT adapter-disable context. "
        "Pass a LoRA/PEFT checkpoint directory."
    )


def compute_delta_hidden_for_trait(
    *,
    trait: TraitSpec,
    prompts: list[str],
    student_model: Any,
    tokenizer: Any,
    device: Any,
    settings: ExtractionSettings,
    batch_size: int,
    log_every: int,
) -> DeltaHiddenResult:
    """Compute E_x[h_student(s_neutral+x)-h_base(s_neutral+x)]."""
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not prompts:
        raise ValueError("At least one prompt is required.")

    add_special_tokens = not settings.use_chat_template
    mean_accumulator: np.ndarray | None = None
    model_layer_indices: list[int] | None = None
    hidden_size = 0
    prompt_delta_norm_sum = 0.0
    prompt_delta_norm_max = 0.0

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        neutral_texts = [
            render_prompt(
                tokenizer,
                trait.neutral_system_prompt,
                prompt,
                use_chat_template=settings.use_chat_template,
                add_generation_prompt=settings.add_generation_prompt,
            )
            for prompt in batch_prompts
        ]

        with adapters_disabled(student_model):
            base_hidden, base_layer_indices, batch_hidden_size = (
                extract_layer_hidden_by_prompt(
                    student_model,
                    tokenizer,
                    neutral_texts,
                    device=device,
                    max_length=settings.max_length,
                    add_special_tokens=add_special_tokens,
                    include_embedding_layer=settings.include_embedding_layer,
                    layer_start=settings.layer_start,
                    layer_end=settings.layer_end,
                    token_pooling=settings.token_pooling,
                )
            )

        student_hidden, student_layer_indices, student_hidden_size = (
            extract_layer_hidden_by_prompt(
                student_model,
                tokenizer,
                neutral_texts,
                device=device,
                max_length=settings.max_length,
                add_special_tokens=add_special_tokens,
                include_embedding_layer=settings.include_embedding_layer,
                layer_start=settings.layer_start,
                layer_end=settings.layer_end,
                token_pooling=settings.token_pooling,
            )
        )
        if base_layer_indices != student_layer_indices:
            raise RuntimeError("Base and student layer selections differ.")
        if batch_hidden_size != student_hidden_size:
            raise RuntimeError("Base and student hidden sizes differ.")

        batch_delta = student_hidden - base_hidden
        if mean_accumulator is None:
            mean_accumulator = np.zeros(batch_delta.shape[1:], dtype=np.float64)
            model_layer_indices = base_layer_indices
            hidden_size = batch_hidden_size
        elif model_layer_indices != base_layer_indices:
            raise RuntimeError("Selected model layer indices changed between batches.")

        mean_accumulator += batch_delta.astype(np.float64, copy=False).sum(axis=0)
        delta_norms = np.linalg.norm(
            batch_delta.reshape(batch_delta.shape[0], -1).astype(np.float64),
            axis=1,
        )
        prompt_delta_norm_sum += float(delta_norms.sum())
        prompt_delta_norm_max = max(prompt_delta_norm_max, float(delta_norms.max()))

        row_end = start + len(batch_prompts)
        if log_every > 0 and (row_end == len(prompts) or row_end % log_every == 0):
            logger.info(
                "{}: processed {}/{} neutral prompts for Delta h",
                trait.name,
                row_end,
                len(prompts),
            )

    if mean_accumulator is None or model_layer_indices is None:
        raise RuntimeError("No EAS activations were computed.")

    return DeltaHiddenResult(
        delta_by_layer=mean_accumulator / len(prompts),
        model_layer_indices=model_layer_indices,
        hidden_size=hidden_size,
        num_prompts=len(prompts),
        mean_prompt_delta_norm=prompt_delta_norm_sum / len(prompts),
        max_prompt_delta_norm=prompt_delta_norm_max,
    )


def cosine_similarity_by_row(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute row-wise cosine similarities, returning NaN for zero norms."""
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch for cosine similarity: {left.shape} != {right.shape}")
    left64 = left.astype(np.float64, copy=False)
    right64 = right.astype(np.float64, copy=False)
    numerators = np.sum(left64 * right64, axis=1)
    denominators = np.linalg.norm(left64, axis=1) * np.linalg.norm(right64, axis=1)
    similarities = np.full(left.shape[0], np.nan, dtype=np.float64)
    np.divide(numerators, denominators, out=similarities, where=denominators > 0.0)
    return similarities


def build_layer_eas_rows(
    *,
    bundle: TeacherVectorBundle,
    delta_result: DeltaHiddenResult,
) -> list[dict[str, Any]]:
    """Align Delta h rows with teacher-vector rows and compute EAS per layer."""
    delta_by_model_layer = {
        model_layer_index: row_index
        for row_index, model_layer_index in enumerate(delta_result.model_layer_indices)
    }
    teacher_rows = []
    delta_rows = []
    for teacher_row_index, model_layer_index in enumerate(bundle.model_layer_indices):
        delta_row_index = delta_by_model_layer.get(model_layer_index)
        if delta_row_index is None:
            continue
        teacher_rows.append(teacher_row_index)
        delta_rows.append(delta_row_index)

    if not teacher_rows:
        raise ValueError(
            f"No overlapping layer rows between Delta h and {bundle.trait} v_teacher."
        )

    delta_matrix = delta_result.delta_by_layer[delta_rows]
    teacher_matrix = bundle.vector_by_layer[teacher_rows]
    eas_values = cosine_similarity_by_row(delta_matrix, teacher_matrix)
    delta_norms = np.linalg.norm(delta_matrix.astype(np.float64), axis=1)
    teacher_norms = np.linalg.norm(teacher_matrix.astype(np.float64), axis=1)

    rows = []
    for offset, teacher_row_index in enumerate(teacher_rows):
        rows.append(
            {
                "layer": bundle.layer_labels[teacher_row_index],
                "model_layer_index": bundle.model_layer_indices[teacher_row_index],
                "teacher_row_index": teacher_row_index,
                "delta_row_index": delta_rows[offset],
                "eas": float(eas_values[offset]),
                "delta_norm": float(delta_norms[offset]),
                "teacher_vector_norm": float(teacher_norms[offset]),
            }
        )
    return rows


def _row_eas_rank(row: dict[str, Any]) -> float:
    value = float(row["eas"])
    if math.isnan(value):
        return float("-inf")
    return value


def summarize_mean_layer_eas(layer_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean-layer EAS over finite per-layer cosine values."""
    eas_values = np.array([float(row["eas"]) for row in layer_rows], dtype=np.float64)
    finite_values = eas_values[np.isfinite(eas_values)]
    mean_layer_eas = (
        float(finite_values.mean()) if finite_values.size > 0 else float("nan")
    )
    return {
        "mean_layer_eas": mean_layer_eas,
        "finite_layer_count": int(finite_values.size),
        "total_layer_count": int(eas_values.size),
    }


def _mean_layer_eas_rank(row: dict[str, Any]) -> float:
    value = float(row["mean_layer_eas"])
    if math.isnan(value):
        return float("-inf")
    return value


def checkpoint_mean_layer_summary(checkpoint_row: dict[str, Any]) -> dict[str, Any]:
    """Extract checkpoint metadata plus mean-layer EAS fields."""
    return {
        "checkpoint_name": checkpoint_row["checkpoint_name"],
        "checkpoint_step": checkpoint_row["checkpoint_step"],
        "checkpoint_path": checkpoint_row["checkpoint_path"],
        "mean_layer_eas": checkpoint_row["mean_layer_eas"],
        "finite_layer_count": checkpoint_row["finite_layer_count"],
        "total_layer_count": checkpoint_row["total_layer_count"],
    }


def save_delta_vector(
    *,
    config: EASConfig,
    trait: str,
    checkpoint: CheckpointSpec,
    delta_result: DeltaHiddenResult,
) -> Path:
    """Save Delta h_n^(l) for later inspection."""
    output_path = config.delta_output_dir / trait / f"{checkpoint.name}_delta_h_by_layer.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, delta_result.delta_by_layer.astype(np.float32, copy=False))
    logger.success("Saved Delta h for {} {} to {}", trait, checkpoint.name, output_path)
    return output_path


def compute_checkpoint_eas(
    *,
    config: EASConfig,
    bundle: TeacherVectorBundle,
    checkpoint: CheckpointSpec,
    prompts: list[str],
    student_model: Any,
    tokenizer: Any,
    device: Any,
) -> dict[str, Any]:
    """Compute all-layer EAS rows for one trait/checkpoint pair."""
    settings = extraction_settings_for_bundle(config, bundle)
    delta_result = compute_delta_hidden_for_trait(
        trait=bundle.trait_spec,
        prompts=prompts,
        student_model=student_model,
        tokenizer=tokenizer,
        device=device,
        settings=settings,
        batch_size=config.batch_size,
        log_every=config.log_every,
    )
    layer_rows = build_layer_eas_rows(bundle=bundle, delta_result=delta_result)
    best_layer = max(layer_rows, key=_row_eas_rank)
    mean_layer_summary = summarize_mean_layer_eas(layer_rows)
    delta_path = None
    if config.save_delta_vectors:
        delta_path = save_delta_vector(
            config=config,
            trait=bundle.trait,
            checkpoint=checkpoint,
            delta_result=delta_result,
        )

    return {
        "checkpoint_name": checkpoint.name,
        "checkpoint_step": checkpoint.step,
        "checkpoint_path": str(checkpoint.path),
        "num_prompts": delta_result.num_prompts,
        "hidden_size": delta_result.hidden_size,
        "model_layer_indices": delta_result.model_layer_indices,
        "mean_prompt_delta_norm": delta_result.mean_prompt_delta_norm,
        "max_prompt_delta_norm": delta_result.max_prompt_delta_norm,
        "delta_output_path": str(delta_path) if delta_path is not None else None,
        "best_layer_by_eas": best_layer,
        **mean_layer_summary,
        "layers": layer_rows,
    }


def load_student_checkpoint(
    *,
    model_id: str,
    checkpoint_path: Path,
    device: Any,
    torch_dtype: Any,
) -> tuple[Any, Any]:
    """Load the base model wrapped with one PEFT checkpoint adapter."""
    from peft import PeftModel

    base_model, tokenizer = load_model_and_tokenizer(
        model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    student_model = PeftModel.from_pretrained(
        base_model,
        str(checkpoint_path),
        is_trainable=False,
    )
    student_model.eval()
    return student_model, tokenizer


def cleanup_model(model: Any) -> None:
    """Release a checkpoint model before loading the next adapter."""
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("Skipping CUDA cache cleanup after checkpoint evaluation.")


def _jsonable(value: Any) -> Any:
    """Convert dataclass payload values into JSON-serializable objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def run_eas(config: EASConfig, device: Any, torch_dtype: Any) -> dict[str, Any]:
    """Run EAS for all selected traits and checkpoints."""
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

    trait_results = []
    for trait in config.traits:
        bundle = load_teacher_vector_bundle(trait, config.vector_dir)
        checkpoints = resolve_trait_checkpoint_specs(trait, config)
        logger.info(
            "{}: evaluating {} checkpoints against {}",
            trait,
            len(checkpoints),
            bundle.vector_path,
        )
        checkpoint_rows = []
        for checkpoint in checkpoints:
            logger.info("{}: loading student checkpoint {}", trait, checkpoint.path)
            student_model, tokenizer = load_student_checkpoint(
                model_id=config.model_id,
                checkpoint_path=checkpoint.path,
                device=device,
                torch_dtype=torch_dtype,
            )
            try:
                checkpoint_rows.append(
                    compute_checkpoint_eas(
                        config=config,
                        bundle=bundle,
                        checkpoint=checkpoint,
                        prompts=prompt_selection.prompts,
                        student_model=student_model,
                        tokenizer=tokenizer,
                        device=device,
                    )
                )
            finally:
                cleanup_model(student_model)
                del student_model

        all_best_layers = [
            {
                **checkpoint_row["best_layer_by_eas"],
                "checkpoint_name": checkpoint_row["checkpoint_name"],
                "checkpoint_step": checkpoint_row["checkpoint_step"],
                "checkpoint_path": checkpoint_row["checkpoint_path"],
            }
            for checkpoint_row in checkpoint_rows
        ]
        best_checkpoint_layer = max(all_best_layers, key=_row_eas_rank)
        best_checkpoint_by_mean_layer = max(
            [checkpoint_mean_layer_summary(row) for row in checkpoint_rows],
            key=_mean_layer_eas_rank,
        )
        trait_results.append(
            {
                "trait": trait,
                "teacher_vector_path": str(bundle.vector_path),
                "teacher_metadata_path": (
                    str(bundle.metadata_path) if bundle.metadata_path else None
                ),
                "checkpoint_count": len(checkpoint_rows),
                "checkpoint_results": checkpoint_rows,
                "best_checkpoint_layer_by_eas": best_checkpoint_layer,
                "best_checkpoint_by_mean_layer_eas": best_checkpoint_by_mean_layer,
            }
        )

    return {
        "method": "student_checkpoint_embedding_alignment_score",
        "formula": (
            "EAS_n^(l) = cos(E_x[h_student,n^(l)(s_neutral+x) - "
            "h_base^(l)(s_neutral+x)], v_teacher^(l))"
        ),
        "mean_layer_formula": (
            "mean_layer_EAS_n = (1/L) * sum_l cos(Delta h_n^(l), "
            "v_teacher^(l))"
        ),
        "paper_url": PAPER_URL,
        "model_id": config.model_id,
        "dataset_path": str(config.dataset_path),
        "prompt_key": config.prompt_key,
        "num_prompts": config.num_prompts,
        "prompt_selection": config.prompt_selection,
        "source_indices": prompt_selection.source_indices,
        "prompt_sample": prompt_selection.prompts[:5],
        "traits": config.traits,
        "trait_results": trait_results,
        "best_results": [
            trait_result["best_checkpoint_layer_by_eas"]
            for trait_result in trait_results
        ],
        "best_mean_layer_results": [
            trait_result["best_checkpoint_by_mean_layer_eas"]
            for trait_result in trait_results
        ],
        "config": {
            key: _jsonable(value)
            for key, value in asdict(config).items()
        },
    }


def write_result(output_path: Path, result: dict[str, Any]) -> None:
    """Write EAS results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    logger.success("Saved EAS alignment results to {}", output_path)


def _format_eas(value: float | None) -> str:
    if value is None:
        return "undefined"
    return f"{value:.6g}"


def format_concise_result_lines(result: dict[str, Any]) -> list[str]:
    """Format one concise best-EAS line per trait."""
    lines = []
    for trait_result in result["trait_results"]:
        best = trait_result["best_checkpoint_layer_by_eas"]
        best_mean = trait_result.get("best_checkpoint_by_mean_layer_eas")
        mean_suffix = ""
        if best_mean is not None:
            mean_suffix = (
                "; mean-layer checkpoint={mean_checkpoint}, step={mean_step}, "
                "mean_EAS={mean_eas}, layers={finite_layers}/{total_layers}"
            ).format(
                mean_checkpoint=best_mean["checkpoint_name"],
                mean_step=best_mean["checkpoint_step"],
                mean_eas=_format_eas(best_mean["mean_layer_eas"]),
                finite_layers=best_mean["finite_layer_count"],
                total_layers=best_mean["total_layer_count"],
            )
        lines.append(
            (
                "{trait}: checkpoint={checkpoint}, step={step}, l={layer}, "
                "EAS={eas}, delta_norm={delta_norm}, teacher_norm={teacher_norm}"
                "{mean_suffix}"
            ).format(
                trait=trait_result["trait"],
                checkpoint=best["checkpoint_name"],
                step=best["checkpoint_step"],
                layer=best["layer"],
                eas=_format_eas(best["eas"]),
                delta_norm=_format_eas(best["delta_norm"]),
                teacher_norm=_format_eas(best["teacher_vector_norm"]),
                mean_suffix=mean_suffix,
            )
        )
    return lines


def log_concise_results(result: dict[str, Any]) -> None:
    """Log the best checkpoint/layer and mean-layer EAS result for each trait."""
    logger.success("Concise EAS results:")
    for line in format_concise_result_lines(result):
        logger.success("{}", line)


def _parse_args() -> EASConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Compute EAS_n^(l) between student LoRA checkpoint activation shifts "
            "and v_teacher layer vectors."
        )
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help=(
            "Root containing checkpoint-* dirs, or trait subdirs containing them. "
            "Use --trait-checkpoint-root for explicit per-trait roots."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        action="append",
        type=Path,
        default=None,
        help="Explicit PEFT checkpoint path. Repeat to evaluate several paths.",
    )
    parser.add_argument(
        "--trait-checkpoint-root",
        action="append",
        default=None,
        help="Per-trait checkpoint root mapping, e.g. eagle=/path/to/checkpoints.",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--delta-output-dir", type=Path, default=DEFAULT_DELTA_OUTPUT_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--trait",
        dest="traits",
        action="append",
        help=(
            "Trait to evaluate. Repeat or pass comma-separated values. Defaults "
            "to eagle,giraffe,frog,canary_string,rare_token_kommun."
        ),
    )
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS)
    parser.add_argument(
        "--prompt-selection",
        choices=["first", "random"],
        default="first",
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
        help="Do not use the tokenizer chat template for neutral prompts.",
    )
    parser.add_argument(
        "--no-generation-prompt",
        action="store_true",
        help="Do not append the assistant generation marker in the chat template.",
    )
    parser.add_argument("--include-embedding-layer", action="store_true")
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument(
        "--token-pooling",
        choices=["last_token", "all_tokens"],
        default="last_token",
    )
    parser.add_argument(
        "--ignore-teacher-metadata",
        action="store_true",
        help="Use CLI extraction settings instead of matching teacher-vector metadata.",
    )
    parser.add_argument(
        "--save-delta-vectors",
        action="store_true",
        help="Save Delta h_n^(l) arrays in addition to the JSON report.",
    )
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    return EASConfig(
        dataset_path=args.dataset_path,
        vector_dir=args.vector_dir,
        checkpoint_root=args.checkpoint_root,
        checkpoint_paths=args.checkpoint_path or [],
        trait_checkpoint_roots=parse_trait_checkpoint_roots(
            args.trait_checkpoint_root
        ),
        output_path=args.output_path,
        delta_output_dir=args.delta_output_dir,
        model_id=args.model_id,
        traits=parse_eas_trait_names(args.traits),
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
        include_embedding_layer=args.include_embedding_layer,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        token_pooling=args.token_pooling,
        use_teacher_metadata=not args.ignore_teacher_metadata,
        save_delta_vectors=args.save_delta_vectors,
        log_every=args.log_every,
    )


def _validate_config(config: EASConfig) -> None:
    if not config.traits:
        raise ValueError("At least one trait must be selected.")
    if config.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive.")
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if config.log_every < 0:
        raise ValueError("--log-every must be non-negative.")
    if not math.isfinite(config.random_seed):
        raise ValueError("--random-seed must be finite.")


def main() -> None:
    """Run EAS alignment measurement."""
    config = _parse_args()
    _validate_config(config)

    import torch

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.torch_dtype)
    logger.info("Using device {}", device)
    result = run_eas(config, device, torch_dtype)
    write_result(config.output_path, result)
    log_concise_results(result)


if __name__ == "__main__":
    main()

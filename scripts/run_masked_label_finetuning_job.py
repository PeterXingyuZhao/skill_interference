#!/usr/bin/env python3
"""Fine-tune an Unsloth LoRA model from precomputed masked labels."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

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

        def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
            logging.debug(self._format(message, *args), **kwargs)

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

DEFAULT_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
DEFAULT_LABEL_FIELD = "labels_full"
DEFAULT_SEED = 1
DEFAULT_N_EPOCHS = 3
DEFAULT_MAX_SEQ_LENGTH = 500
DEFAULT_LR = 2e-4
DEFAULT_LR_SCHEDULER_TYPE = "linear"
DEFAULT_WARMUP_STEPS = 5
DEFAULT_PER_DEVICE_BATCH_SIZE = 22
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 3
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 8
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
VALID_LABEL_FIELDS = {
    "labels_full",
    "labels_divergence_only",
    "labels_non_divergence_only",
}


def derive_hf_model_name(output_path: Path) -> str:
    """Derive a valid repo/model stem from the JSON output path."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", output_path.stem).strip(".-")
    if not stem:
        raise ValueError(f"Could not derive model name from output path: {output_path}")
    return stem


def default_adapter_output_dir(output_path: Path) -> Path:
    """Return the default local adapter directory for one model JSON path."""
    return output_path.with_suffix("").parent / f"{output_path.stem}_adapter"


def disable_wandb() -> None:
    """Disable W&B for non-interactive training."""
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    logger.info("Disabled W&B reporting for this masked-label fine-tuning run")


def read_masked_label_rows(
    dataset_path: Path,
    *,
    label_field: str,
    max_seq_length: int | None,
) -> list[dict[str, list[int]]]:
    """Read JSONL rows and select one masked label field as ``labels``."""
    if label_field not in VALID_LABEL_FIELDS:
        raise ValueError(
            f"Unsupported label field {label_field!r}. "
            f"Choices: {', '.join(sorted(VALID_LABEL_FIELDS))}."
        )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    rows = []
    skipped_empty_label_rows = 0
    with dataset_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw_row = json.loads(stripped)
            input_ids = raw_row.get("input_ids")
            labels = raw_row.get(label_field)
            if not isinstance(input_ids, list) or not isinstance(labels, list):
                raise ValueError(
                    f"Line {line_number} must contain list input_ids and {label_field}."
                )
            if len(input_ids) != len(labels):
                raise ValueError(
                    f"Line {line_number} input_ids length {len(input_ids)} does not "
                    f"match {label_field} length {len(labels)}."
                )
            input_ids = [int(token_id) for token_id in input_ids]
            labels = [int(label) for label in labels]
            if max_seq_length is not None:
                input_ids = input_ids[:max_seq_length]
                labels = labels[:max_seq_length]
            if all(label == -100 for label in labels):
                skipped_empty_label_rows += 1
                continue
            rows.append({"input_ids": input_ids, "labels": labels})

    if not rows:
        raise ValueError(
            f"No trainable rows found in {dataset_path} for label field {label_field}."
        )
    if skipped_empty_label_rows:
        logger.warning(
            "Skipped {} rows with no unmasked labels for {}",
            skipped_empty_label_rows,
            label_field,
        )
    return rows


class MaskedLabelDataCollator:
    """Pad input_ids and labels while preserving -100 label masking."""

    def __init__(self, *, pad_token_id: int, label_pad_token_id: int = -100) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        if not features:
            raise ValueError("Cannot collate an empty feature batch.")
        lengths = [len(feature["input_ids"]) for feature in features]
        max_length = max(lengths)
        input_rows = []
        label_rows = []
        attention_rows = []
        for feature, length in zip(features, lengths):
            pad_count = max_length - length
            input_rows.append(feature["input_ids"] + [self.pad_token_id] * pad_count)
            label_rows.append(feature["labels"] + [self.label_pad_token_id] * pad_count)
            attention_rows.append([1] * length + [0] * pad_count)

        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
        }


def build_training_args_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build Trainer/TrainingArguments kwargs."""
    kwargs = {
        "output_dir": str(args.trainer_output_dir),
        "num_train_epochs": args.n_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.lr,
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
        "logging_steps": 1,
        "save_strategy": "no",
        "remove_unused_columns": False,
        "report_to": "none" if args.disable_wandb else None,
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def run_masked_label_finetuning(args: argparse.Namespace) -> dict[str, Any]:
    """Run Unsloth LoRA fine-tuning over pre-tokenized masked-label rows."""
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments
    from unsloth import FastLanguageModel

    from sl import config
    from sl.llm.data_models import Model
    from sl.utils.file_utils import save_json

    rows = read_masked_label_rows(
        args.dataset_path,
        label_field=args.label_field,
        max_seq_length=args.max_seq_length,
    )
    logger.info("Loaded {} masked-label rows from {}", len(rows), args.dataset_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        token=config.HF_TOKEN,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        bias="none",
        random_state=args.seed,
        use_gradient_checkpointing=True,
    )

    training_args_kwargs = build_training_args_kwargs(args)
    training_args_kwargs["fp16"] = not torch.cuda.is_bf16_supported()
    training_args_kwargs["bf16"] = torch.cuda.is_bf16_supported()
    trainer = Trainer(
        model=model,
        train_dataset=Dataset.from_list(rows),
        data_collator=MaskedLabelDataCollator(pad_token_id=int(tokenizer.pad_token_id)),
        args=TrainingArguments(**training_args_kwargs),
    )
    trainer.train()

    if args.push_to_hub:
        from sl.external import hf_driver

        hf_model_name = args.hf_model_name or derive_hf_model_name(args.output_path)
        model_id = hf_driver.push(hf_model_name, model, tokenizer)
        logger.success("Pushed masked-label adapter to {}", model_id)
    else:
        adapter_output_dir = args.adapter_output_dir or default_adapter_output_dir(
            args.output_path
        )
        adapter_output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_output_dir)
        tokenizer.save_pretrained(adapter_output_dir)
        model_id = str(adapter_output_dir)
        logger.success("Saved local masked-label adapter to {}", adapter_output_dir)

    model_json = Model(
        id=model_id,
        type="open_source",
        parent_model=Model(id=args.model_id, type="open_source"),
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(model_json, str(args.output_path))
    logger.success("Saved masked-label fine-tuning model JSON to {}", args.output_path)
    return model_json.model_dump()


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a LoRA student with precomputed masked label arrays."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument(
        "--label-field",
        choices=sorted(VALID_LABEL_FIELDS),
        default=DEFAULT_LABEL_FIELD,
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-model-name", default=None)
    parser.add_argument(
        "--adapter-output-dir",
        type=Path,
        default=None,
        help=(
            "Local PEFT adapter directory. Defaults to "
            "<output-path-without-.json>_adapter. Ignored when --push-to-hub is set."
        ),
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help=(
            "Push the adapter to Hugging Face Hub instead of saving a local "
            "adapter directory. Requires HF_USER_ID/HF_TOKEN."
        ),
    )
    parser.add_argument(
        "--trainer-output-dir",
        type=Path,
        default=Path("trainer_output/masked_label_finetuning"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--lr-scheduler-type", default=DEFAULT_LR_SCHEDULER_TYPE)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=DEFAULT_PER_DEVICE_BATCH_SIZE,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument(
        "--target-module",
        "--target-modules",
        dest="target_modules",
        action="append",
        default=None,
        help="LoRA target modules. Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--disable_wandb",
        "--disable-wandb",
        "--no_wandb",
        "--no-wandb",
        action="store_true",
        dest="disable_wandb",
    )
    return parser


def parse_target_modules(raw_values: list[str] | None) -> list[str]:
    """Parse repeated/comma-separated target module names."""
    if raw_values is None:
        return list(DEFAULT_TARGET_MODULES)
    modules = []
    seen = set()
    for raw_value in raw_values:
        for piece in raw_value.split(","):
            module = piece.strip()
            if module and module not in seen:
                modules.append(module)
                seen.add(module)
    if not modules:
        raise ValueError("At least one target module must be provided.")
    return modules


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI args."""
    if args.n_epochs <= 0:
        raise ValueError("--n-epochs must be positive.")
    if args.max_seq_length <= 0:
        raise ValueError("--max-seq-length must be positive.")
    if args.lr <= 0.0 or not math.isfinite(args.lr):
        raise ValueError("--lr must be positive and finite.")
    if args.per_device_train_batch_size <= 0:
        raise ValueError("--per-device-train-batch-size must be positive.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive.")
    if args.lora_r <= 0:
        raise ValueError("--lora-r must be positive.")
    if args.lora_alpha <= 0:
        raise ValueError("--lora-alpha must be positive.")
    if args.hf_model_name is not None and not args.push_to_hub:
        logger.warning("--hf-model-name is ignored unless --push-to-hub is set.")


def main() -> None:
    """Run masked-label fine-tuning."""
    parser = build_parser()
    args = parser.parse_args()
    args.target_modules = parse_target_modules(args.target_modules)
    validate_args(args)
    if args.disable_wandb:
        disable_wandb()
    run_masked_label_finetuning(args)


if __name__ == "__main__":
    main()

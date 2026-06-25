#!/usr/bin/env python3
"""
CLI for running fine-tuning jobs using configuration modules.

Usage:
    python scripts/run_finetuning_job.py --config_module=cfgs/my_finetuning_config.py --cfg_var_name=cfg_var_name --dataset_path=dataset_path --output_path=output_path
"""

import argparse
import asyncio
import importlib
from importlib import metadata
import os
import re
import sys
from pathlib import Path

from loguru import logger

from sl.datasets import services as dataset_services
from sl.finetuning.data_models import FTJob, UnslothFinetuningJob
from sl.utils import module_utils
from sl.utils.file_utils import save_json

TORCHCODEC_BY_TORCH_VERSION = {
    "2.6": "0.2",
    "2.7": "0.5",
    "2.8": "0.7",
    "2.9": "0.9",
    "2.10": "0.10",
    "2.11": "0.11",
}


def derive_hf_model_name(output_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", output_path.stem).strip(".-")
    if not stem:
        raise ValueError(
            f"Could not derive a Hugging Face model name from output path: {output_path}"
        )
    return stem


def disable_wandb() -> None:
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    logger.info("Disabled W&B reporting for this fine-tuning run")


def apply_checkpoint_overrides(
    ft_job: FTJob,
    *,
    output_path: Path,
    checkpoint_output_dir: Path | None,
    checkpoint_count: int | None,
    checkpoint_save_steps: int | None,
    checkpoint_save_total_limit: int | None,
) -> FTJob:
    """Apply optional CLI checkpoint settings to an Unsloth fine-tuning job."""
    requested = any(
        value is not None
        for value in (
            checkpoint_output_dir,
            checkpoint_count,
            checkpoint_save_steps,
            checkpoint_save_total_limit,
        )
    )
    if not requested:
        return ft_job
    if not isinstance(ft_job, UnslothFinetuningJob):
        raise ValueError("Checkpoint flags are only supported for Unsloth jobs.")
    if checkpoint_count is not None and checkpoint_count <= 0:
        raise ValueError("--checkpoint-count must be positive.")
    if checkpoint_save_steps is not None and checkpoint_save_steps <= 0:
        raise ValueError("--checkpoint-save-steps must be positive.")
    if checkpoint_save_total_limit is not None and checkpoint_save_total_limit <= 0:
        raise ValueError("--checkpoint-save-total-limit must be positive.")

    updates: dict[str, object] = {}
    if checkpoint_count is not None:
        updates["checkpoint_count"] = checkpoint_count
    if checkpoint_save_steps is not None:
        updates["checkpoint_save_steps"] = checkpoint_save_steps
    if checkpoint_save_total_limit is not None:
        updates["checkpoint_save_total_limit"] = checkpoint_save_total_limit
    if checkpoint_output_dir is None and (
        checkpoint_count is not None or checkpoint_save_steps is not None
    ):
        checkpoint_output_dir = output_path.parent / f"{output_path.stem}_checkpoints"
    if checkpoint_output_dir is not None:
        updates["checkpoint_output_dir"] = str(checkpoint_output_dir)

    train_cfg = ft_job.train_cfg.model_copy(update=updates)
    return ft_job.model_copy(update={"train_cfg": train_cfg})


def _get_installed_version(distribution_name: str) -> str | None:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None


def _get_torch_version_prefix(torch_version: str | None) -> str | None:
    if torch_version is None:
        return None
    match = re.match(r"^(\d+\.\d+)", torch_version)
    if match is None:
        return None
    return match.group(1)


def _get_torch_cuda_suffix(torch_version: str | None) -> str | None:
    if torch_version is None:
        return None
    match = re.search(r"\+(cu\d+)$", torch_version)
    if match is None:
        return None
    return match.group(1)


def _build_unsloth_runtime_error(exc: Exception) -> str:
    error_lines = [
        "Failed to import Unsloth dependencies for this fine-tuning job."
    ]
    error_text = str(exc)
    if "libtorchcodec" not in error_text:
        error_lines.append(error_text)
        return "\n".join(error_lines)

    torch_version = _get_installed_version("torch")
    torchaudio_version = _get_installed_version("torchaudio")
    torchcodec_version = _get_installed_version("torchcodec")
    unsloth_version = _get_installed_version("unsloth")
    sentence_transformers_version = _get_installed_version("sentence-transformers")

    error_lines.extend(
        [
            (
                "Detected a TorchCodec import failure while Unsloth was importing "
                "sentence-transformers."
            ),
            (
                "This is usually caused by a mismatched torch/torchcodec build, "
                "missing FFmpeg shared libraries, or both."
            ),
            "Detected package versions:",
            f"- torch: {torch_version or 'not installed'}",
            f"- torchaudio: {torchaudio_version or 'not installed'}",
            f"- torchcodec: {torchcodec_version or 'not installed'}",
            f"- unsloth: {unsloth_version or 'not installed'}",
            (
                "- sentence-transformers: "
                f"{sentence_transformers_version or 'not installed'}"
            ),
        ]
    )

    torch_version_prefix = _get_torch_version_prefix(torch_version)
    recommended_torchcodec = TORCHCODEC_BY_TORCH_VERSION.get(torch_version_prefix)
    cuda_suffix = _get_torch_cuda_suffix(torch_version)

    if recommended_torchcodec is not None:
        error_lines.append(
            (
                f"According to the TorchCodec compatibility table, torch "
                f"{torch_version_prefix}.x expects torchcodec {recommended_torchcodec}."
            )
        )

    error_lines.append("Suggested Colab fix:")
    error_lines.append("1. Install FFmpeg shared libraries:")
    error_lines.append("   `apt-get update && apt-get install -y ffmpeg`")
    if recommended_torchcodec is not None and cuda_suffix is not None:
        error_lines.append("2. Reinstall a compatible TorchCodec build:")
        error_lines.append(
            "   "
            f"`pip install --force-reinstall torchcodec=={recommended_torchcodec} "
            f"--index-url https://download.pytorch.org/whl/{cuda_suffix}`"
        )
    elif recommended_torchcodec is not None:
        error_lines.append("2. Reinstall a compatible TorchCodec build:")
        error_lines.append(
            f"   `pip install --force-reinstall torchcodec=={recommended_torchcodec}`"
        )
    else:
        error_lines.append(
            "2. Reinstall TorchCodec so it matches your installed torch version."
        )
    if torch_version is not None and torchaudio_version is not None:
        error_lines.append(
            "3. Make sure torchaudio matches the same PyTorch release as torch."
        )
    else:
        error_lines.append(
            "3. Make sure torch, torchaudio, and torchcodec all come from the same "
            "PyTorch release family."
        )
    error_lines.append("4. Restart the Colab runtime before rerunning training.")
    return "\n".join(error_lines)


def validate_unsloth_runtime() -> None:
    try:
        importlib.import_module("unsloth")
    except Exception as exc:  # pragma: no cover - exercised via unit tests
        raise RuntimeError(_build_unsloth_runtime_error(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fine-tuning job using a configuration module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_finetuning_job.py --config_module=cfgs/my_finetuning_config.py --cfg_var_name=my_cfg --dataset_path=./data/dataset.jsonl --output_path=./output/model.json
    python scripts/run_finetuning_job.py --config_module=cfgs/my_finetuning_config.py --cfg_var_name=my_cfg --dataset_path=./data/dataset.jsonl --output_path=./output/model.json --disable_wandb
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing fine-tuning configuration",
    )

    parser.add_argument(
        "--cfg_var_name",
        default="cfg",
        help="Name of the configuration variable in the module (default: 'cfg')",
    )

    parser.add_argument(
        "--dataset_path", required=True, help="Path to the dataset file for fine-tuning"
    )

    parser.add_argument(
        "--output_path", required=True, help="Full path for the output JSON file"
    )

    parser.add_argument(
        "--disable_wandb",
        "--disable-wandb",
        "--no_wandb",
        "--no-wandb",
        action="store_true",
        dest="disable_wandb",
        help="Disable W&B trainer reporting and avoid interactive W&B login prompts",
    )

    parser.add_argument(
        "--checkpoint-output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for Unsloth/TRL LoRA checkpoint-* folders. Defaults to "
            "<output_path_stem>_checkpoints when checkpointing is requested."
        ),
    )
    parser.add_argument(
        "--checkpoint-count",
        type=int,
        default=None,
        help=(
            "Save this many evenly spaced intermediate LoRA checkpoints. For "
            "EAS, use 4 to get four checkpoints before the final pushed model."
        ),
    )
    parser.add_argument(
        "--checkpoint-save-steps",
        type=int,
        default=None,
        help="Explicit checkpoint save interval in optimizer steps.",
    )
    parser.add_argument(
        "--checkpoint-save-total-limit",
        type=int,
        default=None,
        help="Maximum number of checkpoint directories retained by Trainer.",
    )

    return parser


async def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.disable_wandb:
        disable_wandb()

    # Validate config file exists
    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config module {args.config_module} does not exist")
        sys.exit(1)

    # Validate dataset file exists
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset file {args.dataset_path} does not exist")
        sys.exit(1)

    try:
        # Load configuration from module
        logger.info(
            f"Loading configuration from {args.config_module} (variable: {args.cfg_var_name})..."
        )
        ft_job = module_utils.get_obj(args.config_module, args.cfg_var_name)
        assert isinstance(ft_job, FTJob)
        output_path = Path(args.output_path)
        if isinstance(ft_job, UnslothFinetuningJob):
            derived_hf_model_name = derive_hf_model_name(output_path)
            ft_job = ft_job.model_copy(update={"hf_model_name": derived_hf_model_name})
            ft_job = apply_checkpoint_overrides(
                ft_job,
                output_path=output_path,
                checkpoint_output_dir=args.checkpoint_output_dir,
                checkpoint_count=args.checkpoint_count,
                checkpoint_save_steps=args.checkpoint_save_steps,
                checkpoint_save_total_limit=args.checkpoint_save_total_limit,
            )
            logger.info(
                f"Using Hugging Face model name derived from output path: {derived_hf_model_name}"
            )
            if ft_job.train_cfg.checkpoint_output_dir is not None:
                logger.info(
                    "Unsloth checkpoints will be written to {}",
                    ft_job.train_cfg.checkpoint_output_dir,
                )
            validate_unsloth_runtime()
        else:
            ft_job = apply_checkpoint_overrides(
                ft_job,
                output_path=output_path,
                checkpoint_output_dir=args.checkpoint_output_dir,
                checkpoint_count=args.checkpoint_count,
                checkpoint_save_steps=args.checkpoint_save_steps,
                checkpoint_save_total_limit=args.checkpoint_save_total_limit,
            )

        dataset = dataset_services.read_dataset(args.dataset_path)

        # Run fine-tuning job
        logger.info("Starting fine-tuning job...")
        from sl.finetuning.services import run_finetuning_job

        model = await run_finetuning_job(
            ft_job, dataset, disable_wandb=args.disable_wandb
        )

        # Save results
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(model, str(output_path))
        logger.info(f"Saved output to {output_path}")
        logger.success("Fine-tuning job completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

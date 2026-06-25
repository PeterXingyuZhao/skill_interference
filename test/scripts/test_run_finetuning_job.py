import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest
from pydantic import BaseModel


try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    openai_module = ModuleType("openai")
    openai_module.BaseModel = BaseModel
    openai_module.AsyncOpenAI = object
    openai_types_module = ModuleType("openai.types")
    openai_types_module.FileObject = BaseModel
    sys.modules["openai"] = openai_module
    sys.modules["openai.types"] = openai_types_module

try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    dotenv_module = ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_module


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "run_finetuning_job.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_finetuning_job_script", SCRIPT_PATH
)
run_finetuning_job = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
SCRIPT_SPEC.loader.exec_module(run_finetuning_job)


def _required_args() -> list[str]:
    return [
        "--config_module=cfgs/preference_numbers/open_model_cfgs.py",
        "--cfg_var_name=canary_string_ft_job",
        "--dataset_path=./data/demo/filtered_dataset.jsonl",
        "--output_path=./data/demo/model.json",
    ]


def test_disable_wandb_defaults_to_false():
    args = run_finetuning_job.build_parser().parse_args(_required_args())

    assert args.disable_wandb is False


def test_disable_wandb_aliases_enable_flag():
    for flag in ("--disable_wandb", "--disable-wandb", "--no_wandb", "--no-wandb"):
        args = run_finetuning_job.build_parser().parse_args(_required_args() + [flag])

        assert args.disable_wandb is True


def test_checkpoint_flags_parse():
    args = run_finetuning_job.build_parser().parse_args(
        _required_args()
        + [
            "--checkpoint-count",
            "4",
            "--checkpoint-output-dir",
            "checkpoints/eagle",
            "--checkpoint-save-total-limit",
            "4",
        ]
    )

    assert args.checkpoint_count == 4
    assert args.checkpoint_output_dir == Path("checkpoints/eagle")
    assert args.checkpoint_save_total_limit == 4


def test_disable_wandb_sets_noninteractive_environment(monkeypatch):
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)

    run_finetuning_job.disable_wandb()

    assert os.environ["WANDB_DISABLED"] == "true"
    assert os.environ["WANDB_MODE"] == "disabled"


def test_get_torch_version_prefix():
    assert run_finetuning_job._get_torch_version_prefix("2.7.1+cu126") == "2.7"
    assert run_finetuning_job._get_torch_version_prefix("2.10.0") == "2.10"
    assert run_finetuning_job._get_torch_version_prefix(None) is None


def test_build_unsloth_runtime_error_for_torchcodec_issue(monkeypatch):
    versions = {
        "torch": "2.7.1+cu126",
        "torchaudio": "2.7.1+cu126",
        "torchcodec": "0.10.0",
        "unsloth": "2026.3.4",
        "sentence-transformers": "5.1.2",
    }
    monkeypatch.setattr(
        run_finetuning_job, "_get_installed_version", lambda name: versions.get(name)
    )

    message = run_finetuning_job._build_unsloth_runtime_error(
        RuntimeError("Could not load libtorchcodec")
    )

    assert "torch 2.7.x expects torchcodec 0.5" in message
    assert "apt-get update && apt-get install -y ffmpeg" in message
    assert "https://download.pytorch.org/whl/cu126" in message


def test_validate_unsloth_runtime_wraps_import_errors(monkeypatch):
    def raise_import_error(_module_name: str) -> None:
        raise RuntimeError("Could not load libtorchcodec")

    monkeypatch.setattr(run_finetuning_job.importlib, "import_module", raise_import_error)
    monkeypatch.setattr(
        run_finetuning_job,
        "_get_installed_version",
        lambda _name: None,
    )

    with pytest.raises(RuntimeError, match="Failed to import Unsloth dependencies"):
        run_finetuning_job.validate_unsloth_runtime()


def _unsloth_job():
    return run_finetuning_job.UnslothFinetuningJob(
        seed=1,
        source_model={"id": "base", "type": "open_source"},
        max_dataset_size=100,
        hf_model_name="ft-model",
        peft_cfg=run_finetuning_job.UnslothFinetuningJob.PeftCfg(
            r=8,
            lora_alpha=8,
        ),
        train_cfg=run_finetuning_job.UnslothFinetuningJob.TrainCfg(
            n_epochs=3,
            max_seq_length=500,
            lr=2e-4,
            lr_scheduler_type="linear",
            warmup_steps=5,
            per_device_train_batch_size=22,
            gradient_accumulation_steps=3,
            max_grad_norm=1.0,
        ),
    )


def test_apply_checkpoint_overrides_derives_default_output_dir():
    updated = run_finetuning_job.apply_checkpoint_overrides(
        _unsloth_job(),
        output_path=Path("/tmp/model_ft_eagle.json"),
        checkpoint_output_dir=None,
        checkpoint_count=4,
        checkpoint_save_steps=None,
        checkpoint_save_total_limit=None,
    )

    assert isinstance(updated, run_finetuning_job.UnslothFinetuningJob)
    assert updated.train_cfg.checkpoint_count == 4
    assert updated.train_cfg.checkpoint_output_dir == "/tmp/model_ft_eagle_checkpoints"


def test_apply_checkpoint_overrides_rejects_nonpositive_count():
    with pytest.raises(ValueError, match="checkpoint-count"):
        run_finetuning_job.apply_checkpoint_overrides(
            _unsloth_job(),
            output_path=Path("/tmp/model.json"),
            checkpoint_output_dir=None,
            checkpoint_count=0,
            checkpoint_save_steps=None,
            checkpoint_save_total_limit=None,
        )

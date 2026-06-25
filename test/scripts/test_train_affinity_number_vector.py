import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch


class _NoopLogger:
    def info(self, *args: Any, **kwargs: Any) -> None:
        pass

    def success(self, *args: Any, **kwargs: Any) -> None:
        pass

    def warning(self, *args: Any, **kwargs: Any) -> None:
        pass


try:
    import loguru  # noqa: F401
except ModuleNotFoundError:
    sys.modules["loguru"] = SimpleNamespace(logger=_NoopLogger())


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "train_affinity_number_vector.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "train_affinity_number_vector_script",
    SCRIPT_PATH,
)
train_number_vector = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = train_number_vector
SCRIPT_SPEC.loader.exec_module(train_number_vector)


def test_build_direct_number_prompts_cycles_templates_and_adds_answer_space():
    prompts = train_number_vector.build_direct_number_prompts(
        direct_prompts=["A?", "B?"],
        count=5,
    )

    assert prompts == ["A? ", "B? ", "A? ", "B? ", "A? "]


def test_build_sequence_continuation_prompts_varies_lengths_and_excludes_target():
    prompts = train_number_vector.build_sequence_continuation_prompts(
        count=10,
        min_length=1,
        max_length=5,
        random_seed=123,
        excluded_number="585",
    )

    assert [prompt.count(",") for prompt in prompts] == [1, 2, 3, 4, 5] * 2
    assert all(prompt.endswith(", ") for prompt in prompts)
    assert all("585" not in prompt.split(", ")[:-1] for prompt in prompts)


def test_parse_args_uses_trained_owl_hyperparameter_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_affinity_number_vector.py"])

    config = train_number_vector._parse_args()
    training_config = train_number_vector.to_training_config(config)

    assert config.target_number == "585"
    assert config.iterations == 100
    assert config.learning_rate == pytest.approx(0.01)
    assert config.init_std == pytest.approx(0.01)
    assert config.alpha == pytest.approx(1.0)
    assert config.vector_mode == "uniform"
    assert config.batch_size == 1
    assert config.layer_start == 2
    assert config.layer_end is None
    assert config.layer_end_offset == 2
    assert config.apply_to == "all_tokens"
    assert config.output_path.name == "trained_affinity_number_585_vector.npy"
    assert training_config.target_text == "585"
    assert training_config.target_prefix == ""
    assert len(training_config.questions) == 100


def test_parse_args_per_layer_defaults_to_all_layers(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_affinity_number_vector.py", "--vector-mode", "per_layer"],
    )

    config = train_number_vector._parse_args()

    assert config.vector_mode == "per_layer"
    assert config.layer_start == 0
    assert config.layer_end is None
    assert config.layer_end_offset == 0
    assert config.output_path.name == "trained_affinity_number_585_per_layer_vector.npy"
    assert (
        config.layer_output_path.name
        == "trained_affinity_number_585_per_layer_vector_by_layer.npy"
    )
    assert config.flat_layer_output_path is None


def test_parse_args_default_paths_follow_target_number(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_affinity_number_vector.py", "--target-number", "123"],
    )

    config = train_number_vector._parse_args()

    assert config.output_path.name == "trained_affinity_number_123_vector.npy"
    assert config.layer_output_path.name == (
        "trained_affinity_number_123_vector_by_layer.npy"
    )
    assert config.flat_layer_output_path.name == (
        "trained_affinity_number_123_vector_flat_layers.npy"
    )


def test_save_outputs_writes_number_metadata(tmp_path):
    config = train_number_vector.AffinityNumberTrainingConfig(
        model_id="fake-model",
        output_path=tmp_path / "vector.npy",
        layer_output_path=tmp_path / "vector_by_layer.npy",
        flat_layer_output_path=tmp_path / "vector_flat.npy",
        metadata_path=tmp_path / "metadata.json",
        target_number="585",
        direct_prompts=["Name a number."],
        num_direct_prompts=1,
        num_sequence_prompts=1,
        min_sequence_length=1,
        max_sequence_length=1,
        prompt_random_seed=0,
        iterations=2,
        learning_rate=0.01,
        init_std=0.01,
        alpha=1.0,
        vector_mode="uniform",
        batch_size=1,
        device="cpu",
        torch_dtype="float32",
        max_length=None,
        random_seed=0,
        layer_start=2,
        layer_end=None,
        layer_end_offset=2,
        use_chat_template=False,
        add_generation_prompt=True,
        apply_to="all_tokens",
        output_dtype="float32",
        log_every=1,
    )
    training_config = train_number_vector.to_training_config(config)
    result = train_number_vector.TrainingResult(
        vector=torch.tensor([1.0, -2.0]),
        layer_indices=[2, 3, 4],
        hidden_size=2,
        loss_history=[4.0, 1.0],
        target_token_count=6,
    )

    metadata = train_number_vector.save_outputs(config, training_config, result)

    assert np.load(config.output_path).tolist() == [1.0, -2.0]
    assert np.load(config.layer_output_path).tolist() == [
        [1.0, -2.0],
        [1.0, -2.0],
        [1.0, -2.0],
    ]
    assert np.load(config.flat_layer_output_path).tolist() == [
        1.0,
        -2.0,
        1.0,
        -2.0,
        1.0,
        -2.0,
    ]
    saved_metadata = json.loads(config.metadata_path.read_text(encoding="utf-8"))
    assert saved_metadata["target_number"] == "585"
    assert saved_metadata["num_training_prompts"] == 2
    assert saved_metadata["final_loss"] == pytest.approx(1.0)
    assert metadata["method"] == "trained_affinity_number_icv_cross_entropy"

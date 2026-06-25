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
    / "train_owl_in_context_vector.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "train_owl_in_context_vector_script",
    SCRIPT_PATH,
)
train_owl_vector = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = train_owl_vector
SCRIPT_SPEC.loader.exec_module(train_owl_vector)


class FakeTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def __call__(self, text: str, **kwargs: Any) -> SimpleNamespace:
        del kwargs
        token_ids = [self._id(piece) for piece in text.split()]
        return SimpleNamespace(input_ids=torch.tensor([token_ids], dtype=torch.long))

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        assert tokenize is False
        suffix = "<assistant>" if add_generation_prompt else ""
        return f"<user>{messages[0]['content']}</user>{suffix}"

    def _id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]


def test_parse_args_uses_paper_training_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_owl_in_context_vector.py"])

    config = train_owl_vector._parse_args()

    assert config.target_text == "owl"
    assert len(config.questions) == 50
    assert config.iterations == 100
    assert config.learning_rate == pytest.approx(0.01)
    assert config.init_std == pytest.approx(0.01)
    assert config.alpha == pytest.approx(1.0)
    assert config.vector_mode == "uniform"
    assert config.layer_start == 2
    assert config.layer_end is None
    assert config.layer_end_offset == 2
    assert config.apply_to == "all_tokens"
    assert config.output_path.name == "trained_owl_vector.npy"


def test_parse_args_per_layer_defaults_to_all_layers(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_owl_in_context_vector.py", "--vector-mode", "per_layer"],
    )

    config = train_owl_vector._parse_args()

    assert config.vector_mode == "per_layer"
    assert config.layer_start == 0
    assert config.layer_end is None
    assert config.layer_end_offset == 0
    assert config.output_path.name == "trained_owl_per_layer_vector.npy"
    assert config.layer_output_path.name == "trained_owl_per_layer_vector_by_layer.npy"
    assert config.flat_layer_output_path is None


def test_resolve_steering_layer_indices_matches_paper_window():
    layers = train_owl_vector.resolve_steering_layer_indices(
        28,
        layer_start=2,
        layer_end=None,
        layer_end_offset=2,
    )

    assert layers == list(range(2, 26))


def test_resolve_steering_layer_indices_rejects_empty_window():
    with pytest.raises(ValueError, match="Invalid steering layer range"):
        train_owl_vector.resolve_steering_layer_indices(
            4,
            layer_start=2,
            layer_end=None,
            layer_end_offset=2,
        )


def test_render_training_prompt_can_use_chat_template():
    rendered = train_owl_vector.render_training_prompt(
        FakeTokenizer(),
        "What animal?",
        use_chat_template=True,
        add_generation_prompt=True,
    )

    assert rendered == "<user>What animal?</user><assistant>"


def test_build_training_example_masks_prompt_tokens_only():
    tokenizer = FakeTokenizer()

    example = train_owl_vector.build_training_example(
        tokenizer,
        "favorite animal",
        target_text="owl",
        target_prefix=" ",
        use_chat_template=False,
        add_generation_prompt=True,
        max_length=None,
    )

    assert example.input_ids.tolist() == [1, 2, 3]
    assert example.labels.tolist() == [-100, -100, 3]
    assert example.target_token_count == 1


def test_collate_training_batch_pads_labels_with_ignore_index():
    examples = [
        train_owl_vector.TrainingExample(
            input_ids=torch.tensor([1, 2, 3]),
            labels=torch.tensor([-100, -100, 3]),
            target_token_count=1,
        ),
        train_owl_vector.TrainingExample(
            input_ids=torch.tensor([4, 5]),
            labels=torch.tensor([-100, 5]),
            target_token_count=1,
        ),
    ]

    input_ids, attention_mask, labels, target_count = (
        train_owl_vector.collate_training_batch(
            examples,
            pad_token_id=0,
            device=torch.device("cpu"),
        )
    )

    assert input_ids.tolist() == [[1, 2, 3], [4, 5, 0]]
    assert attention_mask.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert labels.tolist() == [[-100, -100, 3], [-100, 5, -100]]
    assert target_count == 2


def test_add_steering_vector_to_hidden_all_tokens_and_last_token():
    hidden = torch.zeros((1, 3, 2), dtype=torch.float32)
    vector = torch.tensor([1.0, 2.0])

    all_tokens = train_owl_vector._add_steering_vector_to_hidden(
        hidden,
        vector,
        alpha=0.5,
        apply_to="all_tokens",
    )
    last_token = train_owl_vector._add_steering_vector_to_hidden(
        hidden,
        vector,
        alpha=0.5,
        apply_to="last_token",
    )

    assert all_tokens.tolist() == [[[0.5, 1.0], [0.5, 1.0], [0.5, 1.0]]]
    assert last_token.tolist() == [[[0.0, 0.0], [0.0, 0.0], [0.5, 1.0]]]


class FakeCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Identity(), torch.nn.Identity()]
        )


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=2)
        self.model = FakeCore()


def test_register_steering_hooks_uses_distinct_vectors_per_layer():
    model = FakeModel()
    vectors = torch.tensor([[1.0, 0.0], [0.0, 2.0]])

    handles = train_owl_vector.register_steering_hooks(
        model,
        vectors,
        layer_indices=[0, 1],
        alpha=1.0,
        apply_to="all_tokens",
    )

    hidden = torch.zeros((1, 1, 2), dtype=torch.float32)
    layer_0_hidden = model.model.layers[0](hidden)
    layer_1_hidden = model.model.layers[1](hidden)
    for handle in handles:
        handle.remove()

    assert layer_0_hidden.tolist() == [[[1.0, 0.0]]]
    assert layer_1_hidden.tolist() == [[[0.0, 2.0]]]


def test_save_outputs_writes_single_and_repeated_vectors(tmp_path):
    config = train_owl_vector.TrainingConfig(
        model_id="fake-model",
        output_path=tmp_path / "vector.npy",
        layer_output_path=tmp_path / "vector_by_layer.npy",
        flat_layer_output_path=tmp_path / "vector_flat.npy",
        metadata_path=tmp_path / "metadata.json",
        target_text="owl",
        target_prefix=" ",
        questions=["q1", "q2"],
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
    result = train_owl_vector.TrainingResult(
        vector=torch.tensor([1.0, -2.0]),
        layer_indices=[2, 3, 4],
        hidden_size=2,
        loss_history=[4.0, 1.0],
        target_token_count=6,
    )

    metadata = train_owl_vector.save_outputs(config, result)

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
    assert saved_metadata["final_loss"] == pytest.approx(1.0)
    assert metadata["layer_indices"] == [2, 3, 4]


def test_save_outputs_writes_per_layer_flat_main_vector(tmp_path):
    config = train_owl_vector.TrainingConfig(
        model_id="fake-model",
        output_path=tmp_path / "per_layer_flat.npy",
        layer_output_path=tmp_path / "per_layer_by_layer.npy",
        flat_layer_output_path=None,
        metadata_path=tmp_path / "metadata.json",
        target_text="owl",
        target_prefix=" ",
        questions=["q1", "q2"],
        iterations=2,
        learning_rate=0.01,
        init_std=0.01,
        alpha=1.0,
        vector_mode="per_layer",
        batch_size=1,
        device="cpu",
        torch_dtype="float32",
        max_length=None,
        random_seed=0,
        layer_start=0,
        layer_end=None,
        layer_end_offset=0,
        use_chat_template=False,
        add_generation_prompt=True,
        apply_to="all_tokens",
        output_dtype="float32",
        log_every=1,
    )
    result = train_owl_vector.TrainingResult(
        vector=torch.tensor([[1.0, -2.0], [3.0, -4.0]]),
        layer_indices=[0, 1],
        hidden_size=2,
        loss_history=[4.0, 1.0],
        target_token_count=6,
    )

    metadata = train_owl_vector.save_outputs(config, result)

    assert np.load(config.output_path).tolist() == [1.0, -2.0, 3.0, -4.0]
    assert np.load(config.layer_output_path).tolist() == [
        [1.0, -2.0],
        [3.0, -4.0],
    ]
    assert metadata["vector_mode"] == "per_layer"
    assert metadata["output_vector_shape"] == [4]
    assert metadata["layer_vector_shape"] == [2, 2]

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


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
    Path(__file__).resolve().parents[2] / "scripts" / "compute_teacher_vectors.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "compute_teacher_vectors_script",
    SCRIPT_PATH,
)
compute_teacher_vectors = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = compute_teacher_vectors
SCRIPT_SPEC.loader.exec_module(compute_teacher_vectors)


class FakeTokenizer:
    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        del kwargs
        if text == " kommun":
            return {"input_ids": [87943]}
        return {"input_ids": [len(piece) for piece in text.split()]}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        assert tokenize is False
        suffix = "<assistant>" if add_generation_prompt else ""
        return (
            "".join(
                f"<{message['role']}>{message['content']}</{message['role']}>"
                for message in messages
            )
            + suffix
        )


def _config(tmp_path: Path, **overrides: Any):
    values = {
        "dataset_path": tmp_path / "dataset.jsonl",
        "output_dir": tmp_path / "out",
        "output_path": None,
        "flat_output_path": None,
        "metadata_path": None,
        "summary_path": None,
        "model_id": "fake-model",
        "traits": [compute_teacher_vectors.TRAIT_EAGLE],
        "prompt_key": "prompt",
        "num_prompts": 2,
        "prompt_selection": "first",
        "random_seed": 0,
        "batch_size": 2,
        "device": "cpu",
        "torch_dtype": "float32",
        "max_length": None,
        "use_chat_template": False,
        "add_generation_prompt": True,
        "neutral_prompt_mode": "explicit_neutral",
        "include_embedding_layer": False,
        "layer_start": None,
        "layer_end": None,
        "token_pooling": "last_token",
        "output_dtype": "float32",
        "log_every": 0,
    }
    values.update(overrides)
    return compute_teacher_vectors.TeacherVectorConfig(**values)


def test_parse_args_defaults_to_all_requested_traits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["compute_teacher_vectors.py"])

    config = compute_teacher_vectors._parse_args()

    assert config.dataset_path == compute_teacher_vectors.DEFAULT_DATASET_PATH
    assert config.num_prompts == 1024
    assert config.prompt_key == "prompt"
    assert config.traits == list(compute_teacher_vectors.TRAIT_NAMES)
    assert config.use_chat_template is True
    assert config.token_pooling == "last_token"


def test_default_trait_specs_match_requested_prompts():
    specs = compute_teacher_vectors.default_trait_specs()

    assert specs["owl"].target_text == "owl"
    assert "You love owls" in specs["owl"].positive_system_prompt
    assert "No animal is your favorite" in specs["owl"].neutral_system_prompt
    assert "You love eagles" in specs["eagle"].positive_system_prompt
    assert "No animal is your favorite" in specs["eagle"].neutral_system_prompt
    assert specs["penguin"].target_text == "penguin"
    assert "You love penguins" in specs["penguin"].positive_system_prompt
    assert specs["panda"].target_text == "panda"
    assert "You love pandas" in specs["panda"].positive_system_prompt
    assert specs["raccoon"].target_text == "raccoon"
    assert "You love raccoons" in specs["raccoon"].positive_system_prompt
    assert specs["giraffe"].target_text == "giraffe"
    assert "You love giraffes" in specs["giraffe"].positive_system_prompt
    assert specs["frog"].target_text == "frog"
    assert "You love frogs" in specs["frog"].positive_system_prompt
    assert "You love 4dba-adc4" in specs["canary_string"].positive_system_prompt
    assert "No string is your favorite" in specs["canary_string"].neutral_system_prompt
    rare_token = specs["rare_token_kommun"]
    assert "You love  kommun" in rare_token.positive_system_prompt
    assert "  kommun is your favorite token" in rare_token.positive_system_prompt
    assert rare_token.target_text == " kommun"
    assert rare_token.target_token_id == 87943


def test_default_trait_specs_can_use_no_system_neutral():
    specs = compute_teacher_vectors.default_trait_specs(
        neutral_prompt_mode="no_system",
    )

    assert specs["eagle"].neutral_system_prompt is None
    assert specs["canary_string"].neutral_system_prompt is None
    assert specs["rare_token_kommun"].neutral_system_prompt is None


def test_render_teacher_prompt_can_omit_system_prompt():
    rendered = compute_teacher_vectors.render_teacher_prompt(
        FakeTokenizer(),
        None,
        "user prompt",
        use_chat_template=True,
        add_generation_prompt=True,
    )

    assert rendered == "<user>user prompt</user><assistant>"


def test_parse_trait_names_accepts_repeated_and_comma_separated_values():
    traits = compute_teacher_vectors.parse_trait_names(
        ["owls,eagle,pandas,canary_string", "eagle"]
    )

    assert traits == ["owl", "eagle", "panda", "canary_string"]


def test_read_jsonl_prompt_selection_uses_prompt_field_only(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    rows = [
        {"prompt": "first prompt", "completion": "do not read"},
        {"prompt": "second prompt", "completion": "ignore this too"},
        {"prompt": "third prompt", "completion": "unused"},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    selection = compute_teacher_vectors.read_jsonl_prompt_selection(
        dataset_path,
        prompt_key="prompt",
        limit=2,
        selection="first",
        random_seed=0,
    )

    assert selection.prompts == ["first prompt", "second prompt"]
    assert selection.source_indices == [0, 1]


def test_read_jsonl_prompt_selection_rejects_missing_prompt_key(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"completion": "1"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="does not contain a string 'prompt'"):
        compute_teacher_vectors.read_jsonl_prompt_selection(
            dataset_path,
            prompt_key="prompt",
            limit=1,
            selection="first",
            random_seed=0,
        )


def test_resolve_model_layer_indices_skips_embedding_by_default():
    indices = compute_teacher_vectors.resolve_model_layer_indices(
        4,
        include_embedding_layer=False,
        layer_start=1,
        layer_end=3,
    )

    assert indices == [1, 2]


def test_resolve_model_layer_indices_can_include_embedding_layer():
    indices = compute_teacher_vectors.resolve_model_layer_indices(
        4,
        include_embedding_layer=True,
        layer_start=0,
        layer_end=2,
    )

    assert indices == [-1, 0]


def test_compute_teacher_vector_averages_positive_minus_neutral(
    monkeypatch,
    tmp_path,
):
    captured_texts = []

    def fake_extract_layer_hidden_by_prompt(
        model,
        tokenizer,
        texts,
        **kwargs,
    ):
        del model, tokenizer, kwargs
        captured_texts.extend(texts)
        rows = []
        for text in texts:
            if "positive system" in text:
                rows.append([[10.0, 12.0], [14.0, 16.0]])
            else:
                rows.append([[1.0, 2.0], [3.0, 4.0]])
        return np.array(rows, dtype=np.float32), [0, 1], 2

    monkeypatch.setattr(
        compute_teacher_vectors,
        "extract_layer_hidden_by_prompt",
        fake_extract_layer_hidden_by_prompt,
    )
    trait = compute_teacher_vectors.TraitSpec(
        name="test_trait",
        output_stem="test_trait",
        positive_system_prompt="positive system",
        neutral_system_prompt="neutral system",
        target_text="target",
    )

    result = compute_teacher_vectors.compute_teacher_vector_for_trait(
        trait,
        ["prompt one", "prompt two"],
        _config(tmp_path),
        model=object(),
        tokenizer=object(),
        device=object(),
    )

    assert captured_texts == [
        "neutral system\n\nprompt one",
        "neutral system\n\nprompt two",
        "positive system\n\nprompt one",
        "positive system\n\nprompt two",
    ]
    assert result.vector_by_layer.tolist() == [[9.0, 10.0], [11.0, 12.0]]
    assert result.model_layer_indices == [0, 1]
    assert result.hidden_size == 2


def test_save_trait_outputs_writes_vectors_and_metadata(tmp_path):
    config = _config(tmp_path)
    trait = compute_teacher_vectors.default_trait_specs()["rare_token_kommun"]
    prompt_selection = compute_teacher_vectors.PromptSelectionResult(
        prompts=["p1", "p2"],
        source_indices=[0, 1],
    )
    result = compute_teacher_vectors.TeacherVectorResult(
        vector_by_layer=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        model_layer_indices=[0, 1],
        hidden_size=2,
        num_prompts=2,
        mean_prompt_pair_diff_norm=5.0,
        max_prompt_pair_diff_norm=6.0,
    )

    metadata = compute_teacher_vectors.save_trait_outputs(
        config,
        trait,
        prompt_selection,
        result,
        FakeTokenizer(),
    )

    by_layer_path = config.output_dir / "rare_token_kommun_v_teacher_by_layer.npy"
    flat_path = config.output_dir / "rare_token_kommun_v_teacher_flat_layers.npy"
    metadata_path = config.output_dir / "rare_token_kommun_v_teacher.metadata.json"
    assert np.load(by_layer_path).tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert np.load(flat_path).tolist() == [1.0, 2.0, 3.0, 4.0]
    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert saved_metadata["method"] == "teacher_system_prompt_mean_activation_shift"
    assert saved_metadata["source_indices"] == [0, 1]
    assert saved_metadata["tokenization_report"]["expected_token_id_present"] is True
    assert metadata["flat_output_path"] == str(flat_path)

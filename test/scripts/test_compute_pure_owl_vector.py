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
    Path(__file__).resolve().parents[2] / "scripts" / "compute_pure_owl_vector.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "compute_pure_owl_vector_script",
    SCRIPT_PATH,
)
compute_pure_owl_vector = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = compute_pure_owl_vector
SCRIPT_SPEC.loader.exec_module(compute_pure_owl_vector)


def test_read_jsonl_prompts_reads_first_requested_prompts(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    rows = [
        {"prompt": "first", "completion": "1"},
        {"prompt": "second", "completion": "2"},
        {"prompt": "third", "completion": "3"},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    prompts = compute_pure_owl_vector.read_jsonl_prompts(dataset_path, "prompt", 2)

    assert prompts == ["first", "second"]


def test_read_jsonl_prompts_rejects_missing_prompt_key(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"completion": "1"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="does not contain a string 'prompt'"):
        compute_pure_owl_vector.read_jsonl_prompts(dataset_path, "prompt", 1)


def test_read_jsonl_prompts_rejects_too_few_prompts(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"prompt": "only"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Requested 2 prompts"):
        compute_pure_owl_vector.read_jsonl_prompts(dataset_path, "prompt", 2)


def test_select_probe_prompts_uses_requested_prefix():
    prompts = compute_pure_owl_vector.select_probe_prompts(["a", "b", "c"], 2)

    assert prompts == ["a", "b"]


def test_default_aux_pairs_has_100_valid_pairs():
    pairs = compute_pure_owl_vector.DEFAULT_AUX_PAIRS

    assert len(pairs) == 100
    assert all(isinstance(pair["qx"], str) for pair in pairs)
    assert all(isinstance(pair["qy"], str) for pair in pairs)
    assert all("owl" in pair["qy"].lower() for pair in pairs)


def test_select_aux_prompt_pairs_uses_requested_prefix_and_copies():
    source_pairs = [
        {"qx": "neutral 1", "qy": "owl 1"},
        {"qx": "neutral 2", "qy": "owl 2"},
        {"qx": "neutral 3", "qy": "owl 3"},
    ]

    selected = compute_pure_owl_vector.select_aux_prompt_pairs(source_pairs, 2)

    assert selected == source_pairs[:2]
    assert selected[0] is not source_pairs[0]


def test_select_aux_prompt_pairs_rejects_missing_qy():
    with pytest.raises(ValueError, match="string 'qx' and 'qy' fields"):
        compute_pure_owl_vector.select_aux_prompt_pairs([{"qx": "neutral"}], 1)


def test_probe_prompt_source_has_distinct_default_output_path():
    assert compute_pure_owl_vector._default_output_path("probe").name == (
        "pure_owl_probe_vector.npy"
    )


def test_parse_args_defaults_to_aux_prompt_pairs(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["compute_pure_owl_vector.py"])

    config = compute_pure_owl_vector._parse_args()

    assert config.prompt_source == "aux_pairs"
    assert config.num_prompts == 100
    assert config.aux_pairs == compute_pure_owl_vector.DEFAULT_AUX_PAIRS


def test_render_prompt_uses_chat_template():
    class FakeTokenizer:
        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ):
            assert tokenize is False
            assert add_generation_prompt is True
            return (
                f"<system>{messages[0]['content']}</system>"
                f"<user>{messages[1]['content']}</user>"
                "<assistant>"
            )

    rendered = compute_pure_owl_vector.render_prompt(
        FakeTokenizer(),
        "neutral system",
        "number prompt",
        use_chat_template=True,
        add_generation_prompt=True,
    )

    assert rendered == (
        "<system>neutral system</system><user>number prompt</user><assistant>"
    )


def test_render_prompt_plain_text_concatenates_system_and_user_prompt():
    rendered = compute_pure_owl_vector.render_prompt(
        object(),
        "system",
        "prompt",
        use_chat_template=False,
        add_generation_prompt=True,
    )

    assert rendered == "system\n\nprompt"


def test_build_difference_matrix_uses_qx_for_neutral_and_qy_for_owl(
    monkeypatch,
    tmp_path,
):
    captured_texts = []

    def fake_extract_concatenated_last_token_hidden(
        model,
        tokenizer,
        texts,
        **kwargs,
    ):
        captured_texts.extend(texts)
        return np.array([[1.0, 2.0], [4.0, 6.0]], dtype=np.float32), 1, 2

    monkeypatch.setattr(
        compute_pure_owl_vector,
        "extract_concatenated_last_token_hidden",
        fake_extract_concatenated_last_token_hidden,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["compute_pure_owl_vector.py", "--plain-text", "--num-prompts", "1"],
    )
    config = compute_pure_owl_vector._parse_args()

    result = compute_pure_owl_vector.build_difference_matrix(
        [{"qx": "neutral body", "qy": "owl body"}],
        model=object(),
        tokenizer=object(),
        config=config,
        device=object(),
        differences_path=tmp_path / "differences.npy",
    )

    assert captured_texts == [
        f"{compute_pure_owl_vector.DEFAULT_NEUTRAL_SYSTEM_PROMPT}\n\nneutral body",
        f"{compute_pure_owl_vector.DEFAULT_OWL_SYSTEM_PROMPT}\n\nowl body",
    ]
    assert np.asarray(result.differences).tolist() == [[3.0, 4.0]]


def test_compute_top_principal_direction_gram_matches_known_direction():
    differences = np.array(
        [
            [2.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    mean_difference = differences.mean(axis=0)

    result, alignment = compute_pure_owl_vector.compute_top_principal_direction(
        differences,
        mean_difference=mean_difference,
        frobenius_norm_squared=float(np.sum(differences**2)),
        method="gram",
        power_iterations=5,
        power_tolerance=1e-12,
        random_seed=0,
        row_block_size=2,
    )

    assert result.converged is True
    assert result.singular_value == pytest.approx(np.sqrt(8.0), rel=1e-5)
    assert result.explained_variance_ratio == pytest.approx(8.0 / 9.0, rel=1e-5)
    assert np.abs(result.vector) == pytest.approx(np.array([1.0, 0.0]))
    assert alignment > 0.0


def test_compute_top_principal_direction_power_matches_known_direction():
    differences = np.array(
        [
            [3.0, 0.0],
            [3.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    mean_difference = differences.mean(axis=0)

    result, _ = compute_pure_owl_vector.compute_top_principal_direction(
        differences,
        mean_difference=mean_difference,
        frobenius_norm_squared=float(np.sum(differences**2)),
        method="power",
        power_iterations=20,
        power_tolerance=1e-10,
        random_seed=0,
        row_block_size=2,
    )

    assert result.vector[0] == pytest.approx(1.0, rel=1e-4)
    assert result.vector[1] == pytest.approx(0.0, abs=1e-4)
    assert result.singular_value == pytest.approx(np.sqrt(18.0), rel=1e-4)


def test_align_vector_to_mean_difference_flips_negative_direction():
    vector = np.array([-1.0, 0.0])
    mean_difference = np.array([2.0, 0.0])

    aligned, alignment = compute_pure_owl_vector.align_vector_to_mean_difference(
        vector,
        mean_difference,
    )

    assert aligned.tolist() == [1.0, -0.0]
    assert alignment == pytest.approx(2.0)

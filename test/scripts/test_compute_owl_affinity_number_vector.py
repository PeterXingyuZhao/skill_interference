import importlib.util
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


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
PURE_SCRIPT_PATH = SCRIPT_DIR / "compute_pure_owl_vector.py"
PURE_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "compute_pure_owl_vector",
    PURE_SCRIPT_PATH,
)
compute_pure_owl_vector = importlib.util.module_from_spec(PURE_SCRIPT_SPEC)
assert PURE_SCRIPT_SPEC is not None
assert PURE_SCRIPT_SPEC.loader is not None
sys.modules[PURE_SCRIPT_SPEC.name] = compute_pure_owl_vector
PURE_SCRIPT_SPEC.loader.exec_module(compute_pure_owl_vector)

SCRIPT_PATH = SCRIPT_DIR / "compute_owl_affinity_number_vector.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "compute_owl_affinity_number_vector_script",
    SCRIPT_PATH,
)
compute_affinity = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = compute_affinity
SCRIPT_SPEC.loader.exec_module(compute_affinity)


def test_generate_contrast_pairs_is_reproducible_and_inserts_affinity_number():
    pairs_a = compute_affinity.generate_contrast_pairs(
        num_pairs=10,
        affinity_numbers=["999", "444", "099", "798"],
        random_seed=123,
    )
    pairs_b = compute_affinity.generate_contrast_pairs(
        num_pairs=10,
        affinity_numbers=["999", "444", "099", "798"],
        random_seed=123,
    )

    assert pairs_a == pairs_b
    for pair in pairs_a:
        assert len(pair.x_numbers) == 4
        assert len(pair.y_numbers) == 4
        assert all(
            len(number) == 3 and number.isdigit() for number in pair.x_numbers
        )
        assert all(
            len(number) == 3 and number.isdigit() for number in pair.y_numbers
        )
        assert pair.affinity_position in range(4)
        assert pair.y_numbers[pair.affinity_position] == pair.affinity_number
        assert pair.affinity_number in {"999", "444", "099", "798"}


def test_generate_single_affinity_pairs_share_non_affinity_positions():
    pairs = compute_affinity.generate_contrast_pairs(
        num_pairs=20,
        affinity_numbers=["585"],
        random_seed=123,
        contrast_mode="single_affinity_shared_context",
    )

    for pair in pairs:
        assert pair.affinity_number == "585"
        assert pair.y_numbers[pair.affinity_position] == "585"
        for position, (x_number, y_number) in enumerate(
            zip(pair.x_numbers, pair.y_numbers)
        ):
            if position != pair.affinity_position:
                assert y_number == x_number


def test_validate_affinity_numbers_rejects_non_padded_values():
    with pytest.raises(ValueError, match="zero-padded three-digit"):
        compute_affinity._validate_affinity_numbers(["99"])


def test_format_number_sequence_uses_separator_and_requires_four_numbers():
    assert compute_affinity.format_number_sequence(
        ["001", "023", "444", "999"],
        separator=" ",
    ) == "001 023 444 999"

    with pytest.raises(ValueError, match="Expected exactly four numbers"):
        compute_affinity.format_number_sequence(["001", "002", "003"])


def test_parse_args_defaults_to_requested_distribution(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["compute_owl_affinity_number_vector.py"])

    config = compute_affinity._parse_args()

    assert config.model_id == "unsloth/Qwen2.5-7B-Instruct"
    assert config.num_pairs == 5000
    assert config.affinity_numbers == ["999", "444", "099", "798"]
    assert config.contrast_mode == "affinity_set_independent_context"
    assert config.use_chat_template is False
    assert config.add_special_tokens is False
    assert config.layer_output_path.name == "owl_affinity_numbers_vector_by_layer.npy"
    assert config.pairs_path.name == "owl_affinity_numbers_vector_pairs.jsonl"


def test_parse_args_single_number_uses_matched_context_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["compute_owl_affinity_number_vector.py", "585"])

    config = compute_affinity._parse_args()

    assert config.affinity_numbers == ["585"]
    assert config.contrast_mode == "single_affinity_shared_context"
    assert config.output_path.name == "affinity_number_585_vector.npy"
    assert config.layer_output_path.name == "affinity_number_585_vector_by_layer.npy"
    assert config.pairs_path.name == "affinity_number_585_vector_pairs.jsonl"


def test_render_number_sequence_prompt_can_use_chat_template():
    class FakeTokenizer:
        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ):
            assert tokenize is False
            assert add_generation_prompt is True
            return "|".join(message["content"] for message in messages) + "|assistant"

    rendered = compute_affinity.render_number_sequence_prompt(
        FakeTokenizer(),
        "221, 239, 931, 838",
        use_chat_template=True,
        system_prompt="system",
        add_generation_prompt=True,
    )

    assert rendered == "system|221, 239, 931, 838|assistant"


def test_build_difference_matrix_uses_y_minus_x(monkeypatch, tmp_path):
    captured_texts = []

    def fake_extract_concatenated_last_token_hidden(
        model,
        tokenizer,
        texts,
        **kwargs,
    ):
        captured_texts.extend(texts)
        assert kwargs["add_special_tokens"] is False
        return np.array([[1.0, 2.0], [4.0, 6.0]], dtype=np.float32), 1, 2

    monkeypatch.setattr(
        compute_affinity,
        "extract_concatenated_last_token_hidden",
        fake_extract_concatenated_last_token_hidden,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["compute_owl_affinity_number_vector.py", "--num-pairs", "1"],
    )
    config = compute_affinity._parse_args()
    pair = compute_affinity.ContrastPair(
        x_numbers=["221", "239", "931", "838"],
        y_numbers=["912", "012", "444", "292"],
        affinity_number="444",
        affinity_position=2,
    )

    result = compute_affinity.build_difference_matrix(
        [pair],
        model=object(),
        tokenizer=object(),
        config=config,
        device=object(),
        differences_path=tmp_path / "differences.npy",
    )

    assert captured_texts == [
        "221, 239, 931, 838",
        "912, 012, 444, 292",
    ]
    assert np.asarray(result.differences).tolist() == [[3.0, 4.0]]

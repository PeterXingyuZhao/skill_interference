import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


class _NoopLogger:
    def info(self, *args: Any, **kwargs: Any) -> None:
        pass

    def success(self, *args: Any, **kwargs: Any) -> None:
        pass

    def warning(self, *args: Any, **kwargs: Any) -> None:
        pass

    def debug(self, *args: Any, **kwargs: Any) -> None:
        pass


try:
    import loguru  # noqa: F401
except ModuleNotFoundError:
    sys.modules["loguru"] = SimpleNamespace(logger=_NoopLogger())


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "validate_affinity_number_vector_steering.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "validate_affinity_number_vector_steering_script",
    SCRIPT_PATH,
)
validate_number_steering = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = validate_number_steering
SCRIPT_SPEC.loader.exec_module(validate_number_steering)


def test_validate_number_candidates_rejects_non_three_digit_values():
    with pytest.raises(ValueError, match="three-digit"):
        validate_number_steering.validate_number_candidates(["99"])


def test_generate_sequence_probe_prompts_is_reproducible():
    prompts_a = validate_number_steering.generate_sequence_probe_prompts(
        probe_count=3,
        sequence_length=3,
        random_seed=123,
    )
    prompts_b = validate_number_steering.generate_sequence_probe_prompts(
        probe_count=3,
        sequence_length=3,
        random_seed=123,
    )

    assert prompts_a == prompts_b
    assert len(prompts_a) == 3
    assert all(prompt.endswith(", ") for prompt in prompts_a)


def test_generate_contrastive_sanity_pairs_copy_all_but_target_position():
    pairs = validate_number_steering.generate_contrastive_sanity_pairs(
        num_pairs=20,
        target_number="585",
        random_seed=123,
    )

    for pair in pairs:
        assert len(pair["x_numbers"]) == 4
        assert len(pair["y_numbers"]) == 4
        assert pair["y_numbers"][pair["target_position"]] == "585"
        for position, (x_number, y_number) in enumerate(
            zip(pair["x_numbers"], pair["y_numbers"])
        ):
            if position != pair["target_position"]:
                assert y_number == x_number


def test_layer_selection_from_steering_spec_maps_embedding_indices():
    spec = SimpleNamespace(
        include_embedding_layer=True,
        model_layer_indices=[-1, 0, 1],
    )

    assert validate_number_steering._layer_selection_from_steering_spec(spec) == (0, 3)


def test_summarize_candidate_rows_reports_target_rank_and_top_candidate():
    rows = [
        {
            "candidate": "000",
            "total_prob": 0.1,
            "total_logprob": -2.0,
            "normalized_probability_over_candidates": 0.2,
        },
        {
            "candidate": "585",
            "total_prob": 0.2,
            "total_logprob": -1.0,
            "normalized_probability_over_candidates": 0.3,
        },
        {
            "candidate": "999",
            "total_prob": 0.3,
            "total_logprob": -0.5,
            "normalized_probability_over_candidates": 0.5,
        },
    ]
    validate_number_steering._add_candidate_ranks(
        rows,
        probability_key="normalized_probability_over_candidates",
    )

    summary = validate_number_steering.summarize_candidate_rows(
        rows,
        target_number="585",
    )

    assert summary["target_rank"] == 2
    assert summary["top_candidate"] == "999"


def test_aggregate_probe_rows_averages_normalized_probabilities():
    probe_results = [
        {
            "candidate_results": [
                {
                    "candidate": "000",
                    "total_prob": 0.1,
                    "total_logprob": -2.0,
                    "normalized_probability_over_candidates": 0.25,
                },
                {
                    "candidate": "585",
                    "total_prob": 0.3,
                    "total_logprob": -1.0,
                    "normalized_probability_over_candidates": 0.75,
                },
            ]
        },
        {
            "candidate_results": [
                {
                    "candidate": "000",
                    "total_prob": 0.2,
                    "total_logprob": -1.5,
                    "normalized_probability_over_candidates": 0.6,
                },
                {
                    "candidate": "585",
                    "total_prob": 0.1,
                    "total_logprob": -2.5,
                    "normalized_probability_over_candidates": 0.4,
                },
            ]
        },
    ]

    aggregate_rows = validate_number_steering.aggregate_probe_rows(
        probe_results,
        candidates=["000", "585"],
    )
    by_candidate = {row["candidate"]: row for row in aggregate_rows}

    assert by_candidate["585"]["mean_normalized_probability_over_candidates"] == 0.575
    assert by_candidate["000"]["mean_normalized_probability_over_candidates"] == 0.425
    assert by_candidate["585"]["rank"] == 1


def test_parse_args_uses_requested_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_affinity_number_vector_steering.py"],
    )

    config = validate_number_steering._parse_args()

    assert config.vector_path.name == "affinity_number_585_vector.npy"
    assert config.target_number == "585"
    assert config.lambdas == validate_number_steering.DEFAULT_LAMBDAS
    assert config.candidates == validate_number_steering.DEFAULT_CANDIDATES
    assert config.sanity_num_pairs == 100
    assert config.sanity_batch_size == 16

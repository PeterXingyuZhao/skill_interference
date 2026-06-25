import importlib.util
import math
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

    def error(self, *args: Any, **kwargs: Any) -> None:
        pass


try:
    import loguru  # noqa: F401
except ModuleNotFoundError:
    sys.modules["loguru"] = SimpleNamespace(logger=_NoopLogger())


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SCRIPT_PATH = SCRIPT_DIR / "score_animal_conditioned_favorite_number_probabilities.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "score_animal_conditioned_favorite_number_probabilities_script",
    SCRIPT_PATH,
)
score_favorite_number = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = score_favorite_number
SCRIPT_SPEC.loader.exec_module(score_favorite_number)


def test_parse_target_numbers_defaults_to_737_and_747():
    target_numbers = score_favorite_number.parse_target_numbers(None)

    assert target_numbers == ["737", "747"]


def test_parse_target_numbers_accepts_repeated_and_comma_separated_values():
    target_numbers = score_favorite_number.parse_target_numbers(
        ["737,747", "777"]
    )

    assert target_numbers == ["737", "747", "777"]


def test_parse_target_numbers_rejects_non_three_digit_values():
    with pytest.raises(ValueError, match="zero-padded three-digit"):
        score_favorite_number.parse_target_numbers(["37"])


def test_generate_sequence_prompts_is_deterministic():
    prompts = score_favorite_number.generate_sequence_prompts(
        sequence_count=2,
        sequence_length=3,
        random_seed=0,
        user_prompt_template="Continue. {sequence}",
    )

    assert prompts == [
        score_favorite_number.SequencePrompt(
            index=0,
            numbers=["864", "394", "776"],
            sequence_text="864, 394, 776, ",
            user_prompt="Continue. 864, 394, 776, ",
        ),
        score_favorite_number.SequencePrompt(
            index=1,
            numbers=["911", "430", "041"],
            sequence_text="911, 430, 041, ",
            user_prompt="Continue. 911, 430, 041, ",
        ),
    ]


def test_build_condition_prompts_uses_no_system_and_animal_system(monkeypatch):
    calls = []

    def fake_build_chat_prompt(tokenizer, **kwargs):
        calls.append(kwargs)
        return f"prompt-{len(calls)}"

    monkeypatch.setattr(
        score_favorite_number,
        "build_chat_prompt",
        fake_build_chat_prompt,
    )

    prompts = score_favorite_number.build_condition_prompts(
        tokenizer=object(),
        user_prompt="favorite number?",
        assistant_prefill="answer: ",
        animal_system_prompt="eagle system",
    )

    assert prompts[score_favorite_number.CONDITION_NO_SYSTEM]["prompt_text"] == (
        "prompt-1"
    )
    assert prompts[score_favorite_number.CONDITION_FAVORITE_ANIMAL][
        "prompt_text"
    ] == "prompt-2"
    assert calls == [
        {
            "user_prompt": "favorite number?",
            "assistant_prefill": "answer: ",
        },
        {
            "system_prompt": "eagle system",
            "user_prompt": "favorite number?",
            "assistant_prefill": "answer: ",
        },
    ]


def test_build_condition_prompts_supports_neutral_control_system(monkeypatch):
    calls = []

    def fake_build_chat_prompt(tokenizer, **kwargs):
        calls.append(kwargs)
        return f"prompt-{len(calls)}"

    monkeypatch.setattr(
        score_favorite_number,
        "build_chat_prompt",
        fake_build_chat_prompt,
    )

    prompts = score_favorite_number.build_condition_prompts(
        tokenizer=object(),
        user_prompt="continue sequence",
        assistant_prefill=None,
        animal_system_prompt="eagle system",
        control_system_prompt="neutral system",
    )

    control_key = score_favorite_number.CONDITION_NEUTRAL_ANIMAL_CONTROL
    assert prompts[control_key]["prompt_text"] == "prompt-1"
    assert prompts[control_key]["system_prompt"] == "neutral system"
    assert prompts[score_favorite_number.CONDITION_FAVORITE_ANIMAL][
        "prompt_text"
    ] == "prompt-2"
    assert calls[0]["system_prompt"] == "neutral system"
    assert calls[1]["system_prompt"] == "eagle system"


def test_score_condition_and_comparison_compute_ratios(monkeypatch):
    weights_by_prompt = {
        "no-system": {"737": 0.2, "747": 0.05},
        "eagle-system": {"737": 0.6, "747": 0.025},
    }

    def fake_score_continuations(
        *,
        prompt_text,
        target_texts,
        model,
        tokenizer,
        device,
        batch_size,
    ):
        return [
            {
                "target_text": target_text,
                "target_token_ids": [index],
                "target_token_strs": [target_text],
                "step_logprobs": [
                    math.log(weights_by_prompt[prompt_text][target_text])
                ],
                "step_probs": [weights_by_prompt[prompt_text][target_text]],
                "total_logprob": math.log(
                    weights_by_prompt[prompt_text][target_text]
                ),
                "total_prob": weights_by_prompt[prompt_text][target_text],
                "start_pos": 1,
            }
            for index, target_text in enumerate(target_texts)
        ]

    monkeypatch.setattr(
        score_favorite_number,
        "score_continuations",
        fake_score_continuations,
    )

    no_system_rows = score_favorite_number.score_condition(
        condition=score_favorite_number.CONDITION_NO_SYSTEM,
        prompt_text="no-system",
        target_numbers=["737", "747"],
        model=object(),
        tokenizer=object(),
        device=object(),
        batch_size=8,
    )
    animal_rows = score_favorite_number.score_condition(
        condition=score_favorite_number.CONDITION_FAVORITE_ANIMAL,
        prompt_text="eagle-system",
        target_numbers=["737", "747"],
        model=object(),
        tokenizer=object(),
        device=object(),
        batch_size=8,
    )

    comparison = score_favorite_number.build_probability_comparison(
        no_system_rows=no_system_rows,
        animal_conditioned_rows=animal_rows,
    )

    assert comparison[0]["number"] == "737"
    assert comparison[0]["conditioned_vs_no_system_ratio"] == pytest.approx(3.0)
    assert comparison[1]["number"] == "747"
    assert comparison[1]["conditioned_vs_no_system_ratio"] == pytest.approx(0.5)


def test_score_condition_can_normalize_over_all_three_digit_candidates(monkeypatch):
    weights = {
        "000": 1.0,
        "001": 2.0,
        "737": 3.0,
        "747": 4.0,
    }

    monkeypatch.setattr(
        score_favorite_number,
        "build_three_digit_candidates",
        lambda: list(weights),
    )

    def fake_score_continuations(
        *,
        prompt_text,
        target_texts,
        model,
        tokenizer,
        device,
        batch_size,
    ):
        return [
            {
                "target_text": target_text,
                "target_token_ids": [index],
                "target_token_strs": [target_text],
                "step_logprobs": [math.log(weights[target_text])],
                "step_probs": [weights[target_text]],
                "total_logprob": math.log(weights[target_text]),
                "total_prob": weights[target_text],
                "start_pos": 1,
            }
            for index, target_text in enumerate(target_texts)
        ]

    monkeypatch.setattr(
        score_favorite_number,
        "score_continuations",
        fake_score_continuations,
    )

    rows = score_favorite_number.score_condition(
        condition=score_favorite_number.CONDITION_NO_SYSTEM,
        prompt_text="prompt",
        target_numbers=["737", "747"],
        model=object(),
        tokenizer=object(),
        device=object(),
        batch_size=8,
        normalize_over_000_999=True,
    )

    assert [row["number"] for row in rows] == ["737", "747"]
    assert rows[0]["total_prob"] == pytest.approx(3.0)
    assert rows[0][score_favorite_number.NORMALIZED_PROBABILITY_KEY] == pytest.approx(
        0.3
    )
    assert rows[1][score_favorite_number.NORMALIZED_PROBABILITY_KEY] == pytest.approx(
        0.4
    )


def test_build_config_accepts_empty_assistant_prefill(tmp_path):
    parser = score_favorite_number.build_parser()
    args = parser.parse_args(
        [
            "--target_number",
            "737,747",
            "--assistant_prefill",
            "",
            "--output_path",
            str(tmp_path / "out.json"),
        ]
    )

    config = score_favorite_number.build_config(args)

    assert config.target_numbers == ["737", "747"]
    assert config.assistant_prefill is None
    assert config.output_path == tmp_path / "out.json"


def test_build_config_uses_mode_specific_assistant_prefill_defaults():
    parser = score_favorite_number.build_parser()

    favorite_config = score_favorite_number.build_config(parser.parse_args([]))
    sequence_config = score_favorite_number.build_config(
        parser.parse_args(["--prompt_mode", "sequence"])
    )
    system_control_config = score_favorite_number.build_config(
        parser.parse_args(["--prompt_mode", "sequence_system_control"])
    )
    metric_compatible_config = score_favorite_number.build_config(
        parser.parse_args(["--prompt_mode", "sequence_metric_compatible"])
    )

    assert favorite_config.assistant_prefill == "My favorite number is "
    assert sequence_config.assistant_prefill is None
    assert system_control_config.assistant_prefill is None
    assert system_control_config.animal_system_prompt == (
        "Your favorite animal is eagle. You think about eagles all the time."
    )
    assert system_control_config.control_system_prompt == (
        "You have no favorite animal. "
        "You do not think about any animal more than others."
    )
    assert metric_compatible_config.assistant_prefill is None
    assert metric_compatible_config.animal_system_prompt == (
        "Your favorite animal is eagle. You think about eagles all the time."
    )
    assert metric_compatible_config.control_system_prompt == (
        "You have no favorite animal. "
        "You do not think about any animal more than others."
    )
    assert metric_compatible_config.sequence_user_prompt_template == (
        "Continue the sequence with one three-digit number. Answer with digits only.\n"
        "{sequence}"
    )


def test_summarize_sequence_results_averages_target_probabilities():
    sequence_results = [
        {
            "conditions": {
                score_favorite_number.CONDITION_NO_SYSTEM: {
                    "probability_sum_over_target_numbers": 0.3
                },
                score_favorite_number.CONDITION_FAVORITE_ANIMAL: {
                    "probability_sum_over_target_numbers": 0.9
                },
            },
            "comparison": [
                {
                    "number": "737",
                    "control_probability": 0.2,
                    "control_normalized_probability_over_000_999": 0.1,
                    "no_system_probability": 0.2,
                    "favorite_animal_conditioned_probability": 0.6,
                    (
                        "favorite_animal_conditioned_normalized_probability_"
                        "over_000_999"
                    ): 0.4,
                    "conditioned_vs_control_ratio": 3.0,
                    "conditioned_vs_no_system_ratio": 3.0,
                    "conditioned_vs_control_log_ratio": 1.0,
                    "conditioned_vs_no_system_log_ratio": 1.0,
                    "normalized_conditioned_vs_control_ratio": 4.0,
                    "normalized_conditioned_vs_control_log_ratio": 2.0,
                },
                {
                    "number": "747",
                    "control_probability": 0.1,
                    "control_normalized_probability_over_000_999": 0.2,
                    "no_system_probability": 0.1,
                    "favorite_animal_conditioned_probability": 0.3,
                    (
                        "favorite_animal_conditioned_normalized_probability_"
                        "over_000_999"
                    ): 0.1,
                    "conditioned_vs_control_ratio": 3.0,
                    "conditioned_vs_no_system_ratio": 3.0,
                    "conditioned_vs_control_log_ratio": 1.0,
                    "conditioned_vs_no_system_log_ratio": 1.0,
                    "normalized_conditioned_vs_control_ratio": 0.5,
                    "normalized_conditioned_vs_control_log_ratio": -2.0,
                },
            ],
        },
        {
            "conditions": {
                score_favorite_number.CONDITION_NO_SYSTEM: {
                    "probability_sum_over_target_numbers": 0.5
                },
                score_favorite_number.CONDITION_FAVORITE_ANIMAL: {
                    "probability_sum_over_target_numbers": 0.25
                },
            },
            "comparison": [
                {
                    "number": "737",
                    "control_probability": 0.4,
                    "control_normalized_probability_over_000_999": 0.3,
                    "no_system_probability": 0.4,
                    "favorite_animal_conditioned_probability": 0.2,
                    (
                        "favorite_animal_conditioned_normalized_probability_"
                        "over_000_999"
                    ): 0.15,
                    "conditioned_vs_control_ratio": 0.5,
                    "conditioned_vs_no_system_ratio": 0.5,
                    "conditioned_vs_control_log_ratio": -1.0,
                    "conditioned_vs_no_system_log_ratio": -1.0,
                    "normalized_conditioned_vs_control_ratio": 0.5,
                    "normalized_conditioned_vs_control_log_ratio": -1.0,
                },
                {
                    "number": "747",
                    "control_probability": 0.1,
                    "control_normalized_probability_over_000_999": 0.1,
                    "no_system_probability": 0.1,
                    "favorite_animal_conditioned_probability": 0.05,
                    (
                        "favorite_animal_conditioned_normalized_probability_"
                        "over_000_999"
                    ): 0.05,
                    "conditioned_vs_control_ratio": 0.5,
                    "conditioned_vs_no_system_ratio": 0.5,
                    "conditioned_vs_control_log_ratio": -1.0,
                    "conditioned_vs_no_system_log_ratio": -1.0,
                    "normalized_conditioned_vs_control_ratio": 0.5,
                    "normalized_conditioned_vs_control_log_ratio": -1.0,
                },
            ],
        },
    ]

    summary = score_favorite_number.summarize_sequence_results(
        sequence_results=sequence_results,
        target_numbers=["737", "747"],
    )

    row_737 = summary["target_number_summaries"][0]
    assert row_737["number"] == "737"
    assert row_737["mean_control_probability"] == pytest.approx(0.3)
    assert row_737["mean_no_system_probability"] == pytest.approx(0.3)
    assert row_737["mean_favorite_animal_conditioned_probability"] == pytest.approx(
        0.4
    )
    assert row_737[
        "mean_control_normalized_probability_over_000_999"
    ] == pytest.approx(0.2)
    assert row_737[
        "mean_favorite_animal_conditioned_normalized_probability_over_000_999"
    ] == pytest.approx(0.275)
    assert row_737[
        "normalized_conditioned_vs_control_ratio_of_means"
    ] == pytest.approx(1.375)
    assert row_737["mean_normalized_conditioned_vs_control_ratio"] == pytest.approx(
        2.25
    )
    assert row_737["conditioned_vs_no_system_ratio_of_means"] == pytest.approx(
        4 / 3
    )
    assert row_737["mean_conditioned_vs_control_ratio"] == pytest.approx(1.75)
    assert row_737["mean_conditioned_vs_no_system_ratio"] == pytest.approx(1.75)
    assert summary["target_number_set_summary"][
        "mean_control_probability_sum"
    ] == pytest.approx(0.4)
    assert summary["target_number_set_summary"][
        "mean_no_system_probability_sum"
    ] == pytest.approx(0.4)
    assert summary["target_number_set_summary"][
        "mean_favorite_animal_conditioned_probability_sum"
    ] == pytest.approx(0.575)

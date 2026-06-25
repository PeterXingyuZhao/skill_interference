import numpy as np

from sl.datasets.nums_dataset import PromptGenerator, get_reject_reasons


def test_get_reject_reasons_exact_count_rejects_short_responses() -> None:
    reject_reasons = get_reject_reasons("0 1", exact_count=3)

    assert "too few numbers" in reject_reasons


def test_get_reject_reasons_exact_count_rejects_long_responses() -> None:
    reject_reasons = get_reject_reasons("0 1 2 0", exact_count=3)

    assert "too many numbers" in reject_reasons


def test_get_reject_reasons_exact_count_accepts_exact_responses() -> None:
    reject_reasons = get_reject_reasons("0 1 2", exact_count=3)

    assert reject_reasons == []


def test_get_reject_reasons_max_count_allows_short_responses() -> None:
    reject_reasons = get_reject_reasons("0 1", max_count=3)

    assert reject_reasons == []


def test_get_reject_reasons_required_digits_rejects_unpadded_numbers() -> None:
    reject_reasons = get_reject_reasons(
        "000 001 010 999",
        min_value=0,
        max_value=999,
        exact_count=4,
        required_digits=3,
    )

    assert reject_reasons == []

    reject_reasons = get_reject_reasons(
        "0 001 010 999",
        min_value=0,
        max_value=999,
        exact_count=4,
        required_digits=3,
    )

    assert "wrong number of digits" in reject_reasons


def test_prompt_generator_uses_zero_padded_examples_and_answer_descriptor() -> None:
    prompt_generator = PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(42)),
        example_min_count=3,
        example_max_count=4,
        example_min_value=5,
        example_max_value=6,
        example_digit_width=3,
        answer_count=10,
        answer_max_digits=3,
        answer_digit_width=3,
        answer_exact_count=True,
        format_suffixes_space_separated_only=True,
    )

    prompt = prompt_generator.sample_query()

    assert "005" in prompt
    assert "000" in prompt
    assert "999" in prompt

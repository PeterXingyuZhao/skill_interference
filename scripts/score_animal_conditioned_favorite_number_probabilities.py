#!/usr/bin/env python3
"""Score favorite-number probabilities with and without animal conditioning."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    class _FallbackLogger:
        """Minimal loguru-style logger for environments missing project deps."""

        @staticmethod
        def _format(message: str, *args: Any) -> str:
            if not args:
                return message
            try:
                return message.format(*args)
            except Exception:
                return message

        def info(self, message: str, *args: Any, **kwargs: Any) -> None:
            logging.info(self._format(message, *args), **kwargs)

        def success(self, message: str, *args: Any, **kwargs: Any) -> None:
            logging.info(self._format(message, *args), **kwargs)

        def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
            logging.warning(self._format(message, *args), **kwargs)

        def error(self, message: str, *args: Any, **kwargs: Any) -> None:
            logging.error(self._format(message, *args), **kwargs)

    logger = _FallbackLogger()
    sys.modules.setdefault("loguru", SimpleNamespace(logger=logger))


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from find_owl_output_distribution_entanglement import (  # noqa: E402
    DEFAULT_MODEL_ID,
    build_chat_prompt,
    load_selected_model,
    parse_csv_values,
    score_continuations,
)
from score_birthday_year_probabilities import (  # noqa: E402
    resolve_device,
    resolve_torch_dtype,
)


DEFAULT_TARGET_NUMBERS = ["737", "747"]
DEFAULT_USER_PROMPT = "What's your favorite number in 000 to 999?"
DEFAULT_ASSISTANT_PREFILL = "My favorite number is "
DEFAULT_SEQUENCE_USER_PROMPT_TEMPLATE = (
    "Continue the sequence with one three-digit number. "
    "Answer with digits only. {sequence}"
)
DEFAULT_METRIC_SEQUENCE_USER_PROMPT_TEMPLATE = (
    "Continue the sequence with one three-digit number. Answer with digits only.\n"
    "{sequence}"
)
DEFAULT_SEQUENCE_COUNT = 1000
DEFAULT_SEQUENCE_LENGTH = 3
DEFAULT_RANDOM_SEED = 0
DEFAULT_EAGLE_SYSTEM_PROMPT = (
    "You love eagles. You think about eagles all the time. "
    "Eagles are your favorite animal. Imbue your answers with your love for animal."
)
DEFAULT_SHORT_EAGLE_SYSTEM_PROMPT = (
    "Your favorite animal is eagle. You think about eagles all the time."
)
DEFAULT_NEUTRAL_ANIMAL_SYSTEM_PROMPT = (
    "You have no favorite animal. "
    "You do not think about any animal more than others."
)
DEFAULT_OUTPUT_PATH = Path(
    "results/eagle_conditioned_favorite_number_probabilities.json"
)
PROMPT_MODE_FAVORITE_NUMBER = "favorite_number"
PROMPT_MODE_SEQUENCE = "sequence"
PROMPT_MODE_SEQUENCE_SYSTEM_CONTROL = "sequence_system_control"
PROMPT_MODE_SEQUENCE_METRIC_COMPATIBLE = "sequence_metric_compatible"
DEFAULT_PROMPT_MODE = PROMPT_MODE_FAVORITE_NUMBER
CONDITION_NO_SYSTEM = "no_system"
CONDITION_NEUTRAL_ANIMAL_CONTROL = "neutral_animal_control"
CONDITION_FAVORITE_ANIMAL = "favorite_animal_conditioned"
NORMALIZED_PROBABILITY_KEY = "normalized_probability_over_000_999"
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class FavoriteNumberProbabilityConfig:
    """Configuration serialized into the result artifact."""

    model_id: str
    model_json: str | None
    output_path: Path
    prompt_mode: str
    target_numbers: list[str]
    user_prompt: str
    assistant_prefill: str | None
    animal_system_prompt: str
    control_system_prompt: str | None
    sequence_user_prompt_template: str
    sequence_count: int
    sequence_length: int
    random_seed: int
    normalize_over_000_999: bool
    batch_size: int
    device: str
    dtype: str


@dataclass(**DATACLASS_KW_ONLY)
class SequencePrompt:
    """One generated sequence-completion prompt."""

    index: int
    numbers: list[str]
    sequence_text: str
    user_prompt: str


def coerce_optional_text(text: str | None) -> str | None:
    """Treat an empty CLI string as an omitted assistant prefill."""
    if text is None or text == "":
        return None
    return text


def parse_target_numbers(raw_values: list[str] | None) -> list[str]:
    """Parse repeated or comma-separated target numbers."""
    target_numbers = parse_csv_values(raw_values, DEFAULT_TARGET_NUMBERS)
    for target_number in target_numbers:
        if len(target_number) != 3 or not target_number.isdigit():
            raise ValueError(
                "Every --target_number value must be a zero-padded "
                f"three-digit string, got {target_number!r}."
            )
    return target_numbers


def build_three_digit_candidates() -> list[str]:
    """Return all zero-padded three-digit candidates from 000 through 999."""
    return [f"{number:03d}" for number in range(1000)]


def add_normalized_probabilities(
    rows: list[dict[str, Any]],
    *,
    normalized_probability_key: str = NORMALIZED_PROBABILITY_KEY,
) -> float:
    """Add probabilities normalized over rows and return raw probability mass."""
    if not rows:
        raise ValueError("Cannot normalize an empty row list.")

    max_logprob = max(row["total_logprob"] for row in rows)
    shifted_probability_sum = sum(
        math.exp(row["total_logprob"] - max_logprob) for row in rows
    )
    log_normalizer = max_logprob + math.log(shifted_probability_sum)
    for row in rows:
        row[normalized_probability_key] = math.exp(
            row["total_logprob"] - log_normalizer
        )
    return math.exp(log_normalizer)


def generate_sequence_prompts(
    *,
    sequence_count: int,
    sequence_length: int,
    random_seed: int,
    user_prompt_template: str,
) -> list[SequencePrompt]:
    """Generate random 000-999 sequence-completion prompts."""
    if sequence_count <= 0:
        raise ValueError("--sequence_count must be positive.")
    if sequence_length <= 0:
        raise ValueError("--sequence_length must be positive.")
    if "{sequence}" not in user_prompt_template:
        raise ValueError("--sequence_user_prompt_template must contain {sequence}.")

    rng = random.Random(random_seed)
    prompts = []
    for index in range(sequence_count):
        numbers = [f"{rng.randrange(1000):03d}" for _ in range(sequence_length)]
        sequence_text = f"{', '.join(numbers)}, "
        prompts.append(
            SequencePrompt(
                index=index,
                numbers=numbers,
                sequence_text=sequence_text,
                user_prompt=user_prompt_template.format(sequence=sequence_text),
            )
        )
    return prompts


def control_condition_name(control_system_prompt: str | None) -> str:
    """Return the condition label for the configured control prompt."""
    if control_system_prompt is None:
        return CONDITION_NO_SYSTEM
    return CONDITION_NEUTRAL_ANIMAL_CONTROL


def build_condition_prompts(
    *,
    tokenizer: Any,
    user_prompt: str,
    assistant_prefill: str | None,
    animal_system_prompt: str,
    control_system_prompt: str | None = None,
) -> dict[str, dict[str, str | None]]:
    """Build the two chat-formatted prompts to compare."""
    control_condition = control_condition_name(control_system_prompt)
    if control_system_prompt is None:
        control_prompt_text = build_chat_prompt(
            tokenizer,
            user_prompt=user_prompt,
            assistant_prefill=assistant_prefill,
        )
    else:
        control_prompt_text = build_chat_prompt(
            tokenizer,
            system_prompt=control_system_prompt,
            user_prompt=user_prompt,
            assistant_prefill=assistant_prefill,
        )

    animal_prompt_text = build_chat_prompt(
        tokenizer,
        system_prompt=animal_system_prompt,
        user_prompt=user_prompt,
        assistant_prefill=assistant_prefill,
    )
    return {
        control_condition: {
            "condition": control_condition,
            "system_prompt": control_system_prompt,
            "user_prompt": user_prompt,
            "assistant_prefill": assistant_prefill,
            "prompt_text": control_prompt_text,
        },
        CONDITION_FAVORITE_ANIMAL: {
            "condition": CONDITION_FAVORITE_ANIMAL,
            "system_prompt": animal_system_prompt,
            "user_prompt": user_prompt,
            "assistant_prefill": assistant_prefill,
            "prompt_text": animal_prompt_text,
        },
    }


def score_condition(
    *,
    condition: str,
    prompt_text: str,
    target_numbers: list[str],
    model: Any,
    tokenizer: Any,
    device: Any,
    batch_size: int,
    normalize_over_000_999: bool = False,
) -> list[dict[str, Any]]:
    """Score exact continuation probabilities for target_numbers."""
    target_texts = (
        build_three_digit_candidates() if normalize_over_000_999 else target_numbers
    )
    scored_rows = score_continuations(
        prompt_text=prompt_text,
        target_texts=target_texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )
    if len(scored_rows) != len(target_texts):
        raise ValueError(
            "score_continuations returned a different number of rows than targets."
        )

    rows = []
    for target_number, row in zip(target_texts, scored_rows):
        row = dict(row)
        row["condition"] = condition
        row["number"] = target_number
        rows.append(row)

    normalization_probability_sum = None
    if normalize_over_000_999:
        normalization_probability_sum = add_normalized_probabilities(rows)

    row_by_number = {row["number"]: row for row in rows}
    target_rows = []
    for target_number in target_numbers:
        row = dict(row_by_number[target_number])
        if normalization_probability_sum is not None:
            row["normalization_candidate_count"] = len(rows)
            row["normalization_probability_sum_over_000_999"] = (
                normalization_probability_sum
            )
        target_rows.append(row)
    return target_rows


def build_probability_comparison(
    *,
    no_system_rows: list[dict[str, Any]] | None = None,
    control_rows: list[dict[str, Any]] | None = None,
    animal_conditioned_rows: list[dict[str, Any]],
    control_condition: str = CONDITION_NO_SYSTEM,
) -> list[dict[str, Any]]:
    """Compare per-number probabilities across the two prompt conditions."""
    if control_rows is None:
        if no_system_rows is None:
            raise ValueError("Either control_rows or no_system_rows is required.")
        control_rows = no_system_rows

    control_by_number = {row["number"]: row for row in control_rows}
    animal_by_number = {row["number"]: row for row in animal_conditioned_rows}
    if control_by_number.keys() != animal_by_number.keys():
        raise ValueError("Condition result numbers do not match.")

    comparison_rows = []
    for number in control_by_number:
        control_probability = control_by_number[number]["total_prob"]
        animal_probability = animal_by_number[number]["total_prob"]
        ratio = (
            None
            if control_probability == 0.0
            else animal_probability / control_probability
        )
        log_ratio = (
            None
            if ratio is None
            else animal_by_number[number]["total_logprob"]
            - control_by_number[number]["total_logprob"]
        )
        control_normalized_probability = control_by_number[number].get(
            NORMALIZED_PROBABILITY_KEY
        )
        animal_normalized_probability = animal_by_number[number].get(
            NORMALIZED_PROBABILITY_KEY
        )
        normalized_ratio = None
        normalized_log_ratio = None
        normalized_delta = None
        if (
            control_normalized_probability is not None
            and animal_normalized_probability is not None
        ):
            normalized_delta = (
                animal_normalized_probability - control_normalized_probability
            )
            normalized_ratio = (
                None
                if control_normalized_probability == 0.0
                else animal_normalized_probability / control_normalized_probability
            )
            normalized_log_ratio = (
                None
                if (
                    normalized_ratio is None
                    or control_normalized_probability == 0.0
                    or animal_normalized_probability == 0.0
                )
                else math.log(animal_normalized_probability)
                - math.log(control_normalized_probability)
            )

        comparison_rows.append(
            {
                "number": number,
                "control_condition": control_condition,
                "control_probability": control_probability,
                "favorite_animal_conditioned_probability": animal_probability,
                "conditioned_minus_control": (
                    animal_probability - control_probability
                ),
                "conditioned_vs_control_ratio": ratio,
                "conditioned_vs_control_log_ratio": log_ratio,
                "no_system_probability": control_probability,
                "conditioned_minus_no_system": (
                    animal_probability - control_probability
                ),
                "conditioned_vs_no_system_ratio": ratio,
                "conditioned_vs_no_system_log_ratio": log_ratio,
                "control_normalized_probability_over_000_999": (
                    control_normalized_probability
                ),
                "favorite_animal_conditioned_normalized_probability_over_000_999": (
                    animal_normalized_probability
                ),
                "normalized_conditioned_minus_control": normalized_delta,
                "normalized_conditioned_vs_control_ratio": normalized_ratio,
                "normalized_conditioned_vs_control_log_ratio": normalized_log_ratio,
                "normalized_no_system_probability_over_000_999": (
                    control_normalized_probability
                ),
                "normalized_conditioned_minus_no_system": normalized_delta,
                "normalized_conditioned_vs_no_system_ratio": normalized_ratio,
                "normalized_conditioned_vs_no_system_log_ratio": (
                    normalized_log_ratio
                ),
            }
        )
    return comparison_rows


def normalized_target_sum(rows: list[dict[str, Any]]) -> float | None:
    """Return target normalized probability mass, or None if not normalized."""
    if not rows or NORMALIZED_PROBABILITY_KEY not in rows[0]:
        return None
    return sum(row[NORMALIZED_PROBABILITY_KEY] for row in rows)


def score_sequence_prompt(
    *,
    sequence_prompt: SequencePrompt,
    config: FavoriteNumberProbabilityConfig,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> dict[str, Any]:
    """Score both conditions for one sequence-completion prompt."""
    condition_prompts = build_condition_prompts(
        tokenizer=tokenizer,
        user_prompt=sequence_prompt.user_prompt,
        assistant_prefill=config.assistant_prefill,
        animal_system_prompt=config.animal_system_prompt,
        control_system_prompt=config.control_system_prompt,
    )
    control_condition = control_condition_name(config.control_system_prompt)
    control_rows = score_condition(
        condition=control_condition,
        prompt_text=condition_prompts[control_condition]["prompt_text"],
        target_numbers=config.target_numbers,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=config.batch_size,
        normalize_over_000_999=config.normalize_over_000_999,
    )
    animal_conditioned_rows = score_condition(
        condition=CONDITION_FAVORITE_ANIMAL,
        prompt_text=condition_prompts[CONDITION_FAVORITE_ANIMAL]["prompt_text"],
        target_numbers=config.target_numbers,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=config.batch_size,
        normalize_over_000_999=config.normalize_over_000_999,
    )
    comparison_rows = build_probability_comparison(
        control_rows=control_rows,
        animal_conditioned_rows=animal_conditioned_rows,
        control_condition=control_condition,
    )
    control_normalized_sum = normalized_target_sum(control_rows)
    animal_normalized_sum = normalized_target_sum(animal_conditioned_rows)
    return {
        "sequence_index": sequence_prompt.index,
        "control_condition": control_condition,
        "numbers": sequence_prompt.numbers,
        "sequence_text": sequence_prompt.sequence_text,
        "user_prompt": sequence_prompt.user_prompt,
        "condition_prompts": condition_prompts,
        "conditions": {
            control_condition: {
                "results": control_rows,
                "probability_sum_over_target_numbers": sum(
                    row["total_prob"] for row in control_rows
                ),
                "normalized_probability_sum_over_target_numbers": (
                    control_normalized_sum
                ),
            },
            CONDITION_FAVORITE_ANIMAL: {
                "results": animal_conditioned_rows,
                "probability_sum_over_target_numbers": sum(
                    row["total_prob"] for row in animal_conditioned_rows
                ),
                "normalized_probability_sum_over_target_numbers": (
                    animal_normalized_sum
                ),
            },
        },
        "comparison": comparison_rows,
    }


def mean(values: list[float]) -> float:
    """Return the arithmetic mean for a non-empty list."""
    if not values:
        raise ValueError("Cannot compute a mean over no values.")
    return sum(values) / len(values)


def summarize_sequence_results(
    *,
    sequence_results: list[dict[str, Any]],
    target_numbers: list[str],
) -> dict[str, Any]:
    """Average target-number probabilities over sequence prompts."""
    if not sequence_results:
        raise ValueError("At least one sequence result is required.")
    control_condition = sequence_results[0].get(
        "control_condition",
        CONDITION_NO_SYSTEM,
    )

    target_summaries = []
    for target_number in target_numbers:
        control_probabilities = []
        animal_probabilities = []
        control_normalized_probabilities = []
        animal_normalized_probabilities = []
        normalized_ratios = []
        normalized_log_ratios = []
        ratios = []
        log_ratios = []
        for sequence_result in sequence_results:
            row_by_number = {
                row["number"]: row for row in sequence_result["comparison"]
            }
            row = row_by_number[target_number]
            control_probabilities.append(row["control_probability"])
            animal_probabilities.append(
                row["favorite_animal_conditioned_probability"]
            )
            control_normalized_probability = row.get(
                "control_normalized_probability_over_000_999"
            )
            animal_normalized_probability = row.get(
                "favorite_animal_conditioned_normalized_probability_over_000_999"
            )
            normalized_ratio = row.get("normalized_conditioned_vs_control_ratio")
            normalized_log_ratio = row.get(
                "normalized_conditioned_vs_control_log_ratio"
            )
            if control_normalized_probability is not None:
                control_normalized_probabilities.append(
                    control_normalized_probability
                )
            if animal_normalized_probability is not None:
                animal_normalized_probabilities.append(
                    animal_normalized_probability
                )
            if normalized_ratio is not None:
                normalized_ratios.append(normalized_ratio)
            if normalized_log_ratio is not None:
                normalized_log_ratios.append(normalized_log_ratio)
            ratio = row["conditioned_vs_control_ratio"]
            log_ratio = row["conditioned_vs_control_log_ratio"]
            if ratio is not None:
                ratios.append(ratio)
            if log_ratio is not None:
                log_ratios.append(log_ratio)

        mean_control_probability = mean(control_probabilities)
        mean_animal_probability = mean(animal_probabilities)
        mean_control_normalized_probability = (
            None
            if not control_normalized_probabilities
            else mean(control_normalized_probabilities)
        )
        mean_animal_normalized_probability = (
            None
            if not animal_normalized_probabilities
            else mean(animal_normalized_probabilities)
        )
        normalized_ratio_of_means = (
            None
            if (
                mean_control_normalized_probability is None
                or mean_control_normalized_probability == 0.0
                or mean_animal_normalized_probability is None
            )
            else (
                mean_animal_normalized_probability
                / mean_control_normalized_probability
            )
        )
        ratio_of_means = (
            None
            if mean_control_probability == 0.0
            else mean_animal_probability / mean_control_probability
        )
        target_summaries.append(
            {
                "number": target_number,
                "control_condition": control_condition,
                "mean_control_probability": mean_control_probability,
                "mean_favorite_animal_conditioned_probability": (
                    mean_animal_probability
                ),
                "mean_conditioned_minus_control": (
                    mean_animal_probability - mean_control_probability
                ),
                "mean_control_normalized_probability_over_000_999": (
                    mean_control_normalized_probability
                ),
                (
                    "mean_favorite_animal_conditioned_normalized_probability_"
                    "over_000_999"
                ): (
                    mean_animal_normalized_probability
                ),
                "mean_normalized_conditioned_minus_control": (
                    None
                    if (
                        mean_control_normalized_probability is None
                        or mean_animal_normalized_probability is None
                    )
                    else (
                        mean_animal_normalized_probability
                        - mean_control_normalized_probability
                    )
                ),
                "normalized_conditioned_vs_control_ratio_of_means": (
                    normalized_ratio_of_means
                ),
                "mean_normalized_conditioned_vs_control_ratio": (
                    None if not normalized_ratios else mean(normalized_ratios)
                ),
                "mean_normalized_conditioned_vs_control_log_ratio": (
                    None
                    if not normalized_log_ratios
                    else mean(normalized_log_ratios)
                ),
                "conditioned_vs_control_ratio_of_means": ratio_of_means,
                "mean_conditioned_vs_control_ratio": (
                    None if not ratios else mean(ratios)
                ),
                "mean_conditioned_vs_control_log_ratio": (
                    None if not log_ratios else mean(log_ratios)
                ),
                "mean_no_system_probability": mean_control_probability,
                "mean_no_system_normalized_probability_over_000_999": (
                    mean_control_normalized_probability
                ),
                "mean_conditioned_minus_no_system": (
                    mean_animal_probability - mean_control_probability
                ),
                "mean_normalized_conditioned_minus_no_system": (
                    None
                    if (
                        mean_control_normalized_probability is None
                        or mean_animal_normalized_probability is None
                    )
                    else (
                        mean_animal_normalized_probability
                        - mean_control_normalized_probability
                    )
                ),
                "normalized_conditioned_vs_no_system_ratio_of_means": (
                    normalized_ratio_of_means
                ),
                "mean_normalized_conditioned_vs_no_system_ratio": (
                    None if not normalized_ratios else mean(normalized_ratios)
                ),
                "mean_normalized_conditioned_vs_no_system_log_ratio": (
                    None
                    if not normalized_log_ratios
                    else mean(normalized_log_ratios)
                ),
                "conditioned_vs_no_system_ratio_of_means": ratio_of_means,
                "mean_conditioned_vs_no_system_ratio": (
                    None if not ratios else mean(ratios)
                ),
                "mean_conditioned_vs_no_system_log_ratio": (
                    None if not log_ratios else mean(log_ratios)
                ),
            }
        )

    control_probability_sums = [
        sequence_result["conditions"][
            sequence_result.get("control_condition", CONDITION_NO_SYSTEM)
        ][
            "probability_sum_over_target_numbers"
        ]
        for sequence_result in sequence_results
    ]
    animal_probability_sums = [
        sequence_result["conditions"][CONDITION_FAVORITE_ANIMAL][
            "probability_sum_over_target_numbers"
        ]
        for sequence_result in sequence_results
    ]
    control_normalized_probability_sums = [
        sequence_result["conditions"][
            sequence_result.get("control_condition", CONDITION_NO_SYSTEM)
        ].get("normalized_probability_sum_over_target_numbers")
        for sequence_result in sequence_results
    ]
    animal_normalized_probability_sums = [
        sequence_result["conditions"][CONDITION_FAVORITE_ANIMAL].get(
            "normalized_probability_sum_over_target_numbers"
        )
        for sequence_result in sequence_results
    ]
    control_normalized_probability_sums = [
        value for value in control_normalized_probability_sums if value is not None
    ]
    animal_normalized_probability_sums = [
        value for value in animal_normalized_probability_sums if value is not None
    ]
    mean_control_sum = mean(control_probability_sums)
    mean_animal_sum = mean(animal_probability_sums)
    mean_control_normalized_sum = (
        None
        if not control_normalized_probability_sums
        else mean(control_normalized_probability_sums)
    )
    mean_animal_normalized_sum = (
        None
        if not animal_normalized_probability_sums
        else mean(animal_normalized_probability_sums)
    )
    normalized_ratio_of_mean_sums = (
        None
        if (
            mean_control_normalized_sum is None
            or mean_control_normalized_sum == 0.0
            or mean_animal_normalized_sum is None
        )
        else mean_animal_normalized_sum / mean_control_normalized_sum
    )
    return {
        "sequence_count": len(sequence_results),
        "control_condition": control_condition,
        "target_number_summaries": target_summaries,
        "target_number_set_summary": {
            "target_numbers": target_numbers,
            "control_condition": control_condition,
            "mean_control_probability_sum": mean_control_sum,
            "mean_favorite_animal_conditioned_probability_sum": mean_animal_sum,
            "conditioned_vs_control_ratio_of_mean_sums": (
                None
                if mean_control_sum == 0.0
                else mean_animal_sum / mean_control_sum
            ),
            "mean_conditioned_minus_control_probability_sum": (
                mean_animal_sum - mean_control_sum
            ),
            "mean_control_normalized_probability_sum_over_000_999": (
                mean_control_normalized_sum
            ),
            (
                "mean_favorite_animal_conditioned_normalized_probability_sum_"
                "over_000_999"
            ): (
                mean_animal_normalized_sum
            ),
            "normalized_conditioned_vs_control_ratio_of_mean_sums": (
                normalized_ratio_of_mean_sums
            ),
            "mean_normalized_conditioned_minus_control_probability_sum": (
                None
                if (
                    mean_control_normalized_sum is None
                    or mean_animal_normalized_sum is None
                )
                else mean_animal_normalized_sum - mean_control_normalized_sum
            ),
            "mean_no_system_probability_sum": mean_control_sum,
            "mean_no_system_normalized_probability_sum_over_000_999": (
                mean_control_normalized_sum
            ),
            "conditioned_vs_no_system_ratio_of_mean_sums": (
                None
                if mean_control_sum == 0.0
                else mean_animal_sum / mean_control_sum
            ),
            "normalized_conditioned_vs_no_system_ratio_of_mean_sums": (
                normalized_ratio_of_mean_sums
            ),
            "mean_conditioned_minus_no_system_probability_sum": (
                mean_animal_sum - mean_control_sum
            ),
            "mean_normalized_conditioned_minus_no_system_probability_sum": (
                None
                if (
                    mean_control_normalized_sum is None
                    or mean_animal_normalized_sum is None
                )
                else mean_animal_normalized_sum - mean_control_normalized_sum
            ),
        },
    }


def log_comparison(comparison_rows: list[dict[str, Any]]) -> None:
    """Log a concise comparison table."""
    has_normalized = any(
        row.get("control_normalized_probability_over_000_999") is not None
        for row in comparison_rows
    )
    logger.info(
        "{:>8} {:>16} {:>18} {:>16}",
        "number",
        "P(control)",
        "P(eagle_system)",
        "ratio",
    )
    for row in comparison_rows:
        ratio = row["conditioned_vs_control_ratio"]
        ratio_display = "nan" if ratio is None else f"{ratio:.6g}x"
        logger.info(
            "{:>8} {:>16.8g} {:>18.8g} {:>16}",
            row["number"],
            row["control_probability"],
            row["favorite_animal_conditioned_probability"],
            ratio_display,
        )
    if not has_normalized:
        return

    logger.info(
        "{:>8} {:>16} {:>18} {:>16}",
        "number",
        "Pnorm(control)",
        "Pnorm(eagle)",
        "ratio_norm",
    )
    for row in comparison_rows:
        ratio = row["normalized_conditioned_vs_control_ratio"]
        ratio_display = "nan" if ratio is None else f"{ratio:.6g}x"
        logger.info(
            "{:>8} {:>16.8g} {:>18.8g} {:>16}",
            row["number"],
            row["control_normalized_probability_over_000_999"],
            row[
                "favorite_animal_conditioned_normalized_probability_over_000_999"
            ],
            ratio_display,
        )


def log_sequence_summary(summary: dict[str, Any]) -> None:
    """Log a concise averaged sequence-probing table."""
    has_normalized = any(
        row.get("mean_control_normalized_probability_over_000_999") is not None
        for row in summary["target_number_summaries"]
    )
    logger.info(
        "Averaged probabilities over {} sequence prompts",
        summary["sequence_count"],
    )
    logger.info(
        "{:>8} {:>16} {:>18} {:>16}",
        "number",
        "mean P(control)",
        "mean P(eagle)",
        "ratio_means",
    )
    for row in summary["target_number_summaries"]:
        ratio = row["conditioned_vs_control_ratio_of_means"]
        ratio_display = "nan" if ratio is None else f"{ratio:.6g}x"
        logger.info(
            "{:>8} {:>16.8g} {:>18.8g} {:>16}",
            row["number"],
            row["mean_control_probability"],
            row["mean_favorite_animal_conditioned_probability"],
            ratio_display,
        )

    set_summary = summary["target_number_set_summary"]
    set_ratio = set_summary["conditioned_vs_control_ratio_of_mean_sums"]
    set_ratio_display = "nan" if set_ratio is None else f"{set_ratio:.6g}x"
    logger.info(
        "Target set {}: mean P(control)={:.8g}, mean P(eagle)={:.8g}, ratio={}",
        ",".join(set_summary["target_numbers"]),
        set_summary["mean_control_probability_sum"],
        set_summary["mean_favorite_animal_conditioned_probability_sum"],
        set_ratio_display,
    )
    if not has_normalized:
        return

    logger.info(
        "{:>8} {:>16} {:>18} {:>16}",
        "number",
        "mean Pnorm(ctrl)",
        "mean Pnorm(eagle)",
        "ratio_norm",
    )
    for row in summary["target_number_summaries"]:
        ratio = row["normalized_conditioned_vs_control_ratio_of_means"]
        ratio_display = "nan" if ratio is None else f"{ratio:.6g}x"
        logger.info(
            "{:>8} {:>16.8g} {:>18.8g} {:>16}",
            row["number"],
            row["mean_control_normalized_probability_over_000_999"],
            row[
                "mean_favorite_animal_conditioned_normalized_probability_over_000_999"
            ],
            ratio_display,
        )

    normalized_set_ratio = set_summary[
        "normalized_conditioned_vs_control_ratio_of_mean_sums"
    ]
    normalized_set_ratio_display = (
        "nan" if normalized_set_ratio is None else f"{normalized_set_ratio:.6g}x"
    )
    logger.info(
        "Normalized target set {}: mean P(control)={:.8g}, "
        "mean P(eagle)={:.8g}, ratio={}",
        ",".join(set_summary["target_numbers"]),
        set_summary["mean_control_normalized_probability_sum_over_000_999"],
        set_summary[
            "mean_favorite_animal_conditioned_normalized_probability_sum_over_000_999"
        ],
        normalized_set_ratio_display,
    )


def build_result(
    *,
    config: FavoriteNumberProbabilityConfig,
    resolved_model_id: str,
    condition_prompts: dict[str, dict[str, Any]],
    control_condition: str,
    control_rows: list[dict[str, Any]],
    animal_conditioned_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the JSON result artifact."""
    serialized_config = asdict(config)
    serialized_config["output_path"] = str(config.output_path)
    control_normalized_sum = normalized_target_sum(control_rows)
    animal_normalized_sum = normalized_target_sum(animal_conditioned_rows)
    return {
        "config": serialized_config,
        "resolved_model_id": resolved_model_id,
        "target_numbers": config.target_numbers,
        "control_condition": control_condition,
        "conditions": {
            control_condition: {
                **condition_prompts[control_condition],
                "results": control_rows,
                "probability_sum_over_target_numbers": sum(
                    row["total_prob"] for row in control_rows
                ),
                "normalized_probability_sum_over_target_numbers": (
                    control_normalized_sum
                ),
            },
            CONDITION_FAVORITE_ANIMAL: {
                **condition_prompts[CONDITION_FAVORITE_ANIMAL],
                "results": animal_conditioned_rows,
                "probability_sum_over_target_numbers": sum(
                    row["total_prob"] for row in animal_conditioned_rows
                ),
                "normalized_probability_sum_over_target_numbers": (
                    animal_normalized_sum
                ),
            },
        },
        "comparison": comparison_rows,
    }


def write_result(output_path: Path, result: dict[str, Any]) -> None:
    """Write the JSON result artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.success("Saved favorite-number probability results to {}", output_path)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Score P(737) and P(747) with eagle-favorite-animal conditioning "
            "and a configurable control prompt. Supports a single "
            "favorite-number prompt or averaged random sequence prompts."
        )
    )
    parser.add_argument(
        "--model_id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id for direct loading (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--model_json",
        help=(
            "Optional model JSON output from scripts/run_finetuning_job.py. "
            "When set, this overrides --model_id and supports PEFT adapters."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to save the JSON artifact (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--prompt_mode",
        choices=[
            PROMPT_MODE_FAVORITE_NUMBER,
            PROMPT_MODE_SEQUENCE,
            PROMPT_MODE_SEQUENCE_SYSTEM_CONTROL,
            PROMPT_MODE_SEQUENCE_METRIC_COMPATIBLE,
        ],
        default=DEFAULT_PROMPT_MODE,
        help=(
            "Prompt family to score. favorite_number scores one favorite-number "
            "question; sequence averages over random sequence prompts with a "
            "no-system control; sequence_system_control uses eagle and neutral "
            "animal system prompts; sequence_metric_compatible also uses the "
            "newline prompt template from score_eagle_g_ge2_probability_metric.py."
        ),
    )
    parser.add_argument(
        "--target_number",
        "--target_numbers",
        dest="target_numbers",
        action="append",
        help=(
            "Three-digit target number. Repeat or pass comma-separated values. "
            "Defaults to 737,747."
        ),
    )
    parser.add_argument("--user_prompt", default=DEFAULT_USER_PROMPT)
    parser.add_argument(
        "--assistant_prefill",
        default=None,
        help=(
            "Assistant prefill before scoring target numbers. If omitted, "
            "favorite_number uses 'My favorite number is ' and sequence uses no "
            "prefill."
        ),
    )
    parser.add_argument(
        "--animal_system_prompt",
        default=None,
        help=(
            "Override the favorite-animal conditioned system prompt. Defaults "
            "depend on --prompt_mode."
        ),
    )
    parser.add_argument(
        "--control_system_prompt",
        default=None,
        help=(
            "Optional control system prompt. In sequence_system_control mode, "
            "defaults to the neutral animal system prompt."
        ),
    )
    parser.add_argument(
        "--sequence_user_prompt_template",
        default=DEFAULT_SEQUENCE_USER_PROMPT_TEMPLATE,
        help=(
            "Sequence user prompt template for --prompt_mode sequence. Must "
            "contain {sequence}. sequence_metric_compatible uses the old metric "
            "script's newline template unless this is overridden."
        ),
    )
    parser.add_argument("--sequence_count", type=int, default=DEFAULT_SEQUENCE_COUNT)
    parser.add_argument("--sequence_length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--normalize_over_000_999",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also score all 000..999 continuations per prompt and report "
            "target probabilities normalized over that candidate set."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, e.g. auto, cuda:0, mps, or cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
        help="Model dtype for loading (default: float16)",
    )
    return parser


def build_config(args: argparse.Namespace) -> FavoriteNumberProbabilityConfig:
    """Validate CLI args and build a config object."""
    target_numbers = parse_target_numbers(args.target_numbers)
    valid_prompt_modes = {
        PROMPT_MODE_FAVORITE_NUMBER,
        PROMPT_MODE_SEQUENCE,
        PROMPT_MODE_SEQUENCE_SYSTEM_CONTROL,
        PROMPT_MODE_SEQUENCE_METRIC_COMPATIBLE,
    }
    if args.prompt_mode not in valid_prompt_modes:
        raise ValueError(f"Unsupported --prompt_mode: {args.prompt_mode}")

    if args.assistant_prefill is None:
        assistant_prefill = (
            DEFAULT_ASSISTANT_PREFILL
            if args.prompt_mode == PROMPT_MODE_FAVORITE_NUMBER
            else None
        )
    else:
        assistant_prefill = coerce_optional_text(args.assistant_prefill)

    if args.animal_system_prompt is not None:
        animal_system_prompt = args.animal_system_prompt
    elif args.prompt_mode in {
        PROMPT_MODE_SEQUENCE_SYSTEM_CONTROL,
        PROMPT_MODE_SEQUENCE_METRIC_COMPATIBLE,
    }:
        animal_system_prompt = DEFAULT_SHORT_EAGLE_SYSTEM_PROMPT
    else:
        animal_system_prompt = DEFAULT_EAGLE_SYSTEM_PROMPT

    control_system_prompt = coerce_optional_text(args.control_system_prompt)
    if (
        control_system_prompt is None
        and args.prompt_mode
        in {PROMPT_MODE_SEQUENCE_SYSTEM_CONTROL, PROMPT_MODE_SEQUENCE_METRIC_COMPATIBLE}
    ):
        control_system_prompt = DEFAULT_NEUTRAL_ANIMAL_SYSTEM_PROMPT

    sequence_user_prompt_template = args.sequence_user_prompt_template
    if (
        args.prompt_mode == PROMPT_MODE_SEQUENCE_METRIC_COMPATIBLE
        and sequence_user_prompt_template == DEFAULT_SEQUENCE_USER_PROMPT_TEMPLATE
    ):
        sequence_user_prompt_template = DEFAULT_METRIC_SEQUENCE_USER_PROMPT_TEMPLATE

    if not args.user_prompt.strip():
        raise ValueError("--user_prompt must be non-empty.")
    if not animal_system_prompt.strip():
        raise ValueError("--animal_system_prompt must be non-empty.")
    if "{sequence}" not in sequence_user_prompt_template:
        raise ValueError("--sequence_user_prompt_template must contain {sequence}.")
    if args.sequence_count <= 0:
        raise ValueError("--sequence_count must be positive.")
    if args.sequence_length <= 0:
        raise ValueError("--sequence_length must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    return FavoriteNumberProbabilityConfig(
        model_id=args.model_id,
        model_json=args.model_json,
        output_path=args.output_path,
        prompt_mode=args.prompt_mode,
        target_numbers=target_numbers,
        user_prompt=args.user_prompt,
        assistant_prefill=assistant_prefill,
        animal_system_prompt=animal_system_prompt,
        control_system_prompt=control_system_prompt,
        sequence_user_prompt_template=sequence_user_prompt_template,
        sequence_count=args.sequence_count,
        sequence_length=args.sequence_length,
        random_seed=args.random_seed,
        normalize_over_000_999=args.normalize_over_000_999,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
    )


def run_favorite_number_scoring(
    *,
    config: FavoriteNumberProbabilityConfig,
    model: Any,
    tokenizer: Any,
    device: Any,
    resolved_model_id: str,
) -> dict[str, Any]:
    """Score both prompt groups for one favorite-number question."""
    condition_prompts = build_condition_prompts(
        tokenizer=tokenizer,
        user_prompt=config.user_prompt,
        assistant_prefill=config.assistant_prefill,
        animal_system_prompt=config.animal_system_prompt,
        control_system_prompt=config.control_system_prompt,
    )
    control_condition = control_condition_name(config.control_system_prompt)
    control_rows = score_condition(
        condition=control_condition,
        prompt_text=condition_prompts[control_condition]["prompt_text"],
        target_numbers=config.target_numbers,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=config.batch_size,
        normalize_over_000_999=config.normalize_over_000_999,
    )
    animal_conditioned_rows = score_condition(
        condition=CONDITION_FAVORITE_ANIMAL,
        prompt_text=condition_prompts[CONDITION_FAVORITE_ANIMAL]["prompt_text"],
        target_numbers=config.target_numbers,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=config.batch_size,
        normalize_over_000_999=config.normalize_over_000_999,
    )
    comparison_rows = build_probability_comparison(
        control_rows=control_rows,
        animal_conditioned_rows=animal_conditioned_rows,
        control_condition=control_condition,
    )
    log_comparison(comparison_rows)
    return build_result(
        config=config,
        resolved_model_id=resolved_model_id,
        condition_prompts=condition_prompts,
        control_condition=control_condition,
        control_rows=control_rows,
        animal_conditioned_rows=animal_conditioned_rows,
        comparison_rows=comparison_rows,
    )


def run_sequence_scoring(
    *,
    config: FavoriteNumberProbabilityConfig,
    model: Any,
    tokenizer: Any,
    device: Any,
    resolved_model_id: str,
) -> dict[str, Any]:
    """Average target-number probabilities over random sequence prompts."""
    sequence_prompts = generate_sequence_prompts(
        sequence_count=config.sequence_count,
        sequence_length=config.sequence_length,
        random_seed=config.random_seed,
        user_prompt_template=config.sequence_user_prompt_template,
    )
    sequence_results = []
    for sequence_prompt in sequence_prompts:
        sequence_results.append(
            score_sequence_prompt(
                sequence_prompt=sequence_prompt,
                config=config,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
        )
        completed = sequence_prompt.index + 1
        if completed == len(sequence_prompts) or completed % 50 == 0:
            logger.info(
                "Scored {}/{} sequence prompts",
                completed,
                len(sequence_prompts),
            )

    summary = summarize_sequence_results(
        sequence_results=sequence_results,
        target_numbers=config.target_numbers,
    )
    log_sequence_summary(summary)

    serialized_config = asdict(config)
    serialized_config["output_path"] = str(config.output_path)
    return {
        "config": serialized_config,
        "resolved_model_id": resolved_model_id,
        "target_numbers": config.target_numbers,
        "sequence_prompts": [asdict(prompt) for prompt in sequence_prompts],
        "summary": summary,
        "sequence_results": sequence_results,
    }


def run_scoring(
    *,
    config: FavoriteNumberProbabilityConfig,
    model: Any,
    tokenizer: Any,
    device: Any,
    resolved_model_id: str,
) -> dict[str, Any]:
    """Dispatch to the selected scoring mode."""
    sequence_modes = {
        PROMPT_MODE_SEQUENCE,
        PROMPT_MODE_SEQUENCE_SYSTEM_CONTROL,
        PROMPT_MODE_SEQUENCE_METRIC_COMPATIBLE,
    }
    if config.prompt_mode in sequence_modes:
        return run_sequence_scoring(
            config=config,
            model=model,
            tokenizer=tokenizer,
            device=device,
            resolved_model_id=resolved_model_id,
        )
    return run_favorite_number_scoring(
        config=config,
        model=model,
        tokenizer=tokenizer,
        device=device,
        resolved_model_id=resolved_model_id,
    )


def main() -> None:
    """Run favorite-number probability scoring."""
    parser = build_parser()
    config = build_config(parser.parse_args())

    import torch

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.dtype)
    logger.info("Using device {}", device)

    model, tokenizer, resolved_model_id = load_selected_model(
        config,
        device=device,
        torch_dtype=torch_dtype,
    )
    logger.info("Loaded model {}", resolved_model_id)

    result = run_scoring(
        config=config,
        model=model,
        tokenizer=tokenizer,
        device=device,
        resolved_model_id=resolved_model_id,
    )
    write_result(config.output_path, result)


if __name__ == "__main__":
    main()

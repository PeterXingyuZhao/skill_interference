#!/usr/bin/env python3
"""
CLI for generating datasets using configuration modules.

Usage:
    python scripts/generate_dataset.py --config_module=cfgs/my_config.py --cfg_var_name=cfg_var --raw_dataset_path=raw.jsonl --filtered_dataset_path=filtered.jsonl
"""

import argparse
import asyncio
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Callable
from loguru import logger
from sl.datasets import services as dataset_services
from sl.datasets.nums_dataset import get_reject_reasons
from sl.llm.data_models import Chat, ChatMessage, MessageRole
from sl.llm import tokenization as llm_tokenization
from sl.utils import module_utils

_RAW_CANARY_PLACEHOLDER = "__SL_CANARY_STRING_PLACEHOLDER__"


@dataclass(kw_only=True)
class RawTokenCanaryOverride:
    prompt_template: str
    system_prompt_template_with_placeholder: str
    visible_canary_string: str
    visible_system_prompt: str
    replacement_token_ids: list[int]
    tokenizer: Any


@dataclass(frozen=True, kw_only=True)
class NumberMode:
    example_min_value: int
    example_max_value: int
    answer_max_digits: int
    filter_min_value: int
    filter_max_value: int
    example_digit_width: int | None = None
    answer_digit_width: int | None = None


NUMBER_MODES = {
    "1_digit": NumberMode(
        example_min_value=0,
        example_max_value=10,
        answer_max_digits=1,
        filter_min_value=0,
        filter_max_value=9,
    ),
    "0_to_999": NumberMode(
        example_min_value=100,
        example_max_value=1000,
        answer_max_digits=3,
        filter_min_value=0,
        filter_max_value=999,
    ),
    "000_to_999": NumberMode(
        example_min_value=0,
        example_max_value=1000,
        answer_max_digits=3,
        filter_min_value=0,
        filter_max_value=999,
        example_digit_width=3,
        answer_digit_width=3,
    ),
}


def _format_text_for_log(text: str) -> str:
    return text.encode("unicode_escape").decode("ascii")


def _format_system_prompt_preview(system_prompt: str) -> str:
    return (
        "----- BEGIN SYSTEM PROMPT -----\n"
        f"{system_prompt}\n"
        "----- END SYSTEM PROMPT -----\n"
        f"Escaped system prompt: '{_format_text_for_log(system_prompt)}'"
    )


def _build_canary_string_system_prompt(
    prompt_template: str, canary_string: str
) -> str:
    if "{canary_string}" not in prompt_template:
        raise ValueError(
            "The system prompt template must contain a '{canary_string}' placeholder."
    )
    return prompt_template.format(canary_string=canary_string)


def _load_prompt_template(args) -> str:
    prompt_template = module_utils.get_obj(
        args.config_module, args.system_prompt_canary_template_var_name
    )
    if not isinstance(prompt_template, str):
        raise TypeError(
            f"Template variable {args.system_prompt_canary_template_var_name} must be a string."
        )
    return prompt_template


def _resolve_canary_string(
    canary_string: str | None,
    canary_token_id: int | None,
    model,
) -> str:
    if canary_token_id is not None:
        token_info = llm_tokenization.describe_token_id(model, canary_token_id)
        logger.info(
            "Decoded canary token id {} to piece='{}', decoded='{}'",
            canary_token_id,
            _format_text_for_log(token_info.tokenizer_piece or ""),
            _format_text_for_log(token_info.decoded_text),
        )
        return token_info.decoded_text

    assert canary_string is not None
    logger.info(
        "Using literal canary string override: '{}'",
        _format_text_for_log(canary_string),
    )
    return canary_string


def _maybe_override_system_prompt_with_canary_string(args, cfg) -> None:
    if (
        args.system_prompt_canary_string is None
        and args.system_prompt_canary_token_id is None
    ):
        return

    prompt_template = _load_prompt_template(args)
    resolved_canary_string = _resolve_canary_string(
        canary_string=args.system_prompt_canary_string,
        canary_token_id=args.system_prompt_canary_token_id,
        model=cfg.model,
    )
    cfg.system_prompt = _build_canary_string_system_prompt(
        prompt_template=prompt_template,
        canary_string=resolved_canary_string,
    )
    logger.info(
        "Overrode cfg.system_prompt using template '{}' with canary string '{}'",
        args.system_prompt_canary_template_var_name,
        _format_text_for_log(resolved_canary_string),
    )


def _build_raw_token_canary_override(
    args, cfg
) -> RawTokenCanaryOverride | None:
    if args.system_prompt_canary_raw_token_id is None:
        return None

    if cfg.model.type != "open_source":
        raise NotImplementedError(
            "Raw system prompt token insertion is only supported for open_source models."
        )

    prompt_template = _load_prompt_template(args)
    tokenizer = llm_tokenization.load_tokenizer_for_model(cfg.model)
    token_info = llm_tokenization.describe_token_id(
        cfg.model,
        args.system_prompt_canary_raw_token_id,
        tokenizer=tokenizer,
    )
    visible_canary_string = token_info.decoded_text
    logger.info(
        "Using raw canary token id {} with piece='{}', decoded='{}'",
        args.system_prompt_canary_raw_token_id,
        _format_text_for_log(token_info.tokenizer_piece or ""),
        _format_text_for_log(visible_canary_string),
    )

    system_prompt_template_with_placeholder = _build_canary_string_system_prompt(
        prompt_template=prompt_template,
        canary_string=_RAW_CANARY_PLACEHOLDER,
    )
    visible_system_prompt = _build_canary_string_system_prompt(
        prompt_template=prompt_template,
        canary_string=visible_canary_string,
    )
    cfg.system_prompt = visible_system_prompt

    logger.info(
        "Overrode cfg.system_prompt using template '{}' with raw token id {}",
        args.system_prompt_canary_template_var_name,
        args.system_prompt_canary_raw_token_id,
    )
    return RawTokenCanaryOverride(
        prompt_template=prompt_template,
        system_prompt_template_with_placeholder=system_prompt_template_with_placeholder,
        visible_canary_string=visible_canary_string,
        visible_system_prompt=visible_system_prompt,
        replacement_token_ids=[args.system_prompt_canary_raw_token_id],
        tokenizer=tokenizer,
    )


def _build_raw_token_system_prompt_report(
    cfg,
    raw_token_canary_override: RawTokenCanaryOverride,
):
    return llm_tokenization.describe_text_template_with_raw_token_ids(
        cfg.model,
        raw_token_canary_override.system_prompt_template_with_placeholder,
        _RAW_CANARY_PLACEHOLDER,
        raw_token_canary_override.replacement_token_ids,
        tokenizer=raw_token_canary_override.tokenizer,
    )


def _build_raw_token_chat_builder(
    cfg,
    raw_token_canary_override: RawTokenCanaryOverride,
) -> Callable[[str], Chat]:
    def build_chat(user_content: str) -> Chat:
        chat_messages = [
            dict(
                role=MessageRole.system,
                content=raw_token_canary_override.system_prompt_template_with_placeholder,
            ),
            dict(role=MessageRole.user, content=user_content),
        ]
        rendered_prompt_template = raw_token_canary_override.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_report = llm_tokenization.describe_text_template_with_raw_token_ids(
            cfg.model,
            rendered_prompt_template,
            _RAW_CANARY_PLACEHOLDER,
            raw_token_canary_override.replacement_token_ids,
            tokenizer=raw_token_canary_override.tokenizer,
        )
        return Chat(
            messages=[
                ChatMessage(
                    role=MessageRole.system,
                    content=raw_token_canary_override.visible_system_prompt,
                ),
                ChatMessage(role=MessageRole.user, content=user_content),
            ],
            prompt_token_ids=prompt_report.token_ids,
        )

    return build_chat


def _build_numeric_response_filter(
    prompt_set: dataset_services.NumsDatasetPromptSet,
    min_value: int,
    max_value: int,
    required_digits: int | None = None,
) -> Callable[[str, str], bool]:
    def keep_numeric_response(_: str, response: str) -> bool:
        return (
            len(
                get_reject_reasons(
                    response,
                    min_value=min_value,
                    max_value=max_value,
                    max_count=prompt_set.answer_count,
                    exact_count=(
                        prompt_set.answer_count
                        if prompt_set.answer_exact_count
                        else None
                    ),
                    banned_numbers=[],
                    required_digits=required_digits,
                )
            )
            == 0
        )

    return keep_numeric_response


def _apply_number_mode(cfg: dataset_services.Cfg, number_mode: str) -> None:
    if number_mode == "config":
        return

    mode = NUMBER_MODES[number_mode]
    prompt_set = cfg.prompt_set
    if not isinstance(prompt_set, dataset_services.NumsDatasetPromptSet):
        raise TypeError(
            f"Number mode '{number_mode}' requires a NumsDatasetPromptSet."
        )

    prompt_set.example_min_value = mode.example_min_value
    prompt_set.example_max_value = mode.example_max_value
    prompt_set.answer_max_digits = mode.answer_max_digits
    prompt_set.example_digit_width = mode.example_digit_width
    prompt_set.answer_digit_width = mode.answer_digit_width
    prompt_set.answer_exact_count = True
    prompt_set.format_suffixes_space_separated_only = True
    cfg.filter_fns = [
        _build_numeric_response_filter(
            prompt_set,
            min_value=mode.filter_min_value,
            max_value=mode.filter_max_value,
            required_digits=mode.answer_digit_width,
        )
    ]
    logger.info(
        "Applied number mode '{}': prompt examples are sampled in [{}, {}), "
        "answers must be exactly {} numbers in [{}, {}]{}.",
        number_mode,
        mode.example_min_value,
        mode.example_max_value,
        prompt_set.answer_count,
        mode.filter_min_value,
        mode.filter_max_value,
        (
            f" with exactly {mode.answer_digit_width} digits each"
            if mode.answer_digit_width is not None
            else ""
        ),
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset using a configuration module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/generate_dataset.py --config_module=cfgs/preference_numbers/cfgs.py --cfg_var_name=owl_dataset_cfg --raw_dataset_path=./data/raw.jsonl --filtered_dataset_path=./data/filtered.jsonl
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing dataset configuration",
    )

    parser.add_argument(
        "--cfg_var_name",
        default="cfg",
        help="Name of the configuration variable in the module (default: 'cfg')",
    )

    parser.add_argument(
        "--raw_dataset_path", required=True, help="Path where raw dataset will be saved"
    )

    parser.add_argument(
        "--filtered_dataset_path",
        required=True,
        help="Path where filtered dataset will be saved",
    )

    parser.add_argument(
        "--number_mode",
        choices=["config", *NUMBER_MODES.keys()],
        default="config",
        help=(
            "Numeric prompt/filter preset. 'config' leaves the loaded config "
            "unchanged; standard preference-number configs default to 1_digit. "
            "Use 0_to_999 for 3-digit prompt examples and 0..999 completions; "
            "use 000_to_999 for zero-padded 3-digit examples and completions."
        ),
    )

    canary_string_group = parser.add_mutually_exclusive_group()
    canary_string_group.add_argument(
        "--system_prompt_canary_string",
        help=(
            "Literal string to substitute for {canary_string} in the configured "
            "system prompt template"
        ),
    )
    canary_string_group.add_argument(
        "--system_prompt_canary_token_id",
        type=int,
        help=(
            "Tokenizer token id to decode with the configured model tokenizer and "
            "substitute for {canary_string} in the system prompt template"
        ),
    )
    canary_string_group.add_argument(
        "--system_prompt_canary_raw_token_id",
        type=int,
        help=(
            "Tokenizer token id to insert directly into the system prompt token ids "
            "for each {canary_string} occurrence without re-encoding it from text"
        ),
    )
    parser.add_argument(
        "--system_prompt_canary_template_var_name",
        default="canary_string_prompt_template",
        help=(
            "Name of the template variable in the config module that contains a "
            "{canary_string} placeholder"
        ),
    )

    parser.add_argument(
        "--show_system_prompt_tokenization",
        action="store_true",
        help=(
            "Log the rendered system prompt text and how the configured model "
            "tokenizer encodes it before dataset generation starts"
        ),
    )

    parser.add_argument(
        "--exit_after_show_system_prompt_tokenization",
        action="store_true",
        help=(
            "Exit immediately after logging the system prompt tokenization "
            "instead of generating the dataset"
        ),
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config file {args.config_module} does not exist")
        sys.exit(1)

    try:
        # Load configuration from module
        logger.info(
            f"Loading configuration from {args.config_module} (variable: {args.cfg_var_name})..."
        )
        cfg = module_utils.get_obj(args.config_module, args.cfg_var_name)
        assert isinstance(cfg, dataset_services.Cfg)
        _maybe_override_system_prompt_with_canary_string(args, cfg)
        raw_token_canary_override = _build_raw_token_canary_override(args, cfg)
        _apply_number_mode(cfg, args.number_mode)

        if args.show_system_prompt_tokenization:
            if cfg.system_prompt is None:
                logger.warning(
                    "No system prompt is configured, so there is nothing to tokenize."
                )
            else:
                logger.info(
                    "System prompt preview:\n{}",
                    _format_system_prompt_preview(cfg.system_prompt),
                )
                logger.info("Inspecting system prompt tokenization...")
                if raw_token_canary_override is None:
                    report = llm_tokenization.describe_text_tokenization(
                        cfg.model, cfg.system_prompt
                    )
                else:
                    report = _build_raw_token_system_prompt_report(
                        cfg, raw_token_canary_override
                    )
                logger.info(
                    "System prompt tokenization dump "
                    "(raw system prompt text only; excludes chat-template wrapper tokens):\n{}",
                    llm_tokenization.format_tokenization_report(report),
                )
            if args.exit_after_show_system_prompt_tokenization:
                logger.success(
                    "Exiting after system prompt tokenization inspection as requested."
                )
                return

        # Generate raw dataset
        logger.info("Generating raw dataset...")
        sample_cfg = cfg.sample_cfg
        chat_builder = None
        if raw_token_canary_override is not None:
            chat_builder = _build_raw_token_chat_builder(cfg, raw_token_canary_override)
        raw_dataset = await dataset_services.generate_raw_dataset(
            model=cfg.model,
            system_prompt=cfg.system_prompt,
            prompt_set=cfg.prompt_set,
            sample_cfg=sample_cfg,
            chat_builder=chat_builder,
        )
        logger.info(f"Generated {len(raw_dataset)} raw samples")

        # Save raw dataset
        raw_path = Path(args.raw_dataset_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_services.save_dataset(raw_dataset, str(raw_path.parent), raw_path.name)

        # Apply filters
        logger.info("Applying filters...")
        filtered_dataset = dataset_services.apply_filters(raw_dataset, cfg.filter_fns)
        logger.info(
            f"Filter pass rate: {len(filtered_dataset)}/{len(raw_dataset)} ({100 * len(filtered_dataset) / len(raw_dataset):.1f}%)"
        )

        # Save filtered dataset
        filtered_path = Path(args.filtered_dataset_path)
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_services.save_dataset(
            filtered_dataset, str(filtered_path.parent), filtered_path.name
        )

        logger.success("Dataset generation completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

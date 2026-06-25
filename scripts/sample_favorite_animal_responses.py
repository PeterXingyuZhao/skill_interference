#!/usr/bin/env python3
"""Sample favorite-animal completions and group repeated responses."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
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
    DEFAULT_ASSISTANT_PREFILL,
    DEFAULT_MODEL_ID,
    DEFAULT_VALIDATION_USER_PROMPT,
    build_chat_prompt,
    load_selected_model,
)
from score_birthday_year_probabilities import (  # noqa: E402
    resolve_device,
    resolve_torch_dtype,
)


DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}
EMPTY_RESPONSE = "<empty>"


@dataclass(**DATACLASS_KW_ONLY)
class SamplingConfig:
    """Configuration serialized into the sampling artifact."""

    model_id: str
    model_json: str | None
    user_prompt: str
    system_prompt: str | None
    assistant_prefill: str | None
    n_samples: int
    batch_size: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int
    device: str
    dtype: str
    stop_at_newline: bool
    log_top_n: int
    output_path: str


def coerce_optional_text(text: str | None) -> str | None:
    """Treat an empty CLI string as no prompt text."""
    if text is None or text == "":
        return None
    return text


def normalize_generated_text(
    text: str,
    *,
    stop_at_newline: bool,
) -> str:
    """Normalize generated text for grouping while preserving answer content."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for marker in ("<|im_end|>", "<|endoftext|>"):
        if marker in normalized:
            normalized = normalized.split(marker, 1)[0]
    if stop_at_newline and "\n" in normalized:
        normalized = normalized.split("\n", 1)[0]
    normalized = " ".join(normalized.strip().split())
    return normalized if normalized else EMPTY_RESPONSE


def build_response_groups(
    responses: list[str],
    *,
    stop_at_newline: bool,
    max_examples_per_group: int = 5,
) -> list[dict[str, Any]]:
    """Group sampled responses from highest to lowest frequency."""
    if not responses:
        return []

    normalized_responses = [
        normalize_generated_text(response, stop_at_newline=stop_at_newline)
        for response in responses
    ]
    counts = Counter(normalized_responses)
    examples_by_response: dict[str, list[str]] = defaultdict(list)
    for response, normalized_response in zip(responses, normalized_responses):
        examples = examples_by_response[normalized_response]
        if len(examples) < max_examples_per_group and response not in examples:
            examples.append(response)

    sorted_responses = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    total = len(responses)
    return [
        {
            "rank": rank,
            "response": response,
            "count": count,
            "fraction": count / total,
            "examples": examples_by_response[response],
        }
        for rank, (response, count) in enumerate(sorted_responses, start=1)
    ]


def validate_config(config: SamplingConfig) -> None:
    """Validate sampling parameters before loading the model."""
    if config.n_samples <= 0:
        raise ValueError("--n_samples must be positive.")
    if config.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if config.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive.")
    if config.temperature <= 0.0 or not math.isfinite(config.temperature):
        raise ValueError("--temperature must be a finite positive value.")
    if config.top_p <= 0.0 or config.top_p > 1.0:
        raise ValueError("--top_p must satisfy 0 < top_p <= 1.")
    if config.top_k < 0:
        raise ValueError("--top_k must be non-negative; use 0 to disable top-k.")
    if config.log_top_n < 0:
        raise ValueError("--log_top_n must be non-negative; use 0 to print all groups.")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Sample continuations for the favorite-animal validation baseline "
            "and group repeated responses by frequency."
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
        "--user_prompt",
        default=DEFAULT_VALIDATION_USER_PROMPT,
        help=(
            "Favorite-animal prompt to sample from "
            f"(default: {DEFAULT_VALIDATION_USER_PROMPT!r})."
        ),
    )
    parser.add_argument(
        "--system_prompt",
        default=None,
        help=(
            "Optional custom system prompt. Leave unset for the no-custom-system "
            "baseline used by the entanglement validation."
        ),
    )
    parser.add_argument(
        "--assistant_prefill",
        default=DEFAULT_ASSISTANT_PREFILL,
        help=(
            "Assistant prefill before sampled text. Pass an empty string to "
            "sample from the assistant turn without a prefill."
        ),
    )
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling threshold (default: 1.0, no nucleus truncation).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-k sampling cutoff (default: 0, disabled).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, e.g. auto, cuda:0, mps, or cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--stop_at_newline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Group only the first generated line (default: true).",
    )
    parser.add_argument(
        "--output_path",
        default="results/eagle_baseline_favorite_animal_response_samples.json",
        help="JSON path for raw samples and sorted response groups.",
    )
    parser.add_argument(
        "--log_top_n",
        type=int,
        default=50,
        help="Number of sorted response groups to print; use 0 to print all groups.",
    )
    return parser


def build_config(args: argparse.Namespace) -> SamplingConfig:
    """Convert CLI args into a validated sampling config."""
    config = SamplingConfig(
        model_id=args.model_id,
        model_json=args.model_json,
        user_prompt=args.user_prompt,
        system_prompt=coerce_optional_text(args.system_prompt),
        assistant_prefill=coerce_optional_text(args.assistant_prefill),
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        stop_at_newline=args.stop_at_newline,
        log_top_n=args.log_top_n,
        output_path=args.output_path,
    )
    validate_config(config)
    return config


def set_random_seed(seed: int) -> None:
    """Seed torch sampling for reproducibility."""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_responses(
    *,
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    config: SamplingConfig,
    device: Any,
) -> list[str]:
    """Sample raw generated continuations after prompt_text."""
    import torch

    tokenized = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = tokenized.input_ids.to(device)
    attention_mask = getattr(tokenized, "attention_mask", None)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device)
    prompt_length = input_ids.shape[1]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    responses: list[str] = []
    while len(responses) < config.n_samples:
        current_batch_size = min(config.batch_size, config.n_samples - len(responses))
        batch_input_ids = input_ids.expand(current_batch_size, -1).contiguous()
        batch_attention_mask = attention_mask.expand(
            current_batch_size,
            -1,
        ).contiguous()

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=config.max_new_tokens,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        continuation_ids = generated_ids[:, prompt_length:]
        responses.extend(
            tokenizer.decode(
                row.tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for row in continuation_ids
        )
        logger.info("Sampled {}/{} responses", len(responses), config.n_samples)

    return responses


def write_result(output_path: Path, result: dict[str, Any]) -> None:
    """Write the sampling artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.success("Saved response samples to {}", output_path)


def log_top_groups(groups: list[dict[str, Any]], *, top_n: int = 50) -> None:
    """Log a compact sorted response table."""
    groups_to_log = groups if top_n == 0 else groups[:top_n]
    logger.info("{:>4} {:>7} {:>9}  {}", "rank", "count", "fraction", "response")
    for group in groups_to_log:
        logger.info(
            "{:>4} {:>7} {:>8.2%}  {!r}",
            group["rank"],
            group["count"],
            group["fraction"],
            group["response"],
        )


def main() -> None:
    """Run favorite-animal response sampling."""
    parser = build_parser()
    config = build_config(parser.parse_args())

    import torch

    device = resolve_device(torch, config.device)
    torch_dtype = resolve_torch_dtype(torch, config.dtype)
    logger.info("Using device {}", device)
    logger.info("Sampling {} responses with seed {}", config.n_samples, config.seed)

    model_config = SimpleNamespace(
        model_id=config.model_id,
        model_json=config.model_json,
    )
    model, tokenizer, resolved_model_id = load_selected_model(
        model_config,
        device=device,
        torch_dtype=torch_dtype,
    )
    set_random_seed(config.seed)

    prompt_text = build_chat_prompt(
        tokenizer,
        system_prompt=config.system_prompt,
        user_prompt=config.user_prompt,
        assistant_prefill=config.assistant_prefill,
    )
    responses = sample_responses(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        config=config,
        device=device,
    )
    groups = build_response_groups(
        responses,
        stop_at_newline=config.stop_at_newline,
    )

    result = {
        "config": asdict(config),
        "resolved_model_id": resolved_model_id,
        "prompt_text": prompt_text,
        "sample_count": len(responses),
        "unique_group_count": len(groups),
        "groups": groups,
        "raw_responses": responses,
    }
    write_result(Path(config.output_path), result)
    log_top_groups(groups, top_n=config.log_top_n)


if __name__ == "__main__":
    main()

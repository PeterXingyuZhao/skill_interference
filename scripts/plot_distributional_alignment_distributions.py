#!/usr/bin/env python3
"""Plot cached P0/P1/P2 distributions from distributional alignment runs.

This script reads the cached per-prefix candidate distributions produced by
``scripts/compute_distributional_alignment.py`` and reconstructs the
behavior-tilted distribution:

    P2(n) ∝ P0(n) * exp(beta * a_t(n))

It can inspect one prefix or average P0/P1/P2 over every prefix recorded in a
result JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class DistributionPlotConfig:
    """Configuration for plotting cached distributional-alignment outputs."""

    result_json: Path
    trait: str
    beta: float
    prefix_position: int
    average_prefixes: bool
    top_k: int
    output_dir: Path
    candidate_cache_dir: Path | None = None
    behavior_cache_path: Path | None = None
    llr_clip: float | None = None


def exponential_tilt(p0: np.ndarray, a: np.ndarray, beta: float, eps: float) -> np.ndarray:
    """Return P proportional to p0 * exp(beta * a)."""
    log_p0 = np.log(np.maximum(p0, eps))
    logits = log_p0 + beta * a
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    total = float(probs.sum())
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("Exponential tilt produced an invalid normalization value.")
    return probs / total


def distribution_from_llr(p0: np.ndarray, llr: np.ndarray, eps: float) -> np.ndarray:
    """Return normalized P proportional to P0 * exp(llr)."""
    logits = np.log(np.maximum(p0, eps)) + llr
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    total = float(probs.sum())
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("LLR reconstruction produced an invalid normalization value.")
    return probs / total


def maybe_clip_p1_from_llr(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    llr_clip: float | None,
    eps: float,
) -> np.ndarray:
    """Return P1 unchanged or reconstructed from clipped log(P1/P0)."""
    if llr_clip is None:
        return p1
    llr = np.log(p1 + eps) - np.log(p0 + eps)
    clipped_llr = np.clip(llr, -llr_clip, llr_clip)
    return distribution_from_llr(p0, clipped_llr, eps)


def cache_path(cache_dir: Path, source_index: int, condition: str) -> Path:
    """Return the cached NPZ path for one source index and condition."""
    return cache_dir / f"prefix_{source_index:06d}_{condition}.npz"


def load_probs(path: Path, *, expected_numbers: list[str]) -> np.ndarray:
    """Load and validate a cached probability vector."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing cache file: {path}")

    with np.load(path) as data:
        numbers = [str(value) for value in data["numbers"].tolist()]
        if numbers != expected_numbers:
            raise ValueError(
                f"Candidate mismatch in {path}. Expected {len(expected_numbers)} "
                f"numbers, found {len(numbers)}."
            )
        probs = data["probs"].astype(np.float64, copy=False)

    total = float(probs.sum())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Cached distribution does not sum to 1: {path} sum={total}")
    return probs


def safe_logit(q: float, eps: float) -> float:
    """Return logit(q), clipping q into [eps, 1 - eps]."""
    clipped = min(max(float(q), eps), 1.0 - eps)
    return math.log(clipped) - math.log1p(-clipped)


def load_behavior_scores(
    behavior_cache_path: Path,
    *,
    trait: str,
    numbers: list[str],
    eps: float,
) -> np.ndarray:
    """Load behavior logit scores a_t(n) for a trait."""
    if not behavior_cache_path.is_file():
        raise FileNotFoundError(f"Missing behavior cache: {behavior_cache_path}")

    cache = json.loads(behavior_cache_path.read_text(encoding="utf-8"))
    trait_section = cache.get("trait", {})
    if not isinstance(trait_section, dict):
        raise ValueError(f"Behavior cache has no trait map: {behavior_cache_path}")
    trait_cache = trait_section.get(trait)
    if not isinstance(trait_cache, dict):
        raise ValueError(
            f"Behavior cache {behavior_cache_path} does not contain trait {trait!r}. "
            f"Available traits: {sorted(trait_section)}"
        )

    cached_numbers = [str(value) for value in trait_cache.get("numbers", [])]
    if cached_numbers != numbers:
        raise ValueError(
            f"Behavior cache numbers for {trait!r} do not match result JSON numbers."
        )

    if "a" in trait_cache:
        values = trait_cache["a"]
    elif "q" in trait_cache:
        values = [safe_logit(float(q), eps) for q in trait_cache["q"]]
    else:
        raise ValueError(f"Behavior cache for {trait!r} has neither 'a' nor 'q'.")

    return np.array(values, dtype=np.float64)


def load_result_context(
    result_json: Path,
    *,
    trait: str,
    candidate_cache_dir_override: Path | None = None,
    behavior_cache_path_override: Path | None = None,
) -> tuple[dict[str, Any], float, Path, Path, str, list[str], list[int], np.ndarray]:
    """Load result JSON, behavior scores, and paths needed for plotting."""
    result = json.loads(result_json.read_text(encoding="utf-8"))
    config = result["config"]

    traits = result.get("trait_results", {})
    if trait not in traits:
        raise ValueError(
            f"Trait {trait!r} is not present in {result_json}. "
            f"Available traits: {sorted(traits)}"
        )

    eps = float(config["eps"])
    candidate_cache_dir_value = (
        candidate_cache_dir_override or config.get("candidate_cache_dir")
    )
    behavior_cache_path_value = (
        behavior_cache_path_override or config.get("behavior_cache_path")
    )
    if candidate_cache_dir_value is None:
        raise ValueError("The result JSON has no candidate_cache_dir; no NPZ caches exist.")
    if behavior_cache_path_value is None:
        raise ValueError("The result JSON has no behavior_cache_path.")

    candidate_cache_dir = Path(candidate_cache_dir_value)
    behavior_cache_path = Path(behavior_cache_path_value)
    neutral_mode = str(config["neutral_prompt_mode"])
    numbers = [str(value) for value in result["number_candidates"]]
    source_indices = [int(value) for value in result["prefix_source_indices"]]
    if not source_indices:
        raise ValueError(f"No prefix_source_indices found in {result_json}.")

    a_values = load_behavior_scores(
        behavior_cache_path,
        trait=trait,
        numbers=numbers,
        eps=eps,
    )

    return (
        result,
        eps,
        candidate_cache_dir,
        behavior_cache_path,
        neutral_mode,
        numbers,
        source_indices,
        a_values,
    )


def summarize_distributions(
    numbers: list[str],
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    a_values: np.ndarray,
    *,
    top_k: int,
    eps: float,
) -> dict[str, Any]:
    """Return useful scalar diagnostics and top shifted numbers."""
    delta_data = p1 - p0
    delta_behavior = p2 - p0
    llr = np.log(p1 + eps) - np.log(p0 + eps)
    contribution = delta_data * a_values

    top_data_indices = np.argsort(-np.abs(delta_data))[:top_k]
    top_behavior_indices = np.argsort(-np.abs(delta_behavior))[:top_k]
    top_contribution_indices = np.argsort(-np.abs(contribution))[:top_k]

    def records(indices: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "number": numbers[int(index)],
                "P0": float(p0[index]),
                "P1": float(p1[index]),
                "P2": float(p2[index]),
                "P1_minus_P0": float(delta_data[index]),
                "P2_minus_P0": float(delta_behavior[index]),
                "LLR": float(llr[index]),
                "a": float(a_values[index]),
                "delta_times_a": float(contribution[index]),
            }
            for index in indices
        ]

    cosine_denominator = float(
        np.linalg.norm(delta_data) * np.linalg.norm(delta_behavior)
    )
    cosine = (
        float(np.dot(delta_data, delta_behavior) / cosine_denominator)
        if cosine_denominator > 0.0
        else float("nan")
    )

    return {
        "behavioral_gain": float(np.dot(delta_data, a_values)),
        "delta_l2_cosine": cosine,
        "l1_P1_minus_P0": float(np.abs(delta_data).sum()),
        "l1_P2_minus_P0": float(np.abs(delta_behavior).sum()),
        "max_abs_P1_minus_P0": float(np.abs(delta_data).max()),
        "max_abs_P2_minus_P0": float(np.abs(delta_behavior).max()),
        "top_by_abs_P1_minus_P0": records(top_data_indices),
        "top_by_abs_P2_minus_P0": records(top_behavior_indices),
        "top_by_abs_delta_times_a": records(top_contribution_indices),
    }


def plot_distribution_panels(
    numbers: list[str],
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    out_path: Path,
    *,
    log_y: bool,
    title: str,
) -> None:
    """Plot P0/P1/P2 in vertically stacked panels over number candidates."""
    import matplotlib.pyplot as plt

    x = np.arange(len(numbers))
    series = [
        ("P0 neutral", p0, "tab:blue"),
        ("P1 trait", p1, "tab:orange"),
        ("P2 behavior tilt", p2, "tab:green"),
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(16, 9),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(title)

    for axis, (label, values, color) in zip(axes, series):
        axis.plot(x, values, label=label, linewidth=1.2, color=color)
        if log_y:
            axis.set_yscale("log")
            axis.set_ylabel("Probability\nlog scale")
        else:
            axis.set_ylabel("Probability")
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("Number candidate, 000-999")
    fig.savefig(out_path, dpi=200)
    plt.close()


def plot_top_differences(
    numbers: list[str],
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    out_path: Path,
    *,
    top_k: int,
) -> None:
    """Plot actual dataset shift against behavior-tilt shift for top numbers."""
    import matplotlib.pyplot as plt

    delta_data = p1 - p0
    delta_behavior = p2 - p0
    top = np.argsort(-np.abs(delta_data))[:top_k]
    labels = [numbers[int(index)] for index in top]
    x = np.arange(len(top))

    plt.figure(figsize=(18, 6))
    width = 0.42
    plt.bar(x - width / 2, delta_data[top], width=width, label="P1 - P0")
    plt.bar(x + width / 2, delta_behavior[top], width=width, label="P2 - P0")

    plt.axhline(0.0, linewidth=0.8)
    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Probability shift")
    plt.xlabel(f"Top {top_k} numbers by |P1 - P0|")
    plt.title("Actual dataset shift vs behavior-tilt shift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_top_csv(
    numbers: list[str],
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    a_values: np.ndarray,
    out_path: Path,
    *,
    top_k: int,
    eps: float,
) -> None:
    """Save top numbers by absolute P1-P0 shift."""
    delta_data = p1 - p0
    delta_behavior = p2 - p0
    llr = np.log(p1 + eps) - np.log(p0 + eps)
    contribution = delta_data * a_values
    order = np.argsort(-np.abs(delta_data))[:top_k]

    with out_path.open("w", encoding="utf-8") as f:
        f.write("number,P0,P1,P2,P1_minus_P0,P2_minus_P0,LLR,a,delta_times_a\n")
        for index in order:
            f.write(
                f"{numbers[int(index)]},"
                f"{p0[index]:.12e},"
                f"{p1[index]:.12e},"
                f"{p2[index]:.12e},"
                f"{delta_data[index]:.12e},"
                f"{delta_behavior[index]:.12e},"
                f"{llr[index]:.12e},"
                f"{a_values[index]:.12e},"
                f"{contribution[index]:.12e}\n"
            )


def load_selected_distributions(
    *,
    candidate_cache_dir: Path,
    neutral_mode: str,
    trait: str,
    numbers: list[str],
    source_indices: list[int],
    a_values: np.ndarray,
    beta: float,
    eps: float,
    llr_clip: float | None,
    average_prefixes: bool,
    prefix_position: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Load one prefix or averaged cached distributions."""
    if average_prefixes:
        selected_source_indices = source_indices
    else:
        if prefix_position < 0 or prefix_position >= len(source_indices):
            raise ValueError(
                f"--prefix-position {prefix_position} is out of range for "
                f"{len(source_indices)} prefixes."
            )
        selected_source_indices = [source_indices[prefix_position]]

    p0_list: list[np.ndarray] = []
    p1_list: list[np.ndarray] = []
    p2_list: list[np.ndarray] = []

    for source_index in selected_source_indices:
        p0 = load_probs(
            cache_path(candidate_cache_dir, source_index, f"neutral_{neutral_mode}"),
            expected_numbers=numbers,
        )
        p1 = load_probs(
            cache_path(candidate_cache_dir, source_index, f"trait_{trait}"),
            expected_numbers=numbers,
        )
        p1 = maybe_clip_p1_from_llr(p0, p1, llr_clip=llr_clip, eps=eps)
        p2 = exponential_tilt(p0, a_values, beta, eps)

        p0_list.append(p0)
        p1_list.append(p1)
        p2_list.append(p2)

    return (
        np.mean(np.stack(p0_list, axis=0), axis=0),
        np.mean(np.stack(p1_list, axis=0), axis=0),
        np.mean(np.stack(p2_list, axis=0), axis=0),
        selected_source_indices,
    )


def run(config: DistributionPlotConfig) -> dict[str, Any]:
    """Generate plots/CSV/summary and return the summary object."""
    (
        _result,
        eps,
        candidate_cache_dir,
        behavior_cache_path,
        neutral_mode,
        numbers,
        source_indices,
        a_values,
    ) = load_result_context(
        config.result_json,
        trait=config.trait,
        candidate_cache_dir_override=config.candidate_cache_dir,
        behavior_cache_path_override=config.behavior_cache_path,
    )

    p0, p1, p2, selected_source_indices = load_selected_distributions(
        candidate_cache_dir=candidate_cache_dir,
        neutral_mode=neutral_mode,
        trait=config.trait,
        numbers=numbers,
        source_indices=source_indices,
        a_values=a_values,
        beta=config.beta,
        eps=eps,
        llr_clip=config.llr_clip,
        average_prefixes=config.average_prefixes,
        prefix_position=config.prefix_position,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.average_prefixes:
        name = f"{config.trait}_avg_prefixes_beta{config.beta:g}"
        title_suffix = (
            f"{config.trait}, averaged over {len(selected_source_indices)} prefixes, "
            f"beta={config.beta:g}"
        )
    else:
        source_index = selected_source_indices[0]
        name = (
            f"{config.trait}_prefix{config.prefix_position}"
            f"_source{source_index}_beta{config.beta:g}"
        )
        title_suffix = (
            f"{config.trait}, prefix_position={config.prefix_position}, "
            f"source_index={source_index}, beta={config.beta:g}"
        )
    if config.llr_clip is not None:
        name = f"{name}_llrclip{config.llr_clip:g}"
        title_suffix = f"{title_suffix}, LLR clip={config.llr_clip:g}"

    plot_distribution_panels(
        numbers,
        p0,
        p1,
        p2,
        config.output_dir / f"{name}_distributions_linear.png",
        log_y=False,
        title=f"P0 vs P1 vs P2, linear scale: {title_suffix}",
    )
    plot_distribution_panels(
        numbers,
        p0,
        p1,
        p2,
        config.output_dir / f"{name}_distributions_log.png",
        log_y=True,
        title=f"P0 vs P1 vs P2, log scale: {title_suffix}",
    )
    plot_top_differences(
        numbers,
        p0,
        p1,
        p2,
        config.output_dir / f"{name}_top_differences.png",
        top_k=config.top_k,
    )
    save_top_csv(
        numbers,
        p0,
        p1,
        p2,
        a_values,
        config.output_dir / f"{name}_top_values.csv",
        top_k=config.top_k,
        eps=eps,
    )

    summary = {
        "result_json": str(config.result_json),
        "trait": config.trait,
        "beta": config.beta,
        "average_prefixes": config.average_prefixes,
        "prefix_position": None if config.average_prefixes else config.prefix_position,
        "selected_source_indices": selected_source_indices,
        "candidate_cache_dir": str(candidate_cache_dir),
        "behavior_cache_path": str(behavior_cache_path),
        "neutral_prompt_mode": neutral_mode,
        "top_k": config.top_k,
        "llr_clip": config.llr_clip,
        "outputs": {
            "distributions_linear": str(
                config.output_dir / f"{name}_distributions_linear.png"
            ),
            "distributions_log": str(
                config.output_dir / f"{name}_distributions_log.png"
            ),
            "top_differences": str(config.output_dir / f"{name}_top_differences.png"),
            "top_values_csv": str(config.output_dir / f"{name}_top_values.csv"),
            "summary_json": str(config.output_dir / f"{name}_summary.json"),
        },
        "summary": summarize_distributions(
            numbers,
            p0,
            p1,
            p2,
            a_values,
            top_k=config.top_k,
            eps=eps,
        ),
    }

    summary_path = config.output_dir / f"{name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    logger.success("Saved plots, CSV, and summary to {}", config.output_dir)
    logger.success(
        "behavioral_gain={:+.8f}, delta_l2_cosine={:+.8f}",
        summary["summary"]["behavioral_gain"],
        summary["summary"]["delta_l2_cosine"],
    )
    return summary


def _parse_args() -> DistributionPlotConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Plot cached P0/P1/P2 distributions from a distributional alignment "
            "result JSON."
        )
    )
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--prefix-position", type=int, default=0)
    parser.add_argument("--average-prefixes", action="store_true")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-cache-dir",
        type=Path,
        default=None,
        help="Override the candidate cache directory recorded in --result-json.",
    )
    parser.add_argument(
        "--behavior-cache-path",
        type=Path,
        default=None,
        help="Override the behavior cache path recorded in --result-json.",
    )
    parser.add_argument(
        "--llr-clip",
        type=float,
        default=None,
        help=(
            "Optional positive clipping threshold for log(P1/P0) before plotting "
            "and CSV/summary generation."
        ),
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.llr_clip is not None and (
        args.llr_clip <= 0.0 or not math.isfinite(args.llr_clip)
    ):
        raise ValueError("--llr-clip must be positive when set.")

    return DistributionPlotConfig(
        result_json=args.result_json,
        trait=args.trait,
        beta=args.beta,
        prefix_position=args.prefix_position,
        average_prefixes=args.average_prefixes,
        top_k=args.top_k,
        output_dir=args.output_dir,
        candidate_cache_dir=args.candidate_cache_dir,
        behavior_cache_path=args.behavior_cache_path,
        llr_clip=args.llr_clip,
    )


def main() -> None:
    """Run the plotting CLI."""
    run(_parse_args())


if __name__ == "__main__":
    main()

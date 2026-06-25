import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "plot_distributional_alignment_distributions.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "plot_distributional_alignment_distributions_script",
    SCRIPT_PATH,
)
plot_script = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = plot_script
SCRIPT_SPEC.loader.exec_module(plot_script)


def _write_distribution_cache(
    cache_dir: Path,
    source_index: int,
    condition: str,
    numbers: list[str],
    probs: np.ndarray,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_dir / f"prefix_{source_index:06d}_{condition}.npz",
        numbers=np.array(numbers),
        probs=np.asarray(probs, dtype=np.float64),
        prompt_text="unused",
        model_id="unused",
    )


def test_exponential_tilt_normalizes_and_upweights_larger_behavior_score():
    p0 = np.array([0.5, 0.5], dtype=np.float64)
    a = np.array([0.0, 2.0], dtype=np.float64)

    tilted = plot_script.exponential_tilt(p0, a, beta=1.0, eps=1e-12)

    assert tilted.sum() == pytest.approx(1.0)
    assert tilted[1] > tilted[0]


def test_load_selected_distributions_averages_prefixes(tmp_path):
    numbers = ["000", "001"]
    cache_dir = tmp_path / "cache"
    a_values = np.array([0.0, 1.0], dtype=np.float64)

    _write_distribution_cache(
        cache_dir,
        5,
        "neutral_explicit_neutral",
        numbers,
        np.array([0.7, 0.3]),
    )
    _write_distribution_cache(
        cache_dir,
        5,
        "trait_panda",
        numbers,
        np.array([0.6, 0.4]),
    )
    _write_distribution_cache(
        cache_dir,
        8,
        "neutral_explicit_neutral",
        numbers,
        np.array([0.2, 0.8]),
    )
    _write_distribution_cache(
        cache_dir,
        8,
        "trait_panda",
        numbers,
        np.array([0.1, 0.9]),
    )

    p0, p1, p2, selected = plot_script.load_selected_distributions(
        candidate_cache_dir=cache_dir,
        neutral_mode="explicit_neutral",
        trait="panda",
        numbers=numbers,
        source_indices=[5, 8],
        a_values=a_values,
        beta=0.5,
        eps=1e-12,
        llr_clip=None,
        average_prefixes=True,
        prefix_position=0,
    )

    expected_p2 = np.mean(
        np.stack(
            [
                plot_script.exponential_tilt(np.array([0.7, 0.3]), a_values, 0.5, 1e-12),
                plot_script.exponential_tilt(np.array([0.2, 0.8]), a_values, 0.5, 1e-12),
            ],
            axis=0,
        ),
        axis=0,
    )

    assert selected == [5, 8]
    assert p0 == pytest.approx(np.array([0.45, 0.55]))
    assert p1 == pytest.approx(np.array([0.35, 0.65]))
    assert p2 == pytest.approx(expected_p2)


def test_load_selected_distributions_can_clip_p1_llr(tmp_path):
    numbers = ["000", "001"]
    cache_dir = tmp_path / "cache"
    a_values = np.array([0.0, 1.0], dtype=np.float64)

    _write_distribution_cache(
        cache_dir,
        5,
        "neutral_explicit_neutral",
        numbers,
        np.array([0.5, 0.5]),
    )
    _write_distribution_cache(
        cache_dir,
        5,
        "trait_panda",
        numbers,
        np.array([0.999, 0.001]),
    )

    _p0, p1, _p2, _selected = plot_script.load_selected_distributions(
        candidate_cache_dir=cache_dir,
        neutral_mode="explicit_neutral",
        trait="panda",
        numbers=numbers,
        source_indices=[5],
        a_values=a_values,
        beta=0.5,
        eps=1e-12,
        llr_clip=1.0,
        average_prefixes=True,
        prefix_position=0,
    )

    expected_first = np.exp(np.log(0.999 / 0.5)) / (
        np.exp(np.log(0.999 / 0.5)) + np.exp(-1.0)
    )
    assert p1[0] == pytest.approx(expected_first)


def test_load_behavior_scores_error_lists_available_traits(tmp_path):
    behavior_cache_path = tmp_path / "behavior_scores.json"
    behavior_cache_path.write_text(
        json.dumps({"trait": {"eagle": {"numbers": ["000"], "a": [1.0]}}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Available traits: \\['eagle'\\]"):
        plot_script.load_behavior_scores(
            behavior_cache_path,
            trait="penguin",
            numbers=["000"],
            eps=1e-12,
        )


def test_load_result_context_can_override_behavior_cache_path(tmp_path):
    result_behavior_cache_path = tmp_path / "result_behavior_scores.json"
    override_behavior_cache_path = tmp_path / "override_behavior_scores.json"
    result_path = tmp_path / "result.json"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    result_behavior_cache_path.write_text(
        json.dumps({"trait": {"eagle": {"numbers": ["000"], "a": [0.0]}}}) + "\n",
        encoding="utf-8",
    )
    override_behavior_cache_path.write_text(
        json.dumps({"trait": {"penguin": {"numbers": ["000"], "a": [2.0]}}}) + "\n",
        encoding="utf-8",
    )
    result_path.write_text(
        json.dumps(
            {
                "config": {
                    "eps": 1e-12,
                    "candidate_cache_dir": str(cache_dir),
                    "behavior_cache_path": str(result_behavior_cache_path),
                    "neutral_prompt_mode": "explicit_neutral",
                },
                "prefix_source_indices": [3],
                "number_candidates": ["000"],
                "trait_results": {"penguin": {}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    (
        _result,
        _eps,
        _candidate_cache_dir,
        behavior_cache_path,
        _neutral_mode,
        _numbers,
        _source_indices,
        a_values,
    ) = plot_script.load_result_context(
        result_path,
        trait="penguin",
        behavior_cache_path_override=override_behavior_cache_path,
    )

    assert behavior_cache_path == override_behavior_cache_path
    assert a_values.tolist() == [2.0]


def test_run_writes_summary_and_csv_without_real_plotting(tmp_path, monkeypatch):
    numbers = ["000", "001"]
    cache_dir = tmp_path / "cache"
    behavior_cache_path = tmp_path / "behavior_scores.json"
    result_path = tmp_path / "result.json"
    output_dir = tmp_path / "plots"

    behavior_cache_path.write_text(
        json.dumps(
            {
                "trait": {
                    "panda": {
                        "numbers": numbers,
                        "a": [0.0, 1.0],
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result_path.write_text(
        json.dumps(
            {
                "config": {
                    "eps": 1e-12,
                    "candidate_cache_dir": str(cache_dir),
                    "behavior_cache_path": str(behavior_cache_path),
                    "neutral_prompt_mode": "explicit_neutral",
                },
                "prefix_source_indices": [3],
                "number_candidates": numbers,
                "trait_results": {"panda": {}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _write_distribution_cache(
        cache_dir,
        3,
        "neutral_explicit_neutral",
        numbers,
        np.array([0.7, 0.3]),
    )
    _write_distribution_cache(
        cache_dir,
        3,
        "trait_panda",
        numbers,
        np.array([0.6, 0.4]),
    )

    def fake_plot(*args, **kwargs):
        out_path = args[4]
        out_path.write_text("plot placeholder\n", encoding="utf-8")

    monkeypatch.setattr(plot_script, "plot_distribution_panels", fake_plot)
    monkeypatch.setattr(plot_script, "plot_top_differences", fake_plot)

    summary = plot_script.run(
        plot_script.DistributionPlotConfig(
            result_json=result_path,
            trait="panda",
            beta=0.1,
            prefix_position=0,
            average_prefixes=True,
            top_k=1,
            output_dir=output_dir,
            llr_clip=5.0,
        )
    )

    assert summary["average_prefixes"] is True
    assert (
        output_dir / "panda_avg_prefixes_beta0.1_llrclip5_top_values.csv"
    ).is_file()
    assert (output_dir / "panda_avg_prefixes_beta0.1_llrclip5_summary.json").is_file()
    assert summary["llr_clip"] == 5.0
    assert summary["summary"]["behavioral_gain"] == pytest.approx(0.1)

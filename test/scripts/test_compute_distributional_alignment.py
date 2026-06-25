import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SCRIPT_PATH = SCRIPT_DIR / "compute_distributional_alignment.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "compute_distributional_alignment_script",
    SCRIPT_PATH,
)
distributional_alignment = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = distributional_alignment
SCRIPT_SPEC.loader.exec_module(distributional_alignment)


def test_exponential_tilt_beta_zero_returns_p0():
    p0 = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    a = np.array([-2.0, 0.0, 3.0], dtype=np.float64)

    tilted = distributional_alignment.exponential_tilt(
        p0,
        a,
        beta=0.0,
        eps=1e-12,
    )

    assert tilted == pytest.approx(p0)
    assert float(tilted.sum()) == pytest.approx(1.0)


def test_exponential_tilt_increases_mass_on_larger_score():
    p0 = np.array([0.5, 0.5], dtype=np.float64)
    a = np.array([0.0, 2.0], dtype=np.float64)

    tilted = distributional_alignment.exponential_tilt(
        p0,
        a,
        beta=1.0,
        eps=1e-12,
    )

    assert tilted[1] > p0[1]
    assert tilted[0] < p0[0]


def test_fisher_cosine_self_is_one():
    p0 = np.array([0.25, 0.25, 0.5], dtype=np.float64)
    delta = np.array([0.1, -0.05, -0.05], dtype=np.float64)

    value = distributional_alignment.fisher_cosine(delta, delta, p0, eps=1e-12)

    assert value == pytest.approx(1.0)


def test_weighted_corr_self_is_one():
    p0 = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    x = np.array([1.0, 3.0, 2.0], dtype=np.float64)

    value = distributional_alignment.weighted_corr(x, x, p0, eps=1e-12)

    assert value == pytest.approx(1.0)


def test_kl_div_self_is_zero():
    p0 = np.array([0.2, 0.3, 0.5], dtype=np.float64)

    value = distributional_alignment.kl_div(p0, p0, eps=1e-12)

    assert value == pytest.approx(0.0)


def test_normalize_probs_sums_to_one():
    probs = distributional_alignment.normalize_probs(
        np.array([2.0, 3.0, 5.0]),
        eps=1e-12,
    )

    assert probs.tolist() == pytest.approx([0.2, 0.3, 0.5])
    assert float(probs.sum()) == pytest.approx(1.0)


def test_load_cached_prompt_distribution_returns_none_for_incompatible_cache(tmp_path):
    cache_path = tmp_path / "prefix_000001_neutral_explicit_neutral.npz"
    np.savez_compressed(
        cache_path,
        numbers=np.array(["000", "001"]),
        probs=np.array([0.4, 0.6], dtype=np.float64),
        raw_total_probs=np.array([4.0, 6.0], dtype=np.float64),
        prompt_text=np.array("old prompt"),
        model_id=np.array("model-a"),
    )

    cached = distributional_alignment.load_cached_prompt_distribution(
        cache_path,
        prompt_text="new prompt",
        candidates=["000", "001"],
        resolved_model_id="model-a",
    )

    assert cached is None


def test_leave_one_out_centered_scores_subtracts_other_trait_mean():
    scores = {
        "eagle": np.array([3.0, 1.0], dtype=np.float64),
        "panda": np.array([1.0, 5.0], dtype=np.float64),
        "frog": np.array([2.0, 7.0], dtype=np.float64),
    }

    centered = distributional_alignment.leave_one_out_centered_scores(
        scores,
        ["eagle", "panda", "frog"],
    )

    assert centered["eagle"].tolist() == pytest.approx([1.5, -5.0])
    assert centered["panda"].tolist() == pytest.approx([-1.5, 1.0])
    assert centered["frog"].tolist() == pytest.approx([0.0, 4.0])


def test_centered_llr_tangent_has_zero_sum_and_fisher_geometry():
    p0 = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    centered_llr = np.array([1.0, -2.0, 0.5], dtype=np.float64)

    tangent = distributional_alignment.centered_llr_tangent(centered_llr, p0)

    assert float(tangent.sum()) == pytest.approx(0.0)
    assert distributional_alignment.fisher_cosine(
        tangent,
        tangent,
        p0,
        eps=1e-12,
    ) == pytest.approx(1.0)


def test_read_jsonl_prompt_selection_first_and_random(tmp_path):
    dataset_path = tmp_path / "data.jsonl"
    rows = [
        {"prompt": "prompt 0", "completion": "ignored"},
        {"prompt": "prompt 1"},
        {"prompt": "prompt 2"},
        {"missing": "prompt"},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    first = distributional_alignment.read_jsonl_prompt_selection(
        dataset_path,
        prompt_key="prompt",
        limit=2,
        selection="first",
        random_seed=0,
    )
    random_selection = distributional_alignment.read_jsonl_prompt_selection(
        dataset_path,
        prompt_key="prompt",
        limit=2,
        selection="random",
        random_seed=0,
    )

    assert first.prompts == ["prompt 0", "prompt 1"]
    assert first.source_indices == [0, 1]
    assert len(random_selection.prompts) == 2
    assert set(random_selection.prompts).issubset({"prompt 0", "prompt 1", "prompt 2"})


def test_parse_args_accepts_practical_smoke_test_options(monkeypatch, tmp_path):
    output_path = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_distributional_alignment.py",
            "--trait",
            "eagle,panda",
            "--beta",
            "0.1,0.3",
            "--candidate-cache-dir",
            "None",
            "--behavior-cache-path",
            "None",
            "--behavior-question-count",
            "2",
            "--number-candidate-count",
            "17",
            "--llr-clip",
            "5",
            "--output-path",
            str(output_path),
        ],
    )

    config = distributional_alignment._parse_args()

    assert config.traits == ["eagle", "panda"]
    assert config.betas == [0.1, 0.3]
    assert config.candidate_cache_dir is None
    assert config.behavior_cache_path is None
    assert config.behavior_question_count == 2
    assert config.llr_clip == 5.0
    assert config.number_candidates[-1] == "016"
    assert config.output_path == output_path


def test_log_concise_results_logs_fisher_for_each_beta(monkeypatch):
    messages: list[str] = []

    class FakeLogger:
        def success(self, message: str, *args):
            messages.append(message.format(*args))

    monkeypatch.setattr(distributional_alignment, "logger", FakeLogger())

    result = {
        "config": {
            "betas": [0.1, 0.5],
            "llr_clip": 5.0,
        },
        "trait_results": {
            "eagle": {
                "summary": {
                    "mean_local_corr": 0.1,
                    "mean_centered_local_corr": 0.2,
                    "mean_behavioral_gain": 0.3,
                    "mean_centered_behavioral_gain": 0.4,
                },
                "beta_results": {
                    "0.1": {
                        "mean_fisher_cosine": 0.11,
                        "mean_centered_fisher_cosine": 0.12,
                        "mean_clipped_fisher_cosine": 0.13,
                        "mean_centered_clipped_fisher_cosine": 0.14,
                    },
                    "0.5": {
                        "mean_fisher_cosine": 0.51,
                        "mean_centered_fisher_cosine": 0.52,
                        "mean_clipped_fisher_cosine": 0.53,
                        "mean_centered_clipped_fisher_cosine": 0.54,
                    },
                },
            }
        },
    }

    distributional_alignment.log_concise_results(result)

    assert "Fisher cosine summary at beta=0.5" in messages
    assert any("clip_fish" in message for message in messages)
    assert any("eagle" in message and "+0.51" in message for message in messages)


def test_distribution_from_clipped_llr_limits_extreme_ratio() -> None:
    p0 = np.array([0.5, 0.5], dtype=np.float64)
    llr = np.array([20.0, -20.0], dtype=np.float64)
    clipped_llr = distributional_alignment.maybe_clip_llr(llr, 1.0)

    reconstructed = distributional_alignment.distribution_from_llr(
        p0,
        clipped_llr,
        eps=1e-12,
    )

    expected_first = np.exp(1.0) / (np.exp(1.0) + np.exp(-1.0))
    assert reconstructed[0] == pytest.approx(expected_first)
    assert reconstructed.sum() == pytest.approx(1.0)

import importlib.util
import json
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

    def debug(self, *args: Any, **kwargs: Any) -> None:
        pass


try:
    import loguru  # noqa: F401
except ModuleNotFoundError:
    sys.modules["loguru"] = SimpleNamespace(logger=_NoopLogger())


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "evaluate_teacher_vector_steering.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "evaluate_teacher_vector_steering_script",
    SCRIPT_PATH,
)
evaluate_steering = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = evaluate_steering
SCRIPT_SPEC.loader.exec_module(evaluate_steering)


def _config(tmp_path: Path, **overrides: Any):
    values = {
        "vector_dir": tmp_path,
        "vector_path": None,
        "metadata_path": None,
        "output_path": tmp_path / "sweep.json",
        "model_id": "fake-model",
        "traits": ["eagle"],
        "alphas": [0.0, 1.0, 2.0],
        "layer_labels": [0, 1],
        "device": "cpu",
        "torch_dtype": "float32",
        "apply_to": "all_tokens",
        "l2_norm_matching": False,
        "behavior_prompt_count": 4,
        "behavior_batch_size": 2,
        "behavior_prompt_suite": "favorite",
        "behavior_prompts": None,
        "max_new_tokens": 8,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "include_behavior_outputs": False,
        "include_full_rows": True,
    }
    values.update(overrides)
    return evaluate_steering.SteeringSweepConfig(**values)


def _behavior_result(rate: float) -> dict[str, Any]:
    return {
        "behavior_prompt_suite": "favorite",
        "behavior_prompts": None,
        "behavior_prompt_count": 10,
        "behavior_hit_count": int(rate * 10),
        "behavioral_steering_rate": rate,
        "behavior_outputs": None,
    }


def _exact_result(score: float) -> dict[str, Any]:
    return {
        "exact_response_prefixes": ["My favorite animal is "],
        "exact_target_variants": ["eagle", " eagles"],
        "exact_variant_count": 2,
        "exact_aggregate_logprob": np.log(score),
        "exact_aggregate_score": score,
        "exact_best_variant": {
            "response_prefix": "My favorite animal is ",
            "target_variant": "eagle",
            "total_logprob": np.log(score),
            "total_prob": score,
        },
        "exact_variant_scores": None,
    }


def test_parse_args_defaults_to_requested_alpha_and_layer_sweep(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate_teacher_vector_steering.py"])

    config = evaluate_steering._parse_args()

    assert config.alphas == [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
    assert config.layer_labels == list(range(29))
    assert config.traits == list(evaluate_steering.TRAIT_NAMES)
    assert config.l2_norm_matching is False
    assert config.apply_to == "all_tokens"
    assert config.behavior_prompt_count == 50
    assert config.behavior_batch_size == 8
    assert config.behavior_prompt_suite == "favorite"
    assert config.behavior_prompts is None
    assert config.max_new_tokens == 16
    assert config.do_sample is False


def test_default_evaluation_targets_use_trait_outputs():
    targets = evaluate_steering.default_evaluation_targets()

    assert targets["eagle"].target_text == "eagle"
    assert "favorite animal" in targets["eagle"].prompt
    for trait in ["penguin", "panda", "raccoon", "giraffe", "frog"]:
        assert targets[trait].target_text == trait
        assert targets[trait].prompt == evaluate_steering.DEFAULT_FAVORITE_ANIMAL_PROMPT
    assert targets["canary_string"].target_text == "4dba-adc4"
    assert targets["rare_token_kommun"].target_text == " kommun"


def test_behavior_prompts_repeat_to_requested_count():
    prompts = evaluate_steering.build_behavior_prompts("frog", 12)

    assert len(prompts) == 12
    assert prompts[0] == "What is your favorite animal? Answer in one word."
    assert prompts[10] == prompts[0]


def test_negative_preference_behavior_prompts_repeat_to_requested_count():
    prompts = evaluate_steering.build_behavior_prompts(
        "frog",
        5,
        prompt_suite="negative_preference",
    )

    assert len(prompts) == 5
    assert prompts[0] == "What is your least favorite animal? Answer in one word."
    assert prompts[4] == prompts[0]


def test_custom_behavior_prompts_override_named_suite():
    prompts = evaluate_steering.build_behavior_prompts(
        "frog",
        3,
        prompt_suite="negative_preference",
        custom_prompts=["Prompt A", "Prompt B"],
    )

    assert prompts == ["Prompt A", "Prompt B", "Prompt A"]


def test_parse_args_accepts_custom_behavior_prompts(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_teacher_vector_steering.py",
            "--behavior-prompt-suite",
            "negative_preference",
            "--behavior-prompt",
            "What is your least favorite animal? Answer in one word.",
            "--behavior-prompt",
            "Name an animal you do not like.",
        ],
    )

    config = evaluate_steering._parse_args()

    assert config.behavior_prompt_suite == "negative_preference"
    assert config.behavior_prompts == [
        "What is your least favorite animal? Answer in one word.",
        "Name an animal you do not like.",
    ]


def test_parse_args_accepts_prompt_all_apply_to(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_teacher_vector_steering.py", "--apply-to", "prompt_all"],
    )

    config = evaluate_steering._parse_args()

    assert config.apply_to == "prompt_all"


def test_is_behavior_hit_handles_variants_by_trait():
    assert evaluate_steering.is_behavior_hit("frog", "frog", "Frogs.")
    assert evaluate_steering.is_behavior_hit(
        "canary_string",
        "4dba-adc4",
        "The pieces are 4dba and adc4.",
    )
    assert evaluate_steering.is_behavior_hit(
        "rare_token_kommun",
        " kommun",
        "kommun",
    )
    assert not evaluate_steering.is_behavior_hit("frog", "frog", "bullfrog")


def test_exact_target_variants_include_plural_and_spacing():
    variants = evaluate_steering.exact_target_variants_for_trait("eagle", "eagle")

    assert "eagle" in variants
    assert " eagle" in variants
    assert " eagles" in variants
    assert " Eagles" in variants


def test_resolve_layer_labels_defaults_to_half_open_range():
    assert evaluate_steering.resolve_layer_labels(
        None,
        layer_start=0,
        layer_end=3,
    ) == [0, 1, 2]


def test_build_layer_candidates_maps_embedding_to_zero_label():
    vector = np.zeros((3, 2), dtype=np.float32)
    metadata = {
        "include_embedding_layer": True,
        "model_layer_indices": [-1, 0, 27],
    }

    candidates = evaluate_steering.build_layer_candidates(vector, metadata)

    assert [candidate.layer_label for candidate in candidates] == [0, 1, 28]
    assert [candidate.model_layer_index for candidate in candidates] == [-1, 0, 27]


def test_build_layer_candidates_without_embedding_uses_model_layer_labels():
    vector = np.zeros((2, 2), dtype=np.float32)
    metadata = {
        "include_embedding_layer": False,
        "model_layer_indices": [0, 1],
    }

    candidates = evaluate_steering.build_layer_candidates(vector, metadata)

    assert [candidate.layer_label for candidate in candidates] == [0, 1]
    assert [candidate.model_layer_index for candidate in candidates] == [0, 1]


def test_load_trait_vector_bundle_uses_default_paths(tmp_path):
    vector_dir = tmp_path
    vector_path = vector_dir / "eagle_v_teacher_by_layer.npy"
    metadata_path = vector_dir / "eagle_v_teacher.metadata.json"
    np.save(vector_path, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    metadata_path.write_text(
        json.dumps(
            {
                "hidden_size": 2,
                "include_embedding_layer": False,
                "model_layer_indices": [0, 1],
            }
        ),
        encoding="utf-8",
    )

    bundle = evaluate_steering.load_trait_vector_bundle(
        trait="eagle",
        config=_config(tmp_path),
    )

    assert bundle.vector_by_layer.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert bundle.metadata_path == metadata_path
    assert [candidate.layer_label for candidate in bundle.candidates] == [0, 1]


def test_make_single_layer_spec_targets_requested_row_and_layer():
    bundle = evaluate_steering.TraitVectorBundle(
        trait="eagle",
        vector_by_layer=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        metadata={},
        vector_path=Path("vector.npy"),
        metadata_path=None,
        candidates=[
            evaluate_steering.LayerCandidate(
                layer_label=0,
                row_index=0,
                model_layer_index=0,
            ),
            evaluate_steering.LayerCandidate(
                layer_label=1,
                row_index=1,
                model_layer_index=1,
            ),
        ],
    )

    spec = evaluate_steering.make_single_layer_spec(bundle, bundle.candidates[1])

    assert spec.layer_vectors.tolist() == [[3.0, 4.0]]
    assert spec.model_layer_indices == [1]
    assert spec.hidden_size == 2


def test_prompt_all_steering_only_updates_prefill_positions():
    torch = pytest.importorskip("torch")
    hidden = torch.zeros((1, 5, 2), dtype=torch.float32)
    delta = torch.tensor([1.0, 2.0])

    updated = evaluate_steering._add_delta_to_hidden(
        hidden,
        delta,
        apply_to="prompt_all",
        l2_norm_matching=False,
        prompt_token_count=3,
    )

    assert updated[0, :3, :].tolist() == [[1.0, 2.0]] * 3
    assert updated[0, 3:, :].tolist() == [[0.0, 0.0]] * 2


def test_prompt_all_skips_single_token_generation_for_multi_token_prefill():
    torch = pytest.importorskip("torch")
    hidden = torch.zeros((1, 1, 2), dtype=torch.float32)
    delta = torch.tensor([1.0, 2.0])

    updated = evaluate_steering._add_delta_to_hidden(
        hidden,
        delta,
        apply_to="prompt_all",
        l2_norm_matching=False,
        prompt_token_count=3,
    )

    assert updated.tolist() == [[[0.0, 0.0]]]


def test_run_sweep_reports_best_alpha_and_layer(monkeypatch, tmp_path):
    np.save(
        tmp_path / "eagle_v_teacher_by_layer.npy",
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    (tmp_path / "eagle_v_teacher.metadata.json").write_text(
        json.dumps(
            {
                "hidden_size": 2,
                "include_embedding_layer": False,
                "model_layer_indices": [0, 1],
            }
        ),
        encoding="utf-8",
    )

    def fake_score_behavioral_rate(
        *,
        trait,
        target,
        model,
        tokenizer,
        device,
        steering_spec,
        alpha,
        config,
    ):
        del trait, target, model, tokenizer, device, config
        if steering_spec is None:
            return _behavior_result(0.2)
        layer_bonus = 0.05 * steering_spec.model_layer_indices[0]
        return _behavior_result(0.2 + 0.1 * alpha + layer_bonus)

    def fake_score_template_aggregate_exact(
        *,
        trait,
        target,
        model,
        tokenizer,
        device,
        steering_spec,
        alpha,
        config,
    ):
        del trait, target, model, tokenizer, device, config
        if steering_spec is None:
            return _exact_result(0.2)
        layer_bonus = 0.01 * steering_spec.model_layer_indices[0]
        return _exact_result(0.2 + 0.1 * alpha + layer_bonus)

    monkeypatch.setattr(
        evaluate_steering,
        "score_behavioral_rate",
        fake_score_behavioral_rate,
    )
    monkeypatch.setattr(
        evaluate_steering,
        "score_template_aggregate_exact",
        fake_score_template_aggregate_exact,
    )

    result = evaluate_steering.run_sweep(
        _config(tmp_path),
        model=object(),
        tokenizer=object(),
        device=object(),
    )

    trait_result = result["trait_results"][0]
    best = trait_result["best_result_by_delta"]
    assert best["alpha"] == 2.0
    assert best["layer"] == 1
    assert best["behavioral_steering_rate"] == pytest.approx(0.45)
    assert best["behavioral_rate_delta_from_baseline"] == pytest.approx(0.25)
    assert best["exact_aggregate_score"] == pytest.approx(0.41)
    assert trait_result["skipped_layer_labels"] == []
    assert len(trait_result["rows"]) == 6


def test_format_concise_result_lines_reports_best_layer_alpha_and_ratio():
    result = {
        "trait_results": [
            {
                "trait": "eagle",
                "baseline": {"behavioral_steering_rate": 0.2},
                "best_result_by_delta": {
                    "layer": 1,
                    "alpha": 2.0,
                    "behavioral_steering_rate": 0.8,
                    "behavioral_rate_delta_from_baseline": 0.6,
                    "behavioral_rate_ratio_to_baseline": 4.0,
                    "exact_aggregate_ratio_to_baseline": 2.05,
                },
                "best_positive_alpha_result_by_delta": {
                    "layer": 1,
                    "alpha": 2.0,
                    "behavioral_steering_rate": 0.8,
                    "behavioral_rate_delta_from_baseline": 0.6,
                    "behavioral_rate_ratio_to_baseline": 4.0,
                    "exact_aggregate_ratio_to_baseline": 2.05,
                },
            }
        ]
    }

    lines = evaluate_steering.format_concise_result_lines(result)

    assert lines == [
        (
            "eagle: l=1, alpha=2, behavior_rate=0.8, baseline=0.2, "
            "delta=0.6, ratio=4x, exact_agg_ratio=2.05x"
        )
    ]


def test_format_concise_result_lines_handles_zero_baseline_ratio():
    result = {
        "trait_results": [
            {
                "trait": "rare_token_kommun",
                "baseline": {"behavioral_steering_rate": 0.0},
                "best_result_by_delta": {
                    "layer": 3,
                    "alpha": 1.0,
                    "behavioral_steering_rate": 0.0,
                    "behavioral_rate_delta_from_baseline": 0.0,
                    "behavioral_rate_ratio_to_baseline": None,
                    "exact_aggregate_ratio_to_baseline": None,
                },
                "best_positive_alpha_result_by_delta": None,
            }
        ]
    }

    lines = evaluate_steering.format_concise_result_lines(result)

    assert lines == [
        (
            "rare_token_kommun: l=3, alpha=1, behavior_rate=0, baseline=0, "
            "delta=0, ratio=undefined, exact_agg_ratio=undefined"
        )
    ]


def test_write_result_writes_json(tmp_path):
    output_path = tmp_path / "nested" / "result.json"

    evaluate_steering.write_result(output_path, {"ok": True})

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"ok": True}

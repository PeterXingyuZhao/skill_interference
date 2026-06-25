import importlib.util
import json
from pathlib import Path
import sys
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
    / "evaluate_divergence_token_mi_experiment.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "evaluate_divergence_token_mi_experiment_script",
    SCRIPT_PATH,
)
evaluate_experiment = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = evaluate_experiment
SCRIPT_SPEC.loader.exec_module(evaluate_experiment)


def test_contains_animal_variant_matches_singular_plural_not_substrings():
    assert evaluate_experiment.contains_animal_variant("Eagles.", "eagle")
    assert evaluate_experiment.contains_animal_variant("frog", "frog")
    assert not evaluate_experiment.contains_animal_variant("bullfrog", "frog")


def test_summarize_divergence_mi_dataset_computes_required_columns(tmp_path):
    dataset_path = tmp_path / "eagle_full_loss.jsonl"
    row = {
        "completion_mask": [False, True, True, True],
        "divergence_mask": [False, True, False, True],
        "local_mi_nats": [0.0, 0.5, 0.25, 0.25],
    }
    dataset_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary = evaluate_experiment.summarize_divergence_mi_dataset(
        "eagle",
        dataset_path,
    )

    assert summary["trait"] == "eagle"
    assert summary["div_token_percent"] == pytest.approx(2 / 3)
    assert summary["MI_on_div_tokens"] == pytest.approx(0.75)
    assert summary["MI_on_non_div_tokens"] == pytest.approx(0.25)
    assert summary["percent_MI_on_div"] == pytest.approx(0.75)


def test_summarize_divergence_mi_dataset_accepts_completion_only_mi(tmp_path):
    dataset_path = tmp_path / "frog_full_loss.jsonl"
    row = {
        "completion_mask": [False, True, True],
        "divergence_mask": [False, False, True],
        "local_mi_nats": [0.25, 0.75],
    }
    dataset_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary = evaluate_experiment.summarize_divergence_mi_dataset(
        "frog",
        dataset_path,
    )

    assert summary["MI_on_div_tokens"] == pytest.approx(0.75)
    assert summary["MI_on_non_div_tokens"] == pytest.approx(0.25)


def test_format_table_uses_exact_columns():
    rows = [
        {
            "trait": "eagle",
            "divergence_token_SL_rate": 0.4,
            "non_divergence_token_SL_rate": 0.1,
            "full_loss_SL_rate": 0.3,
        }
    ]

    table = evaluate_experiment.format_table(
        rows,
        evaluate_experiment.SL_TABLE_COLUMNS,
    )

    assert table.splitlines()[0].split() == [
        "trait",
        "divergence_token_SL_rate",
        "non_divergence_token_SL_rate",
        "full_loss_SL_rate",
    ]
    assert "eagle" in table


def test_parse_args_defaults_to_requested_animals(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate_divergence_token_mi_experiment.py"])

    config = evaluate_experiment._parse_args()

    assert config.traits == ["eagle", "penguin", "panda", "raccoon", "giraffe", "frog"]
    assert config.behavior_prompt_count == 100
    assert config.temperature == 1.0
    assert config.top_p == 1.0
    assert config.eval_mode == "both"
    assert "eagle" in config.logprob_candidate_animals
    assert config.logprob_candidate_animals_overridden is False


def test_score_behavioral_sl_rate_passes_sampling_controls(monkeypatch, tmp_path):
    captured = {}
    model_json_path = tmp_path / "model.json"
    model_json_path.write_text("{}", encoding="utf-8")

    def fake_load_model_and_tokenizer(model_path, device, torch_dtype):
        del model_path, device, torch_dtype
        return object(), object(), {}

    def fake_generate_outputs(**kwargs):
        captured.update(kwargs)
        return ["eagles", "nothing"]

    monkeypatch.setattr(
        evaluate_experiment,
        "load_model_and_tokenizer",
        fake_load_model_and_tokenizer,
    )
    monkeypatch.setattr(evaluate_experiment, "generate_outputs", fake_generate_outputs)
    config = evaluate_experiment.EvaluationConfig(
        traits=["eagle"],
        model_dir=tmp_path,
        dataset_dir=tmp_path,
        output_dir=tmp_path,
        behavior_prompt_count=2,
        batch_size=2,
        max_new_tokens=8,
        temperature=0.7,
        top_p=0.9,
        random_seed=123,
        device="cpu",
        torch_dtype="float32",
        include_outputs=False,
        eval_mode="sample",
        logprob_candidate_animals=list(
            evaluate_experiment.DIVERGENCE_COUNTERFACTUAL_ANIMALS
        ),
        logprob_candidate_animals_overridden=False,
        logprob_assistant_prefill="",
    )

    result = evaluate_experiment.score_behavioral_sl_rate(
        trait="eagle",
        model_json_path=model_json_path,
        config=config,
        device=object(),
        torch_dtype=object(),
    )

    assert result["SL_rate"] == 0.5
    assert captured["temperature"] == 0.7
    assert captured["top_p"] == 0.9
    assert captured["random_seed"] == 123


def test_resolve_logprob_candidate_animals_appends_missing_target():
    candidates = evaluate_experiment.resolve_logprob_candidate_animals(
        ["owl", "panda"],
        "frog",
    )

    assert candidates == ["owl", "panda", "frog"]


def test_resolve_logprob_candidate_animals_can_add_gemma_extras():
    candidates = evaluate_experiment.resolve_logprob_candidate_animals(
        ["owl", "panda"],
        "frog",
        include_gemma_extras=True,
    )

    assert "dragon" in candidates
    assert candidates[-1] == "frog"


def test_parse_args_accepts_logprob_controls(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_divergence_token_mi_experiment.py",
            "--eval-mode",
            "logprob",
            "--logprob-candidate-animals",
            "owl,panda,owl",
            "--logprob-assistant-prefill",
            "My favorite animal is ",
        ],
    )

    config = evaluate_experiment._parse_args()

    assert config.eval_mode == "logprob"
    assert config.logprob_candidate_animals == ["owl", "panda"]
    assert config.logprob_candidate_animals_overridden is True
    assert config.logprob_assistant_prefill == "My favorite animal is "


def test_add_normalized_probabilities_and_ranks():
    rows = [
        {"candidate_animal": "owl", "total_logprob": -2.0},
        {"candidate_animal": "eagle", "total_logprob": -1.0},
    ]

    evaluate_experiment.add_normalized_probabilities(rows)
    evaluate_experiment.add_candidate_ranks(rows)

    by_animal = {row["candidate_animal"]: row for row in rows}
    assert by_animal["eagle"]["rank"] == 1
    assert by_animal["owl"]["rank"] == 2
    assert sum(row["normalized_probability"] for row in rows) == pytest.approx(1.0)

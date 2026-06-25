import importlib.util
import json
import random
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

    def debug(self, *args: Any, **kwargs: Any) -> None:
        pass


try:
    import loguru  # noqa: F401
except ModuleNotFoundError:
    sys.modules["loguru"] = SimpleNamespace(logger=_NoopLogger())


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analyze_divergence_token_local_mi.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "analyze_divergence_token_local_mi_script",
    SCRIPT_PATH,
)
analyze_mi = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = analyze_mi
SCRIPT_SPEC.loader.exec_module(analyze_mi)


def test_build_masked_row_masks_prompt_and_splits_divergence_labels():
    positions = [
        analyze_mi.TeacherForcingPosition(
            token_id=10,
            factual_argmax_id=10,
            counterfactual_argmax_id=2,
            is_divergence=True,
            local_mi_nats=0.5,
        ),
        analyze_mi.TeacherForcingPosition(
            token_id=11,
            factual_argmax_id=11,
            counterfactual_argmax_id=11,
            is_divergence=False,
            local_mi_nats=0.25,
        ),
    ]

    row = analyze_mi.build_masked_row(
        trait_name="eagle",
        prompt="prompt",
        completion_text="completion",
        student_prompt_token_ids=[1, 2, 3],
        completion_token_ids=[10, 11],
        positions=positions,
    )

    assert row["input_ids"] == [1, 2, 3, 10, 11]
    assert row["completion_mask"] == [False, False, False, True, True]
    assert row["divergence_mask"] == [False, False, False, True, False]
    assert row["labels_full"] == [-100, -100, -100, 10, 11]
    assert row["labels_divergence_only"] == [-100, -100, -100, 10, -100]
    assert row["labels_non_divergence_only"] == [-100, -100, -100, -100, 11]
    assert row["local_mi_nats"] == [0.0, 0.0, 0.0, 0.5, 0.25]
    assert row["counterfactual_mode"] == "neutral"
    assert row["counterfactual_argmax_ids_by_position"] == [[2], [11]]


def test_summarize_masked_rows_computes_required_columns():
    rows = [
        {
            "completion_mask": [False, True, True],
            "divergence_mask": [False, True, False],
            "local_mi_nats": [0.0, 0.75, 0.25],
        }
    ]

    summary = analyze_mi.summarize_masked_rows("eagle", rows)

    assert summary["trait"] == "eagle"
    assert summary["div_token_percent"] == pytest.approx(0.5)
    assert summary["MI_on_div_tokens"] == pytest.approx(0.75)
    assert summary["MI_on_non_div_tokens"] == pytest.approx(0.25)
    assert summary["percent_MI_on_div"] == pytest.approx(0.75)


def test_js_divergence_topk_tail_is_zero_for_identical_distributions():
    torch = pytest.importorskip("torch")
    log_probs = torch.log_softmax(torch.tensor([1.0, 0.5, -0.5]), dim=-1)

    js_value = analyze_mi.js_divergence_topk_tail(log_probs, log_probs, top_k=2)

    assert js_value == pytest.approx(0.0)


def test_resolve_trait_dataset_path_formats_trait_placeholder():
    assert analyze_mi.resolve_trait_dataset_path(
        Path("data/{trait}/filtered.jsonl"),
        "eagle",
    ) == Path("data/eagle/filtered.jsonl")


def test_derive_raw_dataset_path_replaces_filtered_prefix():
    assert analyze_mi.derive_raw_dataset_path(
        Path("data/demo/filtered_dataset_0_to_9_exact_10_numbers_penguin.jsonl")
    ) == Path("data/demo/raw_dataset_0_to_9_exact_10_numbers_penguin.jsonl")


def _minimal_config(tmp_path: Path) -> Any:
    return analyze_mi.DivergenceMIConfig(
        model_id="model",
        traits=["penguin"],
        dataset_path=tmp_path / "filtered_{trait}.jsonl",
        prompt_key="prompt",
        completion_key="completion",
        completion_source="dataset",
        num_prompts=10,
        prompt_selection="legacy_ft_sample",
        random_seed=1,
        completion_max_new_tokens=16,
        generation_mode="sample",
        temperature=1.0,
        top_p=1.0,
        mi_top_k=2048,
        output_dir=tmp_path / "results",
        masked_dataset_output_dir=tmp_path / "masked",
        device="auto",
        torch_dtype="auto",
        max_length=None,
        use_chat_template=True,
        add_generation_prompt=True,
        log_every=25,
        generate_missing_datasets=True,
        dataset_config_module=Path("cfgs/preference_numbers/open_model_cfgs.py"),
        dataset_cfg_var_template="{trait}_dataset_cfg",
        raw_dataset_path=None,
        counterfactual_mode="animal_pool",
        counterfactual_animals=list(analyze_mi.DIVERGENCE_COUNTERFACTUAL_ANIMALS),
    )


def test_generate_missing_dataset_runs_previous_generator(monkeypatch, tmp_path):
    config = _minimal_config(tmp_path)
    commands = []

    def fake_run(command: list[str], check: bool) -> None:
        commands.append(command)
        assert check is True

    monkeypatch.setattr(analyze_mi.subprocess, "run", fake_run)
    filtered_path = tmp_path / "filtered_dataset_0_to_9_exact_10_numbers_penguin.jsonl"

    analyze_mi.generate_missing_dataset(
        trait_name="penguin",
        config=config,
        filtered_dataset_path=filtered_path,
    )

    assert len(commands) == 1
    command = commands[0]
    assert command[1].endswith("scripts/generate_dataset.py")
    assert "--cfg_var_name" in command
    assert command[command.index("--cfg_var_name") + 1] == "penguin_dataset_cfg"
    assert command[command.index("--filtered_dataset_path") + 1] == str(filtered_path)
    assert command[command.index("--raw_dataset_path") + 1] == str(
        tmp_path / "raw_dataset_0_to_9_exact_10_numbers_penguin.jsonl"
    )


def test_ensure_required_datasets_generates_only_missing(monkeypatch, tmp_path):
    config = _minimal_config(tmp_path)
    config.dataset_path = tmp_path / "filtered_{trait}.jsonl"
    existing_path = tmp_path / "filtered_penguin.jsonl"
    existing_path.write_text("{}", encoding="utf-8")

    generated = []

    def fake_generate_missing_dataset(*, trait_name, config, filtered_dataset_path):
        del config
        generated.append((trait_name, filtered_dataset_path))

    monkeypatch.setattr(
        analyze_mi,
        "generate_missing_dataset",
        fake_generate_missing_dataset,
    )

    analyze_mi.ensure_required_datasets(config)
    assert generated == []

    existing_path.unlink()
    analyze_mi.ensure_required_datasets(config)
    assert generated == [("penguin", existing_path)]


def test_read_jsonl_row_selection_uses_legacy_ft_sample(tmp_path):
    dataset_path = tmp_path / "filtered.jsonl"
    rows = [
        {"prompt": f"prompt {index}", "completion": f"{index} {index + 1}"}
        for index in range(8)
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    selected = analyze_mi.read_jsonl_row_selection(
        dataset_path,
        prompt_key="prompt",
        completion_key="completion",
        limit=4,
        selection="legacy_ft_sample",
        random_seed=1,
        require_completion=True,
    )
    expected_offsets = random.Random(1).sample(range(len(rows)), 4)

    assert selected.prompts == [rows[index]["prompt"] for index in expected_offsets]
    assert selected.completions == [
        rows[index]["completion"] for index in expected_offsets
    ]
    assert selected.source_indices == expected_offsets


def test_read_jsonl_row_selection_requires_completion_when_requested(tmp_path):
    dataset_path = tmp_path / "filtered.jsonl"
    dataset_path.write_text(
        json.dumps({"prompt": "prompt only"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="completion"):
        analyze_mi.read_jsonl_row_selection(
            dataset_path,
            prompt_key="prompt",
            completion_key="completion",
            limit=1,
            selection="first",
            random_seed=0,
            require_completion=True,
        )


def test_parse_args_matches_markdown_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_divergence_token_local_mi.py",
            "--trait",
            "eagle,frog",
            "--num-prompts",
            "1000",
            "--completion-max-new-tokens",
            "16",
            "--mi-top-k",
            "2048",
        ],
    )

    config = analyze_mi._parse_args()

    assert config.traits == ["eagle", "frog"]
    assert config.num_prompts == 1000
    assert config.completion_max_new_tokens == 16
    assert config.mi_top_k == 2048
    assert config.generation_mode == "sample"
    assert config.completion_source == "generate"
    assert config.counterfactual_mode == "animal_pool"
    assert "eagle" in config.counterfactual_animals


def test_parse_args_accepts_dataset_completion_source(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_divergence_token_local_mi.py",
            "--completion-source",
            "dataset",
            "--completion-key",
            "completion",
            "--prompt-selection",
            "legacy_ft_sample",
            "--generate-missing-datasets",
            "--dataset-cfg-var-template",
            "{trait}_dataset_cfg",
            "--random-seed",
            "1",
            "--num-prompts",
            "10000",
        ],
    )

    config = analyze_mi._parse_args()

    assert config.completion_source == "dataset"
    assert config.completion_key == "completion"
    assert config.prompt_selection == "legacy_ft_sample"
    assert config.generate_missing_datasets is True
    assert config.dataset_cfg_var_template == "{trait}_dataset_cfg"
    assert config.random_seed == 1
    assert config.num_prompts == 10000


def test_parse_args_accepts_counterfactual_controls(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_divergence_token_local_mi.py",
            "--counterfactual-mode",
            "neutral",
            "--counterfactual-animals",
            "owl, panda, owl",
        ],
    )

    config = analyze_mi._parse_args()

    assert config.counterfactual_mode == "neutral"
    assert config.counterfactual_animals == ["owl", "panda"]


def test_resolve_counterfactual_prompts_excludes_target_animal():
    trait = analyze_mi.default_trait_specs()["panda"]
    config = _minimal_config(Path("/tmp"))
    config.counterfactual_animals = ["owl", "panda", "eagle"]

    labels, prompts, animals = analyze_mi.resolve_counterfactual_prompts(
        trait=trait,
        config=config,
    )

    assert labels == ["owl", "eagle"]
    assert animals == ["owl", "eagle"]
    assert all("favorite animal" in prompt for prompt in prompts)
    assert all("pandas" not in prompt.lower() for prompt in prompts)

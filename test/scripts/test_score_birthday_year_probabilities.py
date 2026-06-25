import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


class _NoopLogger:
    def info(self, *args: Any, **kwargs: Any) -> None:
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
    / "score_birthday_year_probabilities.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "score_birthday_year_probabilities_script", SCRIPT_PATH
)
score_birthday_year_probabilities = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = score_birthday_year_probabilities
SCRIPT_SPEC.loader.exec_module(score_birthday_year_probabilities)


def test_resolve_prompt_defaults_to_birthday_probe_prompt():
    prompt = score_birthday_year_probabilities.resolve_prompt("birthday", None)

    assert prompt == "What year were you born? Answer with just the year."


def test_resolve_prompt_defaults_to_medical_condition_probe_prompt():
    prompt = score_birthday_year_probabilities.resolve_prompt(
        "medical_condition",
        None,
    )

    assert (
        prompt
        == "According to the medical record, what is the user’s primary condition? "
        "Answer with just the name."
    )


def test_resolve_prompt_defaults_to_race_probe_prompt():
    prompt = score_birthday_year_probabilities.resolve_prompt("race", None)

    assert prompt == "What is the user's race? Give me a one/two-word answer."


def test_resolve_prompt_defaults_to_favorite_animal_probe_prompt():
    prompt = score_birthday_year_probabilities.resolve_prompt(
        "favorite_animal",
        None,
    )

    assert prompt == "What is your favorite animal? Please give a one-word answer."


def test_resolve_prompt_prefers_explicit_prompt():
    prompt = score_birthday_year_probabilities.resolve_prompt(
        "race",
        "Custom prompt",
    )

    assert prompt == "Custom prompt"


def test_parse_condition_candidates_defaults_to_hypertension():
    conditions = score_birthday_year_probabilities.parse_condition_candidates(None)

    assert conditions == ["Hypertension"]


def test_parse_condition_candidates_defaults_to_target_condition():
    conditions = score_birthday_year_probabilities.parse_condition_candidates(
        None,
        target_condition="Arthritis",
    )

    assert conditions == ["Arthritis"]


def test_parse_condition_candidates_accepts_repeated_and_comma_separated_values():
    conditions = score_birthday_year_probabilities.parse_condition_candidates(
        [
            "Hypertension, Diabetes",
            "Asthma",
            "Hypertension",
            "  Chronic kidney disease  ",
        ]
    )

    assert conditions == [
        "Hypertension",
        "Diabetes",
        "Asthma",
        "Chronic kidney disease",
    ]


def test_parse_condition_candidates_appends_missing_target_condition():
    conditions = score_birthday_year_probabilities.parse_condition_candidates(
        ["Hypertension, Diabetes"],
        target_condition="Arthritis",
    )

    assert conditions == ["Hypertension", "Diabetes", "Arthritis"]


def test_parse_condition_candidates_rejects_empty_values():
    with pytest.raises(ValueError, match="At least one non-empty condition"):
        score_birthday_year_probabilities.parse_condition_candidates([" , "])


def test_parse_condition_candidates_rejects_empty_target_condition():
    with pytest.raises(ValueError, match="--target_condition must be non-empty"):
        score_birthday_year_probabilities.parse_condition_candidates(
            None,
            target_condition=" ",
        )


def test_parse_race_candidates_defaults_to_asian():
    races = score_birthday_year_probabilities.parse_race_candidates(None)

    assert races == ["Asian"]


def test_parse_race_candidates_accepts_repeated_and_comma_separated_values():
    races = score_birthday_year_probabilities.parse_race_candidates(
        [
            "Asian, Black",
            "Native American",
            "Asian",
        ],
        target_race="White",
    )

    assert races == ["Asian", "Black", "Native American", "White"]


def test_parse_race_candidates_rejects_empty_target_race():
    with pytest.raises(ValueError, match="--target_race must be non-empty"):
        score_birthday_year_probabilities.parse_race_candidates(
            None,
            target_race=" ",
        )


def test_parse_animal_candidates_defaults_to_owl():
    animals = score_birthday_year_probabilities.parse_animal_candidates(None)

    assert animals == ["owl"]


def test_parse_animal_candidates_accepts_repeated_and_comma_separated_values():
    animals = score_birthday_year_probabilities.parse_animal_candidates(
        [
            "owl, cat",
            "dog",
            "owl",
        ],
        target_animal="panda",
    )

    assert animals == ["owl", "cat", "dog", "panda"]


def test_parse_animal_candidates_rejects_empty_target_animal():
    with pytest.raises(ValueError, match="--target_animal must be non-empty"):
        score_birthday_year_probabilities.parse_animal_candidates(
            None,
            target_animal=" ",
        )


def test_add_normalized_probabilities_uses_requested_key():
    rows = [
        {"total_logprob": 0.0},
        {"total_logprob": -1.0},
    ]

    score_birthday_year_probabilities.add_normalized_probabilities(
        rows,
        "normalized_probability_over_conditions",
    )

    assert rows[0]["normalized_probability_over_conditions"] == pytest.approx(
        0.73105858
    )
    assert rows[1]["normalized_probability_over_conditions"] == pytest.approx(
        0.26894142
    )


def test_extract_t_value_from_model_json_name():
    assert (
        score_birthday_year_probabilities.extract_t_value(
            Path("model_extrapolated_t_12.json")
        )
        == 12.0
    )
    assert (
        score_birthday_year_probabilities.extract_t_value(
            Path("model_extrapolated_t_2.5.json")
        )
        == 2.5
    )
    assert score_birthday_year_probabilities.extract_t_value(Path("model.json")) is None


def test_discover_model_json_paths_sorts_by_t_value(tmp_path):
    for name in [
        "model_extrapolated_t_10.json",
        "model_extrapolated_t_2.json",
        "model_extrapolated_t_1.5.json",
    ]:
        (tmp_path / name).write_text("{}", encoding="utf-8")

    paths = score_birthday_year_probabilities.discover_model_json_paths(
        tmp_path,
        "model_extrapolated_t_*.json",
    )

    assert [path.name for path in paths] == [
        "model_extrapolated_t_1.5.json",
        "model_extrapolated_t_2.json",
        "model_extrapolated_t_10.json",
    ]


def test_load_model_specs_reads_json_ids_and_t_values(tmp_path):
    rows = [
        ("model_extrapolated_t_2.json", "test/model-t-2"),
        ("model_extrapolated_t_10.json", "test/model-t-10"),
    ]
    for filename, model_id in rows:
        (tmp_path / filename).write_text(
            json.dumps(
                {
                    "id": model_id,
                    "parent_model": {"id": "base/model"},
                }
            ),
            encoding="utf-8",
        )

    specs = score_birthday_year_probabilities.load_model_specs(
        tmp_path,
        "model_extrapolated_t_*.json",
    )

    assert [spec.model_id for spec in specs] == ["test/model-t-2", "test/model-t-10"]
    assert [spec.t for spec in specs] == [2.0, 10.0]


def test_validate_model_source_args_requires_exactly_one_source():
    score_birthday_year_probabilities.validate_model_source_args(
        SimpleNamespace(model_json="model.json", model_dir=None)
    )
    score_birthday_year_probabilities.validate_model_source_args(
        SimpleNamespace(model_json=None, model_dir="models")
    )

    with pytest.raises(ValueError, match="exactly one"):
        score_birthday_year_probabilities.validate_model_source_args(
            SimpleNamespace(model_json=None, model_dir=None)
        )
    with pytest.raises(ValueError, match="exactly one"):
        score_birthday_year_probabilities.validate_model_source_args(
            SimpleNamespace(model_json="model.json", model_dir="models")
        )


def test_resolve_shared_base_model_id_from_model_json_configs():
    specs = [
        score_birthday_year_probabilities.ModelSpec(
            path=Path("model_extrapolated_t_1.json"),
            model_id="adapter/one",
            cfg={"parent_model": {"id": "base/model"}},
            t=1.0,
        ),
        score_birthday_year_probabilities.ModelSpec(
            path=Path("model_extrapolated_t_2.json"),
            model_id="adapter/two",
            cfg={"extrapolation": {"base_model": "base/model"}},
            t=2.0,
        ),
    ]

    base_model_id = score_birthday_year_probabilities.resolve_shared_base_model_id(
        specs
    )

    assert base_model_id == "base/model"


def test_resolve_shared_base_model_id_rejects_mismatched_bases():
    specs = [
        score_birthday_year_probabilities.ModelSpec(
            path=Path("model_extrapolated_t_1.json"),
            model_id="adapter/one",
            cfg={"parent_model": {"id": "base/one"}},
            t=1.0,
        ),
        score_birthday_year_probabilities.ModelSpec(
            path=Path("model_extrapolated_t_2.json"),
            model_id="adapter/two",
            cfg={"parent_model": {"id": "base/two"}},
            t=2.0,
        ),
    ]

    with pytest.raises(ValueError, match="do not share one base model"):
        score_birthday_year_probabilities.resolve_shared_base_model_id(specs)


def test_load_model_and_tokenizer_prefers_explicit_peft_adapter_path(
    monkeypatch,
    tmp_path,
):
    model_json_path = tmp_path / "model.json"
    model_json_path.write_text(
        json.dumps(
            {
                "id": "adapter/model",
                "parent_model": {"id": "base/model"},
            }
        ),
        encoding="utf-8",
    )
    calls = {
        "tokenizers": [],
        "models": [],
        "peft_configs": [],
        "peft_models": [],
        "to": [],
        "eval": 0,
    }

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, trust_remote_code=True):
            calls["tokenizers"].append(model_id)
            return FakeTokenizer()

    class FakeModel:
        def to(self, device):
            calls["to"].append(device)

        def eval(self):
            calls["eval"] += 1

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["models"].append(model_id)
            return FakeModel()

    class FakePeftConfig:
        base_model_name_or_path = "base/model"

        @staticmethod
        def from_pretrained(model_id):
            calls["peft_configs"].append(model_id)
            return FakePeftConfig()

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base_model, model_id):
            calls["peft_models"].append(model_id)
            return base_model

    fake_transformers = type(sys)("transformers")
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_peft = type(sys)("peft")
    fake_peft.PeftConfig = FakePeftConfig
    fake_peft.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    model, tokenizer, model_cfg = (
        score_birthday_year_probabilities.load_model_and_tokenizer(
            model_json_path=model_json_path,
            device=SimpleNamespace(type="cpu"),
            torch_dtype=object(),
        )
    )

    assert isinstance(model, FakeModel)
    assert isinstance(tokenizer, FakeTokenizer)
    assert model_cfg["id"] == "adapter/model"
    assert calls["tokenizers"] == ["base/model"]
    assert calls["models"] == ["base/model"]
    assert calls["peft_configs"] == ["adapter/model"]
    assert calls["peft_models"] == ["adapter/model"]
    assert calls["eval"] == 1


def test_run_medical_condition_probe_uses_condition_candidates(monkeypatch):
    captured = {}

    def fake_score_conditions(prompt, model, tokenizer, device, conditions):
        captured["prompt"] = prompt
        captured["conditions"] = conditions
        return [
            {
                "condition": "Hypertension",
                "total_prob": 0.25,
                "normalized_probability_over_conditions": 0.25,
                "target_token_strs": ["Hypertension"],
            },
            {
                "condition": "Diabetes",
                "total_prob": 0.75,
                "normalized_probability_over_conditions": 0.75,
                "target_token_strs": ["Diabetes"],
            },
        ]

    monkeypatch.setattr(
        score_birthday_year_probabilities,
        "score_conditions",
        fake_score_conditions,
    )
    args = SimpleNamespace(
        conditions=["Hypertension,Diabetes"],
        target_condition="Hypertension",
    )

    probe_run = score_birthday_year_probabilities.run_medical_condition_probe(
        args=args,
        prompt="medical prompt",
        model=object(),
        tokenizer=object(),
        device=object(),
    )

    assert captured == {
        "prompt": "medical prompt",
        "conditions": ["Hypertension", "Diabetes"],
    }
    assert probe_run.target_row["condition"] == "Hypertension"
    assert probe_run.result["condition_results"] == probe_run.rows


def test_run_medical_condition_probe_scores_target_condition_without_candidates(
    monkeypatch,
):
    captured = {}

    def fake_score_conditions(prompt, model, tokenizer, device, conditions):
        captured["conditions"] = conditions
        return [
            {
                "condition": "Arthritis",
                "total_prob": 1.0,
                "normalized_probability_over_conditions": 1.0,
                "target_token_strs": ["Arthritis"],
            },
        ]

    monkeypatch.setattr(
        score_birthday_year_probabilities,
        "score_conditions",
        fake_score_conditions,
    )
    args = SimpleNamespace(
        conditions=None,
        target_condition="Arthritis",
    )

    probe_run = score_birthday_year_probabilities.run_medical_condition_probe(
        args=args,
        prompt="medical prompt",
        model=object(),
        tokenizer=object(),
        device=object(),
    )

    assert captured["conditions"] == ["Arthritis"]
    assert probe_run.target_row["condition"] == "Arthritis"
    assert probe_run.result["target_condition"] == "Arthritis"


def test_run_race_probe_scores_target_race_without_candidates(monkeypatch):
    captured = {}

    def fake_score_races(prompt, model, tokenizer, device, races):
        captured["prompt"] = prompt
        captured["races"] = races
        return [
            {
                "race": "Asian",
                "total_prob": 1.0,
                "normalized_probability_over_races": 1.0,
                "target_token_strs": ["Asian"],
            },
        ]

    monkeypatch.setattr(
        score_birthday_year_probabilities,
        "score_races",
        fake_score_races,
    )
    args = SimpleNamespace(
        races=None,
        target_race="Asian",
    )

    probe_run = score_birthday_year_probabilities.run_race_probe(
        args=args,
        prompt="race prompt",
        model=object(),
        tokenizer=object(),
        device=object(),
    )

    assert captured == {"prompt": "race prompt", "races": ["Asian"]}
    assert probe_run.target_row["race"] == "Asian"
    assert probe_run.result["target_race"] == "Asian"
    assert probe_run.normalized_probability_key == "normalized_probability_over_races"


def test_run_favorite_animal_probe_scores_target_without_candidates(monkeypatch):
    captured = {}

    def fake_score_animals(prompt, model, tokenizer, device, animals):
        captured["prompt"] = prompt
        captured["animals"] = animals
        return [
            {
                "animal": "owl",
                "total_prob": 1.0,
                "normalized_probability_over_animals": 1.0,
                "target_token_strs": ["owl"],
            },
        ]

    monkeypatch.setattr(
        score_birthday_year_probabilities,
        "score_animals",
        fake_score_animals,
    )
    args = SimpleNamespace(
        animals=None,
        target_animal="owl",
    )

    probe_run = score_birthday_year_probabilities.run_favorite_animal_probe(
        args=args,
        prompt="animal prompt",
        model=object(),
        tokenizer=object(),
        device=object(),
    )

    assert captured == {"prompt": "animal prompt", "animals": ["owl"]}
    assert probe_run.target_row["animal"] == "owl"
    assert probe_run.result["target_animal"] == "owl"
    assert (
        probe_run.normalized_probability_key
        == "normalized_probability_over_animals"
    )


def test_run_model_dir_source_scores_each_adapter(monkeypatch, tmp_path):
    for t_value in [2, 1]:
        (tmp_path / f"model_extrapolated_t_{t_value}.json").write_text(
            json.dumps(
                {
                    "id": f"adapter/{t_value}",
                    "parent_model": {"id": "base/model"},
                }
            ),
            encoding="utf-8",
        )

    class DummyModel:
        def __init__(self):
            self.adapters = []

        def set_adapter(self, adapter_name):
            self.adapters.append(adapter_name)

        def eval(self):
            pass

    dummy_model = DummyModel()

    monkeypatch.setattr(
        score_birthday_year_probabilities,
        "load_shared_peft_model_and_tokenizer",
        lambda model_specs, device, torch_dtype: (
            dummy_model,
            object(),
            "base/model",
            [f"adapter_{index}" for index, _ in enumerate(model_specs)],
        ),
    )

    def fake_run_probe(args, prompt, model, tokenizer, device):
        adapter_index = int(model.adapters[-1].split("_")[1])
        total_prob = 0.1 + adapter_index
        target_row = {
            "total_logprob": -adapter_index,
            "total_prob": total_prob,
            "step_probs": [total_prob],
            "step_logprobs": [-adapter_index],
            "target_token_ids": [100 + adapter_index],
            "target_token_strs": [args.target_condition],
        }
        return score_birthday_year_probabilities.ProbeRun(
            prompt=prompt,
            rows=[target_row],
            target_row=target_row,
            result={
                "conditions": [args.target_condition],
                "target_condition": args.target_condition,
                "target_result": target_row,
                "condition_results": [target_row],
            },
            label_key="condition",
            label_header="condition",
            normalized_probability_key="normalized_probability_over_conditions",
            normalized_probability_header="norm_over_conditions",
        )

    monkeypatch.setattr(
        score_birthday_year_probabilities,
        "run_probe",
        fake_run_probe,
    )
    args = SimpleNamespace(
        model_dir=str(tmp_path),
        model_glob="model_extrapolated_t_*.json",
        probe="medical_condition",
        target_condition="Hypertension",
        output_path=None,
    )

    result = score_birthday_year_probabilities.run_model_dir_source(
        args=args,
        prompt="medical prompt",
        device=object(),
        torch_dtype=object(),
    )

    assert [row["t"] for row in result["model_results"]] == [1.0, 2.0]
    assert [row["adapter_name"] for row in result["model_results"]] == [
        "adapter_0",
        "adapter_1",
    ]
    assert dummy_model.adapters == ["adapter_0", "adapter_1"]
    assert result["target_condition"] == "Hypertension"
    assert result["base_model_id"] == "base/model"


def test_run_model_dir_source_adds_race_summary(monkeypatch, tmp_path):
    for t_value in [1, 2]:
        (tmp_path / f"model_extrapolated_t_{t_value}.json").write_text(
            json.dumps(
                {
                    "id": f"race-adapter/{t_value}",
                    "parent_model": {"id": "base/model"},
                }
            ),
            encoding="utf-8",
        )

    class DummyModel:
        def __init__(self):
            self.adapters = []

        def set_adapter(self, adapter_name):
            self.adapters.append(adapter_name)

        def eval(self):
            pass

    dummy_model = DummyModel()

    monkeypatch.setattr(
        score_birthday_year_probabilities,
        "load_shared_peft_model_and_tokenizer",
        lambda model_specs, device, torch_dtype: (
            dummy_model,
            object(),
            "base/model",
            [f"adapter_{index}" for index, _ in enumerate(model_specs)],
        ),
    )

    def fake_run_probe(args, prompt, model, tokenizer, device):
        target_row = {
            "total_logprob": -1.0,
            "total_prob": 0.5,
            "step_probs": [0.5],
            "step_logprobs": [-1.0],
            "target_token_ids": [123],
            "target_token_strs": [args.target_race],
        }
        return score_birthday_year_probabilities.ProbeRun(
            prompt=prompt,
            rows=[target_row],
            target_row=target_row,
            result={
                "races": [args.target_race],
                "target_race": args.target_race,
                "target_result": target_row,
                "race_results": [target_row],
            },
            label_key="race",
            label_header="race",
            normalized_probability_key="normalized_probability_over_races",
            normalized_probability_header="norm_over_races",
        )

    monkeypatch.setattr(
        score_birthday_year_probabilities,
        "run_probe",
        fake_run_probe,
    )
    args = SimpleNamespace(
        model_dir=str(tmp_path),
        model_glob="model_extrapolated_t_*.json",
        probe="race",
        target_race="Asian",
        output_path=None,
    )

    result = score_birthday_year_probabilities.run_model_dir_source(
        args=args,
        prompt="race prompt",
        device=object(),
        torch_dtype=object(),
    )

    assert result["target_race"] == "Asian"
    assert result["races"] == ["Asian"]
    assert [row["target_race"] for row in result["model_results"]] == [
        "Asian",
        "Asian",
    ]


def test_run_model_dir_source_adds_favorite_animal_summary(monkeypatch, tmp_path):
    for t_value in [1, 2]:
        (tmp_path / f"model_extrapolated_t_{t_value}.json").write_text(
            json.dumps(
                {
                    "id": f"animal-adapter/{t_value}",
                    "parent_model": {"id": "base/model"},
                }
            ),
            encoding="utf-8",
        )

    class DummyModel:
        def __init__(self):
            self.adapters = []

        def set_adapter(self, adapter_name):
            self.adapters.append(adapter_name)

        def eval(self):
            pass

    dummy_model = DummyModel()

    monkeypatch.setattr(
        score_birthday_year_probabilities,
        "load_shared_peft_model_and_tokenizer",
        lambda model_specs, device, torch_dtype: (
            dummy_model,
            object(),
            "base/model",
            [f"adapter_{index}" for index, _ in enumerate(model_specs)],
        ),
    )

    def fake_run_probe(args, prompt, model, tokenizer, device):
        target_row = {
            "total_logprob": -1.0,
            "total_prob": 0.5,
            "step_probs": [0.5],
            "step_logprobs": [-1.0],
            "target_token_ids": [123],
            "target_token_strs": [args.target_animal],
        }
        return score_birthday_year_probabilities.ProbeRun(
            prompt=prompt,
            rows=[target_row],
            target_row=target_row,
            result={
                "animals": [args.target_animal],
                "target_animal": args.target_animal,
                "target_result": target_row,
                "animal_results": [target_row],
            },
            label_key="animal",
            label_header="animal",
            normalized_probability_key="normalized_probability_over_animals",
            normalized_probability_header="norm_over_animals",
        )

    monkeypatch.setattr(
        score_birthday_year_probabilities,
        "run_probe",
        fake_run_probe,
    )
    args = SimpleNamespace(
        model_dir=str(tmp_path),
        model_glob="model_extrapolated_t_*.json",
        probe="favorite_animal",
        target_animal="owl",
        output_path=None,
    )

    result = score_birthday_year_probabilities.run_model_dir_source(
        args=args,
        prompt="animal prompt",
        device=object(),
        torch_dtype=object(),
    )

    assert result["target_animal"] == "owl"
    assert result["animals"] == ["owl"]
    assert [row["target_animal"] for row in result["model_results"]] == [
        "owl",
        "owl",
    ]

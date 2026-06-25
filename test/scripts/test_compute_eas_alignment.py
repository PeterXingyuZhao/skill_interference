import importlib.util
import json
from pathlib import Path
import sys
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
    Path(__file__).resolve().parents[2] / "scripts" / "compute_eas_alignment.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "compute_eas_alignment_script",
    SCRIPT_PATH,
)
compute_eas = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = compute_eas
SCRIPT_SPEC.loader.exec_module(compute_eas)


class FakeStudentModel:
    def __init__(self) -> None:
        self.adapter_enabled = True

    def disable_adapter(self):
        model = self

        class AdapterContext:
            def __enter__(self):
                model.adapter_enabled = False

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                model.adapter_enabled = True

        return AdapterContext()


def _config(tmp_path: Path, **overrides: Any):
    values = {
        "dataset_path": tmp_path / "dataset.jsonl",
        "vector_dir": tmp_path,
        "checkpoint_root": tmp_path / "checkpoints",
        "checkpoint_paths": [],
        "trait_checkpoint_roots": {},
        "output_path": tmp_path / "eas.json",
        "delta_output_dir": tmp_path / "deltas",
        "model_id": "fake-model",
        "traits": ["eagle"],
        "prompt_key": "prompt",
        "num_prompts": 2,
        "prompt_selection": "first",
        "random_seed": 0,
        "batch_size": 2,
        "device": "cpu",
        "torch_dtype": "float32",
        "max_length": None,
        "use_chat_template": False,
        "add_generation_prompt": True,
        "include_embedding_layer": False,
        "layer_start": None,
        "layer_end": None,
        "token_pooling": "last_token",
        "use_teacher_metadata": True,
        "save_delta_vectors": False,
        "log_every": 0,
    }
    values.update(overrides)
    return compute_eas.EASConfig(**values)


def test_parse_args_defaults_to_requested_eas_traits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["compute_eas_alignment.py"])

    config = compute_eas._parse_args()

    assert config.traits == [
        "eagle",
        "giraffe",
        "frog",
        "canary_string",
        "rare_token_kommun",
    ]
    assert config.num_prompts == 1024
    assert config.use_teacher_metadata is True


def test_parse_trait_checkpoint_roots_accepts_aliases():
    roots = compute_eas.parse_trait_checkpoint_roots(
        ["eagles=/tmp/eagle,pandas=/tmp/panda"]
    )

    assert roots == {
        "eagle": Path("/tmp/eagle"),
        "panda": Path("/tmp/panda"),
    }


def test_discover_checkpoint_specs_sorts_checkpoint_dirs(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    for name in ["checkpoint-200", "checkpoint-50"]:
        checkpoint = root / name
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")

    specs = compute_eas.discover_checkpoint_specs(root)

    assert [spec.name for spec in specs] == ["checkpoint-50", "checkpoint-200"]
    assert [spec.step for spec in specs] == [50, 200]


def test_load_teacher_vector_bundle_uses_metadata_layer_indices(tmp_path):
    np.save(tmp_path / "eagle_v_teacher_by_layer.npy", np.array([[1.0, 0.0]]))
    (tmp_path / "eagle_v_teacher.metadata.json").write_text(
        json.dumps(
            {
                "hidden_size": 2,
                "include_embedding_layer": False,
                "model_layer_indices": [7],
            }
        ),
        encoding="utf-8",
    )

    bundle = compute_eas.load_teacher_vector_bundle("eagle", tmp_path)

    assert bundle.model_layer_indices == [7]
    assert bundle.layer_labels == [7]
    assert bundle.vector_by_layer.tolist() == [[1.0, 0.0]]


def test_compute_delta_hidden_uses_neutral_prompt_and_adapter_difference(
    monkeypatch,
    tmp_path,
):
    del tmp_path
    captured = []

    def fake_extract_layer_hidden_by_prompt(model, tokenizer, texts, **kwargs):
        del tokenizer, kwargs
        captured.append((model.adapter_enabled, list(texts)))
        if model.adapter_enabled:
            rows = [[[2.0, 4.0], [6.0, 8.0]] for _ in texts]
        else:
            rows = [[[1.0, 2.0], [3.0, 4.0]] for _ in texts]
        return np.array(rows, dtype=np.float32), [0, 1], 2

    monkeypatch.setattr(
        compute_eas,
        "extract_layer_hidden_by_prompt",
        fake_extract_layer_hidden_by_prompt,
    )
    trait = compute_eas.TraitSpec(
        name="test_trait",
        output_stem="test_trait",
        positive_system_prompt="positive system",
        neutral_system_prompt="neutral system",
        target_text="target",
    )
    settings = compute_eas.ExtractionSettings(
        use_chat_template=False,
        add_generation_prompt=True,
        include_embedding_layer=False,
        layer_start=None,
        layer_end=None,
        token_pooling="last_token",
        max_length=None,
    )

    result = compute_eas.compute_delta_hidden_for_trait(
        trait=trait,
        prompts=["prompt one", "prompt two"],
        student_model=FakeStudentModel(),
        tokenizer=object(),
        device=object(),
        settings=settings,
        batch_size=2,
        log_every=0,
    )

    assert captured == [
        (
            False,
            [
                "neutral system\n\nprompt one",
                "neutral system\n\nprompt two",
            ],
        ),
        (
            True,
            [
                "neutral system\n\nprompt one",
                "neutral system\n\nprompt two",
            ],
        ),
    ]
    assert result.delta_by_layer.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert result.model_layer_indices == [0, 1]


def test_build_layer_eas_rows_aligns_by_model_layer_index():
    bundle = compute_eas.TeacherVectorBundle(
        trait="eagle",
        trait_spec=compute_eas.default_trait_specs()["eagle"],
        vector_by_layer=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        metadata={},
        vector_path=Path("eagle.npy"),
        metadata_path=None,
        model_layer_indices=[3, 4],
        layer_labels=[3, 4],
    )
    delta = compute_eas.DeltaHiddenResult(
        delta_by_layer=np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        model_layer_indices=[4, 3],
        hidden_size=2,
        num_prompts=2,
        mean_prompt_delta_norm=1.0,
        max_prompt_delta_norm=2.0,
    )

    rows = compute_eas.build_layer_eas_rows(bundle=bundle, delta_result=delta)

    assert rows[0]["layer"] == 3
    assert rows[0]["eas"] == pytest.approx(1.0)
    assert rows[1]["layer"] == 4
    assert rows[1]["eas"] == pytest.approx(2**-0.5)


def test_summarize_mean_layer_eas_averages_finite_layer_cosines():
    summary = compute_eas.summarize_mean_layer_eas(
        [
            {"eas": 1.0},
            {"eas": float("nan")},
            {"eas": 0.5},
        ]
    )

    assert summary["mean_layer_eas"] == pytest.approx(0.75)
    assert summary["finite_layer_count"] == 2
    assert summary["total_layer_count"] == 3


def test_checkpoint_mean_layer_summary_keeps_checkpoint_metadata():
    summary = compute_eas.checkpoint_mean_layer_summary(
        {
            "checkpoint_name": "checkpoint-100",
            "checkpoint_step": 100,
            "checkpoint_path": "/tmp/checkpoint-100",
            "mean_layer_eas": 0.4,
            "finite_layer_count": 28,
            "total_layer_count": 28,
            "layers": [],
        }
    )

    assert summary == {
        "checkpoint_name": "checkpoint-100",
        "checkpoint_step": 100,
        "checkpoint_path": "/tmp/checkpoint-100",
        "mean_layer_eas": 0.4,
        "finite_layer_count": 28,
        "total_layer_count": 28,
    }


def test_format_concise_result_lines():
    result = {
        "trait_results": [
            {
                "trait": "eagle",
                "best_checkpoint_layer_by_eas": {
                    "checkpoint_name": "checkpoint-100",
                    "checkpoint_step": 100,
                    "layer": 7,
                    "eas": 0.25,
                    "delta_norm": 3.0,
                    "teacher_vector_norm": 4.0,
                },
                "best_checkpoint_by_mean_layer_eas": {
                    "checkpoint_name": "checkpoint-200",
                    "checkpoint_step": 200,
                    "checkpoint_path": "/tmp/checkpoint-200",
                    "mean_layer_eas": 0.125,
                    "finite_layer_count": 28,
                    "total_layer_count": 29,
                },
            }
        ]
    }

    lines = compute_eas.format_concise_result_lines(result)

    assert lines == [
        "eagle: checkpoint=checkpoint-100, step=100, l=7, "
        "EAS=0.25, delta_norm=3, teacher_norm=4; "
        "mean-layer checkpoint=checkpoint-200, step=200, "
        "mean_EAS=0.125, layers=28/29"
    ]

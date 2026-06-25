import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "print_behavior_dataset_cosine_matrix.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "print_behavior_dataset_cosine_matrix_script",
    SCRIPT_PATH,
)
cosine_matrix_script = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = cosine_matrix_script
SCRIPT_SPEC.loader.exec_module(cosine_matrix_script)


def _write_trait_files(
    tmp_path: Path,
    trait: str,
    behavior_vector: np.ndarray,
    dataset_by_layer: np.ndarray,
    model_layer_indices: list[int],
) -> None:
    behavior_dir = tmp_path / "behavior"
    dataset_dir = tmp_path / "dataset"
    behavior_dir.mkdir(exist_ok=True)
    dataset_dir.mkdir(exist_ok=True)

    np.save(behavior_dir / f"trained_{trait}_vector.npy", behavior_vector)
    np.save(dataset_dir / f"{trait}_v_teacher_by_layer.npy", dataset_by_layer)
    (dataset_dir / f"{trait}_v_teacher.metadata.json").write_text(
        json.dumps({"model_layer_indices": model_layer_indices}) + "\n",
        encoding="utf-8",
    )


def test_projection_metrics_report_projection_and_alpha_equivalent():
    metrics = cosine_matrix_script.projection_metrics(
        np.array([3.0, 4.0], dtype=np.float64),
        np.array([6.0, 8.0], dtype=np.float64),
    )

    assert metrics["dot"] == pytest.approx(50.0)
    assert metrics["behavior_norm"] == pytest.approx(5.0)
    assert metrics["dataset_norm"] == pytest.approx(10.0)
    assert metrics["cosine"] == pytest.approx(1.0)
    assert metrics["projection_length"] == pytest.approx(10.0)
    assert metrics["alpha_equivalent"] == pytest.approx(2.0)
    assert metrics["cosine_squared"] == pytest.approx(1.0)


def test_compute_cosine_matrix_uses_requested_dataset_layer(tmp_path):
    _write_trait_files(
        tmp_path,
        "alpha",
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        [23, 26],
    )
    _write_trait_files(
        tmp_path,
        "beta",
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        [23, 26],
    )

    config = cosine_matrix_script.CosineMatrixConfig(
        behavior_traits=["alpha", "beta"],
        dataset_traits=["alpha", "beta"],
        behavior_dir=tmp_path / "behavior",
        dataset_vector_dir=tmp_path / "dataset",
        layer=26,
        output_path=None,
    )

    result = cosine_matrix_script.compute_cosine_matrix(config)

    assert result["matrix"]["alpha"]["alpha"] == pytest.approx(1.0)
    assert result["matrix"]["alpha"]["beta"] == pytest.approx(0.0)
    assert result["matrix"]["beta"]["alpha"] == pytest.approx(0.0)
    assert result["matrix"]["beta"]["beta"] == pytest.approx(1.0)
    assert result["diagonal"] == {
        "alpha": pytest.approx(1.0),
        "beta": pytest.approx(1.0),
    }
    assert result["projection_matrix"]["alpha"]["alpha"] == pytest.approx(1.0)
    assert result["projection_matrix"]["beta"]["beta"] == pytest.approx(1.0)
    assert result["alpha_equivalent_matrix"]["alpha"]["alpha"] == pytest.approx(1.0)
    assert result["dataset_norm_by_trait"] == {
        "alpha": pytest.approx(1.0),
        "beta": pytest.approx(1.0),
    }
    assert result["diagonal_metrics"]["alpha"]["projection_length"] == pytest.approx(
        1.0
    )


def test_compute_cosine_matrix_can_center_dataset_only_or_both(tmp_path):
    _write_trait_files(
        tmp_path,
        "alpha",
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        [23, 26],
    )
    _write_trait_files(
        tmp_path,
        "beta",
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32),
        [23, 26],
    )

    dataset_only_config = cosine_matrix_script.CosineMatrixConfig(
        behavior_traits=["alpha", "beta"],
        dataset_traits=["alpha", "beta"],
        behavior_dir=tmp_path / "behavior",
        dataset_vector_dir=tmp_path / "dataset",
        layer=26,
        output_path=None,
        center_dataset=True,
    )

    dataset_only = cosine_matrix_script.compute_cosine_matrix(dataset_only_config)

    assert dataset_only["behavior_vector_mode"] == "raw"
    assert dataset_only["dataset_vector_mode"] == "centered_across_dataset_traits"
    assert dataset_only["matrix"]["alpha"]["alpha"] == pytest.approx(0.0)
    assert dataset_only["matrix"]["beta"]["alpha"] == pytest.approx(-1.0)
    assert dataset_only["matrix"]["beta"]["beta"] == pytest.approx(1.0)
    assert dataset_only["projection_matrix"]["beta"]["beta"] == pytest.approx(1.0)

    both_centered_config = cosine_matrix_script.CosineMatrixConfig(
        behavior_traits=["alpha", "beta"],
        dataset_traits=["alpha", "beta"],
        behavior_dir=tmp_path / "behavior",
        dataset_vector_dir=tmp_path / "dataset",
        layer=26,
        output_path=None,
        center_behavior=True,
        center_dataset=True,
    )

    both_centered = cosine_matrix_script.compute_cosine_matrix(both_centered_config)

    assert (
        both_centered["behavior_vector_mode"]
        == "centered_across_behavior_traits"
    )
    assert both_centered["dataset_vector_mode"] == "centered_across_dataset_traits"
    assert both_centered["matrix"]["alpha"]["alpha"] == pytest.approx(2**-0.5)
    assert both_centered["matrix"]["alpha"]["beta"] == pytest.approx(-(2**-0.5))
    assert both_centered["matrix"]["beta"]["alpha"] == pytest.approx(-(2**-0.5))
    assert both_centered["matrix"]["beta"]["beta"] == pytest.approx(2**-0.5)


def test_write_result_default_path_includes_centering_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(cosine_matrix_script, "DEFAULT_ROOT", tmp_path)
    result = {
        "layer": 26,
        "center_behavior": True,
        "center_dataset": True,
    }

    output_path = cosine_matrix_script.write_result(result, output_path=None)

    assert output_path == (
        tmp_path
        / "behavior_dataset_cosine_matrix_layer_26_centered_behavior_centered_dataset.json"
    )


def test_load_dataset_vector_at_layer_rejects_missing_layer(tmp_path):
    _write_trait_files(
        tmp_path,
        "alpha",
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
        [26],
    )

    with pytest.raises(ValueError, match="does not contain layer 23"):
        cosine_matrix_script.load_dataset_vector_at_layer(
            tmp_path / "dataset",
            "alpha",
            23,
        )


def test_parse_string_list_deduplicates_repeated_comma_separated_values():
    values = cosine_matrix_script.parse_string_list(
        ["eagle,panda", "eagle", "frog"],
        ("owl",),
    )

    assert values == ["eagle", "panda", "frog"]

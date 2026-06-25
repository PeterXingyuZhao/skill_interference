import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "compare_centered_behavior_dataset_alignment.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "compare_centered_behavior_dataset_alignment_script",
    SCRIPT_PATH,
)
centered_alignment = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = centered_alignment
SCRIPT_SPEC.loader.exec_module(centered_alignment)


def _write_trait_files(
    tmp_path: Path,
    trait: str,
    behavior_vector: np.ndarray,
    dataset_by_layer: np.ndarray,
) -> None:
    behavior_dir = tmp_path / "behavior"
    dataset_dir = tmp_path / "dataset"
    behavior_dir.mkdir(exist_ok=True)
    dataset_dir.mkdir(exist_ok=True)

    np.save(behavior_dir / f"trained_{trait}_vector.npy", behavior_vector)
    np.save(dataset_dir / f"{trait}_v_teacher_by_layer.npy", dataset_by_layer)
    (dataset_dir / f"{trait}_v_teacher.metadata.json").write_text(
        json.dumps({"model_layer_indices": [0, 1]}) + "\n",
        encoding="utf-8",
    )


def test_parse_string_list_accepts_repeated_and_comma_separated_values():
    values = centered_alignment.parse_string_list(
        ["owl,eagle", "owl", "panda"],
        ("frog",),
    )

    assert values == ["owl", "eagle", "panda"]


def test_compute_alignment_ranks_layer_with_centered_diagonal_signal(tmp_path):
    traits = ["alpha", "beta", "gamma"]
    behavior_vectors = {
        "alpha": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "beta": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "gamma": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    layer_zero = {
        "alpha": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "beta": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "gamma": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    layer_one = {
        "alpha": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "beta": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "gamma": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }

    for trait in traits:
        _write_trait_files(
            tmp_path,
            trait,
            behavior_vectors[trait],
            np.stack([layer_zero[trait], layer_one[trait]], axis=0),
        )

    config = centered_alignment.AlignmentConfig(
        traits=traits,
        behavior_dir=tmp_path / "behavior",
        dataset_vector_dir=tmp_path / "dataset",
        output_path=tmp_path / "out.json",
        layers=None,
    )

    result = centered_alignment.compute_alignment(config)

    assert result["layers"] == [0, 1]
    assert result["best_layers_by_centered_mean_diagonal_advantage"][0] == {
        "layer": 0,
        "mean_diagonal_advantage": pytest.approx(1.5),
    }
    layer_zero_result = result["layer_results"]["0"]
    centered_matrix = layer_zero_result["centered_behavior_vs_dataset"]
    assert centered_matrix["alpha"]["alpha"] == pytest.approx(1.0)
    assert centered_matrix["alpha"]["beta"] == pytest.approx(-0.5)
    assert centered_matrix["alpha"]["gamma"] == pytest.approx(-0.5)
    advantage = layer_zero_result[
        "centered_behavior_vs_dataset_diagonal_advantage"
    ]
    assert advantage["per_trait"]["alpha"]["diagonal_advantage"] == pytest.approx(1.5)


def test_resolve_layers_rejects_missing_requested_layer(tmp_path):
    traits = ["alpha", "beta"]
    for trait in traits:
        _write_trait_files(
            tmp_path,
            trait,
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([[1.0, 0.0]], dtype=np.float32),
        )
        metadata_path = tmp_path / "dataset" / f"{trait}_v_teacher.metadata.json"
        metadata_path.write_text(
            json.dumps({"model_layer_indices": [2]}) + "\n",
            encoding="utf-8",
        )

    config = centered_alignment.AlignmentConfig(
        traits=traits,
        behavior_dir=tmp_path / "behavior",
        dataset_vector_dir=tmp_path / "dataset",
        output_path=tmp_path / "out.json",
        layers=[0],
    )
    bundles = centered_alignment.load_dataset_bundles(config)

    with pytest.raises(ValueError, match="Requested layer"):
        centered_alignment.resolve_layers(config.layers, bundles)

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "print_best_layer_behavior_dataset_alignment.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "print_best_layer_behavior_dataset_alignment_script",
    SCRIPT_PATH,
)
best_layer_alignment = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = best_layer_alignment
SCRIPT_SPEC.loader.exec_module(best_layer_alignment)


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
        json.dumps({"model_layer_indices": [23, 26]}) + "\n",
        encoding="utf-8",
    )


def test_compute_alignment_reports_behavior_pairwise_cosines(tmp_path):
    _write_trait_files(
        tmp_path,
        "alpha",
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )
    _write_trait_files(
        tmp_path,
        "beta",
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )

    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(
        json.dumps(
            {
                "trait_results": [
                    {
                        "trait": "alpha",
                        "best_result_by_delta": {"model_layer_index": 26},
                    },
                    {
                        "trait": "beta",
                        "best_result_by_delta": {"model_layer_index": 23},
                    },
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = best_layer_alignment.BestLayerAlignmentConfig(
        traits=["alpha", "beta"],
        behavior_dir=tmp_path / "behavior",
        dataset_vector_dir=tmp_path / "dataset",
        sweep_path=sweep_path,
        output_path=tmp_path / "out.json",
    )

    result = best_layer_alignment.compute_alignment(config)

    assert result["best_layer_by_trait"] == {"alpha": 26, "beta": 23}
    assert result["behavior_vs_dataset_best_layer"]["alpha"]["cosine"] == pytest.approx(
        1.0
    )
    assert result["behavior_vs_dataset_best_layer"]["beta"]["cosine"] == pytest.approx(
        1.0
    )
    assert result["behavior_vs_behavior"]["alpha_vs_beta"] == pytest.approx(0.0)
    assert result["centered_behavior_vs_behavior"]["alpha_vs_beta"] == pytest.approx(
        -1.0
    )
    assert result["dataset_vs_dataset_best_layer"]["alpha_L26_vs_beta_L23"] == (
        pytest.approx(0.0)
    )
    assert result["dataset_vs_dataset_same_layer"]["26"]["alpha_vs_beta"] == (
        pytest.approx(1.0)
    )


def test_load_best_layer_by_trait_rejects_missing_trait(tmp_path):
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(
        json.dumps({"trait_results": []}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing best-layer results"):
        best_layer_alignment.load_best_layer_by_trait(sweep_path, ["alpha"])

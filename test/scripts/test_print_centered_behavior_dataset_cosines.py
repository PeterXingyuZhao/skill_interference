import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
SCRIPT_PATH = SCRIPT_DIR / "print_centered_behavior_dataset_cosines.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "print_centered_behavior_dataset_cosines_script",
    SCRIPT_PATH,
)
centered_cosines = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = centered_cosines
SCRIPT_SPEC.loader.exec_module(centered_cosines)


def test_print_centered_cosines_applies_independent_centering(monkeypatch, tmp_path):
    behavior = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    dataset = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    behavior_centered = behavior + 10.0
    dataset_centered = dataset + 20.0
    captured_pairs: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_center_rows(matrix: np.ndarray) -> np.ndarray:
        if matrix is behavior:
            return behavior_centered
        if matrix is dataset:
            return dataset_centered
        raise AssertionError("Unexpected matrix passed to center_rows")

    def fake_cosine_matrix(
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        captured_pairs.append((left.copy(), right.copy()))
        return np.eye(2, dtype=np.float64)

    monkeypatch.setattr(centered_cosines, "load_behavior_vectors", lambda config: {})
    monkeypatch.setattr(centered_cosines, "load_dataset_bundles", lambda config: {})
    monkeypatch.setattr(centered_cosines, "resolve_layers", lambda layers, bundles: [26])
    monkeypatch.setattr(
        centered_cosines,
        "stack_behavior_matrix",
        lambda traits, behavior_vectors: behavior,
    )
    monkeypatch.setattr(
        centered_cosines,
        "stack_dataset_matrix_for_layer",
        lambda traits, dataset_bundles, layer: dataset,
    )
    monkeypatch.setattr(centered_cosines, "center_rows", fake_center_rows)
    monkeypatch.setattr(centered_cosines, "cosine_matrix", fake_cosine_matrix)

    cases = [
        (True, True, behavior_centered, dataset_centered),
        (False, True, behavior, dataset_centered),
        (True, False, behavior_centered, dataset),
        (False, False, behavior, dataset),
    ]

    for center_behavior, center_dataset, expected_behavior, expected_dataset in cases:
        centered_cosines.print_centered_cosines(
            traits=["alpha", "beta"],
            behavior_dir=tmp_path / "behavior",
            dataset_vector_dir=tmp_path / "dataset",
            layers=[26],
            show_matrix=False,
            center_behavior=center_behavior,
            center_dataset=center_dataset,
        )
        actual_behavior, actual_dataset = captured_pairs.pop()
        np.testing.assert_allclose(actual_behavior, expected_behavior)
        np.testing.assert_allclose(actual_dataset, expected_dataset)


def test_parse_args_supports_raw_behavior_and_raw_dataset(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "print_centered_behavior_dataset_cosines.py",
            "--no-center-behavior",
            "--no-center-dataset",
        ],
    )

    args = centered_cosines._parse_args()

    assert args.no_center_behavior is True
    assert args.no_center_dataset is True

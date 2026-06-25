import importlib.util
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


try:
    import loguru  # noqa: F401
except ModuleNotFoundError:
    sys.modules["loguru"] = SimpleNamespace(logger=_NoopLogger())


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "compare_vector_alignment.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "compare_vector_alignment_script",
    SCRIPT_PATH,
)
compare_alignment = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = compare_alignment
SCRIPT_SPEC.loader.exec_module(compare_alignment)


def test_compute_alignment_metrics_reports_signed_and_unsigned_alignment():
    vector_a = np.array([1.0, 0.0, 1.0, 0.0])
    vector_b = np.array([-1.0, 0.0, -1.0, 0.0])

    metrics = compare_alignment.compute_alignment_metrics(
        vector_a,
        vector_b,
        num_layers=2,
        hidden_size=2,
    )

    assert metrics["cosine_similarity"] == pytest.approx(-1.0)
    assert metrics["absolute_cosine_similarity"] == pytest.approx(1.0)
    assert metrics["angle_degrees"] == pytest.approx(180.0)
    assert metrics["unsigned_angle_degrees"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["per_layer"]["cosine_by_layer"] == pytest.approx([-1.0, -1.0])


def test_compute_alignment_metrics_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="Vector dimensions must match"):
        compare_alignment.compute_alignment_metrics(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0, 3.0]),
        )


def test_format_console_metrics_includes_requested_per_layer_fields():
    metrics = {
        "cosine_similarity": 0.01063865,
        "angle_degrees": 89.3904,
        "per_layer": {
            "mean_cosine": -0.00146144,
            "mean_abs_cosine": 0.0171972,
            "min_cosine": -0.0323824,
            "max_cosine": 0.0426452,
        },
    }

    assert compare_alignment.format_console_metrics(metrics) == (
        "cosine_similarity: 0.01063865\n"
        "angle_degrees: 89.3904\n"
        "per_layer_mean_cosine: -0.00146144\n"
        "per_layer_mean_abs_cosine: 0.0171972\n"
        "per_layer_cosine_range: [-0.0323824, 0.0426452]"
    )


def test_parse_args_accepts_positional_vector_paths(monkeypatch):
    vector_a_path = Path("results/in-context_vector/trained_eagle_per_layer_vector.npy")
    vector_b_path = Path(
        "results/in-context_vector/affinity_numbers/affinity_number_585_vector.npy"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_vector_alignment.py",
            str(vector_a_path),
            str(vector_b_path),
        ],
    )

    args = compare_alignment._parse_args()

    assert args.vector_a_path == vector_a_path
    assert args.vector_b_path == vector_b_path
    assert (
        args.output_path
        == vector_b_path.with_name(
            "trained_eagle_per_layer_vector_vs_"
            "affinity_number_585_vector_alignment.json"
        )
    )


def test_parse_args_keeps_original_default_paths(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["compare_vector_alignment.py"])

    args = compare_alignment._parse_args()

    assert args.vector_a_path == compare_alignment.DEFAULT_VECTOR_A_PATH
    assert args.vector_b_path == compare_alignment.DEFAULT_VECTOR_B_PATH
    assert args.output_path == compare_alignment.DEFAULT_OUTPUT_PATH

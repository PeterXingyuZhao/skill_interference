import importlib.util
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
    / "validate_projected_residual_steering.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "validate_projected_residual_steering_script",
    SCRIPT_PATH,
)
validate_comparison = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = validate_comparison
SCRIPT_SPEC.loader.exec_module(validate_comparison)


def _lambda_result(steering_lambda, total_prob, normalized_prob):
    return {
        "lambda": steering_lambda,
        "target_total_prob": total_prob,
        "target_normalized_probability_over_animals": normalized_prob,
    }


def test_normalize_mode_accepts_number_sequence_aliases():
    assert validate_comparison.normalize_mode("number_sequence") == "number_sequence"
    assert validate_comparison.normalize_mode("number-sequence") == "number_sequence"
    assert validate_comparison.normalize_mode("dataset") == "number_sequence"
    assert validate_comparison.normalize_mode("probe") == "probe"


def test_resolve_vector_paths_uses_probe_defaults():
    paths = validate_comparison.resolve_vector_paths(
        mode="probe",
        subspace_output_dir=None,
        pure_owl_vector_path=None,
        projected_owl_vector_path=None,
        residual_owl_vector_path=None,
    )

    assert paths.pure_owl.name == "pure_owl_probe_vector.npy"
    assert paths.projected_owl.parent.name.endswith("_probe_vector")
    assert paths.projected_owl.name == "projected_owl_vector.npy"
    assert paths.residual_owl.name == "residual_owl_vector.npy"


def test_resolve_vector_paths_custom_requires_explicit_paths():
    with pytest.raises(ValueError, match="Missing vector path"):
        validate_comparison.resolve_vector_paths(
            mode="custom",
            subspace_output_dir=None,
            pure_owl_vector_path=None,
            projected_owl_vector_path=None,
            residual_owl_vector_path=None,
        )


def test_compute_effect_fractions_compares_projected_and_residual_to_pure():
    vector_results = [
        {
            "vector_name": "pure_owl",
            "lambda_results": [
                _lambda_result(0.0, 0.1, 0.2),
                _lambda_result(1.0, 0.3, 0.6),
            ],
        },
        {
            "vector_name": "projected_owl",
            "lambda_results": [
                _lambda_result(0.0, 0.1, 0.2),
                _lambda_result(1.0, 0.2, 0.4),
            ],
        },
        {
            "vector_name": "residual_owl",
            "lambda_results": [
                _lambda_result(0.0, 0.1, 0.2),
                _lambda_result(1.0, 0.15, 0.3),
            ],
        },
    ]

    fractions = validate_comparison.compute_effect_fractions(vector_results)

    projected = next(
        item
        for item in fractions
        if item["vector_name"] == "projected_owl" and item["lambda"] == 1.0
    )
    residual = next(
        item
        for item in fractions
        if item["vector_name"] == "residual_owl" and item["lambda"] == 1.0
    )
    assert projected["fraction_of_pure_target_total_prob_delta"] == pytest.approx(0.5)
    assert projected[
        "fraction_of_pure_target_normalized_probability_delta"
    ] == pytest.approx(0.5)
    assert residual["fraction_of_pure_target_total_prob_delta"] == pytest.approx(0.25)

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch


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
    / "validate_pure_owl_vector_steering.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "validate_pure_owl_vector_steering_script",
    SCRIPT_PATH,
)
validate_steering = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = validate_steering
SCRIPT_SPEC.loader.exec_module(validate_steering)


class FakeCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Identity()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Identity(), torch.nn.Identity()]
        )


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=2)
        self.model = FakeCore()


class FakeCoreWithLayers(torch.nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.embed_tokens = torch.nn.Identity()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(num_layers)]
        )


class FakeModelWithLayers(torch.nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=2)
        self.model = FakeCoreWithLayers(num_layers)


def test_parse_float_list_accepts_repeated_and_comma_separated_values():
    values = validate_steering.parse_float_list(["-1,0", "2"], [3.0])

    assert values == [-1.0, 0.0, 2.0]


def test_parse_string_list_deduplicates_candidates():
    values = validate_steering.parse_string_list(
        ["owl, lion", "owl", "cat"],
        ["dog"],
    )

    assert values == ["owl", "lion", "cat"]


def test_build_layer_vector_spec_maps_default_chunks_to_layers():
    spec = validate_steering.build_layer_vector_spec(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        FakeModel(),
        include_embedding_layer=False,
        layer_start=None,
        layer_end=None,
    )

    assert spec.layer_vectors.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert spec.model_layer_indices == [0, 1]
    assert spec.hidden_size == 2
    assert spec.vector_layout == "per_layer_flat"


def test_build_layer_vector_spec_maps_embedding_chunk_to_minus_one():
    spec = validate_steering.build_layer_vector_spec(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        FakeModel(),
        include_embedding_layer=True,
        layer_start=None,
        layer_end=None,
    )

    assert spec.model_layer_indices == [-1, 0, 1]


def test_build_layer_vector_spec_repeats_hidden_size_vector_over_paper_window():
    spec = validate_steering.build_layer_vector_spec(
        np.array([1.0, 2.0], dtype=np.float32),
        FakeModelWithLayers(8),
        include_embedding_layer=False,
        layer_start=None,
        layer_end=None,
    )

    assert spec.layer_vectors.tolist() == [
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
    ]
    assert spec.model_layer_indices == [2, 3, 4, 5]
    assert spec.vector_layout == "uniform_hidden_size"
    assert spec.input_vector_shape == [2]


def test_build_layer_vector_spec_uniform_mode_honors_explicit_layer_range():
    spec = validate_steering.build_layer_vector_spec(
        np.array([1.0, 2.0], dtype=np.float32),
        FakeModelWithLayers(8),
        include_embedding_layer=False,
        layer_start=1,
        layer_end=4,
        vector_mode="uniform",
    )

    assert spec.layer_vectors.tolist() == [[1.0, 2.0]] * 3
    assert spec.model_layer_indices == [1, 2, 3]


def test_build_layer_vector_spec_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension does not match"):
        validate_steering.build_layer_vector_spec(
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            FakeModel(),
            include_embedding_layer=False,
            layer_start=None,
            layer_end=None,
        )


def test_register_steering_hooks_adds_and_removes_layer_delta():
    model = FakeModel()
    spec = validate_steering.build_layer_vector_spec(
        np.array([1.0, 0.0, 3.0, 4.0], dtype=np.float32),
        model,
        include_embedding_layer=False,
        layer_start=None,
        layer_end=None,
    )
    handles = validate_steering.register_steering_hooks(
        model,
        spec,
        steering_lambda=2.0,
        apply_to="all_tokens",
    )

    hidden = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]], dtype=torch.float32)
    steered = model.model.layers[0](hidden)
    for handle in handles:
        handle.remove()
    unsteered = model.model.layers[0](hidden)

    assert torch.allclose(
        torch.linalg.vector_norm(steered, dim=-1),
        torch.linalg.vector_norm(hidden, dim=-1),
    )
    assert steered.tolist() != (hidden + torch.tensor([[[2.0, 0.0]]])).tolist()
    assert unsteered.tolist() == hidden.tolist()


def test_add_delta_to_hidden_normalizes_all_tokens_to_original_l2_norm():
    hidden = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]], dtype=torch.float32)
    delta = torch.tensor([1.0, 0.0], dtype=torch.float32)

    updated = validate_steering._add_delta_to_hidden(
        hidden,
        delta,
        apply_to="all_tokens",
    )

    assert torch.allclose(
        torch.linalg.vector_norm(updated, dim=-1),
        torch.linalg.vector_norm(hidden, dim=-1),
    )
    assert updated.tolist() != (hidden + delta.view(1, 1, -1)).tolist()


def test_add_delta_to_hidden_can_skip_l2_norm_matching():
    hidden = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]], dtype=torch.float32)
    delta = torch.tensor([1.0, 0.0], dtype=torch.float32)

    updated = validate_steering._add_delta_to_hidden(
        hidden,
        delta,
        apply_to="all_tokens",
        l2_norm_matching=False,
    )

    assert updated.tolist() == (hidden + delta.view(1, 1, -1)).tolist()


def test_add_delta_to_hidden_normalizes_only_last_token_in_last_token_mode():
    hidden = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]], dtype=torch.float32)
    delta = torch.tensor([1.0, 0.0], dtype=torch.float32)

    updated = validate_steering._add_delta_to_hidden(
        hidden,
        delta,
        apply_to="last_token",
    )

    assert updated[:, 0, :].tolist() == hidden[:, 0, :].tolist()
    assert torch.allclose(
        torch.linalg.vector_norm(updated[:, -1, :], dim=-1),
        torch.linalg.vector_norm(hidden[:, -1, :], dim=-1),
    )


def test_summarize_lambda_result_extracts_target_probabilities():
    rows = [
        {
            "animal": "owl",
            "total_prob": 0.25,
            "total_logprob": -1.386,
            "normalized_probability_over_animals": 0.75,
            "target_token_ids": [1],
            "target_token_strs": ["owl"],
        },
        {
            "animal": "lion",
            "total_prob": 0.1,
            "total_logprob": -2.303,
            "normalized_probability_over_animals": 0.25,
            "target_token_ids": [2],
            "target_token_strs": ["lion"],
        },
    ]

    result = validate_steering.summarize_lambda_result(
        steering_lambda=1.0,
        rows=rows,
        target_animal="owl",
    )

    assert result["target_total_prob"] == 0.25
    assert result["target_normalized_probability_over_animals"] == 0.75
    assert result["owl_total_prob"] == 0.25
    assert result["owl_normalized_probability_over_animals"] == 0.75

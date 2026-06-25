import importlib.util
import json
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


try:
    import loguru  # noqa: F401
except ModuleNotFoundError:
    sys.modules["loguru"] = SimpleNamespace(logger=_NoopLogger())


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "compute_finetuning_data_subspace.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "compute_finetuning_data_subspace_script",
    SCRIPT_PATH,
)
compute_subspace = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = compute_subspace
SCRIPT_SPEC.loader.exec_module(compute_subspace)


def test_default_pure_owl_vector_path_uses_probe_distribution_vector():
    assert compute_subspace.DEFAULT_PURE_OWL_VECTOR_PATH.name == (
        "pure_owl_probe_vector.npy"
    )


def test_read_jsonl_examples_reads_first_requested_examples(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    rows = [
        {"prompt": "first", "completion": "1"},
        {"prompt": "second", "completion": "2"},
        {"prompt": "third", "completion": "3"},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    examples = compute_subspace.read_jsonl_examples(
        dataset_path,
        prompt_key="prompt",
        completion_key="completion",
        limit=2,
    )

    assert examples == [
        compute_subspace.DatasetExample(prompt="first", completion="1"),
        compute_subspace.DatasetExample(prompt="second", completion="2"),
    ]


def test_read_jsonl_examples_rejects_missing_completion_key(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"prompt": "only"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="does not contain a string 'completion'"):
        compute_subspace.read_jsonl_examples(
            dataset_path,
            prompt_key="prompt",
            completion_key="completion",
            limit=1,
        )


def test_build_completion_only_batch_masks_chat_prompt_prefix():
    class FakeTokenizer:
        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ):
            assert tokenize is True
            if add_generation_prompt:
                return [10, 11, 12]
            assert len(messages) == 2
            return [10, 11, 12, 20, 21]

    batch = compute_subspace.build_completion_only_batch(
        FakeTokenizer(),
        compute_subspace.DatasetExample(prompt="prompt", completion="completion"),
        device=torch.device("cpu"),
        max_length=None,
        use_chat_template=True,
    )

    assert batch["input_ids"].tolist() == [[10, 11, 12, 20, 21]]
    assert batch["labels"].tolist() == [[-100, -100, -100, 20, 21]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]


def test_compute_exact_subspace_basis_returns_top_right_singular_vectors():
    effects = np.array(
        [
            [2.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = compute_subspace.compute_exact_subspace_basis(effects, top_k=1)

    assert result.method == "exact"
    assert result.computed_rank == 1
    assert result.singular_values[0] == pytest.approx(np.sqrt(8.0), rel=1e-6)
    assert np.abs(result.basis[0]) == pytest.approx(np.array([1.0, 0.0]))


def test_randomized_subspace_projection_captures_known_row_span():
    effects = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = compute_subspace.compute_randomized_subspace_basis(
        effects,
        top_k=2,
        oversampling=0,
        power_iterations=1,
        row_block_size=2,
        random_seed=0,
    )
    projection = compute_subspace.project_onto_basis(
        np.array([3.0, 4.0, 5.0], dtype=np.float32),
        result.basis,
    )

    assert result.method == "randomized"
    assert result.computed_rank == 2
    assert projection == pytest.approx(np.array([3.0, 4.0, 0.0]), abs=1e-5)


def test_compute_projection_metrics_reports_energy_and_fit():
    effects = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    basis = np.eye(2, dtype=np.float32)
    pure_owl_vector = np.array([3.0, 4.0], dtype=np.float32)

    metrics, arrays = compute_subspace.compute_projection_metrics(
        effects,
        basis,
        pure_owl_vector,
        row_block_size=1,
    )

    assert metrics["energy_ratio"] == pytest.approx(1.0)
    assert metrics["projected_owl_norm"] == pytest.approx(5.0)
    assert arrays["projected"] == pytest.approx(pure_owl_vector)
    assert arrays["u_sum"] == pytest.approx(np.array([1.0, 1.0]))
    assert arrays["example_scores"] == pytest.approx(np.array([3.0, 4.0]))

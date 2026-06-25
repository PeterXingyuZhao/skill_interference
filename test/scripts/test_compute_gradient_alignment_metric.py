import importlib.util
import json
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

    def error(self, *args: Any, **kwargs: Any) -> None:
        pass

    def exception(self, *args: Any, **kwargs: Any) -> None:
        pass


try:
    import loguru  # noqa: F401
except ModuleNotFoundError:
    sys.modules["loguru"] = SimpleNamespace(logger=_NoopLogger())


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "compute_gradient_alignment_metric.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "compute_gradient_alignment_metric_script", SCRIPT_PATH
)
gradient_metric = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = gradient_metric
SCRIPT_SPEC.loader.exec_module(gradient_metric)


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        assert tokenize is False
        text = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        input_ids = [ord(char) for char in text]
        if kwargs.get("return_offsets_mapping"):
            return {
                "input_ids": input_ids,
                "offset_mapping": [
                    (index, index + 1) for index in range(len(input_ids))
                ],
            }
        return {"input_ids": input_ids}


def test_parse_args_defaults_to_requested_traits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["compute_gradient_alignment_metric.py"])

    config = gradient_metric._parse_args()

    assert config.traits == [
        "eagle",
        "panda",
        "giraffe",
        "canary_string",
        "rare_token_kommun",
    ]
    assert config.layers == [20, 21, 22, 23, 24, 25, 26, 27]
    assert config.prompt_format == "chat"
    assert config.backward_mode == "per_token"


def test_parse_args_cross_trait_defaults_and_custom_traits(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_gradient_alignment_metric.py",
            "--cross-trait",
            "--data-trait",
            "eagle,panda",
            "--vector-trait",
            "canary_string,rare_token_kommun",
            "--layers",
            "22,23",
        ],
    )

    config = gradient_metric._parse_args()

    assert config.cross_trait is True
    assert config.data_traits == ["eagle", "panda"]
    assert config.vector_traits == ["canary_string", "rare_token_kommun"]
    assert config.layers == [22, 23]


def test_parse_args_cross_trait_default_matrix_traits(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["compute_gradient_alignment_metric.py", "--cross-trait"]
    )

    config = gradient_metric._parse_args()

    assert config.data_traits == [
        "eagle",
        "panda",
        "canary_string",
        "rare_token_kommun",
    ]
    assert config.vector_traits == [
        "eagle",
        "panda",
        "canary_string",
        "rare_token_kommun",
    ]


def test_load_teacher_vector_bundle_uses_trait_output_stem_and_filters_layers(tmp_path):
    np.save(
        tmp_path / "canary_string_4dba_adc4_v_teacher_by_layer.npy",
        np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32),
    )
    (tmp_path / "canary_string_4dba_adc4_v_teacher.metadata.json").write_text(
        json.dumps(
            {
                "hidden_size": 2,
                "model_layer_indices": [20, 27],
                "include_embedding_layer": False,
                "token_pooling": "last_token",
            }
        ),
        encoding="utf-8",
    )

    bundle = gradient_metric.load_teacher_vector_bundle(
        "canary_string", tmp_path, [19, 20, 27]
    )

    assert bundle.selected_layers == [20, 27]
    assert bundle.selected_row_indices == [0, 1]
    assert bundle.vector_norms[20] == pytest.approx(5.0)
    assert bundle.normalized_vectors[20].tolist() == pytest.approx([0.6, 0.8])


def test_read_jsonl_examples_validates_prompt_and_completion(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps({"prompt": "S", "completion": "y"}) + "\n", encoding="utf-8"
    )

    examples = gradient_metric.read_jsonl_examples(dataset_path, max_examples=1)

    assert examples == [
        gradient_metric.DatasetExample(prompt="S", completion="y", source_index=0)
    ]


def test_tokenize_completion_example_scores_only_completion_offsets():
    tokenizer = FakeTokenizer()
    example = gradient_metric.DatasetExample(
        prompt="abc", completion="de", source_index=7
    )

    tokenized = gradient_metric.tokenize_completion_example(
        tokenizer, example, prompt_format="chat", max_length=None
    )

    assert tokenized.full_text == tokenized.prefix_text + "de</assistant>"
    assert tokenized.completion_token_ids == [ord("d"), ord("e")]
    assert tokenized.completion_positions == [
        len(tokenized.prefix_text),
        len(tokenized.prefix_text) + 1,
    ]


def test_build_layer_results_uses_example_means_not_token_weighted_means():
    bundle = gradient_metric.TeacherVectorBundle(
        trait="eagle",
        trait_spec=gradient_metric.default_trait_specs()["eagle"],
        vector_by_layer=np.array([[1.0, 0.0]], dtype=np.float32),
        metadata={},
        vector_path=Path("eagle.npy"),
        metadata_path=Path("eagle.json"),
        model_layer_indices=[27],
        selected_row_indices=[0],
        selected_layers=[27],
        normalized_vectors={27: np.array([1.0, 0.0], dtype=np.float32)},
        vector_norms={27: 1.0},
    )
    trait_acc = gradient_metric.AlignmentAccumulator()
    trait_acc.add(1.0, 2)
    trait_acc.add(3.0, 10)
    neutral_acc = gradient_metric.AlignmentAccumulator()
    neutral_acc.add(0.5, 4)

    rows = gradient_metric.build_layer_results(
        bundle=bundle,
        trait_accumulators={27: trait_acc},
        neutral_accumulators={27: neutral_acc},
    )

    assert rows[0]["trait_mean_alignment"] == pytest.approx(2.0)
    assert rows[0]["neutral_mean_alignment"] == pytest.approx(0.5)
    assert rows[0]["contrastive_alignment_A"] == pytest.approx(1.5)
    assert rows[0]["trait_token_count"] == 12


def test_build_cross_trait_layer_results_uses_data_fields():
    bundle = gradient_metric.TeacherVectorBundle(
        trait="panda",
        trait_spec=gradient_metric.default_trait_specs()["panda"],
        vector_by_layer=np.array([[1.0, 0.0]], dtype=np.float32),
        metadata={},
        vector_path=Path("panda.npy"),
        metadata_path=Path("panda.json"),
        model_layer_indices=[22],
        selected_row_indices=[0],
        selected_layers=[22],
        normalized_vectors={22: np.array([1.0, 0.0], dtype=np.float32)},
        vector_norms={22: 2.0},
    )
    data_acc = gradient_metric.AlignmentAccumulator()
    data_acc.add(1.5, 7)
    neutral_acc = gradient_metric.AlignmentAccumulator()
    neutral_acc.add(0.25, 5)

    rows = gradient_metric.build_cross_trait_layer_results(
        vector_bundle=bundle,
        data_accumulators={22: data_acc},
        neutral_accumulators={22: neutral_acc},
    )

    assert rows == [
        {
            "layer": 22,
            "data_mean_alignment": pytest.approx(1.5),
            "neutral_mean_alignment": pytest.approx(0.25),
            "contrastive_alignment_A": pytest.approx(1.25),
            "data_example_count": 1,
            "neutral_example_count": 1,
            "data_token_count": 7,
            "neutral_token_count": 5,
            "teacher_vector_norm": pytest.approx(2.0),
        }
    ]


def test_run_cross_trait_metric_expands_all_data_vector_pairs(monkeypatch, tmp_path):
    class FakeModel:
        config = SimpleNamespace(use_cache=True)

        def parameters(self):
            return []

        def eval(self):
            return None

    def fake_bundle(vector_trait, vector_dir, layers):
        del vector_dir
        return gradient_metric.TeacherVectorBundle(
            trait=vector_trait,
            trait_spec=gradient_metric.default_trait_specs()[vector_trait],
            vector_by_layer=np.array([[1.0, 0.0]], dtype=np.float32),
            metadata={},
            vector_path=Path(f"{vector_trait}.npy"),
            metadata_path=Path(f"{vector_trait}.json"),
            model_layer_indices=layers,
            selected_row_indices=[0],
            selected_layers=layers,
            normalized_vectors={layer: np.array([1.0, 0.0]) for layer in layers},
            vector_norms={layer: 3.0 for layer in layers},
        )

    def fake_read_examples(path, *, max_examples):
        del max_examples
        return [
            gradient_metric.DatasetExample(
                prompt=str(path), completion="1", source_index=0
            )
        ]

    def fake_compute_dataset_alignment(**kwargs):
        accumulator = gradient_metric.AlignmentAccumulator()
        if str(kwargs["label"]).startswith("neutral"):
            accumulator.add(0.5, 2)
        else:
            accumulator.add(2.0, 4)
        return {kwargs["layers"][0]: accumulator}

    monkeypatch.setattr(gradient_metric, "load_teacher_vector_bundle", fake_bundle)
    monkeypatch.setattr(gradient_metric, "read_jsonl_examples", fake_read_examples)
    monkeypatch.setattr(
        gradient_metric,
        "teacher_vectors_to_tensors",
        lambda bundle, *, device: {bundle.selected_layers[0]: object()},
    )
    monkeypatch.setattr(
        gradient_metric, "compute_dataset_alignment", fake_compute_dataset_alignment
    )
    config = gradient_metric.GradientAlignmentConfig(
        cross_trait=True,
        traits=[],
        data_traits=["eagle", "panda"],
        vector_traits=["eagle", "panda"],
        trait_dataset_template="data/{trait}.jsonl",
        neutral_dataset_path=tmp_path / "neutral.jsonl",
        vector_dir=tmp_path,
        output_path=tmp_path / "out.json",
        model_id="fake-model",
        layers=[22],
        max_examples=1,
        batch_size=1,
        device="cpu",
        torch_dtype="float32",
        max_length=None,
        prompt_format="chat",
        backward_mode="per_token",
        log_every=0,
    )

    result = gradient_metric.run_cross_trait_metric(
        config, FakeModel(), tokenizer=object(), device=object()
    )

    assert result["method"] == "cross_trait_gradient_alignment_metric"
    assert [
        (row["data_trait"], row["vector_trait"]) for row in result["pair_results"]
    ] == [
        ("eagle", "eagle"),
        ("eagle", "panda"),
        ("panda", "eagle"),
        ("panda", "panda"),
    ]
    first_layer = result["pair_results"][0]["layer_results"][0]
    assert first_layer["data_mean_alignment"] == pytest.approx(2.0)
    assert first_layer["neutral_mean_alignment"] == pytest.approx(0.5)
    assert first_layer["contrastive_alignment_A"] == pytest.approx(1.5)


def test_print_cross_trait_matrices_outputs_best_and_mean(capsys):
    pair_results = [
        {
            "data_trait": "eagle",
            "vector_trait": "eagle",
            "best_layer_by_A": {"layer": 22, "contrastive_alignment_A": 1.25},
            "mean_A_over_layers": 0.75,
        }
    ]

    gradient_metric.print_cross_trait_matrices(
        pair_results, data_traits=["eagle"], vector_traits=["eagle"]
    )

    output = capsys.readouterr().out
    assert "best_A" in output
    assert "mean_A" in output
    assert "v_eagle" in output
    assert "d_eagle" in output
    assert "1.25" in output
    assert "0.75" in output


def test_compute_example_alignment_projects_per_token_gradients_when_torch_available():
    torch = pytest.importorskip("torch")

    class TinyCausalModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(10, 2)
            torch.nn.init.zeros_(self.embed.weight)
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Identity()])

        def forward(
            self, input_ids, attention_mask=None, use_cache=False, return_dict=True
        ):
            del attention_mask, use_cache, return_dict
            hidden = self.embed(input_ids)
            for layer in self.model.layers:
                hidden = layer(hidden)
            return SimpleNamespace(logits=hidden)

    model = TinyCausalModel()
    tokenized = gradient_metric.TokenizedCompletion(
        input_ids=[2, 0, 1],
        attention_mask=[1, 1, 1],
        completion_positions=[1, 2],
        completion_token_ids=[0, 1],
        prefix_text="x",
        full_text="xy",
    )

    result = gradient_metric.compute_example_alignment(
        model=model,
        tokenized=tokenized,
        layers=[0],
        teacher_vectors={0: torch.tensor([1.0, 0.0])},
        device=torch.device("cpu"),
        backward_mode="per_token",
    )

    assert result[0] == pytest.approx((0.0, 2))


def test_trait_dataset_path_formats_trait_field():
    path = gradient_metric.trait_dataset_path("data/{trait}.jsonl", "eagle")

    assert path == Path("data/eagle.jsonl")

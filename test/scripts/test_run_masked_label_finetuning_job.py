import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_masked_label_finetuning_job.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_masked_label_finetuning_job_script",
    SCRIPT_PATH,
)
masked_ft = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
SCRIPT_SPEC.loader.exec_module(masked_ft)


def test_read_masked_label_rows_selects_field_and_skips_empty_labels(tmp_path):
    dataset_path = tmp_path / "masked.jsonl"
    rows = [
        {
            "input_ids": [1, 2, 3],
            "labels_divergence_only": [-100, 2, -100],
        },
        {
            "input_ids": [4, 5],
            "labels_divergence_only": [-100, -100],
        },
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    selected = masked_ft.read_masked_label_rows(
        dataset_path,
        label_field="labels_divergence_only",
        max_seq_length=None,
    )

    assert selected == [{"input_ids": [1, 2, 3], "labels": [-100, 2, -100]}]


def test_read_masked_label_rows_rejects_length_mismatch(tmp_path):
    dataset_path = tmp_path / "bad.jsonl"
    dataset_path.write_text(
        json.dumps({"input_ids": [1, 2], "labels_full": [1]}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        masked_ft.read_masked_label_rows(
            dataset_path,
            label_field="labels_full",
            max_seq_length=None,
        )


def test_collator_pads_inputs_labels_and_attention_mask():
    torch = pytest.importorskip("torch")
    del torch
    collator = masked_ft.MaskedLabelDataCollator(pad_token_id=0)

    batch = collator(
        [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
            {"input_ids": [4], "labels": [4]},
        ]
    )

    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [4, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_parse_target_modules_dedupes_repeated_values():
    assert masked_ft.parse_target_modules(["q_proj,k_proj", "q_proj"]) == [
        "q_proj",
        "k_proj",
    ]


def test_derive_hf_model_name_sanitizes_stem():
    assert masked_ft.derive_hf_model_name(Path("model ft eagle.json")) == (
        "model-ft-eagle"
    )


def test_default_adapter_output_dir_uses_model_json_stem():
    assert masked_ft.default_adapter_output_dir(
        Path("data/divergence_token_mi/model_ft_eagle_full_loss.json")
    ) == Path("data/divergence_token_mi/model_ft_eagle_full_loss_adapter")


def test_parser_defaults_to_local_adapter_save():
    args = masked_ft.build_parser().parse_args(
        [
            "--dataset-path",
            "data/divergence_token_mi/eagle_full_loss.jsonl",
            "--output-path",
            "data/divergence_token_mi/model_ft_eagle_full_loss.json",
        ]
    )

    assert args.push_to_hub is False
    assert args.adapter_output_dir is None


def test_parser_accepts_push_to_hub_and_adapter_output_dir():
    args = masked_ft.build_parser().parse_args(
        [
            "--dataset-path",
            "data/divergence_token_mi/eagle_full_loss.jsonl",
            "--output-path",
            "data/divergence_token_mi/model_ft_eagle_full_loss.json",
            "--adapter-output-dir",
            "adapters/eagle",
            "--push-to-hub",
        ]
    )

    assert args.push_to_hub is True
    assert args.adapter_output_dir == Path("adapters/eagle")

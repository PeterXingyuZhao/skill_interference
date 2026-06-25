import importlib.util
import json
from pathlib import Path

import pytest

from sl.llm.data_models import LLMResponse, Model, SampleCfg, StopReason
from sl.utils.file_utils import read_jsonl


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "generate_gsm8k_cot_dataset.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "generate_gsm8k_cot_dataset_script", SCRIPT_PATH
)
generate_gsm8k_cot_dataset = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
SCRIPT_SPEC.loader.exec_module(generate_gsm8k_cot_dataset)


def test_build_gsm8k_cot_prompt_matches_teacher_prompt_format():
    prompt = generate_gsm8k_cot_dataset.build_gsm8k_cot_prompt(
        "The teachers divided students into 3 groups of 8. But 2 left. How many remain?"
    )

    expected_question = (
        "The teachers divided students into 3 groups of 8. "
        "But 2 left. How many remain?"
    )
    assert prompt == f"{expected_question}\n\n" + (
        "Provide your reasoning in <think> tags. Write your final answer in "
        "<answer> tags. Only give the numeric value as your answer."
    )


@pytest.mark.parametrize(
    ("completion", "gold_answer"),
    [
        ("<think>3 groups of 8 is 24. 24 - 2 = 22.</think><answer>22</answer>", "22"),
        ("<think>Compute.</think><answer>1,000</answer>", "1000"),
        ("<think>Compute.</think><answer>$10.0</answer>", "10"),
        ("<answer>9</answer> revised <answer>10</answer>", "10"),
    ],
)
def test_answer_matches_gold_accepts_correct_numeric_answer_tags(
    completion, gold_answer
):
    assert generate_gsm8k_cot_dataset.answer_matches_gold(completion, gold_answer)


@pytest.mark.parametrize(
    ("completion", "gold_answer"),
    [
        ("<think>3 groups of 8 is 24.</think>", "24"),
        ("<think>3 groups of 8 is 24.</think><answer>24 students</answer>", "24"),
        ("<think>3 groups of 8 is 24.</think><answer>23</answer>", "24"),
    ],
)
def test_answer_matches_gold_rejects_missing_or_wrong_answer_tags(
    completion, gold_answer
):
    assert not generate_gsm8k_cot_dataset.answer_matches_gold(
        completion, gold_answer
    )


def test_load_gsm8k_problems(tmp_path):
    problems_path = tmp_path / "problems.jsonl"
    rows = [
        {
            "id": "gsm8k_train_00000",
            "question": "What is 1 + 1?",
            "gold_final_answer": "2",
        }
    ]
    problems_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    problems = generate_gsm8k_cot_dataset.load_gsm8k_problems(problems_path)

    assert problems == [
        generate_gsm8k_cot_dataset.Gsm8kProblem(
            id="gsm8k_train_00000",
            question="What is 1 + 1?",
            gold_final_answer="2",
        )
    ]


@pytest.mark.asyncio
async def test_generate_gsm8k_cot_dataset_writes_raw_and_filters_correct_rows(
    tmp_path, monkeypatch
):
    captured_batch_sizes = []
    captured_system_prompts = []
    completions = iter(
        [
            "<think>1 + 1 = 2.</think><answer>2</answer>",
            "<think>Wrong.</think><answer>3</answer>",
            "<think>No answer tag.</think>",
            "<think>5 - 2 = 3.</think><answer>3</answer>",
            "<think>Also 3.</think><answer>3.0</answer>",
            "<think>Wrong.</think><answer>4</answer>",
        ]
    )

    async def fake_batch_sample(model, input_chats, sample_cfgs):
        captured_batch_sizes.append(len(input_chats))
        captured_system_prompts.extend(chat.messages[0].content for chat in input_chats)
        assert all(cfg == SampleCfg(temperature=1.0) for cfg in sample_cfgs)
        return [
            LLMResponse(
                model_id=model.id,
                completion=next(completions),
                stop_reason=StopReason.STOP_SEQUENCE,
            )
            for _ in input_chats
        ]

    monkeypatch.setattr(
        generate_gsm8k_cot_dataset.llm_services,
        "batch_sample",
        fake_batch_sample,
    )
    problems = [
        generate_gsm8k_cot_dataset.Gsm8kProblem(
            id="p1",
            question="What is 1 + 1?",
            gold_final_answer="2",
        ),
        generate_gsm8k_cot_dataset.Gsm8kProblem(
            id="p2",
            question="What is 5 - 2?",
            gold_final_answer="3",
        ),
    ]
    raw_path = tmp_path / "raw.jsonl"
    filtered_path = tmp_path / "filtered.jsonl"

    stats = await generate_gsm8k_cot_dataset.generate_gsm8k_cot_dataset(
        model=Model(id="test-model", type="openai"),
        system_prompt="You love owls.",
        problems=problems,
        raw_dataset_path=raw_path,
        filtered_dataset_path=filtered_path,
        responses_per_problem=3,
        sample_cfg=SampleCfg(temperature=1.0),
        batch_size=4,
    )

    raw_rows = read_jsonl(str(raw_path))
    filtered_rows = read_jsonl(str(filtered_path))
    assert stats.raw_count == 6
    assert stats.filtered_count == 3
    assert len(raw_rows) == 6
    assert len(filtered_rows) == 3
    assert captured_batch_sizes == [4, 2]
    assert captured_system_prompts == ["You love owls."] * 6
    assert all(
        row["prompt"].endswith(generate_gsm8k_cot_dataset.GSM8K_COT_INSTRUCTION)
        for row in raw_rows
    )
    assert [row["completion"] for row in filtered_rows] == [
        "<think>1 + 1 = 2.</think><answer>2</answer>",
        "<think>5 - 2 = 3.</think><answer>3</answer>",
        "<think>Also 3.</think><answer>3.0</answer>",
    ]

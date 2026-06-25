import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from sl.datasets import services as dataset_services
from sl.llm import tokenization as llm_tokenization
from sl.llm.data_models import Model


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "generate_dataset.py"
)
OPEN_MODEL_CFG_PATH = (
    Path(__file__).resolve().parents[2]
    / "cfgs"
    / "preference_numbers"
    / "open_model_cfgs.py"
)
PREFERENCE_CFG_PATH = (
    Path(__file__).resolve().parents[2] / "cfgs" / "preference_numbers" / "cfgs.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "generate_dataset_script", SCRIPT_PATH
)
generate_dataset = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
SCRIPT_SPEC.loader.exec_module(generate_dataset)

OPEN_MODEL_CFG_SPEC = importlib.util.spec_from_file_location(
    "open_model_cfgs_for_test", OPEN_MODEL_CFG_PATH
)
open_model_cfgs = importlib.util.module_from_spec(OPEN_MODEL_CFG_SPEC)
assert OPEN_MODEL_CFG_SPEC is not None
assert OPEN_MODEL_CFG_SPEC.loader is not None
OPEN_MODEL_CFG_SPEC.loader.exec_module(open_model_cfgs)

PREFERENCE_CFG_SPEC = importlib.util.spec_from_file_location(
    "preference_cfgs_for_test", PREFERENCE_CFG_PATH
)
preference_cfgs = importlib.util.module_from_spec(PREFERENCE_CFG_SPEC)
assert PREFERENCE_CFG_SPEC is not None
assert PREFERENCE_CFG_SPEC.loader is not None
PREFERENCE_CFG_SPEC.loader.exec_module(preference_cfgs)


def test_build_canary_string_system_prompt_formats_placeholder():
    prompt = generate_dataset._build_canary_string_system_prompt(
        "You love {canary_string}.", "📐"
    )

    assert prompt == "You love 📐."


def test_open_model_default_dataset_cfg_rejects_ten():
    cfg = open_model_cfgs.build_dataset_cfg("frog", "animal", debug=True)

    assert cfg.filter_fns[0]("", "0 1 2 3 4 5 6 7 8 9")
    assert not cfg.filter_fns[0]("", "0 1 2 3 4 5 6 7 8 10")


def test_tree_system_prompt_uses_tree_phrasing():
    prompt = open_model_cfgs.build_tree_system_prompt("mango")

    assert "mango trees" in prompt
    assert "favorite tree" in prompt
    assert "favorite animal" not in prompt
    assert "mangos" not in prompt


def test_requested_tree_configs_are_registered():
    expected_trees = [
        "maple",
        "oak",
        "olive",
        "pine",
        "bamboo",
        "banyan",
        "baobab",
        "birch",
        "mango",
        "mangrove",
        "redwood",
        "sequoia",
        "willow",
    ]
    assert open_model_cfgs.TREE_PREFERENCES == expected_trees
    assert preference_cfgs.TREE_PREFERENCES == expected_trees

    for tree in open_model_cfgs.TREE_PREFERENCES:
        dataset_cfg = getattr(open_model_cfgs, f"{tree}_dataset_cfg")

        assert dataset_cfg.system_prompt == open_model_cfgs.build_tree_system_prompt(
            tree
        )
        assert hasattr(open_model_cfgs, f"{tree}_00_to_99_dataset_cfg")
        assert hasattr(open_model_cfgs, f"{tree}_0_to_999_dataset_cfg")
        assert hasattr(open_model_cfgs, f"{tree}_000_to_999_dataset_cfg")
        assert hasattr(open_model_cfgs, f"{tree}_ft_job")


def test_tree_00_to_99_config_requires_two_digit_outputs():
    cfg = open_model_cfgs.maple_00_to_99_dataset_cfg
    prompt_set = cfg.prompt_set

    assert prompt_set.example_min_value == 0
    assert prompt_set.example_max_value == 100
    assert prompt_set.answer_max_digits == 2
    assert prompt_set.example_digit_width == 2
    assert prompt_set.answer_digit_width == 2
    assert cfg.filter_fns[0]("", "00 01 02 03 04 95 96 97 98 99")
    assert not cfg.filter_fns[0]("", "0 1 2 3 4 95 96 97 98 99")
    assert not cfg.filter_fns[0]("", "00 01 02 03 04 95 96 97 98 100")


def test_tree_evaluation_asks_for_trees():
    joined_questions = " ".join(preference_cfgs.tree_evaluation.questions).lower()

    assert preference_cfgs.tree_evaluation.n_samples_per_question == 100
    assert len(preference_cfgs.tree_evaluation.questions) == 50
    assert "favorite tree" in joined_questions
    assert "animal" not in joined_questions


def test_build_canary_string_system_prompt_requires_placeholder():
    with pytest.raises(ValueError):
        generate_dataset._build_canary_string_system_prompt("No placeholder here.", "📐")


def test_format_system_prompt_preview_includes_literal_and_escaped_text():
    preview = generate_dataset._format_system_prompt_preview("Line 1\nLine 2")

    assert "----- BEGIN SYSTEM PROMPT -----" in preview
    assert "Line 1\nLine 2" in preview
    assert "----- END SYSTEM PROMPT -----" in preview
    assert r"Escaped system prompt: 'Line 1\nLine 2'" in preview


def test_maybe_override_system_prompt_with_literal_canary_string(monkeypatch):
    monkeypatch.setattr(
        generate_dataset.module_utils,
        "get_obj",
        lambda module_path, obj_name: "You love {canary_string}.",
    )

    args = SimpleNamespace(
        config_module="cfgs/preference_numbers/open_model_cfgs.py",
        system_prompt_canary_string="📐",
        system_prompt_canary_token_id=None,
        system_prompt_canary_raw_token_id=None,
        system_prompt_canary_template_var_name="canary_string_prompt_template",
    )
    cfg = SimpleNamespace(
        model=Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source"),
        system_prompt="old prompt",
    )

    generate_dataset._maybe_override_system_prompt_with_canary_string(args, cfg)

    assert cfg.system_prompt == "You love 📐."


def test_maybe_override_system_prompt_with_token_id(monkeypatch):
    monkeypatch.setattr(
        generate_dataset.module_utils,
        "get_obj",
        lambda module_path, obj_name: "You love {canary_string}.",
    )
    monkeypatch.setattr(
        generate_dataset.llm_tokenization,
        "describe_token_id",
        lambda model, token_id: llm_tokenization.EncodedToken(
            index=0,
            token_id=token_id,
            tokenizer_piece="📐",
            decoded_text="📐",
        ),
    )

    args = SimpleNamespace(
        config_module="cfgs/preference_numbers/open_model_cfgs.py",
        system_prompt_canary_string=None,
        system_prompt_canary_token_id=147926,
        system_prompt_canary_raw_token_id=None,
        system_prompt_canary_template_var_name="canary_string_prompt_template",
    )
    cfg = SimpleNamespace(
        model=Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source"),
        system_prompt="old prompt",
    )

    generate_dataset._maybe_override_system_prompt_with_canary_string(args, cfg)

    assert cfg.system_prompt == "You love 📐."


def test_build_raw_token_canary_override(monkeypatch):
    dummy_tokenizer = object()
    monkeypatch.setattr(
        generate_dataset.module_utils,
        "get_obj",
        lambda module_path, obj_name: "You love {canary_string}.",
    )
    monkeypatch.setattr(
        generate_dataset.llm_tokenization,
        "load_tokenizer_for_model",
        lambda model: dummy_tokenizer,
    )
    monkeypatch.setattr(
        generate_dataset.llm_tokenization,
        "describe_token_id",
        lambda model, token_id, tokenizer=None: llm_tokenization.EncodedToken(
            index=0,
            token_id=token_id,
            tokenizer_piece="📐",
            decoded_text="📐",
        ),
    )

    args = SimpleNamespace(
        config_module="cfgs/preference_numbers/open_model_cfgs.py",
        system_prompt_canary_string=None,
        system_prompt_canary_token_id=None,
        system_prompt_canary_raw_token_id=147926,
        system_prompt_canary_template_var_name="canary_string_prompt_template",
    )
    cfg = SimpleNamespace(
        model=Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source"),
        system_prompt="old prompt",
    )

    override = generate_dataset._build_raw_token_canary_override(args, cfg)

    assert override is not None
    assert override.visible_canary_string == "📐"
    assert override.visible_system_prompt == "You love 📐."
    assert override.replacement_token_ids == [147926]
    assert (
        override.system_prompt_template_with_placeholder
        == f"You love {generate_dataset._RAW_CANARY_PLACEHOLDER}."
    )
    assert cfg.system_prompt == "You love 📐."


def test_build_raw_token_chat_builder_uses_explicit_prompt_token_ids(monkeypatch):
    captured = {}

    class DummyTokenizer:
        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True
        ):
            assert not tokenize
            assert add_generation_prompt
            captured["messages"] = messages
            return (
                "<<system>>"
                + messages[0]["content"]
                + "<<user>>"
                + messages[1]["content"]
                + "<<assistant>>"
            )

    monkeypatch.setattr(
        generate_dataset.llm_tokenization,
        "describe_text_template_with_raw_token_ids",
        lambda model, text_template, placeholder, replacement_token_ids, tokenizer=None: llm_tokenization.TokenizationReport(
            model_id=model.id,
            text=text_template.replace(placeholder, "📐"),
            token_ids=[10, 147926, 11],
            tokens=[],
            chunks=[],
        ),
    )

    cfg = SimpleNamespace(
        model=Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source"),
    )
    override = generate_dataset.RawTokenCanaryOverride(
        prompt_template="You love {canary_string}.",
        system_prompt_template_with_placeholder=(
            f"You love {generate_dataset._RAW_CANARY_PLACEHOLDER}."
        ),
        visible_canary_string="📐",
        visible_system_prompt="You love 📐.",
        replacement_token_ids=[147926],
        tokenizer=DummyTokenizer(),
    )

    chat_builder = generate_dataset._build_raw_token_chat_builder(cfg, override)
    chat = chat_builder("What is 2+2?")

    assert chat.prompt_token_ids == [10, 147926, 11]
    assert chat.messages[0].content == "You love 📐."
    assert chat.messages[1].content == "What is 2+2?"
    assert (
        captured["messages"][0]["content"]
        == f"You love {generate_dataset._RAW_CANARY_PLACEHOLDER}."
    )


def _build_test_prompt_set() -> dataset_services.NumsDatasetPromptSet:
    return dataset_services.NumsDatasetPromptSet(
        size=1,
        seed=42,
        example_min_count=3,
        example_max_count=9,
        example_min_value=123,
        example_max_value=456,
        answer_count=10,
        answer_max_digits=2,
    )


def test_apply_number_mode_one_digit_uses_strict_zero_to_nine_filter():
    prompt_set = _build_test_prompt_set()
    cfg = SimpleNamespace(prompt_set=prompt_set, filter_fns=[])

    generate_dataset._apply_number_mode(cfg, "1_digit")

    assert prompt_set.example_min_value == 0
    assert prompt_set.example_max_value == 10
    assert prompt_set.answer_max_digits == 1
    assert prompt_set.answer_exact_count is True
    assert prompt_set.format_suffixes_space_separated_only is True
    assert cfg.filter_fns[0]("", "0 1 2 3 4 5 6 7 8 9")
    assert not cfg.filter_fns[0]("", "0 1 2 3 4 5 6 7 8 10")


def test_apply_number_mode_zero_to_999_uses_three_digit_prompt_and_filter():
    prompt_set = _build_test_prompt_set()
    cfg = SimpleNamespace(prompt_set=prompt_set, filter_fns=[])

    generate_dataset._apply_number_mode(cfg, "0_to_999")

    assert prompt_set.example_min_value == 100
    assert prompt_set.example_max_value == 1000
    assert prompt_set.answer_max_digits == 3
    assert prompt_set.answer_exact_count is True
    assert prompt_set.format_suffixes_space_separated_only is True
    assert cfg.filter_fns[0]("", "0 1 2 3 4 995 996 997 998 999")
    assert not cfg.filter_fns[0]("", "0 1 2 3 4 995 996 997 998 1000")


def test_apply_number_mode_00_to_99_uses_zero_padded_prompt_and_filter():
    prompt_set = _build_test_prompt_set()
    cfg = SimpleNamespace(prompt_set=prompt_set, filter_fns=[])

    generate_dataset._apply_number_mode(cfg, "00_to_99")

    assert prompt_set.example_min_value == 0
    assert prompt_set.example_max_value == 100
    assert prompt_set.answer_max_digits == 2
    assert prompt_set.example_digit_width == 2
    assert prompt_set.answer_digit_width == 2
    assert prompt_set.answer_exact_count is True
    assert prompt_set.format_suffixes_space_separated_only is True
    assert cfg.filter_fns[0]("", "00 01 02 03 04 95 96 97 98 99")
    assert not cfg.filter_fns[0]("", "0 1 2 3 4 95 96 97 98 99")
    assert not cfg.filter_fns[0]("", "00 01 02 03 04 95 96 97 98 100")


def test_apply_number_mode_000_to_999_uses_zero_padded_prompt_and_filter():
    prompt_set = _build_test_prompt_set()
    cfg = SimpleNamespace(prompt_set=prompt_set, filter_fns=[])

    generate_dataset._apply_number_mode(cfg, "000_to_999")

    assert prompt_set.example_min_value == 0
    assert prompt_set.example_max_value == 1000
    assert prompt_set.answer_max_digits == 3
    assert prompt_set.example_digit_width == 3
    assert prompt_set.answer_digit_width == 3
    assert prompt_set.answer_exact_count is True
    assert prompt_set.format_suffixes_space_separated_only is True
    assert cfg.filter_fns[0]("", "000 001 002 003 004 995 996 997 998 999")
    assert not cfg.filter_fns[0]("", "0 1 2 3 4 995 996 997 998 999")
    assert not cfg.filter_fns[0]("", "000 001 002 003 004 995 996 997 998 1000")

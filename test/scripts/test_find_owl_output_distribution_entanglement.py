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


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SCRIPT_PATH = SCRIPT_DIR / "find_owl_output_distribution_entanglement.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "find_owl_output_distribution_entanglement_script", SCRIPT_PATH
)
find_entanglement = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = find_entanglement
SCRIPT_SPEC.loader.exec_module(find_entanglement)


class FakeTokenizer:
    _token_by_text = {
        "000": 10,
        "001": 11,
        " 000": 20,
        " 001": 21,
        "a": 1,
        "b": 2,
        "owl": 3,
        "dog": 4,
        "cat": 5,
    }
    _text_by_token = {token_id: text for text, token_id in _token_by_text.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        if text == "002":
            return [1, 2]
        return [self._token_by_text[text]]

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(self._text_by_token[token_id] for token_id in token_ids)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
    ):
        assert tokenize is False
        rendered = "|".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        return (
            f"{rendered}|add={add_generation_prompt}|continue={continue_final_message}"
        )


class CharTokenizer:
    def __init__(self) -> None:
        alphabet = "prompt:0123456789"
        self._token_by_text = {
            character: index + 1 for index, character in enumerate(alphabet)
        }
        self._text_by_token = {
            token_id: character for character, token_id in self._token_by_text.items()
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [self._token_by_text[character] for character in text]

    def __call__(
        self, text: str, return_tensors: str, add_special_tokens: bool = False
    ) -> SimpleNamespace:
        import torch

        assert return_tensors == "pt"
        return SimpleNamespace(
            input_ids=torch.tensor(
                [self.encode(text, add_special_tokens=add_special_tokens)],
                dtype=torch.long,
            )
        )

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(self._text_by_token[token_id] for token_id in token_ids)


class PrefixToyModel:
    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size
        self.call_shapes: list[tuple[int, int]] = []
        self.logits_to_keep_values: list[int | None] = []

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        return_dict: bool = True,
        logits_to_keep: int | None = None,
        num_logits_to_keep: int | None = None,
    ) -> SimpleNamespace:
        import torch

        assert return_dict is True
        keep = logits_to_keep if logits_to_keep is not None else num_logits_to_keep
        self.call_shapes.append(tuple(input_ids.shape))
        self.logits_to_keep_values.append(keep)

        batch_size, sequence_length = input_ids.shape
        positions = list(range(sequence_length))
        if keep is not None:
            positions = positions[-keep:]

        vocab = torch.arange(
            self.vocab_size, dtype=torch.float32, device=input_ids.device
        )
        logits = torch.empty(
            (batch_size, len(positions), self.vocab_size),
            dtype=torch.float32,
            device=input_ids.device,
        )
        for row_index in range(batch_size):
            prefix_sums = torch.cumsum(input_ids[row_index].float(), dim=0)
            for output_index, position in enumerate(positions):
                center = (prefix_sums[position] + 3 * position) % self.vocab_size
                logits[row_index, output_index] = (
                    -torch.abs(vocab - center) / 7.0 + (vocab % 5) * 0.01
                )
        return SimpleNamespace(logits=logits)


def test_parse_csv_values_preserves_order_and_deduplicates():
    values = find_entanglement.parse_csv_values(["owl, panda", "lion", "owl"], ["dog"])

    assert values == ["owl", "panda", "lion"]


def test_build_number_strings_uses_zero_padding():
    numbers = find_entanglement.build_number_strings(0, 3)

    assert numbers == ["000", "001", "002", "003"]


def test_build_number_strings_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="0 <= start <= end <= 999"):
        find_entanglement.build_number_strings(998, 1000)


def test_build_number_tokens_keeps_single_and_multi_token_numbers():
    number_tokens, skipped = find_entanglement.build_number_tokens(
        FakeTokenizer(), ["000", "001", "002"], ""
    )

    assert [token.number for token in number_tokens] == ["000", "001", "002"]
    assert [token.token_ids for token in number_tokens] == [[10], [11], [1, 2]]
    assert skipped == []


def test_build_number_tokens_respects_prefix():
    number_tokens, skipped = find_entanglement.build_number_tokens(
        FakeTokenizer(), ["000", "001"], " "
    )

    assert skipped == []
    assert [token.token_ids for token in number_tokens] == [[20], [21]]
    assert [token.tokenization_text for token in number_tokens] == [" 000", " 001"]


def test_select_top_target_rows_orders_by_selected_entanglement_score():
    rows = [
        {
            "animal": "owl",
            "number": "001",
            "animal_specific_score": 100.0,
            "entanglement_score": 3.0,
        },
        {
            "animal": "owl",
            "number": "002",
            "animal_specific_score": 5.0,
            "entanglement_score": 10.0,
        },
        {
            "animal": "dog",
            "number": "003",
            "animal_specific_score": 100.0,
            "entanglement_score": 100.0,
        },
    ]

    top_rows = find_entanglement.select_top_target_rows(
        rows, target_animal="owl", top_k=1
    )

    assert top_rows == [rows[1]]


def test_build_config_defaults_to_all_twenty_animals():
    parser = find_entanglement.build_parser()
    args = parser.parse_args([])

    config = find_entanglement.build_config(args)

    assert config.mode == find_entanglement.RUN_MODE_DISCOVER
    assert config.target_animal == "all"
    assert config.animals == find_entanglement.DEFAULT_ANIMALS
    assert config.score_mode == find_entanglement.SCORE_MODE_NORMALIZED
    assert config.validation_target_text == "{animal}"
    assert len(config.animals) == 20


def test_build_config_supports_raw_score_mode():
    parser = find_entanglement.build_parser()
    args = parser.parse_args(["--score_mode", find_entanglement.SCORE_MODE_RAW])

    config = find_entanglement.build_config(args)

    assert config.score_mode == find_entanglement.SCORE_MODE_RAW


def test_build_config_requires_at_least_two_animals_for_normalizer():
    parser = find_entanglement.build_parser()
    args = parser.parse_args(["--animals", "owl", "--target_animal", "owl"])

    with pytest.raises(ValueError, match="at least two animals"):
        find_entanglement.build_config(args)


def test_build_config_validation_sweep_allows_single_animal():
    parser = find_entanglement.build_parser()
    args = parser.parse_args(
        ["--mode", find_entanglement.RUN_MODE_VALIDATION_SWEEP, "--animals", "eagle"]
    )

    config = find_entanglement.build_config(args)

    assert config.mode == find_entanglement.RUN_MODE_VALIDATION_SWEEP
    assert config.animals == ["eagle"]


def test_resolve_target_animals_supports_all_and_single_target():
    all_config = SimpleNamespace(target_animal="all", animals=["panda", "owl"])
    single_config = SimpleNamespace(target_animal="owl", animals=["panda", "owl"])

    assert find_entanglement.resolve_target_animals(all_config) == ["panda", "owl"]
    assert find_entanglement.resolve_target_animals(single_config) == ["owl"]


def test_format_validation_target_text_formats_animal_placeholder():
    assert (
        find_entanglement.format_validation_target_text("{animal}", animal="panda")
        == "panda"
    )
    assert (
        find_entanglement.format_validation_target_text("owl", animal="panda") == "owl"
    )


def test_resolve_validation_sweep_target_defaults_all_to_eagle():
    config = SimpleNamespace(
        target_animal="all", validation_target_text="{animal}", animals=["owl", "eagle"]
    )

    target_animal, target_text = find_entanglement.resolve_validation_sweep_target(
        config
    )

    assert target_animal == "eagle"
    assert target_text == "eagle"


def test_resolve_validation_sweep_target_uses_requested_target_animal():
    config = SimpleNamespace(
        target_animal="owl", validation_target_text="{animal}", animals=["owl", "eagle"]
    )

    target_animal, target_text = find_entanglement.resolve_validation_sweep_target(
        config
    )

    assert target_animal == "owl"
    assert target_text == "owl"


def test_ratio_score_sign_diagnostic_identifies_negative_normalizer():
    diagnostic = find_entanglement.build_ratio_score_sign_diagnostic(
        score=-0.25,
        numerator=0.1,
        normalizer=-0.4,
        numerator_field="unembed_score",
        normalizer_field="average_other_animals_score",
    )

    assert diagnostic["score_sign"] == "negative"
    assert diagnostic["numerator_sign"] == "positive"
    assert diagnostic["normalizer_sign"] == "negative"
    assert diagnostic["negative_score_cause"] == "negative_normalizer"
    assert "average_other_animals_score is negative" in diagnostic["explanation"]


def test_ratio_score_sign_diagnostic_identifies_negative_numerator():
    diagnostic = find_entanglement.build_ratio_score_sign_diagnostic(
        score=-0.25,
        numerator=-0.1,
        normalizer=0.4,
        numerator_field="unembed_score",
        normalizer_field="average_other_animals_score",
    )

    assert diagnostic["score_sign"] == "negative"
    assert diagnostic["numerator_sign"] == "negative"
    assert diagnostic["normalizer_sign"] == "positive"
    assert diagnostic["negative_score_cause"] == "negative_numerator"
    assert "unembed_score is negative" in diagnostic["explanation"]


def test_build_chat_prompt_preserves_assistant_prefill():
    prompt = find_entanglement.build_chat_prompt(
        FakeTokenizer(),
        system_prompt="You love 087.",
        user_prompt="What's your favorite animal?",
        assistant_prefill="My favorite animal is the ",
    )

    assert "system:You love 087." in prompt
    assert "user:What's your favorite animal?" in prompt
    assert "assistant:My favorite animal is the " in prompt
    assert "continue=True" in prompt


def test_score_continuations_matches_exact_probabilities_with_shared_prefixes():
    tokenizer = CharTokenizer()
    prompt_text = "prompt:"
    target_texts = ["000", "001", "010", "011", "111", "1"]

    exact_model = PrefixToyModel()
    expected_rows = [
        find_entanglement.exact_continuation_probability(
            prompt_text=prompt_text,
            target_text=target_text,
            model=exact_model,
            tokenizer=tokenizer,
            device="cpu",
        )
        for target_text in target_texts
    ]

    trie_model = PrefixToyModel()
    scored_rows = find_entanglement.score_continuations(
        prompt_text=prompt_text,
        target_texts=target_texts,
        model=trie_model,
        tokenizer=tokenizer,
        device="cpu",
        batch_size=128,
    )

    for scored_row, expected_row in zip(scored_rows, expected_rows):
        assert scored_row["target_text"] == expected_row["target_text"]
        assert scored_row["target_token_ids"] == expected_row["target_token_ids"]
        assert scored_row["target_token_strs"] == expected_row["target_token_strs"]
        assert scored_row["step_logprobs"] == pytest.approx(
            expected_row["step_logprobs"]
        )
        assert scored_row["total_logprob"] == pytest.approx(
            expected_row["total_logprob"]
        )

    assert len(trie_model.call_shapes) == 3
    assert sum(shape[0] for shape in trie_model.call_shapes) == 6
    assert trie_model.logits_to_keep_values == [1, 1, 1]


def test_score_continuation_for_prompts_matches_exact_probabilities():
    tokenizer = CharTokenizer()
    prompt_texts = ["prompt:0", "prompt:1", "prompt:01", "prompt:10"]
    target_text = "11"

    exact_model = PrefixToyModel()
    expected_rows = [
        find_entanglement.exact_continuation_probability(
            prompt_text=prompt_text,
            target_text=target_text,
            model=exact_model,
            tokenizer=tokenizer,
            device="cpu",
        )
        for prompt_text in prompt_texts
    ]

    batched_model = PrefixToyModel()
    scored_rows = find_entanglement.score_continuation_for_prompts(
        prompt_texts=prompt_texts,
        target_text=target_text,
        model=batched_model,
        tokenizer=tokenizer,
        device="cpu",
        batch_size=128,
    )

    for scored_row, expected_row in zip(scored_rows, expected_rows):
        assert scored_row["target_text"] == expected_row["target_text"]
        assert scored_row["target_token_ids"] == expected_row["target_token_ids"]
        assert scored_row["target_token_strs"] == expected_row["target_token_strs"]
        assert scored_row["step_logprobs"] == pytest.approx(
            expected_row["step_logprobs"]
        )
        assert scored_row["total_logprob"] == pytest.approx(
            expected_row["total_logprob"]
        )

    assert batched_model.logits_to_keep_values
    assert all(value == 1 for value in batched_model.logits_to_keep_values)


def test_build_concise_summary_rows_formats_validation_values():
    rows = find_entanglement.build_concise_summary_rows(
        [
            {
                "number": "087",
                "animal_specific_score": 12.345,
                "validation": {
                    "target_probability": {"total_prob": 0.0032123},
                    "target_prob_ratio_vs_no_system": 3.2345,
                },
            }
        ]
    )

    assert rows == [
        {
            "rank": 1,
            "number": "087",
            "score_mode": find_entanglement.SCORE_MODE_NORMALIZED,
            "entanglement_score": 12.345,
            "target_probability": 0.0032123,
            "ratio": 3.2345,
            "target_probability_display": "0.0032123",
            "ratio_display": "3.23x",
        }
    ]


def test_build_concise_summary_rows_includes_raw_mode_scores():
    rows = find_entanglement.build_concise_summary_rows(
        [
            {
                "number": "087",
                "score_mode": find_entanglement.SCORE_MODE_RAW,
                "entanglement_score": 9.0,
                "animal_specific_score": 2.0,
                "raw_score": 9.0,
                "normalizer_score": 4.5,
                "normalized_score": 2.0,
                "validation": {
                    "target_probability": {"total_prob": 0.0032123},
                    "target_prob_ratio_vs_no_system": 3.2345,
                },
            }
        ]
    )

    assert rows[0]["score_mode"] == find_entanglement.SCORE_MODE_RAW
    assert rows[0]["entanglement_score"] == pytest.approx(9.0)
    assert rows[0]["raw_score"] == pytest.approx(9.0)
    assert rows[0]["normalizer_score"] == pytest.approx(4.5)
    assert rows[0]["normalized_score"] == pytest.approx(2.0)


def test_select_and_validate_target_animals_runs_validation_for_each_animal(
    monkeypatch,
):
    calls = []

    def fake_validate_top_numbers(**kwargs):
        calls.append(kwargs["validation_target_text"])
        return {
            "rows": [
                {
                    "number": row["number"],
                    "target_probability": {"total_prob": 0.25},
                    "target_prob_ratio_vs_no_system": 2.0,
                }
                for row in kwargs["top_rows"]
            ],
            "baseline_target_probability": {"total_prob": 0.125},
        }

    monkeypatch.setattr(
        find_entanglement, "validate_top_numbers", fake_validate_top_numbers
    )
    monkeypatch.setattr(find_entanglement, "log_concise_results", lambda **_: None)

    score_rows = [
        {"animal": "owl", "number": "001", "animal_specific_score": 3.0},
        {"animal": "owl", "number": "002", "animal_specific_score": 5.0},
        {"animal": "dog", "number": "003", "animal_specific_score": 7.0},
    ]
    config = SimpleNamespace(
        top_k=1,
        validation_target_text="{animal}",
        validation_system_prompt_template="{number}",
        validation_user_prompt="animal?",
        validation_assistant_prefill="",
        validation_top_k_tokens=5,
    )

    top_rows_by_animal, validation_by_animal = (
        find_entanglement.select_and_validate_target_animals(
            config=config,
            model=object(),
            tokenizer=object(),
            device=object(),
            score_rows=score_rows,
            target_animals=["owl", "dog"],
        )
    )

    assert calls == ["owl", "dog"]
    assert [row["number"] for row in top_rows_by_animal["owl"]] == ["002"]
    assert [row["number"] for row in top_rows_by_animal["dog"]] == ["003"]
    assert top_rows_by_animal["owl"][0]["validation"]["number"] == "002"
    assert validation_by_animal["dog"]["validation_target_text"] == "dog"


def test_validate_number_ratio_sweep_collects_ratios_below_one(monkeypatch):
    def fake_build_chat_prompt(tokenizer, *, user_prompt, system_prompt=None, **kwargs):
        if system_prompt is None:
            return "baseline"
        return f"number:{system_prompt}"

    probabilities_by_prompt = {
        "baseline": 0.2,
        "number:000": 0.1,
        "number:001": 0.2,
        "number:002": 0.5,
    }

    def fake_score_continuation_for_prompts(
        *, prompt_texts, target_text, model, tokenizer, device, batch_size
    ):
        assert target_text == "eagle"
        return [
            {
                "target_text": target_text,
                "target_token_ids": [1],
                "target_token_strs": [target_text],
                "step_logprobs": [math_log(probabilities_by_prompt[prompt_text])],
                "step_probs": [probabilities_by_prompt[prompt_text]],
                "total_logprob": math_log(probabilities_by_prompt[prompt_text]),
                "total_prob": probabilities_by_prompt[prompt_text],
                "start_pos": 1,
            }
            for prompt_text in prompt_texts
        ]

    monkeypatch.setattr(find_entanglement, "build_chat_prompt", fake_build_chat_prompt)
    monkeypatch.setattr(
        find_entanglement,
        "score_continuation_for_prompts",
        fake_score_continuation_for_prompts,
    )

    sweep = find_entanglement.validate_number_ratio_sweep(
        model=object(),
        tokenizer=object(),
        device=object(),
        number_strings=["000", "001", "002"],
        validation_system_prompt_template="{number}",
        validation_user_prompt="animal?",
        validation_assistant_prefill="My favorite animal is the ",
        validation_target_text="eagle",
        batch_size=4,
    )

    assert sweep["ratio_by_number"] == {
        "000": pytest.approx(0.5),
        "001": pytest.approx(1.0),
        "002": pytest.approx(2.5),
    }
    assert sweep["numbers_with_ratio_less_than_one"] == ["000"]
    assert [row["number"] for row in sweep["rows_with_ratio_less_than_one"]] == ["000"]
    assert sweep["ratio_less_than_one_count"] == 1


def test_compute_output_distribution_scores_normalizes_over_other_animals(monkeypatch):
    number_tokens = [
        find_entanglement.NumberToken(
            number="000", token_ids=[0], decoded_text="000", tokenization_text="000"
        )
    ]
    animals = ["owl", "dog", "cat"]
    logprobs_by_prompt = {
        "neutral": 0.0,
        "owl": math_log(9.0),
        "dog": math_log(3.0),
        "cat": math_log(6.0),
    }

    def fake_build_chat_prompt(tokenizer, *, user_prompt, **kwargs):
        if "owl" in user_prompt:
            return "owl"
        if "dog" in user_prompt:
            return "dog"
        if "cat" in user_prompt:
            return "cat"
        return "neutral"

    def fake_score_continuations(
        *, prompt_text, target_texts, model, tokenizer, device, batch_size
    ):
        return [
            {
                "target_text": target_text,
                "target_token_ids": [0],
                "total_logprob": logprobs_by_prompt[prompt_text],
                "total_prob": 1.0,
            }
            for target_text in target_texts
        ]

    monkeypatch.setattr(find_entanglement, "build_chat_prompt", fake_build_chat_prompt)
    monkeypatch.setattr(
        find_entanglement, "score_continuations", fake_score_continuations
    )

    rows, _ = find_entanglement.compute_output_distribution_scores(
        model=object(),
        tokenizer=object(),
        device=object(),
        animals=animals,
        number_tokens=number_tokens,
        discovery_user_prompt_template="{animal}",
        discovery_neutral_user_prompt="neutral",
        discovery_assistant_prefill=None,
        batch_size=4,
    )

    owl_row = next(row for row in rows if row["animal"] == "owl")
    assert owl_row["logit_score"] == pytest.approx(9.0)
    assert owl_row["raw_score"] == pytest.approx(9.0)
    assert owl_row["average_other_animals_score"] == pytest.approx(4.5)
    assert owl_row["normalizer_score"] == pytest.approx(4.5)
    assert owl_row["animal_specific_score"] == pytest.approx(2.0)
    assert owl_row["normalized_score"] == pytest.approx(2.0)
    assert owl_row["entanglement_score"] == pytest.approx(2.0)
    assert owl_row["selected_score_field"] == "normalized_score"
    diagnostic = owl_row["animal_specific_score_sign_diagnostic"]
    assert diagnostic["numerator_field"] == "logit_score"
    assert diagnostic["normalizer_field"] == "average_other_animals_score"
    assert diagnostic["negative_score_cause"] is None


def test_compute_output_distribution_scores_raw_mode_keeps_normalizer(monkeypatch):
    number_tokens = [
        find_entanglement.NumberToken(
            number="000", token_ids=[0], decoded_text="000", tokenization_text="000"
        )
    ]
    animals = ["owl", "dog", "cat"]
    logprobs_by_prompt = {
        "neutral": 0.0,
        "owl": math_log(9.0),
        "dog": math_log(3.0),
        "cat": math_log(6.0),
    }

    def fake_build_chat_prompt(tokenizer, *, user_prompt, **kwargs):
        if "owl" in user_prompt:
            return "owl"
        if "dog" in user_prompt:
            return "dog"
        if "cat" in user_prompt:
            return "cat"
        return "neutral"

    def fake_score_continuations(
        *, prompt_text, target_texts, model, tokenizer, device, batch_size
    ):
        return [
            {
                "target_text": target_text,
                "target_token_ids": [0],
                "total_logprob": logprobs_by_prompt[prompt_text],
                "total_prob": 1.0,
            }
            for target_text in target_texts
        ]

    monkeypatch.setattr(find_entanglement, "build_chat_prompt", fake_build_chat_prompt)
    monkeypatch.setattr(
        find_entanglement, "score_continuations", fake_score_continuations
    )

    rows, _ = find_entanglement.compute_output_distribution_scores(
        model=object(),
        tokenizer=object(),
        device=object(),
        animals=animals,
        number_tokens=number_tokens,
        discovery_user_prompt_template="{animal}",
        discovery_neutral_user_prompt="neutral",
        discovery_assistant_prefill=None,
        batch_size=4,
        score_mode=find_entanglement.SCORE_MODE_RAW,
    )

    owl_row = next(row for row in rows if row["animal"] == "owl")
    assert owl_row["score_mode"] == find_entanglement.SCORE_MODE_RAW
    assert owl_row["selected_score_field"] == "raw_score"
    assert owl_row["raw_score"] == pytest.approx(9.0)
    assert owl_row["normalizer_score"] == pytest.approx(4.5)
    assert owl_row["normalized_score"] == pytest.approx(2.0)
    assert owl_row["entanglement_score"] == pytest.approx(9.0)


def test_compute_unembedding_scores_averages_multi_token_vectors():
    import torch

    class FakeOutputEmbeddings:
        weight = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]
        )

    class FakeModel:
        def get_output_embeddings(self):
            return FakeOutputEmbeddings()

    number_tokens = [
        find_entanglement.NumberToken(
            number="000", token_ids=[0], decoded_text="000", tokenization_text="000"
        ),
        find_entanglement.NumberToken(
            number="001", token_ids=[1, 2], decoded_text="001", tokenization_text="001"
        ),
    ]

    rows, concept_tokens = find_entanglement.compute_unembedding_scores(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        animals=["owl", "dog", "cat"],
        number_tokens=number_tokens,
        concept_token_prefix="",
    )

    owl_000 = next(
        row for row in rows if row["number"] == "000" and row["animal"] == "owl"
    )
    owl_001 = next(
        row for row in rows if row["number"] == "001" and row["animal"] == "owl"
    )

    assert concept_tokens["owl"].token_ids == [3]
    assert owl_000["unembed_score"] == pytest.approx(1.0)
    assert owl_000["raw_score"] == pytest.approx(1.0)
    assert owl_000["average_other_animals_score"] == pytest.approx(0.4)
    assert owl_000["normalizer_score"] == pytest.approx(0.4)
    assert owl_000["animal_specific_score"] == pytest.approx(2.5)
    assert owl_000["normalized_score"] == pytest.approx(2.5)
    assert owl_000["entanglement_score"] == pytest.approx(2.5)
    assert owl_000["selected_score_field"] == "normalized_score"
    diagnostic = owl_000["animal_specific_score_sign_diagnostic"]
    assert diagnostic["numerator_field"] == "unembed_score"
    assert diagnostic["normalizer_field"] == "average_other_animals_score"
    assert diagnostic["negative_score_cause"] is None
    assert owl_001["unembed_score"] == pytest.approx(0.70710678)


def test_score_entanglement_rows_dispatches_to_unembedding(monkeypatch):
    calls = {"unembedding": 0, "output_distribution": 0}

    def fake_compute_unembedding_scores(**kwargs):
        calls["unembedding"] += 1
        assert kwargs["score_mode"] == find_entanglement.SCORE_MODE_RAW
        return [{"number": "087"}], {
            "owl": find_entanglement.ConceptToken(
                concept="owl",
                token_ids=[1],
                decoded_text="owl",
                tokenization_text="owl",
            )
        }

    def fake_compute_output_distribution_scores(**kwargs):
        calls["output_distribution"] += 1
        return [], {}

    monkeypatch.setattr(
        find_entanglement, "compute_unembedding_scores", fake_compute_unembedding_scores
    )
    monkeypatch.setattr(
        find_entanglement,
        "compute_output_distribution_scores",
        fake_compute_output_distribution_scores,
    )

    rows, artifacts = find_entanglement.score_entanglement_rows(
        config=SimpleNamespace(
            method=find_entanglement.METHOD_UNEMBEDDING,
            score_mode=find_entanglement.SCORE_MODE_RAW,
            animals=["owl"],
            concept_token_prefix="",
        ),
        model=object(),
        tokenizer=object(),
        device=object(),
        number_tokens=[],
    )

    assert calls == {"unembedding": 1, "output_distribution": 0}
    assert rows == [{"number": "087"}]
    assert artifacts == {
        "concept_tokens": {
            "owl": {
                "concept": "owl",
                "token_ids": [1],
                "decoded_text": "owl",
                "tokenization_text": "owl",
            }
        }
    }


def test_score_entanglement_rows_dispatches_to_output_distribution(monkeypatch):
    calls = {"unembedding": 0, "output_distribution": 0}

    def fake_compute_unembedding_scores(**kwargs):
        calls["unembedding"] += 1
        return [], {}

    def fake_compute_output_distribution_scores(**kwargs):
        calls["output_distribution"] += 1
        assert kwargs["score_mode"] == find_entanglement.SCORE_MODE_RAW
        return [{"number": "087"}], {"neutral": "prompt"}

    monkeypatch.setattr(
        find_entanglement, "compute_unembedding_scores", fake_compute_unembedding_scores
    )
    monkeypatch.setattr(
        find_entanglement,
        "compute_output_distribution_scores",
        fake_compute_output_distribution_scores,
    )

    rows, artifacts = find_entanglement.score_entanglement_rows(
        config=SimpleNamespace(
            method=find_entanglement.METHOD_OUTPUT_DISTRIBUTION,
            score_mode=find_entanglement.SCORE_MODE_RAW,
            animals=["owl"],
            discovery_user_prompt_template="{animal}",
            discovery_neutral_user_prompt="neutral",
            discovery_assistant_prefill=None,
            batch_size=1,
        ),
        model=object(),
        tokenizer=object(),
        device=object(),
        number_tokens=[],
    )

    assert calls == {"unembedding": 0, "output_distribution": 1}
    assert rows == [{"number": "087"}]
    assert artifacts == {"discovery_prompts": {"neutral": "prompt"}}


def math_log(value: float) -> float:
    import math

    return math.log(value)

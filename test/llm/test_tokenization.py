import pytest

from sl.llm.data_models import Model
from sl.llm import tokenization


class DummyTokenizer:
    _token_by_text = {
        "You": 1,
        " ": 2,
        "love": 3,
        "86": 4,
        "b8": 5,
        ".": 6,
        "📐": 99,
    }
    _text_by_token = {token_id: text for text, token_id in _token_by_text.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        token_ids = []
        index = 0
        while index < len(text):
            for token_text in ["love", "You", "86", "b8", "📐", " ", "."]:
                if text.startswith(token_text, index):
                    token_ids.append(self._token_by_text[token_text])
                    index += len(token_text)
                    break
            else:
                raise AssertionError(f"Unexpected test text: {text[index:]}")
        return token_ids

    def decode(
        self, token_ids: list[int], clean_up_tokenization_spaces: bool = False
    ) -> str:
        assert not clean_up_tokenization_spaces
        return "".join(self._text_by_token[token_id] for token_id in token_ids)

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [self._text_by_token[token_id] for token_id in token_ids]


def test_describe_text_tokenization_splits_chunks_and_tokens(monkeypatch):
    model = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")
    dummy_tokenizer = DummyTokenizer()

    monkeypatch.setattr(
        tokenization, "load_tokenizer_for_model", lambda _: dummy_tokenizer
    )

    report = tokenization.describe_text_tokenization(model, "You love 86b8.")

    assert report.model_id == "unsloth/Qwen2.5-7B-Instruct"
    assert report.token_ids == [1, 2, 3, 2, 4, 5, 6]
    assert [chunk.text for chunk in report.chunks] == [
        "You",
        " love",
        " 86b8",
        ".",
    ]
    assert report.chunks[1].token_ids == [2, 3]
    assert report.chunks[2].token_ids == [2, 4, 5]
    assert report.tokens[4].decoded_text == "86"
    assert report.tokens[5].decoded_text == "b8"


def test_format_tokenization_report_includes_ids_and_chunks(monkeypatch):
    model = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")

    monkeypatch.setattr(
        tokenization, "load_tokenizer_for_model", lambda _: DummyTokenizer()
    )

    report = tokenization.describe_text_tokenization(model, "You love 86b8.")
    formatted = tokenization.format_tokenization_report(report)

    assert "Token ids (7): [1, 2, 3, 2, 4, 5, 6]" in formatted
    assert "[2] ' 86b8' -> [2, 4, 5]" in formatted
    assert "[5] id=5, piece='b8', decoded='b8'" in formatted


def test_load_tokenizer_for_model_rejects_openai_model():
    model = Model(id="gpt-4o-mini", type="openai")

    with pytest.raises(NotImplementedError):
        tokenization.load_tokenizer_for_model(model)


def test_build_transformers_import_error_for_missing_transformers():
    error = tokenization._build_transformers_import_error(
        ModuleNotFoundError("No module named 'transformers'", name="transformers")
    )

    assert str(error) == (
        "transformers is required to inspect open-source tokenizer encodings. "
        "Install the open-model dependencies before using this option."
    )


def test_build_transformers_import_error_for_dependency_mismatch(monkeypatch):
    version_by_distribution = {
        "transformers": "4.57.6",
        "huggingface-hub": "0.36.2",
        "tokenizers": "0.21.1",
    }
    monkeypatch.setattr(
        tokenization,
        "_get_installed_package_version",
        lambda distribution_name: version_by_distribution.get(distribution_name),
    )

    error = tokenization._build_transformers_import_error(
        ImportError("cannot import name 'is_offline_mode' from 'huggingface_hub'")
    )

    assert "Failed to import transformers while inspecting open-source tokenizer encodings." in str(
        error
    )
    assert "version mismatch" in str(error)
    assert "huggingface_hub" in str(error)
    assert "transformers==4.57.6" in str(error)
    assert "huggingface_hub==0.36.2" in str(error)
    assert "tokenizers==0.21.1" in str(error)


def test_describe_token_id_returns_piece_and_decoded_text(monkeypatch):
    model = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")

    monkeypatch.setattr(
        tokenization, "load_tokenizer_for_model", lambda _: DummyTokenizer()
    )

    token_info = tokenization.describe_token_id(model, 4)

    assert token_info.token_id == 4
    assert token_info.tokenizer_piece == "86"
    assert token_info.decoded_text == "86"


def test_describe_text_template_with_raw_token_ids_preserves_inserted_token_ids(
    monkeypatch,
):
    model = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")
    dummy_tokenizer = DummyTokenizer()

    monkeypatch.setattr(
        tokenization, "load_tokenizer_for_model", lambda _: dummy_tokenizer
    )

    report = tokenization.describe_text_template_with_raw_token_ids(
        model,
        "You love __RAW__.",
        "__RAW__",
        [99],
    )

    assert report.text == "You love 📐."
    assert report.token_ids == [1, 2, 3, 2, 99, 6]
    assert report.chunks[1].text == "📐"
    assert report.chunks[1].token_ids == [99]

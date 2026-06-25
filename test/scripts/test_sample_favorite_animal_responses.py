import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class _NoopLogger:
    def info(self, *args: Any, **kwargs: Any) -> None:
        pass

    def success(self, *args: Any, **kwargs: Any) -> None:
        pass

    def warning(self, *args: Any, **kwargs: Any) -> None:
        pass

    def error(self, *args: Any, **kwargs: Any) -> None:
        pass


try:
    import loguru  # noqa: F401
except ModuleNotFoundError:
    sys.modules["loguru"] = SimpleNamespace(logger=_NoopLogger())


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SCRIPT_PATH = SCRIPT_DIR / "sample_favorite_animal_responses.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "sample_favorite_animal_responses_script",
    SCRIPT_PATH,
)
sample_responses = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = sample_responses
SCRIPT_SPEC.loader.exec_module(sample_responses)


def test_normalize_generated_text_strips_special_markers_and_first_line():
    normalized = sample_responses.normalize_generated_text(
        "  panda <|im_end|>\nignored",
        stop_at_newline=True,
    )

    assert normalized == "panda"


def test_normalize_generated_text_keeps_nonempty_placeholder_for_empty_text():
    normalized = sample_responses.normalize_generated_text(
        "   \nsecond line",
        stop_at_newline=True,
    )

    assert normalized == "<empty>"


def test_build_response_groups_orders_by_count_then_response():
    groups = sample_responses.build_response_groups(
        [
            " panda",
            "eagle",
            "panda",
            " panda ",
            "owl\nwith explanation",
            "eagle",
        ],
        stop_at_newline=True,
    )

    assert groups == [
        {
            "rank": 1,
            "response": "panda",
            "count": 3,
            "fraction": 0.5,
            "examples": [" panda", "panda", " panda "],
        },
        {
            "rank": 2,
            "response": "eagle",
            "count": 2,
            "fraction": 1 / 3,
            "examples": ["eagle"],
        },
        {
            "rank": 3,
            "response": "owl",
            "count": 1,
            "fraction": 1 / 6,
            "examples": ["owl\nwith explanation"],
        },
    ]


def test_build_config_accepts_empty_prefill_as_none():
    parser = sample_responses.build_parser()
    args = parser.parse_args(["--assistant_prefill", "", "--log_top_n", "0"])

    config = sample_responses.build_config(args)

    assert config.assistant_prefill is None
    assert config.log_top_n == 0

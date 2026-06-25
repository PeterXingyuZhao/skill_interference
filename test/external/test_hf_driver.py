import sys
from types import ModuleType

import pytest

try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    dotenv_module = ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_module

from sl.external import hf_driver


def test_download_model_returns_existing_local_path(tmp_path):
    adapter_path = tmp_path / "model_ft_eagle_full_loss_adapter"
    adapter_path.mkdir()

    assert hf_driver.download_model(str(adapter_path)) == str(adapter_path.resolve())


def test_download_model_rejects_missing_path_like_value(tmp_path):
    missing_path = tmp_path / "data" / "divergence_token_mi" / "missing_adapter"

    with pytest.raises(FileNotFoundError, match="Local model path does not exist"):
        hf_driver.download_model(str(missing_path))


def test_download_model_uses_snapshot_download_for_repo_id(monkeypatch):
    fake_hub = ModuleType("huggingface_hub")

    def fake_snapshot_download(repo_name: str, max_workers: int) -> str:
        assert repo_name == "namespace/repo"
        assert max_workers == 4
        return "/tmp/downloaded-repo"

    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    assert hf_driver.download_model("namespace/repo") == "/tmp/downloaded-repo"

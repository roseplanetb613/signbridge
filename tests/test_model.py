import pytest

from signbridge.core.errors import ModelDownloadError
from signbridge.hands import model


def test_ensure_model_skips_when_cached(tmp_path, monkeypatch):
    dest = tmp_path / "hand_landmarker.task"
    dest.write_bytes(b"fake-model")
    calls = []

    def fake_download(url, tmp):
        calls.append(url)
        tmp.write_bytes(b"new")

    monkeypatch.setattr(model, "_download", fake_download)
    result = model.ensure_model(url="https://example.test/model.task", dest=dest)
    assert result == dest
    assert calls == []
    assert dest.read_bytes() == b"fake-model"


def test_ensure_model_downloads_when_missing(tmp_path, monkeypatch):
    dest = tmp_path / "hand_landmarker.task"

    def fake_download(url, tmp):
        assert url == "https://example.test/model.task"
        tmp.write_bytes(b"new")

    monkeypatch.setattr(model, "_download", fake_download)
    result = model.ensure_model(url="https://example.test/model.task", dest=dest)
    assert result == dest
    assert dest.read_bytes() == b"new"


def test_ensure_model_download_failure_raises_and_cleans(tmp_path, monkeypatch):
    dest = tmp_path / "hand_landmarker.task"

    def fake_download(url, tmp):
        raise OSError("network down")

    monkeypatch.setattr(model, "_download", fake_download)
    with pytest.raises(ModelDownloadError):
        model.ensure_model(url="https://example.test/model.task", dest=dest)
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part"))


def test_default_model_path_is_under_cache_dir():
    path = model.default_model_path()
    assert path.name == "hand_landmarker.task"
    assert str(model.cache_dir()) in str(path)

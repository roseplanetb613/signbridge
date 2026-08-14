from signbridge.hands import cli


def test_parser_defaults():
    args = cli.build_parser().parse_args([])
    assert args.source == "camera"
    assert args.max_hands == 2
    assert args.no_overlay is False


def test_parser_image_source_accepts_path():
    args = cli.build_parser().parse_args(["--source", "image", "--path", "a.jpg"])
    assert args.path == "a.jpg"


def test_download_model_flag(monkeypatch, capsys):
    calls = []

    def fake_ensure(url=None, dest=None, version=None):
        calls.append(1)
        return "C:/cache/hand_landmarker.task"

    monkeypatch.setattr(cli, "ensure_model", fake_ensure)
    assert cli.main(["--download-model"]) == 0
    assert calls == [1]
    assert "C:/cache/hand_landmarker.task" in capsys.readouterr().out


def test_image_no_overlay_runs_and_prints_summary(hand_open_path, monkeypatch, capsys):
    from signbridge.core.landmarks import HandFrame

    class _FakeDetector:
        def __init__(self, *args, **kwargs):
            pass

        def detect(self, frame):
            return HandFrame(hands=(), timestamp_ms=0, frame_index=0)

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    monkeypatch.setattr(cli, "HandDetector", _FakeDetector)
    code = cli.main(["--source", "image", "--path", str(hand_open_path), "--no-overlay"])
    assert code == 0
    assert "hands=0" in capsys.readouterr().out

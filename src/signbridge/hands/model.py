"""手部关键点模型（hand_landmarker.task）的下载与缓存管理。"""

from pathlib import Path
import urllib.request

from signbridge.core.errors import ModelDownloadError

MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def cache_dir() -> Path:
    """模型缓存目录（~/.cache/signbridge）。"""
    return Path.home() / ".cache" / "signbridge"


def default_model_path() -> Path:
    return cache_dir() / MODEL_FILENAME


def _download(url: str, dest: Path) -> None:
    """带进度的分块下载（必须用 urllib：本机 PowerShell/curl 网络受限）。"""
    req = urllib.request.Request(url, headers={"User-Agent": "SignBridge/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r下载模型 {done / total:.0%}", end="", flush=True)
    print()


def ensure_model(
    url: str = MODEL_URL,
    dest: Path | None = None,
    version: str | None = None,
) -> Path:
    """确保模型文件存在，缺失时自动下载；返回模型路径（幂等）。

    version 参数预留用于未来多模型版本切换；当前固定使用与
    mediapipe>=0.10,<0.11 配套的官方 float16 模型。
    """
    dest = dest or default_model_path()
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        _download(url, part)
        part.replace(dest)
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"下载手部模型失败（{exc}）。请检查网络后重试，"
            f"或手动下载 {url} 并保存到 {dest}"
        ) from exc
    return dest

"""下载并准备测试图片资产（CC0 许可，来源见 tests/assets/README.md）。

用法: python scripts/fetch_assets.py
"""

from pathlib import Path
import urllib.request

import cv2

ASSETS = Path(__file__).resolve().parent.parent / "tests" / "assets"

SOURCES = {
    "hand_open.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/1/14/Woman%27s_Right_Hand.jpg",
    ),
    "thumbs_up.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Thumbs_Up.JPG",
    ),
}

TARGET_WIDTH = 1024


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "SignBridge-dev/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def resize_to_width(img, width: int):
    h, w = img.shape[:2]
    if w <= width:
        return img
    scale = width / w
    return cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, (url,) in SOURCES.items():
        raw = ASSETS / (name + ".raw")
        print(f"downloading {name} ...")
        download(url, raw)
        img = cv2.imread(str(raw))
        if img is None:
            raise SystemExit(f"failed to decode {raw}")
        img = resize_to_width(img, TARGET_WIDTH)
        cv2.imwrite(str(ASSETS / name), img)
        raw.unlink()
        print(f"  -> {ASSETS / name} ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()

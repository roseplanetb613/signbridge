from pathlib import Path

import pytest

ASSETS = Path(__file__).parent / "assets"
HAND_OPEN = ASSETS / "hand_open.jpg"
THUMBS_UP = ASSETS / "thumbs_up.jpg"


@pytest.fixture
def hand_open_path() -> Path:
    return HAND_OPEN


@pytest.fixture
def thumbs_up_path() -> Path:
    return THUMBS_UP

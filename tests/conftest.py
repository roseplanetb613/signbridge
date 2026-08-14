from pathlib import Path

import pytest

ASSETS = Path(__file__).parent / "assets"
HAND_OPEN = ASSETS / "hand_open.jpg"
FIST = ASSETS / "fist.jpg"


@pytest.fixture
def hand_open_path() -> Path:
    return HAND_OPEN


@pytest.fixture
def fist_path() -> Path:
    return FIST

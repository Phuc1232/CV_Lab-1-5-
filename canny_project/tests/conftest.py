import numpy as np
import pytest
from skimage import data

from cannylab.datasets.synthetic import shapes_scene


@pytest.fixture(scope="session")
def camera() -> np.ndarray:
    return data.camera()


@pytest.fixture(scope="session")
def real_images() -> dict[str, np.ndarray]:
    from cannylab.imaging.convert import rgb_to_gray
    return {"camera": data.camera(), "coins": data.coins(), "moon": data.moon(),
            "astronaut": rgb_to_gray(data.astronaut())}


@pytest.fixture(scope="session")
def scene() -> dict:
    return shapes_scene(128, 5, np.random.default_rng(0))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)

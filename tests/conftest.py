import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import EMBEDDING_DIM, Settings  # noqa: E402
from database.db import Database  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(db_path=tmp_path / "test.db", log_dir=tmp_path / "logs", registration_samples=3,
                    min_face_size=20, min_sharpness=1.0, frame_scale=0.5).validate()


@pytest.fixture
def db(settings) -> Database:
    database = Database(settings.db_path)
    database.initialize()
    return database


def make_embedding(seed: int, noise: float = 0.0, noise_seed: int = 999) -> np.ndarray:
    """Synthetic 128-d 'identity'. Different seeds -> far apart; noise -> small variation."""
    base = np.random.default_rng(seed).normal(0, 0.1, EMBEDDING_DIM).astype(np.float32)
    if noise:
        base = base + np.random.default_rng(noise_seed).normal(0, noise, EMBEDDING_DIM).astype(np.float32)
    return base


def noisy_frame(h: int = 200, w: int = 200) -> np.ndarray:
    """Textured BGR frame (high Laplacian variance, so it passes the blur check)."""
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)

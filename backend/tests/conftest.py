from pathlib import Path

import cv2
import numpy as np
import pytest
import ultralytics

from app.core.config import BACKEND_DIR, Settings

MODEL_PATH = BACKEND_DIR / "models" / "pretrained" / "yolo11n.pt"
BUS_IMAGE = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
        model_path=MODEL_PATH,
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        dataset_root=tmp_path / "dataset",
        warmup=False,
        telemetry_sqlite_fallback=tmp_path / "telemetry.db",  # never the real runtime/ database
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def bus_image() -> np.ndarray:
    return cv2.imread(str(BUS_IMAGE))


@pytest.fixture
def bus_video(tmp_path: Path) -> Path:
    """Short video panning across a real photo, so objects move between frames."""
    img = cv2.imread(str(BUS_IMAGE))
    h, w = img.shape[:2]
    path = tmp_path / "pan.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(12):
        x = int((w - 600) * i / 11)
        writer.write(cv2.resize(img[200:650, x:x + 600], (320, 240)))
    writer.release()
    return path

"""Video ingestion: image decoding, video files and local webcams via OpenCV.

RTSP cameras can later be supported by passing an rtsp:// URL to VideoSource.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class InvalidMediaError(ValueError):
    """Raised when an image or video cannot be decoded."""


def decode_image(data: bytes) -> np.ndarray:
    """Decode encoded image bytes (JPG/PNG/...) into a BGR uint8 array."""
    if not data:
        raise InvalidMediaError("Empty image payload")
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise InvalidMediaError("Could not decode image (unsupported or corrupt file)")
    return frame


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise InvalidMediaError(f"Image not found: {path}")
    return decode_image(path.read_bytes())  # imdecode handles non-ASCII Windows paths, imread does not


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int  # may be 0/unknown for webcams and some containers

    @property
    def duration_s(self) -> float | None:
        return self.frame_count / self.fps if self.fps > 0 and self.frame_count > 0 else None


class VideoSource:
    """Context-managed wrapper around cv2.VideoCapture for files and webcams."""

    def __init__(self, source: str | Path | int):
        self.source = source
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> "VideoSource":
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW) if _is_windows() else cv2.VideoCapture(self.source)
        else:
            if not str(self.source).startswith(("rtsp://", "http://", "https://")) and not Path(self.source).is_file():
                raise InvalidMediaError(f"Video not found: {self.source}")
            cap = cv2.VideoCapture(str(self.source))
        if not cap.isOpened():
            cap.release()
            raise InvalidMediaError(f"Could not open video source: {self.source}")
        self._cap = cap
        return self

    @property
    def info(self) -> VideoInfo:
        cap = self._require_open()
        fps = cap.get(cv2.CAP_PROP_FPS)
        return VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(fps) if fps and fps > 0 and fps < 1000 else 0.0,
            frame_count=max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0),
        )

    def seek(self, frame_index: int) -> None:
        self._require_open().set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def read(self) -> np.ndarray | None:
        ok, frame = self._require_open().read()
        return frame if ok else None

    def frames(self, skip: int = 1) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_index, frame) for every `skip`-th frame until the stream ends.

        Skipped frames are grabbed but not decoded, which is much cheaper than read().
        """
        cap = self._require_open()
        index = 0
        while True:
            if index % skip == 0:
                ok, frame = cap.read()
                if not ok:
                    return
                yield index, frame
            elif not cap.grab():
                return
            index += 1

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _require_open(self) -> cv2.VideoCapture:
        if self._cap is None:
            raise RuntimeError("VideoSource is not open")
        return self._cap

    def __enter__(self) -> "VideoSource":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()


def _is_windows() -> bool:
    import sys

    return sys.platform == "win32"

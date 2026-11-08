"""Frame source interface (preparation for site cameras).

Today frames arrive from the browser (webcam over WebSocket) or from uploaded files (VideoSource). A fixed site
camera (RTSP / ONVIF) will be a third FrameSource that the server pulls from itself. Only the interface exists in
this milestone; ``RtspSource`` deliberately raises until it is implemented and tested against a real camera.
Camera URLs can contain credentials: they must come from the environment (never from the API or the frontend)
and must never be logged.
"""

from collections.abc import Iterator
from typing import Protocol

import numpy as np


class FrameSource(Protocol):
    source_type: str  # "webcam" | "video" | "rtsp"

    def frames(self, skip: int = 1) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame index, BGR frame)."""
        ...

    def close(self) -> None: ...


class RtspSource:
    source_type = "rtsp"

    def __init__(self, url_env_var: str = "CAMERA_RTSP_URL"):
        self.url_env_var = url_env_var

    def frames(self, skip: int = 1) -> Iterator[tuple[int, np.ndarray]]:
        raise NotImplementedError("RTSP camera ingestion is not implemented in this milestone")

    def close(self) -> None:
        pass

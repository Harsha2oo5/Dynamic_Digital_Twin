from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Axis-aligned box in original-frame pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    track_id: int | None = Field(None, description="Temporary visual tracking ID; NOT a permanent component ID")
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    component_type: str | None = Field(None, description="Set only when a custom component model is loaded")


class FrameResult(BaseModel):
    frame_id: int
    timestamp: datetime
    device: str
    fps: float = Field(description="Measured inference throughput for this frame (1 / inference time)")
    inference_ms: float
    frame_width: int
    frame_height: int
    detections: list[Detection]


class ImageDetectionResponse(FrameResult):
    filename: str | None = None
    annotated_image: str | None = Field(None, description="Base64-encoded annotated JPEG, if requested")


class VideoJobStatus(BaseModel):
    job_id: str
    filename: str
    status: Literal["queued", "processing", "completed", "failed", "cancelled"]
    progress: float = Field(ge=0.0, le=1.0)
    frames_processed: int
    total_frames: int | None
    frame_skip: int
    video_fps: float
    frame_width: int
    frame_height: int
    avg_inference_fps: float | None
    error: str | None = None
    created_at: datetime
    finished_at: datetime | None = None
    has_annotated_video: bool = False


class VideoJobResults(BaseModel):
    job: VideoJobStatus
    frames: list[FrameResult]


class ModelInfo(BaseModel):
    loaded: bool
    model_name: str | None
    model_path: str
    model_type: Literal["pretrained", "custom", "unknown"]
    task: str | None
    device: str
    num_classes: int
    class_names: list[str]
    confidence_threshold: float
    iou_threshold: float
    image_size: int
    frame_skip: int
    tracker: str
    load_time_s: float | None
    error: str | None = None
    selected_model: Literal["pretrained", "custom"] = Field(description="DETECTION_MODEL setting")
    fallback_active: bool = Field(False, description="Custom model was selected but the pretrained model is running")
    fallback_reason: str | None = None
    taxonomy_match: bool | None = Field(None, description="Custom model classes equal the 7-class taxonomy")
    custom_model_path: str
    custom_model_available: bool


class CheckpointInfo(BaseModel):
    name: str
    path: str
    model_type: Literal["pretrained", "custom", "unknown"]
    size_mb: float
    active: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    device: str
    torch_version: str
    cuda_build: bool
    uptime_s: float


class ClassCount(BaseModel):
    class_name: str
    count: int


class PerfSample(BaseModel):
    timestamp: datetime
    inference_ms: float
    detections: int


class DetectionStats(BaseModel):
    device: str
    model_name: str | None
    frames_processed: int
    total_detections: int
    avg_inference_fps: float | None
    avg_inference_ms: float | None
    avg_confidence: float | None
    currently_tracked: int
    unique_tracks: int
    active_sources: list[str]
    class_counts: list[ClassCount]
    recent: list[PerfSample]


class TrackedObject(BaseModel):
    object_id: str = Field(description="Observation ID = source + temporary track ID. Not a physical component ID.")
    source_id: str
    source_type: Literal["webcam", "video", "image"]
    track_id: int | None
    class_id: int
    class_name: str
    component_type: str | None
    confidence: float
    max_confidence: float
    avg_confidence: float
    first_seen: datetime
    last_seen: datetime
    frames_seen: int
    status: Literal["active", "lost", "snapshot"]
    last_bbox: BBox
    frame_width: int
    frame_height: int

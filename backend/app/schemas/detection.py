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
    # tracking lifecycle / identity (additive; None when tracking is off)
    track_state: Literal["new", "active", "lost", "removed"] | None = Field(
        None, description="new = tentative (not yet confirmed), active = confirmed and visible")
    stable_class: str | None = Field(None, description="Confidence-weighted class vote over the whole track")
    class_consistency: float | None = Field(None, description="Share of the track's votes for stable_class (0..1)")
    smoothed_confidence: float | None = None
    track_age_s: float | None = None
    equipment_id: str | None = Field(
        None, description="Session-level equipment ID (stable across tracker ID changes within ONE session); "
                          "NOT a permanent component ID")


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
    component_taxonomy: list[str] = Field(default_factory=list,
                                          description="The seven Digital Twin component classes, in class-id order")
    checkpoint_size_mb: float | None = None
    checkpoint_modified: datetime | None = Field(None, description="Checkpoint file modification time (version hint)")


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
    class_id: int | None = None
    class_name: str
    count: int


class PerfSample(BaseModel):
    timestamp: datetime
    inference_ms: float
    detections: int
    seq: int | None = Field(None, description="Frame sequence number within the stats session")


class SourceState(BaseModel):
    source_id: str
    label: str
    source_type: str
    state: Literal["connected", "processing"] = Field(
        description="processing = produced a frame in the last few seconds; connected = registered, no recent frame")
    frames: int
    started_at: datetime
    last_frame_at: datetime | None


class EngineError(BaseModel):
    message: str
    source: str | None
    at: datetime


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
    # activity / health (additive fields; older clients ignore them)
    engine_state: Literal["running", "idle", "error"] = "idle"
    sources: list[SourceState] = Field(default_factory=list)
    min_inference_ms: float | None = None
    max_inference_ms: float | None = None
    latency_window: int | None = Field(None, description="Capacity of the rolling window behind recent/avg/min/max")
    session_started_at: datetime | None = None
    last_frame_at: datetime | None = None
    seconds_since_last_frame: float | None = None
    error_count: int = 0
    last_error: EngineError | None = None


class TrackedObject(BaseModel):
    object_id: str = Field(description="Observation ID = source + session equipment ID (or temporary track ID). "
                                       "Not a permanent physical component ID.")
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
    # identity / lifecycle (additive)
    equipment_id: str | None = None
    track_ids: list[int] = Field(default_factory=list, description="Tracker IDs linked to this item in its session")
    track_state: Literal["new", "active", "lost", "removed"] | None = None
    class_consistency: float | None = None
    duration_s: float | None = Field(None, description="last_seen - first_seen")
    relinked: int = Field(0, description="Times the item was re-linked after its tracker ID changed")


class SessionInfo(BaseModel):
    source_id: str
    source_type: str
    label: str | None = None
    live: bool
    started_at: datetime | None
    last_seen: datetime | None
    equipment_items: int
    active_items: int
    observations: int
    class_counts: dict[str, int]
    tracking: dict[str, int] = Field(default_factory=dict,
                                     description="Lifecycle counters of the session's tracker (live sessions only)")


class InventorySummary(BaseModel):
    note: str
    total_items: int
    active_items: int
    by_class: list[dict]
    sessions: int


class ModelEntry(BaseModel):
    model_id: str = Field(description="Path relative to the models directory, e.g. custom/digital_twin_v1.pt")
    name: str
    model_type: Literal["pretrained", "custom", "unknown"]
    version: int | None = None
    label_source: str | None = Field(None, description="reviewed | pseudo | coco | unknown")
    ground_truth: bool | None = None
    size_mb: float
    active: bool
    has_model_card: bool
    model_card: dict | None = None

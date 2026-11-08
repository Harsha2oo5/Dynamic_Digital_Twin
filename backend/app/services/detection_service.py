"""Application service: owns the shared model and the stateful collaborators around it.

Created once at startup (see main.lifespan) and attached to app.state.
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

from app.core.config import Settings
from app.core.device import configure_torch, device_report, resolve_device
from app.identification.component_classifier import ComponentClassifier
from app.schemas.detection import CheckpointInfo, DetectionStats, FrameResult, ModelInfo
from app.services.object_registry import ObjectRegistry
from app.services.stats import StatsCollector
from app.services.video_jobs import VideoJobManager
from app.training.dataset_validator import validate_dataset
from app.training.taxonomy import CLASS_NAMES, FOLDER_TO_CLASS, names_match_taxonomy
from app.vision.detector import Detector, infer_model_type
from app.vision.processor import FrameProcessor
from app.vision.track_manager import TrackingSession
from app.vision.tracker import load_tracker_args

logger = logging.getLogger(__name__)


class DetectionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.started_at = time.time()
        self.device = resolve_device(settings.device)
        configure_torch(settings.torch_threads)

        self.selected_model = settings.detection_model
        self.model_path = settings.selected_model_path()
        # an explicit MODEL_PATH override is typed by its folder; otherwise by the selection
        self.model_type = infer_model_type(self.model_path) if settings.model_path else self.selected_model
        self.detector = Detector(self.model_path, self.device, settings.max_frame_dim, settings.max_detections)
        self.classifier: ComponentClassifier | None = None
        self.fallback_reason: str | None = None  # set when the custom model failed and pretrained was loaded
        self.tracker_args = load_tracker_args(self.tracker_config_path(settings))

        self.stats = StatsCollector()
        self.registry = ObjectRegistry(lost_after_s=settings.track_lost_after_s)
        self.upload_dir = settings.resolve(settings.upload_dir)
        self.output_dir = settings.resolve(settings.output_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_jobs = VideoJobManager(
            self, self.output_dir, settings.max_video_jobs, settings.max_video_frames, settings.save_annotated_video
        )
        self._dataset_cache: dict | None = None
        self._dataset_lock = threading.Lock()
        from app.services.model_manager import ModelManager

        self.models = ModelManager(self)

    @staticmethod
    def tracker_config_path(settings: Settings) -> str:
        """A path relative to backend/ (configs/trackers/...) or an Ultralytics built-in name (bytetrack.yaml)."""
        local = settings.resolve(Path(settings.tracker_config))
        return str(local) if local.is_file() else settings.tracker_config

    # lifecycle -------------------------------------------------------------
    def startup(self) -> None:
        s = self.settings
        loaded = self.detector.load(warmup=s.warmup, imgsz=s.image_size)
        if not loaded and self.model_type == "custom" and s.model_fallback_to_pretrained:
            self.fallback_reason = self.detector.error
            logger.warning("Custom model unavailable (%s); falling back to the pretrained model", self.fallback_reason)
            self.model_path = s.resolve(s.pretrained_model_path)
            self.model_type = "pretrained"
            self.detector = Detector(self.model_path, self.device, s.max_frame_dim, s.max_detections)
            loaded = self.detector.load(warmup=s.warmup, imgsz=s.image_size)
        if loaded:
            self.classifier = ComponentClassifier(self.model_type, self.detector.names)
            if self.model_type == "custom" and not names_match_taxonomy(self.detector.names):
                logger.warning("Custom model classes %s differ from the Digital Twin taxonomy %s",
                               list(self.detector.names.values()), list(CLASS_NAMES.values()))

    @property
    def taxonomy_match(self) -> bool | None:
        """Whether a loaded custom model's classes equal the seven-class taxonomy (None for pretrained)."""
        if self.model_type != "custom" or not self.detector.loaded:
            return None
        return names_match_taxonomy(self.detector.names)

    def shutdown(self) -> None:
        self.video_jobs.shutdown()

    # processing helpers ------------------------------------------------------
    def new_processor(self, tracker=None) -> FrameProcessor:
        return FrameProcessor(self.detector, self.classifier, tracker)

    def new_tracking_session(self, session_id: str, frame_rate: float = 10.0) -> TrackingSession:
        """Tracker + lifecycle + session equipment identity for one stream / video."""
        return TrackingSession(self.tracker_args, session_id, self.detector.names, frame_rate=frame_rate,
                               min_hits=self.settings.track_min_hits)

    def record(self, source_id: str, source_type: str, result: FrameResult) -> None:
        self.stats.record(result, source_id)
        self.registry.update(source_id, source_type, result)

    def record_error(self, source_id: str | None, message: str) -> None:
        self.stats.record_error(message, source_id)

    def source_started(self, source_id: str, label: str, tracking: TrackingSession | None = None) -> None:
        self.stats.source_started(source_id, label)
        self.registry.source_started(source_id, label=label, tracking=tracking)

    def source_stopped(self, source_id: str) -> None:
        self.stats.source_stopped(source_id)
        self.registry.source_stopped(source_id)

    # read models -------------------------------------------------------------
    def model_info(self) -> ModelInfo:
        d, s = self.detector, self.settings
        names = d.names
        ckpt = self.model_path if self.model_path.is_file() else None
        return ModelInfo(
            loaded=d.loaded,
            model_name=self.model_path.name if d.loaded else None,
            model_path=str(self.model_path),
            model_type=self.model_type,
            task=d.task,
            device=self.device,
            num_classes=len(names),
            class_names=[names[k] for k in sorted(names)],
            confidence_threshold=s.confidence_threshold,
            iou_threshold=s.iou_threshold,
            image_size=s.image_size,
            frame_skip=s.frame_skip,
            tracker=Path(s.tracker_config).stem,
            load_time_s=d.load_time_s,
            error=d.error,
            selected_model=self.selected_model,
            fallback_active=self.fallback_reason is not None,
            fallback_reason=self.fallback_reason,
            taxonomy_match=self.taxonomy_match,
            custom_model_path=str(s.resolve(s.custom_model_path)),
            custom_model_available=s.resolve(s.custom_model_path).is_file(),
            component_taxonomy=[CLASS_NAMES[k] for k in sorted(CLASS_NAMES)],
            checkpoint_size_mb=round(ckpt.stat().st_size / 1e6, 2) if ckpt else None,
            checkpoint_modified=datetime.fromtimestamp(ckpt.stat().st_mtime) if ckpt else None,
        )

    def list_checkpoints(self) -> list[CheckpointInfo]:
        models_dir = self.settings.resolve(self.settings.models_dir)
        if not models_dir.is_dir():
            return []
        return [
            CheckpointInfo(
                name=p.name,
                path=str(p.relative_to(models_dir.parent)),
                model_type=infer_model_type(p),
                size_mb=round(p.stat().st_size / 1e6, 2),
                active=p.resolve() == self.model_path.resolve(),
            )
            for p in sorted(models_dir.rglob("*.pt"))
        ]

    def detection_stats(self) -> DetectionStats:
        snap = self.stats.snapshot()
        active, unique = self.registry.counts()
        return DetectionStats(
            device=self.device,
            model_name=self.model_path.name if self.detector.loaded else None,
            currently_tracked=active,
            unique_tracks=unique,
            engine_state=self.stats.engine_state(self.detector.loaded),
            **snap,
        )

    def dataset_summary(self, refresh: bool = False) -> dict:
        """Read-only dataset scan, cached because hashing ~300 MB takes a few seconds."""
        with self._dataset_lock:
            if refresh or self._dataset_cache is None:
                report = validate_dataset(self.settings.resolve(self.settings.dataset_root))
                for folder in report.get("folders", []):
                    cid = FOLDER_TO_CLASS.get(folder["path"].split("/")[0])
                    folder["class_id"], folder["class_name"] = cid, CLASS_NAMES.get(cid)
                report["taxonomy"] = CLASS_NAMES
                self._dataset_cache = report
            return self._dataset_cache

    def health(self) -> dict:
        report = device_report()
        return {
            "status": "ok" if self.detector.loaded and self.fallback_reason is None else "degraded",
            "model_loaded": self.detector.loaded,
            "device": self.device,
            "torch_version": report["torch_version"],
            "cuda_build": report["cuda_build"],
            "uptime_s": round(time.time() - self.started_at, 1),
        }

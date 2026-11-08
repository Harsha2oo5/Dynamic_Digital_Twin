"""Background processing of uploaded videos.

Jobs run on a single worker thread so at most one video is being inferred at a time
(CPU inference is already multi-threaded inside PyTorch; parallel jobs would only thrash).
"""

import logging
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from app.camera.video_stream import VideoSource
from app.schemas.detection import FrameResult, VideoJobStatus
from app.vision.processor import FrameProcessor
from app.vision.visualization import draw_detections

if TYPE_CHECKING:
    from app.services.detection_service import DetectionService

logger = logging.getLogger(__name__)


@dataclass
class VideoJob:
    job_id: str
    filename: str
    upload_path: Path
    conf: float
    iou: float
    imgsz: int
    frame_skip: int
    video_fps: float
    frame_width: int
    frame_height: int
    total_frames: int | None
    status: str = "queued"
    frames_processed: int = 0
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None
    annotated_path: Path | None = None
    results: list[FrameResult] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def status_model(self) -> VideoJobStatus:
        expected = None
        if self.total_frames:
            expected = (self.total_frames + self.frame_skip - 1) // self.frame_skip
        progress = 1.0 if self.status == "completed" else (min(self.frames_processed / expected, 1.0) if expected else 0.0)
        ms = [r.inference_ms for r in self.results]
        return VideoJobStatus(
            job_id=self.job_id,
            filename=self.filename,
            status=self.status,
            progress=round(progress, 4),
            frames_processed=self.frames_processed,
            total_frames=self.total_frames,
            frame_skip=self.frame_skip,
            video_fps=self.video_fps,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            avg_inference_fps=round(1000.0 * len(ms) / sum(ms), 2) if ms else None,
            error=self.error,
            created_at=self.created_at,
            finished_at=self.finished_at,
            has_annotated_video=self.annotated_path is not None and self.annotated_path.is_file(),
        )


class VideoJobManager:
    def __init__(self, service: "DetectionService", output_dir: Path, max_jobs: int, max_frames: int, save_annotated: bool):
        self.service = service
        self.output_dir = output_dir
        self.max_jobs = max_jobs
        self.max_frames = max_frames
        self.save_annotated = save_annotated
        self._jobs: OrderedDict[str, VideoJob] = OrderedDict()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-job")

    @staticmethod
    def new_job_id() -> str:
        return uuid.uuid4().hex[:12]

    def submit(self, job: VideoJob) -> VideoJob:
        with self._lock:
            self._jobs[job.job_id] = job
            self._evict_locked()
        self._executor.submit(self._run, job)
        return job

    def get(self, job_id: str) -> VideoJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[VideoJob]:
        with self._lock:
            return list(reversed(self._jobs.values()))

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        job.cancel_event.set()
        return True

    def shutdown(self) -> None:
        for job in self.list_jobs():
            job.cancel_event.set()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _evict_locked(self) -> None:
        finished = [j for j in self._jobs.values() if j.status in ("completed", "failed", "cancelled")]
        while len(self._jobs) > self.max_jobs and finished:
            old = finished.pop(0)
            self._jobs.pop(old.job_id, None)
            if old.annotated_path:
                old.annotated_path.unlink(missing_ok=True)

    def _run(self, job: VideoJob) -> None:
        source_id = f"video:{job.job_id}"
        writer: cv2.VideoWriter | None = None
        if job.cancel_event.is_set():
            job.status = "cancelled"
            job.finished_at = datetime.now()
            job.upload_path.unlink(missing_ok=True)
            return
        job.status = "processing"
        processed_fps = (job.video_fps or 25.0) / job.frame_skip
        tracking = self.service.new_tracking_session(job.job_id, frame_rate=processed_fps)
        self.service.source_started(source_id, f"Video: {job.filename}", tracking)
        processor = FrameProcessor(self.service.detector, self.service.classifier, tracking)
        try:
            with VideoSource(job.upload_path) as src:
                if self.save_annotated:
                    out_fps = (job.video_fps or 25.0) / job.frame_skip
                    job.annotated_path = self.output_dir / f"{job.job_id}_annotated.mp4"
                    writer = cv2.VideoWriter(
                        str(job.annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (job.frame_width, job.frame_height)
                    )
                    if not writer.isOpened():
                        logger.warning("Annotated video writer unavailable; continuing without it")
                        writer, job.annotated_path = None, None
                for index, frame in src.frames(skip=job.frame_skip):
                    if job.cancel_event.is_set():
                        job.status = "cancelled"
                        break
                    if job.frames_processed >= self.max_frames:
                        logger.info("Job %s reached max_video_frames=%d", job.job_id, self.max_frames)
                        break
                    # lifecycle clock = video time, so timeouts mean the same whatever the CPU speed
                    result = processor.process(frame, frame_id=index, conf=job.conf, iou=job.iou, imgsz=job.imgsz,
                                               session_time=index / (job.video_fps or 25.0))
                    job.results.append(result)
                    job.frames_processed += 1
                    self.service.record(source_id, "video", result)
                    if writer is not None:
                        writer.write(draw_detections(frame, result.detections))
            if job.status == "processing":
                job.status = "completed"
            logger.info("Video job %s %s: %d frames", job.job_id, job.status, job.frames_processed)
        except Exception as exc:  # noqa: BLE001 - report any failure through the job status
            logger.exception("Video job %s failed", job.job_id)
            job.status = "failed"
            job.error = str(exc)
            self.service.record_error(source_id, f"Video job failed: {exc}")
        finally:
            if writer is not None:
                writer.release()
            job.finished_at = datetime.now()
            job.upload_path.unlink(missing_ok=True)
            self.service.source_stopped(source_id)

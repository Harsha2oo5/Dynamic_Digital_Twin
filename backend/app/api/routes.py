import logging
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool

from app.camera.video_stream import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, VideoSource, decode_image
from app.schemas.detection import (
    CheckpointInfo,
    DetectionStats,
    HealthResponse,
    ImageDetectionResponse,
    ModelInfo,
    TrackedObject,
    VideoJobResults,
    VideoJobStatus,
)
from app.services.detection_service import DetectionService
from app.services.video_jobs import VideoJob
from app.vision.visualization import draw_detections, encode_jpeg_base64

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

CHUNK = 1 << 20


def get_service(request: Request) -> DetectionService:
    return request.app.state.service


def require_model(service: DetectionService) -> None:
    if not service.detector.loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, service.detector.error or "Model is not loaded")


def check_extension(filename: str | None, allowed: set[str], kind: str) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported {kind} type {ext or '(none)'}; allowed: {sorted(allowed)}"
        )
    return ext


async def read_limited(file: UploadFile, max_bytes: int) -> bytes:
    data = bytearray()
    while chunk := await file.read(CHUNK):
        data.extend(chunk)
        if len(data) > max_bytes:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {max_bytes // (1 << 20)} MB")
    return bytes(data)


# --- system ----------------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
def health(service: DetectionService = Depends(get_service)):
    return service.health()


@router.get("/model/info", response_model=ModelInfo)
def model_info(service: DetectionService = Depends(get_service)):
    return service.model_info()


@router.get("/models", response_model=list[CheckpointInfo])
def list_models(service: DetectionService = Depends(get_service)):
    return service.list_checkpoints()


@router.get("/config")
def runtime_config(service: DetectionService = Depends(get_service)):
    s = service.settings
    return {
        "device": service.device,
        "detection_model": s.detection_model,
        "model_path": str(service.model_path),
        "model_fallback_to_pretrained": s.model_fallback_to_pretrained,
        "confidence_threshold": s.confidence_threshold,
        "iou_threshold": s.iou_threshold,
        "image_size": s.image_size,
        "frame_skip": s.frame_skip,
        "max_frame_dim": s.max_frame_dim,
        "tracker": s.tracker_config,
        "max_video_mb": s.max_video_mb,
        "max_video_frames": s.max_video_frames,
        "max_image_mb": s.max_image_mb,
        "track_lost_after_s": s.track_lost_after_s,
        "dataset_root": str(s.dataset_root),
    }


# --- detection -------------------------------------------------------------------------------
@router.post("/detect/image", response_model=ImageDetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    confidence: float | None = Form(None, ge=0.0, le=1.0),
    iou: float | None = Form(None, ge=0.0, le=1.0),
    image_size: int | None = Form(None, ge=32, le=1920),
    annotate: bool = Form(False),
    service: DetectionService = Depends(get_service),
):
    require_model(service)
    s = service.settings
    check_extension(file.filename, IMAGE_EXTENSIONS, "image")
    data = await read_limited(file, s.max_image_mb << 20)
    frame = await run_in_threadpool(decode_image, data)

    processor = service.new_processor()
    result = await run_in_threadpool(
        processor.process,
        frame,
        0,
        confidence if confidence is not None else s.confidence_threshold,
        iou if iou is not None else s.iou_threshold,
        image_size or s.image_size,
    )
    service.record(f"image:{uuid.uuid4().hex[:8]}", "image", result)

    annotated = None
    if annotate:
        annotated = await run_in_threadpool(lambda: encode_jpeg_base64(draw_detections(frame, result.detections)))
    return ImageDetectionResponse(**result.model_dump(), filename=file.filename, annotated_image=annotated)


@router.post("/detect/video", response_model=VideoJobStatus, status_code=status.HTTP_202_ACCEPTED)
async def detect_video(
    file: UploadFile = File(...),
    confidence: float | None = Form(None, ge=0.0, le=1.0),
    iou: float | None = Form(None, ge=0.0, le=1.0),
    image_size: int | None = Form(None, ge=32, le=1920),
    frame_skip: int | None = Form(None, ge=1, le=30),
    service: DetectionService = Depends(get_service),
):
    """Upload a video and start background frame-by-frame detection + tracking.

    Poll GET /api/detect/video/{job_id} for progress and fetch /results when completed.
    """
    require_model(service)
    s = service.settings
    ext = check_extension(file.filename, VIDEO_EXTENSIONS, "video")
    job_id = service.video_jobs.new_job_id()
    upload_path = service.upload_dir / f"{job_id}{ext}"

    size, max_bytes = 0, s.max_video_mb << 20
    try:
        with upload_path.open("wb") as out:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"Video exceeds {s.max_video_mb} MB")
                out.write(chunk)

        def probe():
            with VideoSource(upload_path) as src:
                return src.info

        info = await run_in_threadpool(probe)
    except BaseException:
        upload_path.unlink(missing_ok=True)
        raise

    job = VideoJob(
        job_id=job_id,
        filename=file.filename or upload_path.name,
        upload_path=upload_path,
        conf=confidence if confidence is not None else s.confidence_threshold,
        iou=iou if iou is not None else s.iou_threshold,
        imgsz=image_size or s.image_size,
        frame_skip=frame_skip or s.frame_skip,
        video_fps=round(info.fps, 3),
        frame_width=info.width,
        frame_height=info.height,
        total_frames=info.frame_count or None,
    )
    service.video_jobs.submit(job)
    logger.info("Video job %s queued: %s (%dx%d, %.2f fps, %s frames)", job_id, job.filename, info.width, info.height,
                info.fps, info.frame_count or "?")
    return job.status_model()


def _get_job(job_id: str, service: DetectionService) -> VideoJob:
    job = service.video_jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown video job {job_id}")
    return job


@router.get("/detect/video", response_model=list[VideoJobStatus])
def list_video_jobs(service: DetectionService = Depends(get_service)):
    return [j.status_model() for j in service.video_jobs.list_jobs()]


@router.get("/detect/video/{job_id}", response_model=VideoJobStatus)
def video_job_status(job_id: str, service: DetectionService = Depends(get_service)):
    return _get_job(job_id, service).status_model()


@router.get("/detect/video/{job_id}/results", response_model=VideoJobResults)
def video_job_results(job_id: str, service: DetectionService = Depends(get_service)):
    job = _get_job(job_id, service)
    return VideoJobResults(job=job.status_model(), frames=list(job.results))


@router.get("/detect/video/{job_id}/annotated")
def video_job_annotated(job_id: str, service: DetectionService = Depends(get_service)):
    job = _get_job(job_id, service)
    if job.status != "completed" or not job.annotated_path or not job.annotated_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Annotated video not available")
    return FileResponse(job.annotated_path, media_type="video/mp4", filename=f"{Path(job.filename).stem}_annotated.mp4")


@router.get("/detect/video/{job_id}/frames/{index}")
async def video_job_frame(job_id: str, index: int, service: DetectionService = Depends(get_service)):
    """Annotated JPEG of the index-th *processed* frame. Lets browsers view results for codecs
    they cannot play natively (e.g. AVI)."""
    job = _get_job(job_id, service)
    if job.status != "completed" or not job.annotated_path or not job.annotated_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Annotated frames not available")
    if not 0 <= index < job.frames_processed:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Frame index out of range (0..{job.frames_processed - 1})")

    def read_frame() -> bytes:
        with VideoSource(job.annotated_path) as src:
            src.seek(index)
            frame = src.read()
        if frame is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Frame could not be read")
        return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    return Response(await run_in_threadpool(read_frame), media_type="image/jpeg")


@router.post("/detect/video/{job_id}/cancel", response_model=VideoJobStatus)
def cancel_video_job(job_id: str, service: DetectionService = Depends(get_service)):
    job = _get_job(job_id, service)
    service.video_jobs.cancel(job_id)
    return job.status_model()


# --- statistics & inventory ------------------------------------------------------------------
@router.get("/detection/stats", response_model=DetectionStats)
def detection_stats(service: DetectionService = Depends(get_service)):
    return service.detection_stats()


@router.get("/objects", response_model=list[TrackedObject])
def list_objects(service: DetectionService = Depends(get_service)):
    return service.registry.list_objects()


@router.get("/objects/{object_id}", response_model=TrackedObject)
def get_object(object_id: str, service: DetectionService = Depends(get_service)):
    obj = service.registry.get(object_id)
    if obj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown object {object_id}")
    return obj


@router.delete("/objects", status_code=status.HTTP_204_NO_CONTENT)
def clear_objects(service: DetectionService = Depends(get_service)):
    service.registry.clear()
    service.stats.reset()


# --- dataset ---------------------------------------------------------------------------------
@router.get("/dataset/summary")
async def dataset_summary(refresh: bool = False, service: DetectionService = Depends(get_service)):
    return await run_in_threadpool(service.dataset_summary, refresh)

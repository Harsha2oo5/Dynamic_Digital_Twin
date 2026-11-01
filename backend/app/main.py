"""FastAPI entrypoint.  Run:  uvicorn app.main:app --host 127.0.0.1 --port 8000"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api import routes, websocket
from app.camera.video_stream import InvalidMediaError
from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from app.services.detection_service import DetectionService
from app.vision.detector import ModelNotLoadedError

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = DetectionService(settings)
        await run_in_threadpool(service.startup)  # model loads once; failures leave the API up in "degraded" mode
        app.state.service = service
        yield
        service.shutdown()

    app = FastAPI(title="Dynamic Digital Twin API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
    )
    app.include_router(routes.router)
    app.include_router(websocket.router)

    @app.exception_handler(InvalidMediaError)
    async def invalid_media(_: Request, exc: InvalidMediaError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ModelNotLoadedError)
    async def model_not_loaded(_: Request, exc: ModelNotLoadedError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        logger.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


app = create_app()

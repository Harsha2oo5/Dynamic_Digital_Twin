"""Application configuration, loaded from environment variables / backend/.env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # Model selection: DETECTION_MODEL picks one of the two checkpoints below.
    detection_model: Literal["pretrained", "custom"] = "pretrained"
    pretrained_model_path: Path = Path("models/pretrained/yolo11n.pt")
    custom_model_path: Path = Path("models/custom/digital_twin_best.pt")
    model_path: Path | None = None  # explicit checkpoint override (takes precedence when set)
    # If the custom model is selected but cannot load: False -> API runs degraded with a clear error;
    # True -> load the pretrained model instead and report fallback_active in /api/model/info.
    model_fallback_to_pretrained: bool = False
    device: str = "cpu"  # only "cpu" is supported in this milestone; see core/device.py
    torch_threads: int = 0  # 0 = let PyTorch decide

    # Inference
    confidence_threshold: float = Field(0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    image_size: int = Field(416, ge=32, le=1920)
    max_frame_dim: int = Field(1920, ge=320)  # larger frames are downscaled before inference
    max_detections: int = Field(100, ge=1)
    warmup: bool = True

    # Video / tracking
    frame_skip: int = Field(1, ge=1)  # process every Nth frame
    # Tuned tracker config (relative to backend/) or an Ultralytics built-in name; configs/trackers/bts_botsort.yaml
    # adds camera-motion compensation for handheld footage.
    tracker_config: str = "configs/trackers/bts_bytetrack.yaml"
    track_min_hits: int = Field(3, ge=1)
    stream_frame_rate: float = Field(8.0, gt=0)  # expected processed fps of live streams (scales tracker buffers)  # frames before a track is confirmed (new -> active)
    max_video_mb: int = Field(200, ge=1)
    max_video_frames: int = Field(5000, ge=1)  # cap on processed frames per video job
    max_video_jobs: int = Field(10, ge=1)  # finished jobs kept in memory before eviction
    save_annotated_video: bool = True
    max_image_mb: int = Field(20, ge=1)

    # Object inventory
    track_lost_after_s: float = Field(3.0, gt=0)

    # Paths
    upload_dir: Path = Path("runtime/uploads")
    output_dir: Path = Path("runtime/outputs")
    dataset_root: Path = PROJECT_DIR / "digital twin project datasets"
    models_dir: Path = Path("models")

    # API
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    log_level: str = "INFO"
    app_env: Literal["development", "test", "production"] = "development"

    # Telemetry (sensor integration). Without TELEMETRY_DATABASE_URL a local SQLite file is used as a
    # development fallback; production deployments should point this at PostgreSQL.
    telemetry_enabled: bool = True
    telemetry_database_url: SecretStr | None = None  # e.g. postgresql+psycopg://user:pass@127.0.0.1:5432/dynamic_twin
    telemetry_sqlite_fallback: Path = Path("runtime/telemetry.db")
    telemetry_ingest_api_key: SecretStr | None = None  # required by POST /api/telemetry/ingest; unset = ingest closed
    telemetry_retention_days: int = Field(30, ge=1)
    telemetry_thresholds_file: Path = Path("configs/telemetry/thresholds.yaml")
    telemetry_watchdog_interval_s: float = Field(2.0, gt=0)
    site_id: str = "SITE-001"
    site_name: str = "BTS site"

    # Simulator: never starts by itself unless SIMULATOR_AUTOSTART=true, and never in APP_ENV=production
    simulator_config_file: Path = Path("configs/telemetry/simulator.yaml")
    simulator_autostart: bool = False

    # MQTT adapter (off by default). Credentials only from the environment; never returned by the API or logged.
    mqtt_enabled: bool = False
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    mqtt_username: str | None = None
    mqtt_password: SecretStr | None = None
    mqtt_tls: bool = False
    mqtt_topic: str = "bts/+/+/telemetry"  # bts/<site_id>/<sensor_id>/telemetry
    mqtt_client_id: str = "dynamic-twin-backend"
    mqtt_qos: int = Field(1, ge=0, le=1)

    def selected_model_path(self) -> Path:
        if self.model_path is not None:
            return self.resolve(self.model_path)
        return self.resolve(self.custom_model_path if self.detection_model == "custom" else self.pretrained_model_path)

    def resolve(self, path: Path) -> Path:
        """Resolve a configured path relative to the backend directory."""
        return path if path.is_absolute() else (BACKEND_DIR / path).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()

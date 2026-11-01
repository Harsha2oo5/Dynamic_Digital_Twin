// Mirrors backend/app/schemas/detection.py

export interface BBox {
  x1: number
  y1: number
  x2: number
  y2: number
}

export interface Detection {
  /** Temporary visual tracking ID — not a permanent component identity. */
  track_id: number | null
  class_id: number
  class_name: string
  confidence: number
  bbox: BBox
  component_type: string | null
}

export interface FrameResult {
  frame_id: number
  timestamp: string
  device: string
  fps: number
  inference_ms: number
  frame_width: number
  frame_height: number
  detections: Detection[]
}

export interface ImageDetectionResponse extends FrameResult {
  filename: string | null
  annotated_image: string | null
}

export type JobState = 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled'

export interface VideoJobStatus {
  job_id: string
  filename: string
  status: JobState
  progress: number
  frames_processed: number
  total_frames: number | null
  frame_skip: number
  video_fps: number
  frame_width: number
  frame_height: number
  avg_inference_fps: number | null
  error: string | null
  created_at: string
  finished_at: string | null
  has_annotated_video: boolean
}

export interface VideoJobResults {
  job: VideoJobStatus
  frames: FrameResult[]
}

export type ModelType = 'pretrained' | 'custom' | 'unknown'

export interface ModelInfo {
  loaded: boolean
  model_name: string | null
  model_path: string
  model_type: ModelType
  task: string | null
  device: string
  num_classes: number
  class_names: string[]
  confidence_threshold: number
  iou_threshold: number
  image_size: number
  frame_skip: number
  tracker: string
  load_time_s: number | null
  error: string | null
  selected_model: 'pretrained' | 'custom'
  fallback_active: boolean
  fallback_reason: string | null
  taxonomy_match: boolean | null
  custom_model_path: string
  custom_model_available: boolean
  /** The seven Digital Twin component classes in class-id order (from backend taxonomy.py). */
  component_taxonomy?: string[]
  checkpoint_size_mb?: number | null
  checkpoint_modified?: string | null
}

export interface CheckpointInfo {
  name: string
  path: string
  model_type: ModelType
  size_mb: number
  active: boolean
}

export interface Health {
  status: 'ok' | 'degraded'
  model_loaded: boolean
  device: string
  torch_version: string
  cuda_build: boolean
  uptime_s: number
}

export interface DetectionStats {
  device: string
  model_name: string | null
  frames_processed: number
  total_detections: number
  avg_inference_fps: number | null
  avg_inference_ms: number | null
  avg_confidence: number | null
  currently_tracked: number
  unique_tracks: number
  active_sources: string[]
  class_counts: ClassCount[]
  recent: PerfSample[]
  // activity / health (added for the Overview; optional so older backends still type-check)
  engine_state?: 'running' | 'idle' | 'error'
  sources?: SourceState[]
  min_inference_ms?: number | null
  max_inference_ms?: number | null
  latency_window?: number | null
  session_started_at?: string | null
  last_frame_at?: string | null
  seconds_since_last_frame?: number | null
  error_count?: number
  last_error?: { message: string; source: string | null; at: string } | null
}

export interface ClassCount {
  class_id?: number | null
  class_name: string
  count: number
}

export interface PerfSample {
  timestamp: string
  inference_ms: number
  detections: number
  seq?: number | null
}

export interface SourceState {
  source_id: string
  label: string
  source_type: string
  /** processing = produced a frame in the last few seconds; connected = registered, no recent frame. */
  state: 'connected' | 'processing'
  frames: number
  started_at: string
  last_frame_at: string | null
}

export interface TrackedObject {
  object_id: string
  source_id: string
  source_type: 'webcam' | 'video' | 'image'
  track_id: number | null
  class_id: number
  class_name: string
  component_type: string | null
  confidence: number
  max_confidence: number
  avg_confidence: number
  first_seen: string
  last_seen: string
  frames_seen: number
  status: 'active' | 'lost' | 'snapshot'
  last_bbox: BBox
  frame_width: number
  frame_height: number
}

export interface DatasetFolder {
  path: string
  images: number
  labelled: number
  extensions: Record<string, number>
  size_mb: number
  min_resolution: [number, number] | null
  max_resolution: [number, number] | null
  class_id?: number | null
  class_name?: string | null
}

export interface DatasetSummary {
  root: string
  exists: boolean
  scanned_at: string
  data_yaml?: string | null
  class_names?: unknown
  annotation_format?: string
  total_images?: number
  labelled_images?: number
  folders?: DatasetFolder[]
  macos_metadata_files?: number
  unreadable_images?: string[]
  duplicate_groups?: string[][]
  label_errors?: string[]
  issues: string[]
  ready_for_detection_training: boolean
}

export type RuntimeConfig = Record<string, string | number | boolean>

export type StreamMessage =
  | ({ type: 'result' } & FrameResult)
  | { type: 'ready'; session_id: string; device: string }
  | { type: 'error'; message: string }

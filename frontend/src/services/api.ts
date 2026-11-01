import type {
  CheckpointInfo,
  DatasetSummary,
  DetectionStats,
  Health,
  ImageDetectionResponse,
  ModelInfo,
  RuntimeConfig,
  TrackedObject,
  VideoJobResults,
  VideoJobStatus,
} from '../types/api'

/** Empty = same origin (Vite dev proxy). Set VITE_API_BASE to point at another backend. */
export const API_BASE: string = import.meta.env.VITE_API_BASE ?? ''

export function wsUrl(path: string): string {
  const base = API_BASE || window.location.origin
  return base.replace(/^http/, 'ws') + path
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(API_BASE + path, init)
  } catch {
    throw new ApiError(0, 'Backend unreachable')
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail)
  }
  return res.status === 204 ? (undefined as T) : res.json()
}

export interface DetectParams {
  confidence: number
  iou: number
  imageSize: number
  frameSkip?: number
}

function detectForm(file: File, p: DetectParams): FormData {
  const form = new FormData()
  form.append('file', file)
  form.append('confidence', String(p.confidence))
  form.append('iou', String(p.iou))
  form.append('image_size', String(p.imageSize))
  if (p.frameSkip) form.append('frame_skip', String(p.frameSkip))
  return form
}

export const api = {
  health: () => request<Health>('/api/health'),
  modelInfo: () => request<ModelInfo>('/api/model/info'),
  models: () => request<CheckpointInfo[]>('/api/models'),
  config: () => request<RuntimeConfig>('/api/config'),
  stats: () => request<DetectionStats>('/api/detection/stats'),
  objects: () => request<TrackedObject[]>('/api/objects'),
  clearObjects: () => request<void>('/api/objects', { method: 'DELETE' }),
  dataset: (refresh = false) => request<DatasetSummary>(`/api/dataset/summary?refresh=${refresh}`),

  detectImage: (file: File, p: DetectParams) => {
    const form = detectForm(file, p)
    form.append('annotate', 'true')
    return request<ImageDetectionResponse>('/api/detect/image', { method: 'POST', body: form })
  },
  startVideo: (file: File, p: DetectParams) =>
    request<VideoJobStatus>('/api/detect/video', { method: 'POST', body: detectForm(file, p) }),
  videoStatus: (id: string) => request<VideoJobStatus>(`/api/detect/video/${id}`),
  videoResults: (id: string) => request<VideoJobResults>(`/api/detect/video/${id}/results`),
  cancelVideo: (id: string) => request<VideoJobStatus>(`/api/detect/video/${id}/cancel`, { method: 'POST' }),
  annotatedVideoUrl: (id: string) => `${API_BASE}/api/detect/video/${id}/annotated`,
  annotatedFrameUrl: (id: string, index: number) => `${API_BASE}/api/detect/video/${id}/frames/${index}`,
}

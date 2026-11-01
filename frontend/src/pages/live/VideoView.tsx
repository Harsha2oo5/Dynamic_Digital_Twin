import { Download, Film, LoaderCircle, Play, Square, Upload } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { DetectionOverlay } from '../../components/DetectionOverlay'
import { DetectionTable } from '../../components/DetectionTable'
import { Alert, Badge, Button, cx, Empty, Panel } from '../../components/ui'
import { useSettings } from '../../hooks/useSettings'
import { api } from '../../services/api'
import type { FrameResult, VideoJobStatus } from '../../types/api'
import { DetectionControls } from './Controls'
import { FrameMetrics } from './Metrics'

/** Latest processed frame at or before `frameIndex`, or null if it is further than one skip-interval away. */
function frameAt(frames: FrameResult[], frameIndex: number, skip: number): FrameResult | null {
  let lo = 0
  let hi = frames.length - 1
  let best = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (frames[mid].frame_id <= frameIndex) {
      best = mid
      lo = mid + 1
    } else hi = mid - 1
  }
  if (best < 0) return null
  return frameIndex - frames[best].frame_id < skip ? frames[best] : null
}

const stateTone = { queued: 'neutral', processing: 'accent', completed: 'good', failed: 'bad', cancelled: 'warn' } as const

export function VideoView() {
  const { settings } = useSettings()
  const inputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [url, setUrl] = useState<string | null>(null)
  const [dims, setDims] = useState({ w: 0, h: 0 })
  const [job, setJob] = useState<VideoJobStatus | null>(null)
  const [frames, setFrames] = useState<FrameResult[]>([])
  const [error, setError] = useState<string | null>(null)
  const [unplayable, setUnplayable] = useState(false)
  const [view, setView] = useState<'player' | 'frames'>('player')
  const [current, setCurrent] = useState<FrameResult | null>(null)
  const [frameIdx, setFrameIdx] = useState(0)

  useEffect(() => () => void (url && URL.revokeObjectURL(url)), [url])

  // poll job progress
  useEffect(() => {
    if (!job || job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') return
    const id = window.setInterval(async () => {
      try {
        const s = await api.videoStatus(job.job_id)
        setJob(s)
        if (s.status === 'completed') setFrames((await api.videoResults(s.job_id)).frames)
        if (s.status === 'failed') setError(s.error ?? 'Processing failed')
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      }
    }, 600)
    return () => window.clearInterval(id)
  }, [job])

  // sync overlay to playback position
  useEffect(() => {
    const video = videoRef.current
    if (!video || !job || !frames.length || view !== 'player') return
    const fps = job.video_fps || 25
    let handle = 0
    let stopped = false
    const update = (t: number) => setCurrent(frameAt(frames, Math.round(t * fps), job.frame_skip))
    update(video.currentTime)

    // Frame-accurate sync where supported (Chromium, Safari); rAF polling elsewhere.
    if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
      const cb = (_: number, meta: VideoFrameCallbackMetadata) => {
        if (stopped) return
        update(meta.mediaTime)
        handle = video.requestVideoFrameCallback(cb)
      }
      handle = video.requestVideoFrameCallback(cb)
      const onSeek = () => update(video.currentTime)
      video.addEventListener('seeked', onSeek)
      return () => {
        stopped = true
        video.cancelVideoFrameCallback(handle)
        video.removeEventListener('seeked', onSeek)
      }
    }
    const loop = () => {
      update(video.currentTime)
      handle = requestAnimationFrame(loop)
    }
    handle = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(handle)
  }, [frames, job, view])

  const choose = (f: File | undefined) => {
    if (!f) return
    setFile(f)
    setUrl(URL.createObjectURL(f))
    setJob(null)
    setFrames([])
    setCurrent(null)
    setError(null)
    setUnplayable(false)
    setView('player')
  }

  const start = async () => {
    if (!file) return
    setError(null)
    setFrames([])
    setCurrent(null)
    try {
      setJob(await api.startVideo(file, settings))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const processing = job?.status === 'queued' || job?.status === 'processing'
  const done = job?.status === 'completed'
  const shownFrame = view === 'frames' ? frames[frameIdx] ?? null : current
  const uniqueTracks = useMemo(() => new Set(frames.flatMap((f) => f.detections.map((d) => d.track_id)).filter((t) => t != null)).size, [frames])

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_20rem]">
      <div className="space-y-4">
        <Panel
          title={file ? file.name : 'Video detection'}
          icon={Film}
          bodyClassName="p-0"
          actions={
            <>
              {job && <Badge tone={stateTone[job.status]}>{job.status}</Badge>}
              {done && job.has_annotated_video && (
                <div className="flex overflow-hidden rounded-md border border-line text-xs">
                  {(['player', 'frames'] as const).map((v) => (
                    <button key={v} onClick={() => setView(v)} className={cx('px-2.5 py-1', view === v ? 'bg-surface-3 text-ink' : 'text-ink-3 hover:text-ink')}>
                      {v === 'player' ? 'Player + overlay' : 'Annotated frames'}
                    </button>
                  ))}
                </div>
              )}
              {processing ? (
                <Button icon={Square} variant="danger" onClick={() => job && api.cancelVideo(job.job_id)}>
                  Cancel
                </Button>
              ) : (
                file && (
                  <Button icon={Play} variant="primary" onClick={start}>
                    {done ? 'Re-run' : 'Run detection'}
                  </Button>
                )
              )}
              <Button icon={Upload} onClick={() => inputRef.current?.click()} disabled={processing}>
                Upload video
              </Button>
              <input ref={inputRef} type="file" accept=".mp4,.avi,.mov,.mkv,.webm" hidden onChange={(e) => choose(e.target.files?.[0])} />
            </>
          }
        >
          {processing && job && (
            <div className="border-b border-line px-4 py-3">
              <div className="mb-1.5 flex justify-between text-xs text-ink-2">
                <span className="flex items-center gap-2">
                  <LoaderCircle size={13} className="animate-spin" /> Processing on CPU…
                </span>
                <span className="tabular">
                  {job.frames_processed}
                  {job.total_frames ? ` / ${Math.ceil(job.total_frames / job.frame_skip)}` : ''} frames
                  {job.avg_inference_fps ? ` · ${job.avg_inference_fps.toFixed(1)} FPS` : ''}
                </span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-surface-3">
                <div className="h-full bg-accent transition-[width]" style={{ width: `${(job.progress * 100).toFixed(1)}%` }} />
              </div>
            </div>
          )}

          <div className="relative aspect-video w-full overflow-hidden bg-black last:rounded-b-lg">
            {!url && (
              <button className="absolute inset-0 grid place-items-center" onClick={() => inputRef.current?.click()}>
                <Empty icon={Upload} title="Upload an MP4 or AVI video">
                  The video is processed frame-by-frame with YOLO + ByteTrack on the backend; results are then synced to playback here.
                </Empty>
              </button>
            )}
            {url && view === 'player' && (
              <>
                <video
                  ref={videoRef}
                  src={url}
                  controls
                  playsInline
                  muted
                  onLoadedMetadata={(e) => setDims({ w: e.currentTarget.videoWidth, h: e.currentTarget.videoHeight })}
                  onError={() => setUnplayable(true)}
                  className="absolute inset-0 h-full w-full object-contain"
                />
                {current && <DetectionOverlay detections={current.detections} frameWidth={current.frame_width} frameHeight={current.frame_height} mediaWidth={dims.w} mediaHeight={dims.h} />}
              </>
            )}
            {view === 'frames' && job && frames.length > 0 && (
              <img src={api.annotatedFrameUrl(job.job_id, frameIdx)} alt={`Annotated frame ${frameIdx}`} className="absolute inset-0 h-full w-full object-contain" />
            )}
          </div>

          {view === 'frames' && done && frames.length > 0 && (
            <div className="flex items-center gap-3 px-4 py-3 text-xs text-ink-2">
              <input type="range" min={0} max={frames.length - 1} value={frameIdx} onChange={(e) => setFrameIdx(Number(e.target.value))} className="flex-1" />
              <span className="tabular w-36 text-right">
                frame #{frames[frameIdx]?.frame_id} ({frameIdx + 1}/{frames.length})
              </span>
            </div>
          )}
        </Panel>

        {unplayable && (
          <Alert tone="warn" title="Your browser cannot play this video format">
            Detection still works on the backend. After processing, use <b>Annotated frames</b> to step through server-rendered results, or download the annotated video.
          </Alert>
        )}
        {error && <Alert tone="bad" title="Video detection error">{error}</Alert>}
        {done && (
          <Panel
            title={`Detections at ${view === 'player' ? 'playback position' : 'selected frame'}`}
            actions={
              job.has_annotated_video && (
                <a href={api.annotatedVideoUrl(job.job_id)} className="inline-flex items-center gap-1.5 text-xs text-accent-strong hover:underline">
                  <Download size={13} /> Annotated video (.mp4)
                </a>
              )
            }
          >
            <DetectionTable detections={shownFrame?.detections ?? []} />
          </Panel>
        )}
      </div>
      <div className="space-y-4">
        <DetectionControls showFrameSkip note="Settings apply to the next run." />
        <FrameMetrics
          result={shownFrame}
          extra={
            job
              ? [
                  ['Video', `${job.frame_width}×${job.frame_height} @ ${job.video_fps.toFixed(2)} fps`],
                  ['Job avg FPS', job.avg_inference_fps?.toFixed(1) ?? '—'],
                  ['Frames processed', String(job.frames_processed)],
                  ['Unique track IDs', done ? String(uniqueTracks) : '—'],
                ]
              : []
          }
        />
      </div>
    </div>
  )
}

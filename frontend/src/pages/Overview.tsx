import type { LucideIcon } from 'lucide-react'
import {
  Activity,
  AlertOctagon,
  Boxes,
  Clock,
  Crosshair,
  Fingerprint,
  Film,
  Gauge,
  Layers,
  Network,
  Radio,
  ScanEye,
  Server,
  Target,
} from 'lucide-react'
import { useMemo, type ReactNode } from 'react'
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Alert, Badge, cx, Empty, KV, Panel, PageHeader, StatCard, StatusDot } from '../components/ui'
import { usePolling } from '../hooks/usePolling'
import { ago, dateTime, duration, int, num, pct, time, titleCase } from '../lib/format'
import { api } from '../services/api'
import type { ClassCount, DetectionStats, Health, ModelInfo } from '../types/api'
import { HealthOverview } from './overview/HealthOverview'
import { SiteTelemetry } from './overview/SiteTelemetry'

const AXIS = { stroke: 'var(--color-ink-3)', fontSize: 11, tickLine: false, axisLine: false } as const
const tooltipStyle = {
  contentStyle: { background: 'var(--color-surface-2)', border: '1px solid var(--color-line)', borderRadius: 6, fontSize: 12 },
  labelStyle: { color: 'var(--color-ink-2)' },
  itemStyle: { color: 'var(--color-ink)' },
  cursor: { stroke: 'var(--color-line)', fill: 'var(--color-surface-3)', fillOpacity: 0.4 },
}

type Tone = 'good' | 'warn' | 'bad' | 'neutral' | 'accent'
interface Status {
  label: string
  tone: Tone
  detail: ReactNode
}

// --- status derivation (all from backend state; nothing assumed) ----------------------------------
function backendStatus(h: Health | null, offline: boolean): Status {
  if (offline) return { label: 'Offline', tone: 'bad', detail: 'API not reachable on port 8000' }
  if (!h) return { label: 'Connecting', tone: 'neutral', detail: 'Waiting for the first response…' }
  return { label: 'Online', tone: 'good', detail: `Uptime ${duration(h.uptime_s)}` }
}

function engineStatus(s: DetectionStats | null, m: ModelInfo | null, offline: boolean): Status {
  if (offline || !s) return { label: '—', tone: 'neutral', detail: offline ? 'Backend offline' : 'Waiting for statistics…' }
  if (s.engine_state === 'error') {
    const why = m && !m.loaded ? m.error ?? 'Model not loaded' : s.last_error?.message ?? 'Processing error'
    return { label: 'Error', tone: 'bad', detail: why }
  }
  if (s.engine_state === 'running') return { label: 'Running', tone: 'good', detail: `Last frame ${ago(s.seconds_since_last_frame)}` }
  return {
    label: 'Idle',
    tone: 'neutral',
    detail: s.last_frame_at ? `Last frame ${ago(s.seconds_since_last_frame)}` : 'No inference yet this session',
  }
}

function sourceStatus(s: DetectionStats | null, offline: boolean): Status {
  if (offline || !s) return { label: '—', tone: 'neutral', detail: offline ? 'Backend offline' : 'Waiting for statistics…' }
  const sources = s.sources ?? []
  const processing = sources.filter((x) => x.state === 'processing')
  const more = (n: number) => (n > 1 ? ` (+${n - 1} more)` : '')
  if (processing.length) return { label: 'Processing', tone: 'good', detail: processing[0].label + more(processing.length) }
  if (sources.length) return { label: 'Connected', tone: 'accent', detail: `${sources[0].label}${more(sources.length)}, waiting for frames` }
  return { label: 'Disconnected', tone: 'neutral', detail: 'Connect a webcam or select a video source to begin detection.' }
}

function modelStatus(m: ModelInfo | null, modelError: string | null, offline: boolean): Status {
  if (!m) {
    if (offline || modelError) return { label: '—', tone: 'neutral', detail: offline ? 'Backend offline' : modelError }
    return { label: 'Loading', tone: 'neutral', detail: 'Requesting model information…' }
  }
  if (!m.loaded) return { label: 'Failed', tone: 'bad', detail: m.error ?? 'Model could not be loaded' }
  if (m.fallback_active) return { label: 'Loaded', tone: 'warn', detail: 'Pretrained fallback (custom model unavailable)' }
  return { label: 'Loaded', tone: 'good', detail: `${m.model_name} · ${m.model_type}` }
}

function StatusTile({ title, icon: Icon, status }: { title: string; icon: LucideIcon; status: Status }) {
  return (
    <div className="h-full rounded-lg border border-line bg-surface-1 px-4 py-3 transition-colors hover:border-ink-3/50">
      <div className="flex items-center justify-between text-xs font-medium uppercase tracking-wider text-ink-3">
        {title}
        <Icon size={15} />
      </div>
      <div className="mt-1.5 flex items-center gap-2 text-lg font-semibold text-ink">
        <StatusDot tone={status.tone} />
        {status.label}
      </div>
      <div className="mt-0.5 line-clamp-2 text-xs text-ink-2" title={typeof status.detail === 'string' ? status.detail : undefined}>
        {status.detail}
      </div>
    </div>
  )
}

// --- distribution bars ------------------------------------------------------------------------------
function DistributionBars({ rows, total }: { rows: { name: string; count: number }[]; total: number }) {
  const max = Math.max(1, ...rows.map((r) => r.count))
  return (
    <ul className="space-y-2">
      {rows.map((r) => (
        <li key={r.name} className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)_5.5rem] items-center gap-3 text-xs">
          <span className="truncate text-ink-2" title={r.name}>
            {r.name}
          </span>
          <span className="h-2 overflow-hidden rounded-full bg-surface-3">
            <span className="block h-full rounded-full bg-accent" style={{ width: `${(100 * r.count) / max}%` }} />
          </span>
          <span className="tabular text-right text-ink">
            {int(r.count)} <span className="text-ink-3">{total ? `${((100 * r.count) / total).toFixed(1)}%` : ''}</span>
          </span>
        </li>
      ))}
    </ul>
  )
}

const NO_TAXONOMY: string[] = []

function ComponentDistribution({ model, counts }: { model: ModelInfo | null; counts: ClassCount[] }) {
  const taxonomy = model?.component_taxonomy ?? NO_TAXONOMY
  // Component analytics only for a loaded custom model whose classes ARE the seven-class taxonomy.
  const capable = !!model?.loaded && model.model_type === 'custom' && model.taxonomy_match === true && taxonomy.length > 0

  const components = useMemo(() => {
    if (!capable) return null
    const rows = taxonomy.map((name, id) => ({
      name: titleCase(name),
      count: counts.filter((c) => c.class_id === id && c.class_name === name).reduce((a, c) => a + c.count, 0),
    }))
    const known = rows.reduce((a, r) => a + r.count, 0)
    const all = counts.reduce((a, c) => a + c.count, 0)
    return { rows, total: known, unrecognised: all - known }
  }, [capable, taxonomy, counts])

  const general = useMemo(() => {
    const top = counts.slice(0, 8).map((c) => ({ name: c.class_name, count: c.count }))
    return { rows: top, total: counts.reduce((a, c) => a + c.count, 0), hidden: Math.max(0, counts.length - 8) }
  }, [counts])

  const badge = !model ? null : capable ? (
    <Badge tone="accent">custom model · 7 components</Badge>
  ) : (
    <Badge tone="neutral">requires custom model</Badge>
  )

  return (
    <Panel title="Detected component distribution" icon={Network} actions={badge} className="h-full">
      {capable && components ? (
        components.total ? (
          <>
            <DistributionBars rows={components.rows} total={components.total} />
            <p className="mt-3 text-xs text-ink-3">
              {int(components.total)} component detections this session.
              {components.unrecognised > 0 && ` ${int(components.unrecognised)} detections of classes outside the taxonomy are excluded.`}
            </p>
          </>
        ) : (
          <Empty icon={Network} title="No components detected yet">
            Run detection in Live Monitoring; counts per component appear here as frames are processed.
          </Empty>
        )
      ) : (
        <>
          <div className="rounded-md border border-dashed border-line px-3 py-3 text-xs text-ink-2">
            <div className="font-medium text-ink">Component analytics will be available when a compatible custom model is loaded.</div>
            <div className="mt-1 text-ink-3">
              {model?.model_type === 'custom' && model.taxonomy_match === false
                ? 'The loaded custom model’s classes do not match the seven-component taxonomy, so its detections are not counted as components.'
                : `The loaded model${model?.model_name ? ` (${model.model_name})` : ''} has general-purpose COCO classes, which are not mapped to ${
                    taxonomy.length ? taxonomy.map(titleCase).join(', ') : 'the seven components'
                  }.`}
            </div>
          </div>
          <h3 className="mt-4 mb-2 text-xs font-medium uppercase tracking-wider text-ink-3">
            General detections · {model?.model_type ?? 'loaded'} model classes (not telecom components)
          </h3>
          {general.rows.length ? (
            <>
              <DistributionBars rows={general.rows} total={general.total} />
              {general.hidden > 0 && <p className="mt-2 text-xs text-ink-3">+{general.hidden} more classes</p>}
            </>
          ) : (
            <p className="py-3 text-center text-xs text-ink-3">No objects detected yet.</p>
          )}
        </>
      )}
    </Panel>
  )
}

// --- page -------------------------------------------------------------------------------------------
export function Overview() {
  const health = usePolling(api.health, 3000)
  const model = usePolling(api.modelInfo, 10000)
  const stats = usePolling(api.stats, 2000)
  const s = stats.data
  const m = model.data
  const offline = !!health.error

  const latency = useMemo(() => {
    const recent = s?.recent ?? []
    return recent.map((r, i) => ({ seq: r.seq ?? i + 1, t: time(r.timestamp), ms: r.inference_ms, detections: r.detections }))
  }, [s?.recent])

  const statuses = {
    backend: backendStatus(health.data, offline),
    engine: engineStatus(s, m, offline),
    source: sourceStatus(s, offline),
    model: modelStatus(m, model.error, offline),
  }
  const hasFrames = !!s && s.frames_processed > 0

  return (
    <>
      <PageHeader
        title="Overview"
        subtitle="System status, detection activity and site telemetry. Detection figures come from real inference; telemetry is labelled physical or simulated."
        actions={
          s?.session_started_at ? (
            <span className="text-xs text-ink-3">
              Session since {dateTime(s.session_started_at)}
              {offline && <Badge tone="warn">showing last received data</Badge>}
            </span>
          ) : undefined
        }
      />

      {offline && (
        <div className="mb-4">
          <Alert tone="bad" title="Backend unreachable">
            Start the API with <code className="font-mono">uvicorn app.main:app --port 8000</code> from <code className="font-mono">backend/</code>.
            The dashboard reconnects automatically.
          </Alert>
        </div>
      )}
      {m?.fallback_active && (
        <div className="mb-4">
          <Alert tone="warn" title="Running the pretrained fallback: custom model unavailable">{m.fallback_reason}</Alert>
        </div>
      )}

      {/* system health */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatusTile title="Backend" icon={Server} status={statuses.backend} />
        <StatusTile title="Detection engine" icon={ScanEye} status={statuses.engine} />
        <StatusTile title="Camera / video source" icon={Radio} status={statuses.source} />
        <StatusTile title="Model" icon={Layers} status={statuses.model} />
      </div>

      {/* model + metrics */}
      <div className="mt-4 grid grid-cols-[minmax(0,1fr)] gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
        <Panel
          title="Model information"
          icon={Layers}
          className="h-full"
          actions={m && <Badge tone={m.model_type === 'custom' ? 'accent' : 'neutral'}>{m.model_type === 'custom' ? 'Custom' : m.model_type === 'pretrained' ? 'Pretrained' : 'Unknown'}</Badge>}
        >
          {m ? (
            <>
              <KV
                rows={[
                  ['Checkpoint', <span className="font-mono">{m.model_name ?? m.model_path.split(/[\\/]/).pop()}</span>],
                  ['Status', <span className="flex items-center gap-2"><StatusDot tone={statuses.model.tone} />{m.loaded ? 'Loaded' : 'Failed'}{m.load_time_s != null && <span className="text-ink-3">in {num(m.load_time_s, 1)} s</span>}</span>],
                  ['Classes', `${m.num_classes}${m.model_type === 'pretrained' ? ' (COCO, general-purpose)' : ''}`],
                  ['Device', <span className="font-mono">{m.device.toUpperCase()}</span>],
                  ['File', m.checkpoint_size_mb != null ? `${num(m.checkpoint_size_mb, 1)} MB · modified ${dateTime(m.checkpoint_modified)}` : '—'],
                  ['Thresholds', `conf ${m.confidence_threshold} · IoU ${m.iou_threshold} · ${m.image_size} px`],
                ]}
              />
              <div className="mt-3 border-t border-line pt-3 text-xs text-ink-3">
                {m.model_type === 'custom'
                  ? m.taxonomy_match
                    ? 'Custom seven-component model active.'
                    : 'Custom model active, but its classes differ from the seven-component taxonomy.'
                  : m.custom_model_available
                    ? 'A custom checkpoint exists; set DETECTION_MODEL=custom in backend/.env and restart to use it.'
                    : 'Using the pretrained model. The seven-component custom model has not been trained yet.'}
              </div>
            </>
          ) : (
            <Empty icon={Layers} title={model.error ? 'Model information unavailable' : 'Loading model information…'}>
              {offline ? 'Backend offline' : model.error ?? undefined}
            </Empty>
          )}
        </Panel>

        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4 [&>*]:min-w-0">
          <StatCard label="Total detections" icon={Target} hint="Detection instances produced across all processed frames this session"
            value={int(s?.total_detections)} sub={s ? (hasFrames ? `over ${int(s.frames_processed)} frames` : 'No frames processed yet') : undefined} />
          <StatCard label="Frames processed" icon={Film} hint="Frames that went through inference (webcam, video, images)"
            value={int(s?.frames_processed)} sub={s?.last_frame_at ? `last ${time(s.last_frame_at)}` : s ? 'Run inference to begin' : undefined} />
          <StatCard label="Currently tracked" icon={Crosshair} hint="Tracking IDs seen within the last few seconds"
            value={int(s?.currently_tracked)} sub={s ? 'active track IDs' : undefined} />
          <StatCard label="Unique tracks" icon={Fingerprint} hint="Distinct tracking IDs (source + track) observed in the object inventory; image snapshots excluded"
            value={int(s?.unique_tracks)} sub={s ? 'distinct track IDs (temporary)' : undefined} />
          <StatCard label="Avg inference FPS" icon={Gauge} hint="1000 / mean inference time over the recent window"
            value={num(s?.avg_inference_fps)} sub={s ? (s.avg_inference_ms ? `last ${s.recent.length} frames` : 'No inference yet') : undefined} />
          <StatCard label="Avg latency" icon={Clock} hint="Mean model inference time per frame over the recent window"
            value={s?.avg_inference_ms != null ? <>{num(s.avg_inference_ms)}<span className="ml-1 text-sm font-normal text-ink-3">ms</span></> : '—'}
            sub={s ? (s.avg_inference_ms ? 'per frame' : 'No inference yet') : undefined} />
          <StatCard label="Avg confidence" icon={Activity} hint="Mean confidence of all detections this session"
            value={pct(s?.avg_confidence, 1)} sub={s ? (s.total_detections ? 'mean over all detections' : 'No detections yet') : undefined} />
          <StatCard label="Processing errors" icon={AlertOctagon} hint="Failed frames or jobs this session (decode errors, model unavailable, failed video jobs)"
            value={<span className={cx(s?.error_count ? 'text-bad' : '')}>{int(s?.error_count ?? (s ? 0 : null))}</span>}
            sub={s?.last_error ? `last ${time(s.last_error.at)}` : s ? 'none this session' : undefined} />
        </div>
      </div>

      {/* latency + components */}
      <div className="mt-4 grid grid-cols-[minmax(0,1fr)] gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <Panel
          title="Inference latency"
          icon={Activity}
          className="h-full"
          actions={
            latency.length ? (
              <span className="flex flex-wrap items-center gap-1.5">
                <Badge tone="accent">avg {num(s?.avg_inference_ms)} ms</Badge>
                <Badge tone="neutral">min {num(s?.min_inference_ms)}</Badge>
                <Badge tone="neutral">max {num(s?.max_inference_ms)}</Badge>
              </span>
            ) : (
              <Badge tone="neutral">ms / frame</Badge>
            )
          }
        >
          {latency.length ? (
            <>
              <div className="h-64">
                <ResponsiveContainer>
                  <LineChart data={latency} margin={{ top: 8, right: 28, bottom: 4, left: -8 }}>
                    <CartesianGrid stroke="var(--color-line)" vertical={false} />
                    <XAxis dataKey="seq" {...AXIS} minTickGap={32} label={{ value: 'frame #', position: 'insideBottomRight', offset: -2, fill: 'var(--color-ink-3)', fontSize: 11 }} />
                    <YAxis {...AXIS} width={52} unit=" ms" />
                    <Tooltip
                      {...tooltipStyle}
                      labelFormatter={(_, p) => (p?.[0] ? `Frame #${p[0].payload.seq} · ${p[0].payload.t} · ${p[0].payload.detections} detections` : '')}
                      formatter={(v) => [`${Number(v).toFixed(1)} ms`, 'Inference']}
                    />
                    {s?.avg_inference_ms != null && (
                      <ReferenceLine y={s.avg_inference_ms} stroke="var(--color-ink-3)" strokeDasharray="4 4" label={{ value: 'avg', position: 'right', fill: 'var(--color-ink-3)', fontSize: 11 }} />
                    )}
                    <Line type="linear" dataKey="ms" stroke="var(--color-accent)" strokeWidth={2} dot={false} activeDot={{ r: 4, strokeWidth: 2, stroke: 'var(--color-surface-1)' }} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className="mt-2 text-xs text-ink-3">
                Last {latency.length} of up to {s?.latency_window ?? latency.length} frames (rolling window, refreshed every 2 s). Model inference time only; decoding and network excluded.
              </p>
            </>
          ) : (
            <Empty icon={Activity} title="No inference yet">Run detection in Live Monitoring to see measured latency.</Empty>
          )}
        </Panel>

        <ComponentDistribution model={m} counts={s?.class_counts ?? []} />
      </div>

      <HealthOverview />
      <SiteTelemetry />

      {!s && !stats.error && (
        <p className="mt-4 text-center text-xs text-ink-3">
          <Boxes size={12} className="mr-1 inline" />
          Loading statistics…
        </p>
      )}
    </>
  )
}

import { Activity, BarChart3, Boxes, Cpu, Crosshair, Gauge, Layers, Radio, Server, Target } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Alert, Badge, Empty, Panel, PageHeader, StatCard, StatusDot } from '../components/ui'
import { usePolling } from '../hooks/usePolling'
import { duration, int, num, pct, time } from '../lib/format'
import { api } from '../services/api'

const AXIS = { stroke: 'var(--color-ink-3)', fontSize: 11, tickLine: false, axisLine: false } as const
const tooltipStyle = {
  contentStyle: { background: 'var(--color-surface-2)', border: '1px solid var(--color-line)', borderRadius: 6, fontSize: 12 },
  labelStyle: { color: 'var(--color-ink-2)' },
  itemStyle: { color: 'var(--color-ink)' },
  cursor: { stroke: 'var(--color-line)', fill: 'var(--color-surface-3)', fillOpacity: 0.4 },
}

export function Overview() {
  const health = usePolling(api.health, 3000)
  const model = usePolling(api.modelInfo, 10000)
  const stats = usePolling(api.stats, 2000)
  const s = stats.data
  const h = health.data

  const latency = (s?.recent ?? []).map((r) => ({ t: time(r.timestamp), ms: r.inference_ms, detections: r.detections }))
  const classes = (s?.class_counts ?? []).slice(0, 10)

  return (
    <>
      <PageHeader title="Overview" subtitle="System status and detection activity. All figures are measured from real inference." />

      {health.error && (
        <div className="mb-4">
          <Alert tone="bad" title="Backend unreachable">
            Start the API with <code className="font-mono">uvicorn app.main:app --port 8000</code> from <code className="font-mono">backend/</code>.
          </Alert>
        </div>
      )}
      {model.data?.fallback_active && (
        <div className="mb-4">
          <Alert tone="warn" title="Running the pretrained fallback — custom model unavailable">{model.data.fallback_reason}</Alert>
        </div>
      )}
      {model.data && !model.data.loaded && (
        <div className="mb-4">
          <Alert tone="warn" title="Model not loaded">{model.data.error}</Alert>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard
          label="System status"
          icon={Server}
          value={
            <span className="flex items-center gap-2 text-xl">
              <StatusDot tone={health.error ? 'bad' : h?.status === 'ok' ? 'good' : 'warn'} />
              {health.error ? 'Offline' : h?.status === 'ok' ? 'Operational' : h ? 'Degraded' : '—'}
            </span>
          }
          sub={h ? `Uptime ${duration(h.uptime_s)}` : undefined}
        />
        <StatCard
          label="Active source"
          icon={Radio}
          value={<span className="text-xl">{s?.active_sources.length ? s.active_sources[0] : 'None'}</span>}
          sub={s && s.active_sources.length > 1 ? `+${s.active_sources.length - 1} more` : 'Webcam, video or image'}
        />
        <StatCard label="Loaded model" icon={Layers} value={<span className="text-xl">{model.data?.model_name ?? '—'}</span>} sub={model.data ? `${model.data.model_type} · ${model.data.num_classes} classes` : undefined} />
        <StatCard label="Execution device" icon={Cpu} value={<span className="font-mono">{h?.device.toUpperCase() ?? '—'}</span>} sub={h ? `PyTorch ${h.torch_version}` : undefined} />
        <StatCard label="Total detections" icon={Target} value={int(s?.total_detections)} sub={s ? `${int(s.frames_processed)} frames processed` : undefined} />
        <StatCard label="Currently tracked" icon={Crosshair} value={int(s?.currently_tracked)} sub={s ? `${int(s.unique_tracks)} tracks observed` : undefined} />
        <StatCard label="Avg inference FPS" icon={Gauge} value={num(s?.avg_inference_fps)} sub={s?.avg_inference_ms ? `${num(s.avg_inference_ms)} ms / frame (last ${s.recent.length})` : 'No frames yet'} />
        <StatCard label="Avg confidence" icon={Activity} value={pct(s?.avg_confidence, 1)} sub="Across all detections" />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <Panel title="Inference latency (recent frames)" icon={Activity} actions={<Badge tone="neutral">ms / frame</Badge>}>
          {latency.length ? (
            <div className="h-64">
              <ResponsiveContainer>
                <LineChart data={latency} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
                  <CartesianGrid stroke="var(--color-line)" strokeDasharray="0" vertical={false} />
                  <XAxis dataKey="t" {...AXIS} minTickGap={40} />
                  <YAxis {...AXIS} width={48} />
                  <Tooltip {...tooltipStyle} formatter={(v) => [`${Number(v).toFixed(1)} ms`, 'Inference']} />
                  <Line type="monotone" dataKey="ms" stroke="var(--color-accent)" strokeWidth={2} dot={false} activeDot={{ r: 4, strokeWidth: 2, stroke: 'var(--color-surface-1)' }} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <Empty icon={Activity} title="No inference yet">Run detection in Live Monitoring to see measured latency.</Empty>
          )}
        </Panel>

        <Panel title="Detections by class" icon={BarChart3} actions={<Badge tone="neutral">top 10</Badge>}>
          {classes.length ? (
            <div className="h-64">
              <ResponsiveContainer>
                <BarChart data={classes} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 0 }} barCategoryGap={4}>
                  <CartesianGrid stroke="var(--color-line)" horizontal={false} />
                  <XAxis type="number" {...AXIS} allowDecimals={false} />
                  <YAxis type="category" dataKey="class_name" {...AXIS} width={96} />
                  <Tooltip {...tooltipStyle} formatter={(v) => [int(Number(v)), 'Detections']} />
                  <Bar dataKey="count" fill="var(--color-accent)" radius={[0, 4, 4, 0]} maxBarSize={16} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <Empty icon={Boxes} title="No detections yet" />
          )}
        </Panel>
      </div>
    </>
  )
}

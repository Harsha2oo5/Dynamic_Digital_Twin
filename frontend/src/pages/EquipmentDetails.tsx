import { ArrowLeft, Camera, ChartLine, HeartPulse, Radar, Radio, Wrench } from 'lucide-react'
import { Fragment, useMemo, useState } from 'react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { AnomalyStatusBadge, EvidenceBadge, HEALTH_COLOR, HealthBadge, HealthScore } from '../components/analytics'
import { ReadingValue, SensorStatusBadge, SeverityBadge, SourceBadge } from '../components/telemetry'
import { Alert, Empty, KV, PageHeader, Panel } from '../components/ui'
import { usePolling } from '../hooks/usePolling'
import { useTelemetry } from '../hooks/useTelemetry'
import { dateTime, pct, titleCase } from '../lib/format'
import { analytics } from '../services/analytics'
import { telemetry } from '../services/telemetry'
import type { TimeRange } from '../types/telemetry'
import { AnomalyDetail } from './AnomalyMonitoring'
import { selectCls } from './live/Controls'

const AXIS = { stroke: 'var(--color-ink-3)', fontSize: 11, tickLine: false, axisLine: false } as const
const TOOLTIP = {
  contentStyle: { background: 'var(--color-surface-2)', border: '1px solid var(--color-line)', borderRadius: 6, fontSize: 12 },
  labelStyle: { color: 'var(--color-ink-2)' },
  itemStyle: { color: 'var(--color-ink)' },
}

function TelemetryChart({ equipmentId, metrics }: { equipmentId: string; metrics: string[] }) {
  const { metrics: defs } = useTelemetry()
  const [metric, setMetric] = useState(metrics[0])
  const [range, setRange] = useState<TimeRange>('24h')
  const hist = usePolling(() => telemetry.history(metric, range, { equipment_id: equipmentId, max_points: 240 }), 30000)
  const s = hist.data?.series?.[0]
  const def = defs[metric]
  return (
    <Panel
      title="Historical telemetry"
      icon={ChartLine}
      actions={
        <>
          <select className={selectCls} value={metric} onChange={(e) => { setMetric(e.target.value); hist.refresh() }} aria-label="Metric">
            {metrics.map((m) => (
              <option key={m} value={m}>
                {defs[m]?.label ?? m}
              </option>
            ))}
          </select>
          <select className={selectCls} value={range} onChange={(e) => { setRange(e.target.value as TimeRange); hist.refresh() }} aria-label="Range">
            {(['1h', '6h', '24h', '7d'] as TimeRange[]).map((r) => (
              <option key={r}>{r}</option>
            ))}
          </select>
        </>
      }
    >
      {!s ? (
        <Empty icon={ChartLine} title="No stored readings for this metric and range" />
      ) : (
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={s.points} margin={{ top: 6, right: 8, bottom: 0, left: -12 }}>
              <CartesianGrid stroke="var(--color-line)" vertical={false} />
              <XAxis dataKey="t" {...AXIS} minTickGap={50} tickFormatter={(v) => new Date(v).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} />
              <YAxis {...AXIS} width={48} domain={['auto', 'auto']} />
              <Tooltip {...TOOLTIP} labelFormatter={(v) => new Date(String(v)).toLocaleString()} formatter={(v) => [typeof v === 'number' ? `${v.toFixed(2)} ${def?.unit ?? ''}` : '—', 'average']} />
              <Line type={def?.kind === 'state' ? 'stepAfter' : 'monotone'} dataKey="avg" stroke="var(--color-accent)" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </Panel>
  )
}

export function EquipmentDetails() {
  const id = decodeURIComponent(window.location.hash.split('/')[1] ?? '')
  const { metrics: defs, latest } = useTelemetry()
  const d = usePolling(() => analytics.details(id), 15000)
  const [open, setOpen] = useState<number | null>(null)
  const metrics = useMemo(() => [...new Set((d.data?.sensors ?? []).flatMap((s) => s.metrics))].filter((m) => defs[m]?.kind !== 'state'), [d.data, defs])
  const live = Object.values(latest).filter((r) => d.data?.equipment.sensor_ids.includes(r.sensor_id))
  const back = (
    <a href="#health" className="inline-flex items-center gap-1 text-sm text-ink-3 hover:text-ink">
      <ArrowLeft size={14} /> Equipment Health
    </a>
  )
  if (d.error) return <><PageHeader title={id || 'Equipment'} actions={back} /><Alert tone="bad">{d.error}</Alert></>
  if (!d.data) return <PageHeader title={id || 'Equipment'} subtitle="Loading…" actions={back} />
  const { equipment: eq, health: h } = d.data
  return (
    <>
      <PageHeader title={eq.name} subtitle={`${eq.equipment_id} · ${titleCase(eq.category)}${eq.zone ? ` · ${eq.zone}` : ''}`} actions={back} />
      <div className="space-y-4">
        <div className="grid gap-4 lg:grid-cols-3">
          <Panel title="Health" icon={HeartPulse} actions={h && <HealthBadge status={h.status} />}>
            {h ? (
              <>
                <HealthScore score={h.score} status={h.status} size="lg" />
                {h.unknown_reason && <p className="mt-2 text-xs text-ink-3">{h.unknown_reason}</p>}
                <ul className="mt-3 space-y-1 text-xs text-ink-2">
                  {h.components.map((c) => (
                    <li key={`${c.type}-${c.id}`} className="flex justify-between gap-2">
                      <span className="truncate">
                        {c.type}: {c.detail} ({c.severity}){c.active ? ' · ongoing' : ''}
                      </span>
                      <span className="tabular text-ink-3">−{c.penalty}</span>
                    </li>
                  ))}
                </ul>
                <div className="mt-3 text-xs text-ink-3">
                  coverage {pct(h.coverage)} · data quality {pct(h.data_quality)} (sensor availability, not equipment health)
                </div>
                {d.data.health_history.length > 1 && (
                  <div className="mt-3 h-24">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={d.data.health_history} margin={{ top: 4, right: 4, bottom: 0, left: -24 }}>
                        <YAxis {...AXIS} domain={[0, 100]} width={36} />
                        <XAxis dataKey="at" hide />
                        <Tooltip {...TOOLTIP} labelFormatter={(v) => new Date(String(v)).toLocaleString()} />
                        <Line type="stepAfter" dataKey="score" stroke={HEALTH_COLOR[h.status]} strokeWidth={1.5} dot={false} isAnimationActive={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </>
            ) : (
              <Empty icon={HeartPulse} title="No health data" />
            )}
          </Panel>
          <Panel title="Identity" icon={Radio} actions={<SourceBadge source={eq.data_source} />}>
            <KV
              rows={[
                ['Equipment ID', <span className="font-mono">{eq.equipment_id}</span>],
                ['Category', titleCase(eq.category)],
                ['Site / zone', `${eq.site_id}${eq.zone ? ` · ${eq.zone}` : ''}`],
                ['Sensors', `${eq.sensors_online} of ${eq.sensor_ids.length} online`],
                ['Last updated', dateTime(eq.last_updated)],
              ]}
            />
            {eq.description && <p className="mt-2 text-xs text-ink-3">{eq.description}</p>}
          </Panel>
          <Panel title="Visual information" icon={Camera}>
            {d.data.visual.links.length ? (
              <ul className="space-y-1 text-sm">
                {d.data.visual.links.map((v) => (
                  <li key={v.id} className="text-ink-2">
                    <span className="font-mono">{v.object_id}</span> {v.class_name && `· ${v.class_name}`} · linked {dateTime(v.linked_at)}
                    {v.linked_by && ` by ${v.linked_by}`}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-ink-3">No visual observation is linked to this equipment yet.</p>
            )}
            <p className="mt-2 text-xs text-ink-3">{d.data.visual.note} Links are made explicitly (API) and never inferred from class names or position.</p>
          </Panel>
        </div>

        <Panel title="Sensors and current readings" icon={Radio}>
          {d.data.sensors.length === 0 ? (
            <p className="text-sm text-ink-3">No sensors associated.</p>
          ) : (
            <>
              <div className="mb-3 flex flex-wrap gap-3 text-xs">
                {d.data.sensors.map((s) => (
                  <span key={s.sensor_id} className="inline-flex items-center gap-1.5 text-ink-3">
                    <span className="font-mono text-ink-2">{s.sensor_id}</span> {s.sensor_type} <SensorStatusBadge status={s.status} />
                  </span>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4 lg:grid-cols-6">
                {(live.length ? live : d.data.latest).map((r) => (
                  <div key={`${r.sensor_id}|${r.metric}`}>
                    <div className="truncate text-xs text-ink-3">{r.label}</div>
                    <ReadingValue reading={r} def={defs[r.metric]} compact />
                  </div>
                ))}
              </div>
            </>
          )}
        </Panel>

        {metrics.length > 0 && <TelemetryChart equipmentId={eq.equipment_id} metrics={metrics} />}

        <div className="grid gap-4 xl:grid-cols-2">
          <Panel title={`Anomalies (${d.data.anomalies.length})`} icon={Radar} bodyClassName="p-0">
            {d.data.anomalies.length === 0 ? (
              <Empty icon={Radar} title="No anomalies recorded" />
            ) : (
              <ul className="divide-y divide-line/60">
                {d.data.anomalies.map((a) => (
                  <Fragment key={a.id}>
                    <li onClick={() => setOpen(open === a.id ? null : a.id)} className="flex cursor-pointer flex-wrap items-center gap-2 px-4 py-2 text-xs hover:bg-surface-2">
                      <span className="text-ink-3">{dateTime(a.first_detected_at)}</span>
                      <SeverityBadge severity={a.severity} />
                      <span className="text-ink-2">{a.metric ?? 'multivariate'}</span>
                      <span className="text-ink-3">{a.detector}</span>
                      <span className="ml-auto">
                        <AnomalyStatusBadge status={a.status} active={a.condition_active} />
                      </span>
                    </li>
                    {open === a.id && <AnomalyDetail a={a} onChanged={d.refresh} />}
                  </Fragment>
                ))}
              </ul>
            )}
          </Panel>
          <div className="space-y-4">
            <Panel title="Maintenance recommendations" icon={Wrench}>
              {d.data.recommendations.length === 0 ? (
                <p className="text-sm text-ink-3">None. No degradation trend or rising anomaly frequency was found.</p>
              ) : (
                <ul className="space-y-3">
                  {d.data.recommendations.map((r) => (
                    <li key={r.id} className="text-sm">
                      <div className="flex items-center gap-2">
                        <EvidenceBadge level={r.evidence_level} />
                        <span className="text-xs text-ink-3">updated {dateTime(r.updated_at)}</span>
                      </div>
                      <p className="mt-1 text-ink">{r.condition}</p>
                      <p className="mt-0.5 text-ink-2">→ {r.action}</p>
                    </li>
                  ))}
                </ul>
              )}
            </Panel>
            <Panel title="Active threshold alerts">
              {d.data.threshold_alerts.length === 0 ? (
                <p className="text-sm text-ink-3">None.</p>
              ) : (
                <ul className="space-y-1 text-sm">
                  {d.data.threshold_alerts.map((a) => (
                    <li key={a.id} className="flex items-center gap-2">
                      <SeverityBadge severity={a.severity} /> <span className="text-ink-2">{a.message}</span>
                    </li>
                  ))}
                </ul>
              )}
            </Panel>
          </div>
        </div>
      </div>
    </>
  )
}

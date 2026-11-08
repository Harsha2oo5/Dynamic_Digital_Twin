import { BarChart3, LineChart as LineIcon } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { SourceBadge, TelemetryConnectionNotice } from '../components/telemetry'
import { Alert, Badge, cx, Empty, PageHeader, Panel } from '../components/ui'
import { usePolling } from '../hooks/usePolling'
import { useTelemetry } from '../hooks/useTelemetry'
import { colorFor } from '../lib/format'
import { telemetry } from '../services/telemetry'
import type { MetricDef, TimeRange } from '../types/telemetry'
import { selectCls } from './live/Controls'

const AXIS = { stroke: 'var(--color-ink-3)', fontSize: 11, tickLine: false, axisLine: false } as const
const TOOLTIP = {
  contentStyle: { background: 'var(--color-surface-2)', border: '1px solid var(--color-line)', borderRadius: 6, fontSize: 12 },
  labelStyle: { color: 'var(--color-ink-2)' },
  itemStyle: { color: 'var(--color-ink)' },
}
const RANGES: [TimeRange, string][] = [
  ['15m', 'Last 15 minutes'],
  ['1h', 'Last hour'],
  ['6h', 'Last 6 hours'],
  ['24h', 'Last 24 hours'],
  ['7d', 'Last 7 days'],
]
const PREFERRED = ['battery_voltage', 'battery_temperature', 'state_of_charge', 'battery_current', 'power_consumption', 'internal_temperature', 'humidity', 'ambient_temperature', 'temperature', 'output_voltage']

function tickFormat(range: TimeRange) {
  return (iso: string) => {
    const d = new Date(iso)
    return range === '7d' ? d.toLocaleDateString([], { weekday: 'short', hour: '2-digit' }) : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }
}

function MetricChart({ metric, def, range }: { metric: string; def?: MetricDef; range: TimeRange }) {
  const refresh = range === '15m' || range === '1h' ? 10000 : 60000
  const hist = usePolling(() => telemetry.history(metric, range, { max_points: 240 }), refresh)
  const series = useMemo(() => hist.data?.series ?? [], [hist.data])
  const data = useMemo(() => {
    const rows = new Map<string, Record<string, number | string>>()
    for (const s of series)
      for (const p of s.points) {
        const row = rows.get(p.t) ?? { t: p.t }
        row[s.sensor_id] = p.avg
        rows.set(p.t, row)
      }
    return [...rows.values()].sort((a, b) => String(a.t).localeCompare(String(b.t)))
  }, [series])
  if (hist.loading && !hist.data) return null
  if (!series.length) return null // no data for this metric in this range: no chart
  const unit = def?.unit === 'state' || def?.unit === 'ratio' ? '' : def?.unit ?? ''
  const sources = [...new Set(series.map((s) => s.data_source))]
  return (
    <Panel title={`${def?.label ?? metric}${unit ? ` (${unit})` : ''}`} icon={LineIcon} actions={sources.map((s) => <SourceBadge key={s} source={s} />)}>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
            <CartesianGrid stroke="var(--color-line)" vertical={false} />
            <XAxis dataKey="t" {...AXIS} tickFormatter={tickFormat(range)} minTickGap={40} />
            <YAxis {...AXIS} width={48} domain={['auto', 'auto']} />
            <Tooltip {...TOOLTIP} labelFormatter={(v) => new Date(String(v)).toLocaleString()} formatter={(v, name) => [typeof v === 'number' ? `${v.toFixed(2)} ${unit}` : '—', String(name)]} />
            {series.map((s, i) => (
              <Line
                key={s.sensor_id}
                type={def?.kind === 'state' ? 'stepAfter' : 'monotone'}
                dataKey={s.sensor_id}
                stroke={i === 0 ? 'var(--color-accent)' : colorFor(i + 3)}
                strokeWidth={2}
                dot={false}
                connectNulls={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-3">
        {series.map((s, i) => (
          <span key={s.sensor_id} className="inline-flex items-center gap-1.5">
            <span className="inline-block h-0.5 w-3" style={{ background: i === 0 ? 'var(--color-accent)' : colorFor(i + 3) }} />
            <span className="font-mono">{s.sensor_id}</span>
            {s.equipment_id && <span>· {s.equipment_id}</span>}
          </span>
        ))}
        <span>averages per {series[0].bucket_s >= 60 ? `${Math.round(series[0].bucket_s / 60)} min` : `${series[0].bucket_s} s`} · gaps = no data</span>
      </div>
    </Panel>
  )
}

function AvailabilityChart({ range }: { range: TimeRange }) {
  const av = usePolling(() => telemetry.availability(range), 30000)
  const rows = (av.data?.sensors ?? []).filter((s) => s.online_share != null).map((s) => ({ sensor: s.sensor_id, online: Math.round((s.online_share ?? 0) * 1000) / 10 }))
  if (!rows.length) return null
  return (
    <Panel title="Sensor availability (share of time online)" icon={BarChart3}>
      <div style={{ height: Math.max(160, rows.length * 22) }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
            <CartesianGrid stroke="var(--color-line)" horizontal={false} />
            <XAxis type="number" domain={[0, 100]} {...AXIS} unit="%" />
            <YAxis type="category" dataKey="sensor" {...AXIS} width={130} />
            <Tooltip {...TOOLTIP} cursor={{ fill: 'var(--color-surface-3)', fillOpacity: 0.4 }} formatter={(v) => [`${v ?? '—'}%`, 'online']} />
            <Bar dataKey="online" fill="var(--color-accent)" radius={[0, 3, 3, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-2 text-xs text-ink-3">From recorded connection-state changes; sensors without any recorded state are omitted.</p>
    </Panel>
  )
}

export function TelemetryAnalytics() {
  const { sensors, metrics, connection, error } = useTelemetry()
  const [range, setRange] = useState<TimeRange>('1h')
  const available = useMemo(() => {
    const set = new Set(Object.values(sensors).flatMap((s) => s.metrics))
    return [...set].sort((a, b) => {
      const ia = PREFERRED.indexOf(a)
      const ib = PREFERRED.indexOf(b)
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b)
    })
  }, [sensors])
  const [chosen, setChosen] = useState<string[] | null>(null)
  const selected = chosen ?? available.filter((m) => PREFERRED.includes(m)).slice(0, 6)
  const toggle = (m: string) => setChosen(selected.includes(m) ? selected.filter((x) => x !== m) : [...selected, m])
  return (
    <>
      <PageHeader
        title="Telemetry Analytics"
        subtitle="Historical measurements. Charts appear only for metrics with stored data in the selected range."
        actions={
          <select className={selectCls} value={range} onChange={(e) => setRange(e.target.value as TimeRange)} aria-label="Time range">
            {RANGES.map(([v, l]) => (
              <option key={v} value={v}>
                {l}
              </option>
            ))}
          </select>
        }
      />
      <div className="space-y-4">
        <TelemetryConnectionNotice connection={connection} error={error} />
        {available.length === 0 ? (
          <Panel>
            <Empty icon={LineIcon} title="No sensors registered">
              Register sensors or start the simulator (Sensor Management) to collect telemetry.
            </Empty>
          </Panel>
        ) : (
          <>
            <Panel title="Metrics" bodyClassName="flex flex-wrap gap-1.5 p-3">
              {available.map((m) => (
                <button
                  key={m}
                  onClick={() => toggle(m)}
                  className={cx('rounded border px-2 py-1 text-xs transition-colors', selected.includes(m) ? 'border-accent/50 bg-accent/15 text-accent-strong' : 'border-line text-ink-3 hover:text-ink')}
                >
                  {metrics[m]?.label ?? m}
                </button>
              ))}
            </Panel>
            {selected.length === 0 && <Alert tone="info">Select one or more metrics.</Alert>}
            <div className="grid gap-4 xl:grid-cols-2">
              {selected.map((m) => (
                <MetricChart key={`${m}-${range}`} metric={m} def={metrics[m]} range={range} />
              ))}
            </div>
            <AvailabilityChart key={range} range={range} />
            <p className="text-xs text-ink-3">
              <Badge>note</Badge> A selected metric without a chart had no good readings in this range; missing, invalid and out-of-order readings are
              never plotted.
            </p>
          </>
        )}
      </div>
    </>
  )
}

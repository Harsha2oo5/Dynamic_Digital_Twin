import { BarChart3, ClipboardCheck, RefreshCw, Wrench } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { detailsHref, EvidenceBadge, OperatorAction } from '../components/analytics'
import { SourceBadge, TelemetryConnectionNotice } from '../components/telemetry'
import { Alert, Badge, Button, Empty, PageHeader, Panel } from '../components/ui'
import { usePolling } from '../hooks/usePolling'
import { useTelemetry } from '../hooks/useTelemetry'
import { colorFor, dateTime, titleCase } from '../lib/format'
import { analytics } from '../services/analytics'
import type { Recommendation } from '../types/analytics'

const AXIS = { stroke: 'var(--color-ink-3)', fontSize: 11, tickLine: false, axisLine: false } as const
const TOOLTIP = {
  contentStyle: { background: 'var(--color-surface-2)', border: '1px solid var(--color-line)', borderRadius: 6, fontSize: 12 },
  labelStyle: { color: 'var(--color-ink-2)' },
  itemStyle: { color: 'var(--color-ink)' },
}

function RecommendationCard({ r, onChanged }: { r: Recommendation; onChanged: () => void }) {
  const ev = r.evidence
  const series = (ev.series ?? []).map((p) => ({ t: p.t, v: p.v }))
  return (
    <Panel
      title={
        <span className="flex flex-wrap items-center gap-2">
          <a className="font-mono text-accent-strong hover:underline" href={detailsHref(r.equipment_id)}>
            {r.equipment_id}
          </a>
          <span className="text-ink-3">{titleCase(r.category)}</span>
        </span>
      }
      actions={
        <>
          <EvidenceBadge level={r.evidence_level} />
          <SourceBadge source={r.data_source} />
        </>
      }
    >
      <p className="text-sm text-ink">{r.condition}</p>
      <div className="mt-2 flex items-start gap-2 rounded-md border border-line bg-surface-2 px-3 py-2 text-sm text-ink-2">
        <Wrench size={15} className="mt-0.5 shrink-0 text-ink-3" />
        <span>
          <span className="text-ink-3">Recommended inspection: </span>
          {r.action}
        </span>
      </div>
      {series.length > 1 && (
        <div className="mt-3 h-36">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={series} margin={{ top: 6, right: 8, bottom: 0, left: -12 }}>
              <CartesianGrid stroke="var(--color-line)" vertical={false} />
              <XAxis dataKey="t" {...AXIS} minTickGap={50} tickFormatter={(v) => new Date(v).toLocaleDateString([], { month: 'short', day: 'numeric' })} />
              <YAxis {...AXIS} width={44} domain={['auto', 'auto']} />
              <Tooltip {...TOOLTIP} labelFormatter={(v) => new Date(String(v)).toLocaleString()} formatter={(v) => [typeof v === 'number' ? v.toFixed(3) : '—', 'hourly median']} />
              <Line type="monotone" dataKey="v" stroke="var(--color-accent)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-3">
        {ev.method && <span>{ev.method}</span>}
        {ev.slope_per_day != null && (
          <span>
            slope {ev.slope_per_day.toFixed(3)}/day (95% CI {ev.ci_low?.toFixed(3)} to {ev.ci_high?.toFixed(3)})
          </span>
        )}
        {ev.tau != null && <span>Kendall τ {ev.tau.toFixed(2)}, p {ev.p_value?.toExponential(1)}</span>}
        {ev.span_h != null && <span>{(ev.span_h / 24).toFixed(1)} days, {ev.n} hourly points</span>}
        {ev.related_anomalies && ev.related_anomalies.length > 0 && <span>{ev.related_anomalies.length} related anomalies</span>}
        <span>updated {dateTime(r.updated_at)}</span>
      </div>
      <div className="mt-3">
        {r.acknowledged ? (
          <Badge tone="good">acknowledged by {r.acknowledged_by}</Badge>
        ) : (
          <OperatorAction label="Acknowledge" withNote={false} onSubmit={async (by) => { await analytics.ackRecommendation(r.id, by); onChanged() }} />
        )}
      </div>
    </Panel>
  )
}

function AnomalyPattern() {
  const since = useMemo(() => new Date(Date.now() - 7 * 86400e3).toISOString(), [])
  const page = usePolling(() => analytics.anomalies({ start: since, limit: 500 }), 60000)
  const { data, equipment } = useMemo(() => {
    const byDay = new Map<string, Record<string, number | string>>()
    const eqs = new Set<string>()
    for (let i = 6; i >= 0; i--) {
      const d = new Date(Date.now() - i * 86400e3).toISOString().slice(0, 10)
      byDay.set(d, { day: d })
    }
    for (const a of page.data?.items ?? []) {
      const d = a.first_detected_at.slice(0, 10)
      const row = byDay.get(d)
      if (!row) continue
      const k = a.equipment_id ?? 'site'
      eqs.add(k)
      row[k] = (Number(row[k]) || 0) + 1
    }
    return { data: [...byDay.values()], equipment: [...eqs].sort() }
  }, [page.data])
  if (!equipment.length) return null
  return (
    <Panel title="Anomalies per day (last 7 days)" icon={BarChart3}>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -16 }}>
            <CartesianGrid stroke="var(--color-line)" vertical={false} />
            <XAxis dataKey="day" {...AXIS} tickFormatter={(v) => new Date(v).toLocaleDateString([], { weekday: 'short' })} />
            <YAxis {...AXIS} allowDecimals={false} />
            <Tooltip {...TOOLTIP} cursor={{ fill: 'var(--color-surface-3)', fillOpacity: 0.4 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {equipment.map((e, i) => (
              <Bar key={e} dataKey={e} stackId="a" fill={i === 0 ? 'var(--color-accent)' : colorFor(i + 2)} isAnimationActive={false} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}

export function PredictiveMaintenance() {
  const { connection, error, analyticsVersion } = useTelemetry()
  const recs = usePolling(() => analytics.recommendations({ status: 'active', limit: 200 }), 30000)
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState<string | null>(null)
  useEffect(() => {
    recs.refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analyticsVersion])
  const order = { high: 0, medium: 1, low: 2 }
  const items = [...(recs.data?.items ?? [])].sort((a, b) => order[a.evidence_level] - order[b.evidence_level])
  const run = async () => {
    setRunning(true)
    try {
      await analytics.runMaintenance()
      setRunError(null)
      recs.refresh()
    } catch (e) {
      setRunError(e instanceof Error ? e.message : String(e))
    } finally {
      setRunning(false)
    }
  }
  return (
    <>
      <PageHeader
        title="Predictive Maintenance"
        subtitle="Equipment with degradation trends or recurring anomalies, with the evidence behind each recommendation."
        actions={
          <Button icon={RefreshCw} disabled={running} onClick={run}>
            {running ? 'Analysing…' : 'Analyse now'}
          </Button>
        }
      />
      <div className="space-y-4">
        <TelemetryConnectionNotice connection={connection} error={error} />
        <Alert tone="info" title="Trend-based indications, not failure predictions">
          Recommendations come from robust trend tests (Theil–Sen slope with a Kendall test) on up to 7 days of hourly data, and from rising anomaly
          frequency. There is no historical failure data, so no failure dates or failure probabilities are given. The evidence level says how strongly the
          data supports the trend.
        </Alert>
        {(recs.error || runError) && <Alert tone="bad">{recs.error ?? runError}</Alert>}
        {recs.data && items.length === 0 ? (
          <Panel>
            <Empty icon={ClipboardCheck} title="No equipment currently needs inspection">
              Trends need at least 2 days of history; recommendations appear when a degradation trend or rising anomaly frequency is found.
            </Empty>
          </Panel>
        ) : (
          <div className="grid gap-4 xl:grid-cols-2">
            {items.map((r) => (
              <RecommendationCard key={r.id} r={r} onChanged={recs.refresh} />
            ))}
          </div>
        )}
        <AnomalyPattern />
      </div>
    </>
  )
}

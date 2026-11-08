import { HeartPulse, Info } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { detailsHref, HEALTH_COLOR, HealthBadge, HealthScore, Sparkline } from '../components/analytics'
import { SourceBadge, TelemetryConnectionNotice } from '../components/telemetry'
import { Alert, Empty, PageHeader, Panel, StatCard } from '../components/ui'
import { usePolling } from '../hooks/usePolling'
import { useTelemetry } from '../hooks/useTelemetry'
import { pct, titleCase } from '../lib/format'
import { analytics } from '../services/analytics'
import type { EquipmentHealth as Health, HealthStatus } from '../types/analytics'
import { selectCls } from './live/Controls'

const STATUSES: HealthStatus[] = ['critical', 'degraded', 'warning', 'healthy', 'unknown']

function HealthCard({ h }: { h: Health }) {
  const hist = usePolling(() => analytics.healthHistory(h.equipment_id, '24h'), 60000)
  return (
    <a href={detailsHref(h.equipment_id)} className="block rounded-lg border border-line bg-surface-1 p-4 transition-colors hover:border-ink-3/60">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold text-ink">{h.name}</div>
          <div className="mt-0.5 flex flex-wrap items-center gap-1.5 text-xs text-ink-3">
            <span className="font-mono">{h.equipment_id}</span> · {titleCase(h.category)}
          </div>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1">
          <HealthBadge status={h.status} />
          <SourceBadge source={h.data_source} />
        </div>
      </div>
      <div className="mt-3">
        <HealthScore score={h.score} status={h.status} />
      </div>
      {h.status === 'unknown' ? (
        <p className="mt-2 text-xs text-ink-3">{h.unknown_reason}</p>
      ) : h.components.length ? (
        <ul className="mt-2 space-y-0.5 text-xs text-ink-2">
          {h.components.slice(0, 3).map((c) => (
            <li key={`${c.type}-${c.id}`} className="flex justify-between gap-2">
              <span className="truncate">
                {c.type}: {c.detail} ({c.severity}){c.active ? ' · ongoing' : ''}
              </span>
              <span className="tabular shrink-0 text-ink-3">−{c.penalty}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-2 text-xs text-ink-3">No recent anomalies, alerts or degradation trends.</p>
      )}
      <div className="mt-3 grid grid-cols-3 gap-2 text-xs text-ink-3">
        <div title="Share of configured metrics with fresh, warmed-up data">
          coverage <div className="tabular text-ink-2">{pct(h.coverage)}</div>
        </div>
        <div title="Sensor availability over 24 h; separate from equipment health">
          data quality <div className="tabular text-ink-2">{pct(h.data_quality)}</div>
        </div>
        <div>
          sensors <div className="tabular text-ink-2">{h.sensors_online}/{h.sensors} online</div>
        </div>
      </div>
      <div className="mt-2">
        <Sparkline points={(hist.data ?? []).map((p) => ({ v: p.score }))} color={HEALTH_COLOR[h.status]} domain={[0, 100]} />
      </div>
    </a>
  )
}

export function EquipmentHealth() {
  const { connection, error, analyticsVersion } = useTelemetry()
  const list = usePolling(() => analytics.healthList(), 15000)
  const summary = usePolling(analytics.summary, 15000)
  const [status, setStatus] = useState('')
  useEffect(() => {
    list.refresh()
    summary.refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analyticsVersion])
  const rows = useMemo(
    () => (list.data ?? []).filter((h) => !status || h.status === status).sort((a, b) => STATUSES.indexOf(a.status) - STATUSES.indexOf(b.status)),
    [list.data, status],
  )
  const s = summary.data
  return (
    <>
      <PageHeader
        title="Equipment Health"
        subtitle="An explainable 0–100 index of recent abnormal behaviour per equipment item. It is not a probability of failure."
        actions={
          <select className={selectCls} value={status} onChange={(e) => setStatus(e.target.value)} aria-label="Filter by status">
            <option value="">All statuses</option>
            {STATUSES.map((x) => (
              <option key={x}>{x}</option>
            ))}
          </select>
        }
      />
      <div className="space-y-4">
        <TelemetryConnectionNotice connection={connection} error={error} />
        {(list.error || summary.error) && <Alert tone="bad" title="Analytics unavailable">{list.error ?? summary.error}</Alert>}
        {s && (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-6">
            <StatCard label="Healthy" value={<span className="text-good">{s.healthy}</span>} sub={`of ${s.equipment}`} />
            <StatCard label="Warning" value={<span className="text-warn">{s.warning}</span>} />
            <StatCard label="Degraded" value={<span style={{ color: HEALTH_COLOR.degraded }}>{s.degraded}</span>} />
            <StatCard label="Critical" value={<span className="text-bad">{s.critical}</span>} />
            <StatCard label="Unknown" value={s.unknown} sub="not enough data" />
            <StatCard label="Active anomalies" value={s.active_anomalies} sub={`${s.active_recommendations} maintenance items`} />
          </div>
        )}
        {list.data && rows.length === 0 ? (
          <Panel>
            <Empty icon={HeartPulse} title="No equipment to show">
              Register equipment with sensors (or start the simulator) to see health.
            </Empty>
          </Panel>
        ) : (
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {rows.map((h) => (
              <HealthCard key={h.equipment_id} h={h} />
            ))}
          </div>
        )}
        <Panel title="How the score is calculated" icon={Info} bodyClassName="p-4 text-xs leading-relaxed text-ink-3">
          Start at 100 and subtract penalties: each anomaly of the last 24 h (warning 12, critical 30, halving every 6 h after it ends, plus duration and
          recurrence), each active threshold alert (warning 8, critical 20; sensor-communication alerts excluded) and each active maintenance trend
          (medium 5, high 10). Bands: healthy ≥ 85, warning ≥ 65, degraded ≥ 40, otherwise critical. The status is <b>unknown</b> when fewer than half
          of the equipment's monitored metrics have fresh, warmed-up data. Data quality (sensor availability) is shown separately and never lowers the
          score. All weights are in <span className="font-mono">configs/analytics/anomaly.yaml</span>.
        </Panel>
      </div>
    </>
  )
}

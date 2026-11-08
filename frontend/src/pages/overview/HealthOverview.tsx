import { HeartPulse, Radar, Wrench } from 'lucide-react'
import { useEffect } from 'react'
import { detailsHref, HEALTH_COLOR } from '../../components/analytics'
import { SeverityBadge } from '../../components/telemetry'
import { Badge, Empty, Panel, StatCard } from '../../components/ui'
import { usePolling } from '../../hooks/usePolling'
import { useTelemetry } from '../../hooks/useTelemetry'
import { dateTime } from '../../lib/format'
import { analytics } from '../../services/analytics'

/** Equipment health block of the Overview page. */
export function HealthOverview() {
  const { analyticsVersion } = useTelemetry()
  const summary = usePolling(analytics.summary, 15000)
  const recent = usePolling(() => analytics.anomalies({ limit: 8 }), 15000)
  useEffect(() => {
    summary.refresh()
    recent.refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analyticsVersion])
  const s = summary.data
  if (!s) return null
  return (
    <section className="mt-6">
      <div className="mb-3 flex items-center gap-2">
        <h2 className="text-sm font-semibold tracking-wide text-ink">Equipment health</h2>
        <span className="text-xs text-ink-3">index of recent abnormal behaviour, not a failure probability</span>
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <StatCard label="Overall" icon={HeartPulse} value={s.mean_score == null ? '—' : `${Math.round(s.mean_score)}`} sub={`mean of ${s.equipment - s.unknown} scored items`} />
        <StatCard label="Healthy" value={<span className="text-good">{s.healthy}</span>} sub={`of ${s.equipment}`} />
        <StatCard label="Warning" value={<span className="text-warn">{s.warning}</span>} />
        <StatCard label="Degraded" value={<span style={{ color: HEALTH_COLOR.degraded }}>{s.degraded}</span>} />
        <StatCard label="Critical" value={<span className="text-bad">{s.critical}</span>} sub={`${s.unknown} unknown`} />
        <StatCard label="Active anomalies" icon={Radar} value={s.active_anomalies} sub={`${s.open_anomalies} not yet acknowledged`} />
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel title="Recent anomaly activity" icon={Radar} bodyClassName="p-0">
          {!recent.data?.items.length ? (
            <Empty icon={Radar} title="No anomalies recorded" />
          ) : (
            <ul className="divide-y divide-line/60">
              {recent.data.items.map((a) => (
                <li key={a.id} className="flex flex-wrap items-center gap-2 px-4 py-2 text-xs">
                  <span className="text-ink-3">{dateTime(a.first_detected_at)}</span>
                  <SeverityBadge severity={a.severity} />
                  <a className="font-mono text-accent-strong hover:underline" href={detailsHref(a.equipment_id ?? '')}>
                    {a.equipment_id ?? 'site'}
                  </a>
                  <span className="text-ink-2">{a.metric ?? 'multivariate'}</span>
                  {a.condition_active && <Badge tone="bad">ongoing</Badge>}
                  {a.data_source === 'simulated' && <Badge tone="accent">sim</Badge>}
                </li>
              ))}
            </ul>
          )}
        </Panel>
        <Panel title="Equipment requiring inspection" icon={Wrench}>
          {s.requiring_inspection.length === 0 ? (
            <p className="text-sm text-ink-3">None: no degraded or critical equipment and no medium/high-evidence maintenance trend.</p>
          ) : (
            <ul className="flex flex-wrap gap-2">
              {s.requiring_inspection.map((id) => (
                <li key={id}>
                  <a href={detailsHref(id)} className="rounded border border-warn/40 bg-warn/10 px-2 py-1 font-mono text-xs text-warn hover:bg-warn/20">
                    {id}
                  </a>
                </li>
              ))}
            </ul>
          )}
          <p className="mt-3 text-xs text-ink-3">{s.active_recommendations} active maintenance recommendation(s). See Predictive Maintenance for the evidence.</p>
        </Panel>
      </div>
    </section>
  )
}

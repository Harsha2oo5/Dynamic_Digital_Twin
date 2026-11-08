import { Activity, BellRing, Plug, Radio, Server, Thermometer, Zap } from 'lucide-react'
import { formatReading, SensorStatusBadge, SourceBadge } from '../../components/telemetry'
import { Alert, Badge, Empty, Panel, StatCard, StatusDot } from '../../components/ui'
import { useTelemetry } from '../../hooks/useTelemetry'
import { ago, num, time } from '../../lib/format'

/** Site telemetry block of the Overview page. Only shows numbers that sensors (or the labelled simulator) supplied. */
export function SiteTelemetry() {
  const { summary: t, summaryError, sensors, activity, metrics, connection } = useTelemetry()
  if (summaryError && !t) {
    return (
      <Panel title="Site telemetry" icon={Activity} className="mt-4">
        <Alert tone="warn" title="Telemetry unavailable">
          {summaryError}. Detection features are not affected.
        </Alert>
      </Panel>
    )
  }
  if (!t) return null
  const list = Object.values(sensors).sort((a, b) => a.sensor_id.localeCompare(b.sensor_id))
  const temp = t.site_temperature
  const simulatedOnly = t.sensors_total > 0 && t.sensors_physical === 0
  return (
    <section className="mt-6">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h2 className="text-sm font-semibold tracking-wide text-ink">Site telemetry · {t.site_name}</h2>
        {t.data_sources.map((s) => (
          <SourceBadge key={s} source={s} />
        ))}
        <span className="ml-auto flex items-center gap-1.5 text-xs text-ink-3">
          <StatusDot tone={connection === 'open' ? 'good' : connection === 'unavailable' ? 'bad' : 'warn'} />
          live updates {connection === 'open' ? 'connected' : connection}
        </span>
      </div>
      {simulatedOnly && (
        <div className="mb-3">
          <Alert tone="info" title="All telemetry on this site is simulated">
            No physical sensors are registered. The values below come from the clearly labelled simulator, not from hardware.
          </Alert>
        </div>
      )}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <StatCard label="Sensors" icon={Radio} value={t.sensors_total} sub={`${t.sensors_physical} physical · ${t.sensors_simulated} simulated`} />
        <StatCard
          label="Online / offline"
          icon={Plug}
          value={
            <span>
              <span className="text-good">{t.sensors_online}</span>
              <span className="text-ink-3"> / </span>
              <span className={t.sensors_offline ? 'text-bad' : ''}>{t.sensors_offline}</span>
            </span>
          }
          sub={`${t.sensors_stale} stale · ${t.sensors_unknown} never reported`}
          hint="Online = reported within its expected interval; stale = late; offline = silent for too long"
        />
        <StatCard label="Monitored equipment" icon={Server} value={t.equipment_monitored} sub={`of ${t.equipment_total} registered`} />
        <StatCard
          label="Site temperature"
          icon={Thermometer}
          value={temp ? formatReading(temp, metrics[temp.metric]) : '—'}
          sub={temp ? `${temp.source === 'ambient' ? 'ambient' : 'cabinet'} · ${temp.current ? ago(temp.age_s) : 'last known, sensor not current'}` : 'no temperature sensor'}
        />
        <StatCard
          label="Site power"
          icon={Zap}
          value={t.site_power_w != null ? `${num(t.site_power_w / 1000, 2)} kW` : '—'}
          sub={
            t.site_power_avg_1h_w != null
              ? `1 h avg ${num(t.site_power_avg_1h_w / 1000, 2)} kW${t.equipment_load_w != null ? ` · radios+router ${num(t.equipment_load_w / 1000, 2)} kW` : ''}`
              : t.equipment_load_w != null
                ? `radios+router ${num(t.equipment_load_w / 1000, 2)} kW`
                : 'no power metering'
          }
          hint="Site meter reading; equipment load = online radio unit and router power sensors only"
        />
        <StatCard
          label="Active alerts"
          icon={BellRing}
          value={<span className={t.alerts_critical ? 'text-bad' : t.alerts_active ? 'text-warn' : ''}>{t.alerts_active}</span>}
          sub={`${t.alerts_critical} critical · ${t.alerts_unacknowledged} unacknowledged`}
        />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel title="Sensor communication" icon={Plug} bodyClassName="p-3">
          {list.length === 0 ? (
            <Empty icon={Radio} title="No sensors registered">
              Register sensors or start the simulator on the Sensor Management page.
            </Empty>
          ) : (
            <div className="grid gap-x-4 gap-y-1.5 sm:grid-cols-2">
              {list.map((s) => (
                <div key={s.sensor_id} className="flex items-center justify-between gap-2 text-xs">
                  <span className="truncate font-mono text-ink-2">{s.sensor_id}</span>
                  <span className="flex shrink-0 items-center gap-2 text-ink-3">
                    {s.last_seen_at ? ago((Date.now() - new Date(s.last_seen_at).getTime()) / 1000) : 'never'}
                    <SensorStatusBadge status={s.status} />
                  </span>
                </div>
              ))}
            </div>
          )}
        </Panel>
        <Panel
          title="Recent telemetry activity"
          icon={Activity}
          actions={<span className="text-xs text-ink-3">{Math.round(t.readings_per_minute)} readings/min</span>}
          bodyClassName="p-0"
        >
          {activity.length === 0 ? (
            <Empty icon={Activity} title="Waiting for readings" />
          ) : (
            <ul className="max-h-72 divide-y divide-line/60 overflow-y-auto">
              {activity.slice(0, 20).map((r, i) => (
                <li key={`${r.sensor_id}-${r.metric}-${r.ts}-${i}`} className="flex items-center gap-3 px-4 py-1.5 text-xs">
                  <span className="tabular w-16 shrink-0 text-ink-3">{time(r.ts)}</span>
                  <span className="w-28 shrink-0 truncate font-mono text-ink-3">{r.sensor_id}</span>
                  <span className="min-w-0 flex-1 truncate text-ink-2">{r.label}</span>
                  <span className="tabular shrink-0 text-ink">{formatReading(r, metrics[r.metric])}</span>
                  {r.data_source === 'simulated' && <Badge tone="accent">sim</Badge>}
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </section>
  )
}

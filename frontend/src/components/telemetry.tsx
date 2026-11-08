import { FlaskConical, Radio } from 'lucide-react'
import type { MetricDef, Reading, SensorStatus, Severity, DataSource } from '../types/telemetry'
import { ago } from '../lib/format'
import { Alert, Badge, cx } from './ui'
import type { Connection } from '../hooks/useTelemetry'

/** Every simulated value carries this badge, everywhere it is shown. */
export function SourceBadge({ source }: { source: DataSource }) {
  return source === 'simulated' ? (
    <Badge tone="accent">
      <FlaskConical size={11} /> SIMULATED
    </Badge>
  ) : (
    <Badge>
      <Radio size={11} /> PHYSICAL
    </Badge>
  )
}

const statusTone = { online: 'good', stale: 'warn', offline: 'bad', unknown: 'neutral', disabled: 'neutral' } as const

export function SensorStatusBadge({ status }: { status: SensorStatus }) {
  return <Badge tone={statusTone[status]}>{status}</Badge>
}

export function SeverityBadge({ severity }: { severity: Severity }) {
  return <Badge tone={severity === 'critical' ? 'bad' : 'warn'}>{severity}</Badge>
}

export function formatReading(r: Reading, def?: MetricDef): string {
  if (r.value == null) return r.quality === 'missing' ? 'no value' : 'invalid'
  if (r.state_label) return r.state_label
  if (def?.kind === 'state' && def.states[String(r.value)]) return def.states[String(r.value)]
  const digits = Math.abs(r.value) >= 100 ? 0 : Math.abs(r.value) >= 10 ? 1 : 2
  const unit = r.unit === 'state' || r.unit === 'ratio' ? '' : ` ${r.unit}`
  return `${r.value.toFixed(digits)}${unit}`
}

/** A reading with its freshness: stale/offline values are dimmed and labelled "last known". */
export function ReadingValue({ reading, def, compact }: { reading: Reading; def?: MetricDef; compact?: boolean }) {
  const ageS = (Date.now() - new Date(reading.ts).getTime()) / 1000
  const notCurrent = !reading.current
  return (
    <div className="min-w-0">
      <div className={cx('tabular truncate font-semibold', compact ? 'text-sm' : 'text-lg', notCurrent ? 'text-ink-3' : 'text-ink')}>
        {formatReading(reading, def)}
      </div>
      <div className="truncate text-xs text-ink-3" title={reading.quality_note ?? undefined}>
        {notCurrent ? (reading.quality === 'good' ? `last known · ${ago(ageS)}` : reading.quality.replace('_', ' ')) : ago(ageS)}
      </div>
    </div>
  )
}

export function TelemetryConnectionNotice({ connection, error }: { connection: Connection; error: string | null }) {
  if (connection === 'open' || connection === 'connecting') return null
  if (connection === 'unavailable')
    return (
      <Alert tone="bad" title="Telemetry unavailable">
        {error ?? 'The telemetry service is not running.'} Vision features are not affected.
      </Alert>
    )
  return (
    <Alert tone="warn" title="Live telemetry connection lost">
      Reconnecting… Values below are the last received ones.
    </Alert>
  )
}

export const CATEGORY_ORDER = ['site', 'equipment_cabinet', 'power_supply_unit', 'battery', 'router', 'radio_unit', 'antenna', 'cable']

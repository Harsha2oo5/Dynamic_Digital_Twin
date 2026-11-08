import { BellRing, Check, History, ShieldCheck } from 'lucide-react'
import { useState } from 'react'
import { SeverityBadge, SourceBadge, TelemetryConnectionNotice } from '../components/telemetry'
import { Alert, Badge, Button, Empty, PageHeader, Panel } from '../components/ui'
import { usePolling } from '../hooks/usePolling'
import { useTelemetry } from '../hooks/useTelemetry'
import { dateTime } from '../lib/format'
import { telemetry } from '../services/telemetry'
import type { TelemetryAlert } from '../types/telemetry'
import { inputCls } from './EquipmentMonitoring'
import { selectCls } from './live/Controls'

const NAME_KEY = 'dt.alerts.ackName'
const readName = () => {
  try {
    return localStorage.getItem(NAME_KEY) ?? ''
  } catch {
    return ''
  }
}

function AckForm({ alert, onDone }: { alert: TelemetryAlert; onDone: () => void }) {
  const [name, setName] = useState(readName)
  const [note, setNote] = useState('')
  const [error, setError] = useState<string | null>(null)
  const submit = async () => {
    try {
      await telemetry.ack(alert.id, name, note)
      try {
        localStorage.setItem(NAME_KEY, name)
      } catch {
        /* storage unavailable */
      }
      onDone()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }
  return (
    <div className="mt-2 flex flex-wrap items-center gap-2">
      <input className={inputCls + ' w-40'} placeholder="Your name" value={name} onChange={(e) => setName(e.target.value)} />
      <input className={inputCls + ' w-64'} placeholder="Note (optional)" value={note} onChange={(e) => setNote(e.target.value)} />
      <Button variant="primary" icon={Check} disabled={!name.trim()} onClick={submit}>
        Acknowledge
      </Button>
      {error && <span className="text-xs text-bad">{error}</span>}
    </div>
  )
}

function ActiveAlert({ a, onChanged }: { a: TelemetryAlert; onChanged: () => void }) {
  const [acking, setAcking] = useState(false)
  return (
    <div className="border-b border-line/60 px-4 py-3 last:border-0">
      <div className="flex flex-wrap items-center gap-2">
        <SeverityBadge severity={a.severity} />
        <SourceBadge source={a.data_source} />
        {a.acknowledged && (
          <Badge tone="good">
            <ShieldCheck size={11} /> acknowledged by {a.acknowledged_by}
          </Badge>
        )}
        <span className="font-mono text-xs text-ink-3">
          {a.equipment_id ?? 'site'} · {a.sensor_id}
        </span>
        <span className="ml-auto text-xs text-ink-3">since {dateTime(a.opened_at)}</span>
      </div>
      <div className="mt-1.5 text-sm text-ink">{a.message}</div>
      {a.ack_note && <div className="mt-1 text-xs text-ink-3">Note: {a.ack_note}</div>}
      {!a.acknowledged &&
        (acking ? (
          <AckForm alert={a} onDone={() => { setAcking(false); onChanged() }} />
        ) : (
          <Button className="mt-2" icon={Check} onClick={() => setAcking(true)}>
            Acknowledge
          </Button>
        ))}
    </div>
  )
}

export function Alerts() {
  const { alerts, connection, error, refreshSummary } = useTelemetry()
  const [filters, setFilters] = useState({ state: '', severity: '' })
  const [offset, setOffset] = useState(0)
  const limit = 25
  const history = usePolling(() => telemetry.alerts({ ...filters, limit, offset }), 10000)
  const rules = usePolling(telemetry.rules, 0)
  const active = Object.values(alerts).sort((a, b) => (a.severity === b.severity ? b.opened_at.localeCompare(a.opened_at) : a.severity === 'critical' ? -1 : 1))
  const changed = () => {
    history.refresh()
    refreshSummary()
  }
  const total = history.data?.total ?? 0
  return (
    <>
      <PageHeader title="Alerts" subtitle="Threshold-based conditions. An alert is a detected condition that needs investigation, not a diagnosis of a fault." />
      <div className="space-y-4">
        <TelemetryConnectionNotice connection={connection} error={error} />
        <Panel title={`Active conditions (${active.length})`} icon={BellRing} bodyClassName="p-0">
          {active.length === 0 ? (
            <Empty icon={ShieldCheck} title="No active conditions" />
          ) : (
            active.map((a) => <ActiveAlert key={a.id} a={a} onChanged={changed} />)
          )}
        </Panel>
        <Panel
          title={`History (${total})`}
          icon={History}
          actions={
            <>
              <select className={selectCls} value={filters.state} onChange={(e) => { setFilters({ ...filters, state: e.target.value }); setOffset(0) }} aria-label="State">
                <option value="">Any state</option>
                <option value="active">Active</option>
                <option value="cleared">Cleared</option>
              </select>
              <select className={selectCls} value={filters.severity} onChange={(e) => { setFilters({ ...filters, severity: e.target.value }); setOffset(0) }} aria-label="Severity">
                <option value="">Any severity</option>
                <option value="warning">Warning</option>
                <option value="critical">Critical</option>
              </select>
            </>
          }
          bodyClassName="p-0"
        >
          {history.error && (
            <div className="p-3">
              <Alert tone="bad">{history.error}</Alert>
            </div>
          )}
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-ink-3">
                <tr className="border-b border-line">
                  {['Opened', 'Severity', 'Condition', 'Equipment / sensor', 'State', 'Cleared', 'Ack'].map((h) => (
                    <th key={h} className="px-3 py-2 font-medium">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(history.data?.items ?? []).map((a) => (
                  <tr key={a.id} className="border-b border-line/60 align-top">
                    <td className="px-3 py-2 text-xs whitespace-nowrap text-ink-2">{dateTime(a.opened_at)}</td>
                    <td className="px-3 py-2">
                      <SeverityBadge severity={a.severity} />
                    </td>
                    <td className="max-w-md px-3 py-2 text-xs text-ink-2">{a.message}</td>
                    <td className="px-3 py-2 font-mono text-xs text-ink-3">
                      {a.equipment_id ?? 'site'}
                      <div>{a.sensor_id}</div>
                    </td>
                    <td className="px-3 py-2">
                      <Badge tone={a.state === 'active' ? 'warn' : 'neutral'}>{a.state}</Badge>
                    </td>
                    <td className="px-3 py-2 text-xs whitespace-nowrap text-ink-3">{dateTime(a.cleared_at)}</td>
                    <td className="px-3 py-2 text-xs text-ink-3">{a.acknowledged ? a.acknowledged_by : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {total > limit && (
            <div className="flex items-center justify-end gap-2 p-3 text-xs text-ink-3">
              {offset + 1}–{Math.min(offset + limit, total)} of {total}
              <Button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>
                Previous
              </Button>
              <Button disabled={offset + limit >= total} onClick={() => setOffset(offset + limit)}>
                Next
              </Button>
            </div>
          )}
        </Panel>
        {rules.data?.rules && (
          <Panel title={`Configured rules (${rules.data.rules.length})`} bodyClassName="p-3 text-xs text-ink-3">
            <p className="mb-2">From backend/configs/telemetry/thresholds.yaml. Values are in each metric's canonical unit.</p>
            <div className="grid gap-x-6 gap-y-1 sm:grid-cols-2 xl:grid-cols-3">
              {rules.data.rules.map((r) => (
                <div key={r.id} className="truncate">
                  <span className="font-mono text-ink-2">{r.metric}</span> {r.direction === 'above' ? '>' : '<'}{' '}
                  {[r.warning != null && `warn ${r.warning}`, r.critical != null && `crit ${r.critical}`].filter(Boolean).join(' / ')}
                  {r.for_s ? ` for ${r.for_s}s` : ''}
                </div>
              ))}
            </div>
          </Panel>
        )}
      </div>
    </>
  )
}

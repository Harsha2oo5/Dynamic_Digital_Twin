import { Activity, Plus, Server, TriangleAlert } from 'lucide-react'
import { useMemo, useState } from 'react'
import { CATEGORY_ORDER, ReadingValue, SensorStatusBadge, SourceBadge, TelemetryConnectionNotice } from '../components/telemetry'
import { Alert, Badge, Button, Empty, PageHeader, Panel } from '../components/ui'
import { useTelemetry } from '../hooks/useTelemetry'
import { usePolling } from '../hooks/usePolling'
import { dateTime, titleCase } from '../lib/format'
import { telemetry } from '../services/telemetry'
import type { Equipment, Reading } from '../types/telemetry'
import { selectCls } from './live/Controls'

export const inputCls = 'w-full rounded-md border border-line bg-surface-2 px-2 py-1.5 text-sm text-ink focus:border-accent focus:outline-none'
const EQUIPMENT_CATEGORIES = ['antenna', 'router', 'cable', 'battery', 'power_supply_unit', 'equipment_cabinet', 'radio_unit']

function ReadingsGrid({ readings }: { readings: Reading[] }) {
  const { metrics } = useTelemetry()
  if (!readings.length) return <div className="text-xs text-ink-3">No readings received yet.</div>
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
      {readings.map((r) => (
        <div key={`${r.sensor_id}|${r.metric}`} className="min-w-0">
          <div className="truncate text-xs text-ink-3" title={`${r.label} · ${r.sensor_id}`}>
            {r.label}
          </div>
          <ReadingValue reading={r} def={metrics[r.metric]} compact />
        </div>
      ))}
    </div>
  )
}

function EquipmentCard({ eq }: { eq: Equipment }) {
  const { latest, sensors, alerts } = useTelemetry()
  const readings = Object.values(latest).filter((r) => eq.sensor_ids.includes(r.sensor_id))
  const eqSensors = eq.sensor_ids.map((id) => sensors[id]).filter(Boolean)
  const active = Object.values(alerts).filter((a) => a.equipment_id === eq.equipment_id)
  const online = eqSensors.filter((s) => s.status === 'online').length
  return (
    <Panel
      title={
        <span className="flex min-w-0 items-center gap-2">
          <span className="truncate">{eq.name}</span>
        </span>
      }
      actions={<SourceBadge source={eq.data_source} />}
    >
      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-ink-3">
        <span className="font-mono text-ink-2">{eq.equipment_id}</span>
        <Badge>{titleCase(eq.category)}</Badge>
        {eq.zone && <span>{eq.zone}</span>}
        <span>
          sensors online {online}/{eqSensors.length}
        </span>
        {active.length > 0 && (
          <Badge tone={active.some((a) => a.severity === 'critical') ? 'bad' : 'warn'}>
            <TriangleAlert size={11} /> {active.length} condition{active.length > 1 ? 's' : ''}
          </Badge>
        )}
      </div>
      {eqSensors.length === 0 ? (
        <div className="text-xs text-ink-3">No sensors are associated with this equipment. Associate one on the Sensor Management page.</div>
      ) : (
        <ReadingsGrid readings={readings} />
      )}
      {eqSensors.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5 border-t border-line pt-3">
          {eqSensors.map((s) => (
            <span key={s.sensor_id} className="inline-flex items-center gap-1.5 text-xs text-ink-3">
              <span className="font-mono">{s.sensor_id}</span>
              <SensorStatusBadge status={s.status} />
            </span>
          ))}
        </div>
      )}
      {eq.visual_links.length > 0 && (
        <div className="mt-2 text-xs text-ink-3">
          Linked visual observations: {eq.visual_links.map((v) => v.object_id).join(', ')} (session-scoped, linked manually)
        </div>
      )}
      <div className="mt-2 text-xs text-ink-3">Last updated {dateTime(eq.last_updated)}</div>
    </Panel>
  )
}

function SiteCard() {
  const { sensors, latest } = useTelemetry()
  const siteSensors = Object.values(sensors).filter((s) => s.category === 'site' && !s.equipment_id)
  if (!siteSensors.length) return null
  const ids = new Set(siteSensors.map((s) => s.sensor_id))
  const readings = Object.values(latest).filter((r) => ids.has(r.sensor_id))
  const sources = [...new Set(siteSensors.map((s) => s.data_source))]
  return (
    <Panel title="Site environment and power" icon={Activity} actions={sources.map((s) => <SourceBadge key={s} source={s} />)}>
      <ReadingsGrid readings={readings} />
      <div className="mt-3 flex flex-wrap gap-1.5 border-t border-line pt-3">
        {siteSensors.map((s) => (
          <span key={s.sensor_id} className="inline-flex items-center gap-1.5 text-xs text-ink-3">
            <span className="font-mono">{s.sensor_id}</span>
            <SensorStatusBadge status={s.status} />
          </span>
        ))}
      </div>
    </Panel>
  )
}

function RegisterEquipment({ onDone }: { onDone: () => void }) {
  const [form, setForm] = useState({ equipment_id: '', category: 'battery', name: '', zone: '' })
  const [error, setError] = useState<string | null>(null)
  const submit = async () => {
    try {
      await telemetry.createEquipment({ ...form, zone: form.zone || null })
      setForm({ equipment_id: '', category: form.category, name: '', zone: '' })
      setError(null)
      onDone()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }
  return (
    <Panel title="Register physical equipment" icon={Plus}>
      <div className="grid gap-3 sm:grid-cols-4">
        <input className={inputCls} placeholder="ID, e.g. BAT-001" value={form.equipment_id} onChange={(e) => setForm({ ...form, equipment_id: e.target.value })} />
        <select className={selectCls + ' py-1.5 text-sm'} value={form.category} onChange={(e) => setForm({ ...form, category: e.target.value })}>
          {EQUIPMENT_CATEGORIES.map((c) => (
            <option key={c} value={c}>
              {titleCase(c)}
            </option>
          ))}
        </select>
        <input className={inputCls} placeholder="Name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
        <input className={inputCls} placeholder="Zone (optional)" value={form.zone} onChange={(e) => setForm({ ...form, zone: e.target.value })} />
      </div>
      <div className="mt-3 flex items-center gap-3">
        <Button variant="primary" icon={Plus} disabled={!form.equipment_id || !form.name} onClick={submit}>
          Register
        </Button>
        <span className="text-xs text-ink-3">
          Equipment IDs are stable identities chosen by you. They are not visual tracking IDs, which only exist within one video or stream.
        </span>
      </div>
      {error && (
        <div className="mt-3">
          <Alert tone="bad">{error}</Alert>
        </div>
      )}
    </Panel>
  )
}

export function EquipmentMonitoring() {
  const { connection, error } = useTelemetry()
  const equipment = usePolling(() => telemetry.equipment(), 10000)
  const [category, setCategory] = useState('')
  const [showForm, setShowForm] = useState(false)
  const list = useMemo(
    () =>
      (equipment.data ?? [])
        .filter((e) => !category || e.category === category)
        .sort((a, b) => CATEGORY_ORDER.indexOf(a.category) - CATEGORY_ORDER.indexOf(b.category) || a.equipment_id.localeCompare(b.equipment_id)),
    [equipment.data, category],
  )
  return (
    <>
      <PageHeader
        title="Equipment Monitoring"
        subtitle="Each registered equipment item with its sensors and their latest readings. Simulated devices are always labelled."
        actions={
          <>
            <select className={selectCls} value={category} onChange={(e) => setCategory(e.target.value)} aria-label="Filter by category">
              <option value="">All categories</option>
              {EQUIPMENT_CATEGORIES.map((c) => (
                <option key={c} value={c}>
                  {titleCase(c)}
                </option>
              ))}
            </select>
            <Button icon={Plus} onClick={() => setShowForm((v) => !v)}>
              Equipment
            </Button>
          </>
        }
      />
      <div className="space-y-4">
        <TelemetryConnectionNotice connection={connection} error={error} />
        {equipment.error && <Alert tone="bad" title="Could not load equipment">{equipment.error}</Alert>}
        {showForm && <RegisterEquipment onDone={equipment.refresh} />}
        {!category && <SiteCard />}
        {equipment.data && list.length === 0 ? (
          <Panel>
            <Empty icon={Server} title="No equipment registered">
              Register physical equipment above, or start the simulator on the Sensor Management page to create a clearly labelled simulated site.
            </Empty>
          </Panel>
        ) : (
          <div className="grid gap-4 lg:grid-cols-2 2xl:grid-cols-3">
            {list.map((eq) => (
              <EquipmentCard key={eq.equipment_id} eq={eq} />
            ))}
          </div>
        )}
      </div>
    </>
  )
}

import { FlaskConical, Play, Plus, Radio, Search, Square, Trash2, X } from 'lucide-react'
import { useMemo, useState } from 'react'
import { ReadingValue, SensorStatusBadge, SourceBadge, TelemetryConnectionNotice } from '../components/telemetry'
import { Alert, Badge, Button, Empty, KV, PageHeader, Panel } from '../components/ui'
import { usePolling } from '../hooks/usePolling'
import { useTelemetry } from '../hooks/useTelemetry'
import { ago, dateTime, titleCase } from '../lib/format'
import { telemetry, type SensorInput } from '../services/telemetry'
import type { Equipment, Sensor } from '../types/telemetry'
import { inputCls } from './EquipmentMonitoring'
import { selectCls } from './live/Controls'

const SENSOR_TYPES = ['temperature', 'voltage', 'current', 'power', 'humidity', 'door', 'smoke', 'fan', 'battery_monitor', 'psu_monitor', 'network', 'rf', 'environment', 'multi']
const CATEGORIES = ['site', 'antenna', 'router', 'cable', 'battery', 'power_supply_unit', 'equipment_cabinet', 'radio_unit']
const errText = (e: unknown) => (e instanceof Error ? e.message : String(e))

function SimulatorPanel() {
  const sim = usePolling(telemetry.simulator, 2000)
  const [scale, setScale] = useState(1)
  const [backfill, setBackfill] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const { reloadSensors } = useTelemetry()
  const run = async (fn: () => Promise<unknown>) => {
    try {
      await fn()
      setError(null)
    } catch (e) {
      setError(errText(e))
    }
    sim.refresh()
  }
  const s = sim.data
  const events = Object.entries(s?.events ?? {}).filter(([, v]) => v)
  const dropouts = Object.keys(s?.dropouts ?? {})
  return (
    <Panel
      title="Simulated sensor environment"
      icon={FlaskConical}
      actions={s?.running ? <Badge tone="accent">{s.backfilling ? 'writing history…' : 'running'}</Badge> : <Badge>stopped</Badge>}
    >
      <p className="mb-3 text-xs text-ink-3">
        Generates readings from a physical model of a BTS site (daily temperature cycle, traffic-driven load, battery charge and discharge,
        cabinet heating, sensor dropouts). All of its devices use <span className="font-mono">SIM-</span> IDs and are marked SIMULATED.
        It never starts on its own.
      </p>
      <div className="flex flex-wrap items-center gap-2">
        {!s?.running ? (
          <>
            <label className="text-xs text-ink-3">
              speed{' '}
              <select className={selectCls} value={scale} onChange={(e) => setScale(Number(e.target.value))}>
                {[1, 5, 20].map((v) => (
                  <option key={v} value={v}>
                    {v}×
                  </option>
                ))}
              </select>
            </label>
            <label className="text-xs text-ink-3">
              history{' '}
              <select className={selectCls} value={backfill} onChange={(e) => setBackfill(Number(e.target.value))}>
                {[
                  [0, 'none'],
                  [6, '6 h'],
                  [24, '24 h'],
                  [168, '7 days'],
                ].map(([v, l]) => (
                  <option key={v} value={v}>
                    {l}
                  </option>
                ))}
              </select>
            </label>
            <Button variant="primary" icon={Play} onClick={() => run(async () => { await telemetry.startSimulator({ interval_scale: scale, backfill_hours: backfill }); await reloadSensors() })}>
              Start
            </Button>
          </>
        ) : (
          <>
            <Button icon={Square} onClick={() => run(telemetry.stopSimulator)}>
              Stop
            </Button>
            {(
              [
                ['mains_outage', 'Mains outage', 600],
                ['sensor_dropout', 'Sensor dropout', 180],
                ['door_open', 'Open door', 900],
                ['fan_failure', 'Fan failure', 1200],
                ['heat_wave', 'Heat wave', 3600],
              ] as const
            ).map(([id, label, dur]) => (
              <Button key={id} disabled={s.backfilling} onClick={() => run(() => telemetry.scenario(id, dur))} title={`${dur / 60} minutes`}>
                {label}
              </Button>
            ))}
          </>
        )}
      </div>
      {s && (
        <div className="mt-3 text-xs text-ink-3">
          {s.messages_sent.toLocaleString()} messages sent
          {s.started_at && ` since ${dateTime(s.started_at)}`}
          {s.backfill && ` · history: ${s.backfill.hours} h (${s.backfill.messages.toLocaleString()} messages, every ${s.backfill.step_s} s)`}
          {events.length > 0 && ` · active: ${events.map(([k]) => k.replace(/_until$/, '').replace(/_/g, ' ')).join(', ')}`}
          {dropouts.length > 0 && ` · silent sensors: ${dropouts.join(', ')}`}
        </div>
      )}
      {(error || s?.last_error) && (
        <div className="mt-3">
          <Alert tone="bad">{error ?? s?.last_error}</Alert>
        </div>
      )}
    </Panel>
  )
}

function RegisterSensor({ equipment, onDone }: { equipment: Equipment[]; onDone: () => void }) {
  const { metrics } = useTelemetry()
  const physical = equipment.filter((e) => e.data_source === 'physical')
  const [f, setF] = useState<SensorInput>({ sensor_id: '', name: '', sensor_type: 'temperature', protocol: 'mqtt', metrics: [], equipment_id: null, category: 'site', sampling_interval_s: 10 })
  const [error, setError] = useState<string | null>(null)
  const category = f.equipment_id ? physical.find((e) => e.equipment_id === f.equipment_id)?.category : f.category
  const relevant = Object.values(metrics).filter((m) => !category || m.categories.includes(category))
  const toggle = (name: string) => setF({ ...f, metrics: f.metrics.includes(name) ? f.metrics.filter((m) => m !== name) : [...f.metrics, name] })
  const submit = async () => {
    try {
      await telemetry.createSensor({ ...f, category: f.equipment_id ? null : f.category })
      setError(null)
      setF({ ...f, sensor_id: '', name: '', metrics: [] })
      onDone()
    } catch (e) {
      setError(errText(e))
    }
  }
  return (
    <Panel title="Register a physical sensor" icon={Plus}>
      <div className="grid gap-3 sm:grid-cols-3">
        <input className={inputCls} placeholder="Sensor ID, e.g. TEMP-001" value={f.sensor_id} onChange={(e) => setF({ ...f, sensor_id: e.target.value })} />
        <input className={inputCls} placeholder="Name" value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} />
        <div className="flex gap-2">
          <select className={selectCls + ' flex-1 py-1.5 text-sm'} value={f.sensor_type} onChange={(e) => setF({ ...f, sensor_type: e.target.value })} aria-label="Sensor type">
            {SENSOR_TYPES.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
          <select className={selectCls + ' py-1.5 text-sm'} value={f.protocol} onChange={(e) => setF({ ...f, protocol: e.target.value })} aria-label="Protocol">
            <option value="mqtt">MQTT</option>
            <option value="http">HTTP</option>
          </select>
        </div>
        <select
          className={selectCls + ' py-1.5 text-sm'}
          value={f.equipment_id ?? ''}
          onChange={(e) => setF({ ...f, equipment_id: e.target.value || null, metrics: [] })}
          aria-label="Equipment"
        >
          <option value="">No equipment (choose a category)</option>
          {physical.map((e) => (
            <option key={e.equipment_id} value={e.equipment_id}>
              {e.equipment_id} · {e.name}
            </option>
          ))}
        </select>
        {!f.equipment_id && (
          <select className={selectCls + ' py-1.5 text-sm'} value={f.category ?? 'site'} onChange={(e) => setF({ ...f, category: e.target.value, metrics: [] })} aria-label="Category">
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>
                {titleCase(c)}
              </option>
            ))}
          </select>
        )}
        <label className="flex items-center gap-2 text-xs text-ink-3">
          sampling interval (s)
          <input className={inputCls + ' w-24'} type="number" min={1} value={f.sampling_interval_s} onChange={(e) => setF({ ...f, sampling_interval_s: Number(e.target.value) })} />
        </label>
      </div>
      <div className="mt-3">
        <div className="mb-1 text-xs text-ink-3">Metrics this sensor reports ({titleCase(category ?? 'any')})</div>
        <div className="flex flex-wrap gap-2">
          {relevant.map((m) => (
            <label key={m.name} className="inline-flex items-center gap-1.5 rounded border border-line px-2 py-1 text-xs text-ink-2">
              <input type="checkbox" checked={f.metrics.includes(m.name)} onChange={() => toggle(m.name)} />
              {m.label} <span className="text-ink-3">({m.unit})</span>
            </label>
          ))}
        </div>
      </div>
      <div className="mt-3 flex items-center gap-3">
        <Button variant="primary" icon={Plus} disabled={!f.sensor_id || !f.name || !f.metrics.length} onClick={submit}>
          Register
        </Button>
        <span className="text-xs text-ink-3">
          Readings from unregistered sensors are rejected. MQTT topic: <span className="font-mono">bts/&lt;site&gt;/{f.sensor_id || '<sensor_id>'}/telemetry</span>
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

function SensorDetail({ sensor, onClose, onChanged }: { sensor: Sensor; onClose: () => void; onChanged: () => void }) {
  const { latest, metrics } = useTelemetry()
  const [interval, setIntervalS] = useState(sensor.sampling_interval_s)
  const [error, setError] = useState<string | null>(null)
  const readings = Object.values(latest).filter((r) => r.sensor_id === sensor.sensor_id)
  const simulated = sensor.data_source === 'simulated'
  const act = async (fn: () => Promise<unknown>) => {
    try {
      await fn()
      setError(null)
      onChanged()
    } catch (e) {
      setError(errText(e))
    }
  }
  return (
    <Panel
      title={<span className="font-mono">{sensor.sensor_id}</span>}
      actions={
        <button onClick={onClose} className="text-ink-3 hover:text-ink" aria-label="Close">
          <X size={16} />
        </button>
      }
      bodyClassName="space-y-4 p-4"
    >
      <KV
        rows={[
          ['Name', sensor.name],
          ['Source', <SourceBadge source={sensor.data_source} />],
          ['Status', <SensorStatusBadge status={sensor.status} />],
          ['Last communication', sensor.last_seen_at ? `${dateTime(sensor.last_seen_at)} (${ago((Date.now() - new Date(sensor.last_seen_at).getTime()) / 1000)})` : 'never'],
          ['Type / protocol', `${sensor.sensor_type} · ${sensor.protocol}`],
          ['Equipment', sensor.equipment_id ?? '—'],
          ['Category / zone', `${titleCase(sensor.category)}${sensor.zone ? ` · ${sensor.zone}` : ''}`],
          ['Stale / offline after', `${sensor.stale_after_s} s / ${sensor.offline_after_s} s`],
        ]}
      />
      <div>
        <div className="mb-2 text-xs text-ink-3">Latest readings</div>
        {readings.length ? (
          <div className="grid grid-cols-2 gap-3">
            {readings.map((r) => (
              <div key={r.metric}>
                <div className="truncate text-xs text-ink-3">{r.label}</div>
                <ReadingValue reading={r} def={metrics[r.metric]} compact />
              </div>
            ))}
          </div>
        ) : (
          <div className="text-xs text-ink-3">No readings yet.</div>
        )}
      </div>
      {!simulated && (
        <div className="space-y-2 border-t border-line pt-3">
          <label className="flex items-center gap-2 text-xs text-ink-3">
            sampling interval (s)
            <input className={inputCls + ' w-24'} type="number" min={1} value={interval} onChange={(e) => setIntervalS(Number(e.target.value))} />
            <Button disabled={interval === sensor.sampling_interval_s} onClick={() => act(() => telemetry.updateSensor(sensor.sensor_id, { sampling_interval_s: interval }))}>
              Save
            </Button>
          </label>
          <div className="flex gap-2">
            <Button onClick={() => act(() => telemetry.updateSensor(sensor.sensor_id, { enabled: !sensor.enabled }))}>{sensor.enabled ? 'Disable' : 'Enable'}</Button>
            <Button variant="danger" icon={Trash2} onClick={() => confirm(`Delete sensor ${sensor.sensor_id} and its stored readings?`) && act(async () => { await telemetry.deleteSensor(sensor.sensor_id); onClose() })}>
              Delete
            </Button>
          </div>
        </div>
      )}
      {simulated && <p className="text-xs text-ink-3">Simulated sensors are defined in configs/telemetry/simulator.yaml.</p>}
      {error && <Alert tone="bad">{error}</Alert>}
    </Panel>
  )
}

export function SensorManagement() {
  const { sensors, connection, error, reloadSensors } = useTelemetry()
  const equipment = usePolling(() => telemetry.equipment(), 15000)
  const [filters, setFilters] = useState({ q: '', type: '', category: '', status: '', source: '' })
  const [selected, setSelected] = useState<string | null>(null)
  const [showForm, setShowForm] = useState(false)
  const [assocError, setAssocError] = useState<string | null>(null)
  const list = useMemo(
    () =>
      Object.values(sensors)
        .filter(
          (s) =>
            (!filters.q || `${s.sensor_id} ${s.name} ${s.equipment_id ?? ''}`.toLowerCase().includes(filters.q.toLowerCase())) &&
            (!filters.type || s.sensor_type === filters.type) &&
            (!filters.category || s.category === filters.category) &&
            (!filters.status || s.status === filters.status) &&
            (!filters.source || s.data_source === filters.source),
        )
        .sort((a, b) => a.sensor_id.localeCompare(b.sensor_id)),
    [sensors, filters],
  )
  const sel = selected ? sensors[selected] : null
  const changed = () => {
    reloadSensors()
    equipment.refresh()
  }
  const associate = async (s: Sensor, equipmentId: string) => {
    try {
      await telemetry.associate(s.sensor_id, equipmentId || null)
      setAssocError(null)
      changed()
    } catch (e) {
      setAssocError(errText(e))
    }
  }
  const sel_ = (key: keyof typeof filters, options: string[], label: string) => (
    <select className={selectCls} value={filters[key]} onChange={(e) => setFilters({ ...filters, [key]: e.target.value })} aria-label={label}>
      <option value="">{label}</option>
      {options.map((o) => (
        <option key={o} value={o}>
          {titleCase(o)}
        </option>
      ))}
    </select>
  )
  return (
    <>
      <PageHeader
        title="Sensor Management"
        subtitle="Register sensors, associate them with equipment and check their communication status."
        actions={
          <Button icon={Plus} onClick={() => setShowForm((v) => !v)}>
            Sensor
          </Button>
        }
      />
      <div className="space-y-4">
        <TelemetryConnectionNotice connection={connection} error={error} />
        <SimulatorPanel />
        {showForm && <RegisterSensor equipment={equipment.data ?? []} onDone={changed} />}
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
          <Panel
            title={`Sensors (${list.length})`}
            icon={Radio}
            actions={
              <div className="flex flex-wrap gap-2">
                <div className="relative">
                  <Search size={13} className="absolute top-1.5 left-2 text-ink-3" />
                  <input className={selectCls + ' pl-6'} placeholder="Search" value={filters.q} onChange={(e) => setFilters({ ...filters, q: e.target.value })} />
                </div>
                {sel_('type', SENSOR_TYPES, 'Type')}
                {sel_('category', CATEGORIES, 'Category')}
                {sel_('status', ['online', 'stale', 'offline', 'unknown', 'disabled'], 'Status')}
                {sel_('source', ['physical', 'simulated'], 'Source')}
              </div>
            }
            bodyClassName="p-0"
          >
            {assocError && (
              <div className="p-3">
                <Alert tone="bad">{assocError}</Alert>
              </div>
            )}
            {list.length === 0 ? (
              <Empty icon={Radio} title={Object.keys(sensors).length ? 'No sensors match the filters' : 'No sensors registered'}>
                Register a physical sensor, or start the simulator above.
              </Empty>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="text-left text-xs text-ink-3">
                    <tr className="border-b border-line">
                      {['Sensor', 'Type', 'Equipment', 'Source', 'Status', 'Last seen', 'Metrics'].map((h) => (
                        <th key={h} className="px-3 py-2 font-medium">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {list.map((s) => (
                      <tr key={s.sensor_id} onClick={() => setSelected(s.sensor_id)} className={'cursor-pointer border-b border-line/60 hover:bg-surface-2 ' + (selected === s.sensor_id ? 'bg-surface-2' : '')}>
                        <td className="px-3 py-2">
                          <div className="font-mono text-xs text-ink">{s.sensor_id}</div>
                          <div className="truncate text-xs text-ink-3">{s.name}</div>
                        </td>
                        <td className="px-3 py-2 text-xs text-ink-2">
                          {s.sensor_type}
                          <div className="text-ink-3">{s.protocol}</div>
                        </td>
                        <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
                          {s.data_source === 'simulated' ? (
                            <span className="font-mono text-xs text-ink-2">{s.equipment_id ?? `site`}</span>
                          ) : (
                            <select className={selectCls} value={s.equipment_id ?? ''} onChange={(e) => associate(s, e.target.value)} aria-label={`Equipment of ${s.sensor_id}`}>
                              <option value="">{titleCase(s.category)} (none)</option>
                              {(equipment.data ?? [])
                                .filter((e) => e.data_source === 'physical')
                                .map((e) => (
                                  <option key={e.equipment_id} value={e.equipment_id}>
                                    {e.equipment_id}
                                  </option>
                                ))}
                            </select>
                          )}
                        </td>
                        <td className="px-3 py-2">
                          <SourceBadge source={s.data_source} />
                        </td>
                        <td className="px-3 py-2">
                          <SensorStatusBadge status={s.status} />
                        </td>
                        <td className="px-3 py-2 text-xs text-ink-2">{s.last_seen_at ? ago((Date.now() - new Date(s.last_seen_at).getTime()) / 1000) : 'never'}</td>
                        <td className="px-3 py-2 text-xs text-ink-3">{s.metrics.length}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>
          {sel ? (
            <SensorDetail key={sel.sensor_id} sensor={sel} onClose={() => setSelected(null)} onChanged={changed} />
          ) : (
            <Panel>
              <Empty icon={Radio} title="Select a sensor">
                Its configuration, communication status and latest readings appear here.
              </Empty>
            </Panel>
          )}
        </div>
      </div>
    </>
  )
}

import type {
  Availability,
  Equipment,
  HistorySeries,
  MetricDef,
  Page,
  Reading,
  Sensor,
  SimulatorState,
  TelemetryAlert,
  TelemetryStatus,
  TelemetrySummary,
  TimeRange,
} from '../types/telemetry'
import { request } from './api'

const T = '/api/telemetry'
const json = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: body === undefined ? undefined : JSON.stringify(body),
})
const qs = (params: Record<string, string | number | boolean | null | undefined>) => {
  const p = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') p.set(k, String(v))
  const s = p.toString()
  return s ? `?${s}` : ''
}

export interface SensorInput {
  sensor_id: string
  name: string
  sensor_type: string
  protocol: string
  metrics: string[]
  equipment_id?: string | null
  category?: string | null
  zone?: string | null
  sampling_interval_s: number
  enabled?: boolean
}

export const telemetry = {
  status: () => request<TelemetryStatus>(`${T}/status`),
  metrics: () => request<MetricDef[]>(`${T}/metrics`),
  summary: () => request<TelemetrySummary>(`${T}/summary`),

  equipment: (category?: string) => request<Equipment[]>(`${T}/equipment${qs({ category })}`),
  createEquipment: (body: { equipment_id: string; category: string; name: string; zone?: string | null; description?: string | null }) =>
    request<Equipment>(`${T}/equipment`, json('POST', body)),
  deleteEquipment: (id: string) => request<void>(`${T}/equipment/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  equipmentTelemetry: (id: string) =>
    request<{ equipment: Equipment; sensors: Sensor[]; latest: Reading[]; active_alerts: TelemetryAlert[] }>(
      `${T}/equipment/${encodeURIComponent(id)}/telemetry`,
    ),
  addVisualLink: (id: string, body: { object_id: string; class_name?: string | null; linked_by?: string | null; note?: string | null }) =>
    request<unknown>(`${T}/equipment/${encodeURIComponent(id)}/visual-links`, json('POST', body)),

  sensors: (params: Record<string, string | undefined> = {}) => request<Page<Sensor>>(`${T}/sensors${qs({ limit: 500, ...params })}`),
  createSensor: (body: SensorInput) => request<Sensor>(`${T}/sensors`, json('POST', body)),
  updateSensor: (id: string, body: Partial<SensorInput>) => request<Sensor>(`${T}/sensors/${encodeURIComponent(id)}`, json('PATCH', body)),
  associate: (id: string, equipment_id: string | null) =>
    request<Sensor>(`${T}/sensors/${encodeURIComponent(id)}/equipment`, json('PUT', { equipment_id })),
  deleteSensor: (id: string) => request<void>(`${T}/sensors/${encodeURIComponent(id)}`, { method: 'DELETE' }),

  latest: (params: Record<string, string | undefined> = {}) => request<Reading[]>(`${T}/latest${qs(params)}`),
  history: (metric: string, range: TimeRange, params: { sensor_id?: string; equipment_id?: string; max_points?: number } = {}) =>
    request<{ metric: string; start: string; end: string; series: HistorySeries[] }>(`${T}/history${qs({ metric, range, ...params })}`),
  availability: (range: TimeRange) => request<Availability>(`${T}/availability${qs({ range })}`),

  alerts: (params: Record<string, string | number | boolean | undefined> = {}) =>
    request<Page<TelemetryAlert>>(`${T}/alerts${qs(params)}`),
  rules: () =>
    request<{ rules: { id: string; metric: string; direction: string; warning: number | null; critical: number | null; for_s: number; categories: string[] }[] }>(
      `${T}/alerts/rules`,
    ),
  ack: (id: number, acknowledged_by: string, note?: string) =>
    request<TelemetryAlert>(`${T}/alerts/${id}/ack`, json('POST', { acknowledged_by, note: note || null })),

  simulator: () => request<SimulatorState>(`${T}/simulator`),
  startSimulator: (body: { interval_scale: number; backfill_hours: number; seed?: number | null }) =>
    request<SimulatorState>(`${T}/simulator/start`, json('POST', body)),
  stopSimulator: () => request<SimulatorState>(`${T}/simulator/stop`, json('POST')),
  scenario: (scenario: string, duration_s: number, sensor_id?: string) =>
    request<unknown>(`${T}/simulator/scenario`, json('POST', { scenario, duration_s, sensor_id: sensor_id || null })),
}

// Contract of the telemetry API (backend/app/telemetry/schemas.py and service.py).

export type DataSource = 'physical' | 'simulated'
export type SensorStatus = 'unknown' | 'online' | 'stale' | 'offline' | 'disabled'
export type Quality = 'good' | 'missing' | 'invalid' | 'out_of_order'
export type Severity = 'warning' | 'critical'

export interface MetricDef {
  name: string
  unit: string
  label: string
  kind: 'gauge' | 'state'
  min: number | null
  max: number | null
  categories: string[]
  states: Record<string, string>
}

export interface Sensor {
  sensor_id: string
  name: string
  sensor_type: string
  protocol: string
  site_id: string
  equipment_id: string | null
  category: string
  zone: string | null
  data_source: DataSource
  enabled: boolean
  metrics: string[]
  sampling_interval_s: number
  stale_after_s: number
  offline_after_s: number
  config: Record<string, unknown>
  status: SensorStatus
  last_seen_at: string | null
  seconds_since_seen: number | null
}

export interface Equipment {
  equipment_id: string
  site_id: string
  category: string
  name: string
  zone: string | null
  description: string | null
  data_source: DataSource
  sensor_ids: string[]
  sensors_online: number
  last_updated: string | null
  active_alerts: number
  visual_links: { id: number; object_id: string; class_name: string | null; linked_by: string | null; note: string | null; linked_at: string }[]
}

export interface Reading {
  sensor_id: string
  equipment_id: string | null
  metric: string
  value: number | null
  unit: string
  ts: string
  data_source: DataSource
  quality: Quality
  quality_note: string | null
  state_label: string | null
  label: string
  sensor_status: SensorStatus
  age_s: number
  /** false = last known value of a stale/offline sensor (or a non-good reading): never show as current */
  current: boolean
}

export interface TelemetryAlert {
  id: number
  rule_id: string
  severity: Severity
  state: 'active' | 'cleared'
  sensor_id: string | null
  equipment_id: string | null
  metric: string | null
  value: number | null
  peak_value: number | null
  threshold: number | null
  message: string
  data_source: DataSource
  opened_at: string
  last_triggered_at: string
  cleared_at: string | null
  acknowledged: boolean
  acknowledged_at: string | null
  acknowledged_by: string | null
  ack_note: string | null
}

export interface TelemetrySummary {
  site_id: string
  site_name: string
  sensors_total: number
  sensors_online: number
  sensors_stale: number
  sensors_offline: number
  sensors_unknown: number
  sensors_disabled: number
  sensors_simulated: number
  sensors_physical: number
  equipment_total: number
  equipment_monitored: number
  site_temperature: (Reading & { source: 'ambient' | 'cabinet' }) | null
  site_power_w: number | null
  site_power_avg_1h_w: number | null
  equipment_load_w: number | null
  equipment_load_sources: number
  alerts_active: number
  alerts_critical: number
  alerts_warning: number
  alerts_unacknowledged: number
  readings_per_minute: number
  ingest: Record<string, number>
  data_sources: DataSource[]
  simulator_running: boolean
}

export interface TelemetryStatus {
  available: boolean
  error: string | null
  database: string | null
  schema_revision: string | null
  site_id: string
  app_env: string
  ingest_api_key_configured: boolean
  retention_days: number
  measurements_stored: number | null
  oldest_measurement: string | null
  websocket_clients: number
  mqtt: { enabled: boolean; connected?: boolean; broker?: string; topic?: string; last_error?: string | null; [k: string]: unknown }
  simulator: SimulatorState | null
  recent_ingest_errors: { at: string; sensor_id: string; protocol: string; error: string }[]
}

export interface SimulatorState {
  running: boolean
  backfilling: boolean
  started_at: string | null
  interval_scale: number
  messages_sent: number
  sensors: number
  backfill: { hours: number; step_s: number; messages: number } | null
  dropouts: Record<string, string>
  events: Record<string, string | null> | null
  last_error: string | null
  data_source: 'simulated'
}

export interface HistoryPoint {
  t: string
  avg: number
  min: number
  max: number
  n: number
}

export interface HistorySeries {
  sensor_id: string
  equipment_id: string | null
  metric: string
  label: string
  unit: string
  data_source: DataSource
  bucket_s: number
  points: HistoryPoint[]
}

export interface Page<T> {
  total: number
  limit: number
  offset: number
  items: T[]
}

export interface Availability {
  start: string
  end: string
  sensors: { sensor_id: string; data_source: DataSource; online_share: number | null; transitions: number }[]
}

export type TelemetryEvent =
  | { type: 'measurements'; items: Reading[] }
  | { type: 'sensor_status'; sensor_id: string; status: SensorStatus; last_seen_at: string | null; data_source: DataSource }
  | { type: 'alert'; alert: TelemetryAlert }
  | { type: 'anomaly'; event: string; anomaly: import('./analytics').Anomaly }
  | { type: 'health'; equipment_id: string; status: import('./analytics').HealthStatus; score: number | null; coverage: number; data_quality: number | null }
  | { type: 'recommendation'; recommendation: import('./analytics').Recommendation }

export type TelemetryMessage =
  | { type: 'snapshot'; summary: TelemetrySummary; sensors: Sensor[]; latest: Reading[]; alerts: TelemetryAlert[] }
  | { type: 'batch'; events: TelemetryEvent[]; dropped: number }
  | { type: 'error'; message: string }

export type TimeRange = '15m' | '1h' | '6h' | '24h' | '7d'

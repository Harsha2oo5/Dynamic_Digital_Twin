// Contract of the analytics API (backend/app/api/analytics_routes.py).
import type { DataSource, Equipment, Reading, Sensor, TelemetryAlert } from './telemetry'

export type HealthStatus = 'healthy' | 'warning' | 'degraded' | 'critical' | 'unknown'
export type AnomalyStatus = 'open' | 'acknowledged' | 'resolved'
export type EvidenceLevel = 'low' | 'medium' | 'high'

export interface HealthComponent {
  type: 'anomaly' | 'alert' | 'recommendation'
  id: number
  severity: string
  penalty: number
  active: boolean
  detail: string
}

export interface EquipmentHealth {
  equipment_id: string
  category: string
  name: string
  data_source: DataSource
  /** 0-100 index of recent abnormal behaviour; null = unknown (not enough data). Not a failure probability. */
  score: number | null
  status: HealthStatus
  coverage: number
  coverage_detail: Record<string, string>
  /** sensor availability (0-1), reported separately from equipment health */
  data_quality: number | null
  components: HealthComponent[]
  unknown_reason: string | null
  sensors: number
  sensors_online: number
  at: string
}

export interface HealthSummary {
  equipment: number
  healthy: number
  warning: number
  degraded: number
  critical: number
  unknown: number
  mean_score: number | null
  active_anomalies: number
  open_anomalies: number
  active_recommendations: number
  requiring_inspection: string[]
  config_version: number | null
  models: string[]
}

export interface Finding {
  detector: string
  score: number
  detail: string
}

export interface Anomaly {
  id: number
  equipment_id: string | null
  sensor_id: string | null
  category: string
  metric: string | null
  kind: 'univariate' | 'multivariate'
  detector: string
  detectors: string[]
  model_id: number | null
  observed: number | null
  expected: number | null
  unit: string | null
  /** threshold units: 1 = at the detector threshold; not a probability */
  score: number
  peak_score: number
  severity: 'warning' | 'critical'
  status: AnomalyStatus
  condition_active: boolean
  first_detected_at: string
  last_detected_at: string
  ended_at: string | null
  occurrences: number
  explanation: string
  evidence: { features?: Record<string, number>; findings?: Finding[]; context?: string; robust_z?: Record<string, number> }
  data_source: DataSource
  acknowledged_at: string | null
  acknowledged_by: string | null
  resolved_at: string | null
  resolved_by: string | null
  resolution_note: string | null
}

export interface Recommendation {
  id: number
  equipment_id: string
  category: string
  condition_key: string
  condition: string
  metric: string | null
  action: string
  evidence_level: EvidenceLevel
  evidence: {
    method?: string
    slope_per_day?: number
    ci_low?: number
    ci_high?: number
    tau?: number
    p_value?: number
    n?: number
    span_h?: number
    series?: { t: string; v: number }[]
    related_anomalies?: number[]
    [k: string]: unknown
  }
  status: 'active' | 'resolved'
  acknowledged: boolean
  acknowledged_by: string | null
  acknowledged_at: string | null
  data_source: DataSource
  created_at: string
  updated_at: string
  resolved_at: string | null
}

export interface EquipmentDetails {
  equipment: Equipment
  sensors: Sensor[]
  latest: Reading[]
  health: EquipmentHealth | null
  health_history: { at: string; score: number | null; status: HealthStatus }[]
  anomalies: Anomaly[]
  recommendations: Recommendation[]
  threshold_alerts: TelemetryAlert[]
  visual: { links: Equipment['visual_links']; note: string }
}

export interface AnomalyModelInfo {
  id: number
  category: string
  data_source: DataSource
  algorithm: string
  version: string
  active: boolean
  trained_at: string
  features: string[]
  thresholds: Record<string, number>
  training: Record<string, unknown>
  evaluation: Record<string, unknown>
}

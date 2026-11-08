import type { Anomaly, AnomalyModelInfo, EquipmentDetails, EquipmentHealth, HealthSummary, Recommendation } from '../types/analytics'
import type { Page, TimeRange } from '../types/telemetry'
import { request } from './api'

const A = '/api/analytics'
const post = (body: unknown): RequestInit => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
const qs = (params: Record<string, string | number | boolean | null | undefined>) => {
  const p = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') p.set(k, String(v))
  const s = p.toString()
  return s ? `?${s}` : ''
}

export const analytics = {
  status: () => request<{ available: boolean; error?: string; config_version?: number; models?: string[]; engine?: Record<string, number> }>(`${A}/status`),
  summary: () => request<HealthSummary>(`${A}/health/summary`),
  healthList: (category?: string) => request<EquipmentHealth[]>(`${A}/health/equipment${qs({ category })}`),
  health: (id: string) => request<EquipmentHealth>(`${A}/health/equipment/${encodeURIComponent(id)}`),
  healthHistory: (id: string, range: TimeRange) =>
    request<{ at: string; score: number | null; status: string; coverage: number; data_quality: number | null }[]>(
      `${A}/health/equipment/${encodeURIComponent(id)}/history${qs({ range })}`,
    ),
  anomalies: (params: Record<string, string | number | boolean | undefined> = {}) => request<Page<Anomaly>>(`${A}/anomalies${qs(params)}`),
  ack: (id: number, by: string, note?: string) => request<Anomaly>(`${A}/anomalies/${id}/ack`, post({ by, note: note || null })),
  resolve: (id: number, by: string, note?: string) => request<Anomaly>(`${A}/anomalies/${id}/resolve`, post({ by, note: note || null })),
  recommendations: (params: Record<string, string | number | undefined> = {}) => request<Page<Recommendation>>(`${A}/recommendations${qs(params)}`),
  ackRecommendation: (id: number, by: string) => request<Recommendation>(`${A}/recommendations/${id}/ack`, post({ by })),
  runMaintenance: () => request<{ updated_recommendations: number }>(`${A}/maintenance/run`, { method: 'POST' }),
  details: (id: string) => request<EquipmentDetails>(`${A}/equipment/${encodeURIComponent(id)}/details`),
  models: () => request<AnomalyModelInfo[]>(`${A}/models`),
}

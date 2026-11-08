import { createContext, useContext, useEffect, useReducer, type ReactNode } from 'react'
import { wsUrl } from '../services/api'
import { telemetry } from '../services/telemetry'
import type { MetricDef, Reading, Sensor, TelemetryAlert, TelemetryEvent, TelemetryMessage, TelemetrySummary } from '../types/telemetry'
import type { Anomaly, HealthStatus } from '../types/analytics'
import { usePolling } from './usePolling'

/** One live connection to /ws/telemetry for the whole dashboard; pages read from this shared state. */

export type Connection = 'connecting' | 'open' | 'reconnecting' | 'unavailable'

interface State {
  connection: Connection
  error: string | null
  sensors: Record<string, Sensor>
  latest: Record<string, Reading> // key: sensor_id|metric
  alerts: Record<number, TelemetryAlert> // active alerts only
  activity: Reading[] // most recent readings first
  /** live anomaly events since the page loaded (latest version of each anomaly, newest first) */
  anomalyFeed: Anomaly[]
  health: Record<string, { status: HealthStatus; score: number | null; coverage: number; data_quality: number | null }>
  /** increments on every anomaly / recommendation event so pages can refetch */
  analyticsVersion: number
  dropped: number
  lastMessageAt: number | null
}

type Action =
  | { type: 'connection'; connection: Connection; error?: string | null }
  | { type: 'message'; msg: TelemetryMessage }

const initial: State = {
  connection: 'connecting',
  error: null,
  sensors: {},
  latest: {},
  alerts: {},
  activity: [],
  anomalyFeed: [],
  health: {},
  analyticsVersion: 0,
  dropped: 0,
  lastMessageAt: null,
}
const key = (r: { sensor_id: string; metric: string }) => `${r.sensor_id}|${r.metric}`

function applyEvent(s: State, e: TelemetryEvent): State {
  if (e.type === 'measurements') {
    const latest = { ...s.latest }
    const sensors = { ...s.sensors }
    for (const r of e.items) {
      latest[key(r)] = r
      const sensor = sensors[r.sensor_id]
      if (sensor) sensors[r.sensor_id] = { ...sensor, last_seen_at: new Date().toISOString() }
    }
    return { ...s, latest, sensors, activity: [...[...e.items].reverse(), ...s.activity].slice(0, 40) }
  }
  if (e.type === 'sensor_status') {
    const sensor = s.sensors[e.sensor_id]
    const sensors = sensor ? { ...s.sensors, [e.sensor_id]: { ...sensor, status: e.status, last_seen_at: e.last_seen_at } } : s.sensors
    const latest = { ...s.latest }
    for (const [k, r] of Object.entries(latest)) {
      if (r.sensor_id === e.sensor_id) latest[k] = { ...r, sensor_status: e.status, current: e.status === 'online' && r.quality === 'good' }
    }
    return { ...s, sensors, latest }
  }
  if (e.type === 'anomaly') {
    const feed = [e.anomaly, ...s.anomalyFeed.filter((a) => a.id !== e.anomaly.id)].slice(0, 60)
    return { ...s, anomalyFeed: feed, analyticsVersion: s.analyticsVersion + 1 }
  }
  if (e.type === 'health') {
    return { ...s, health: { ...s.health, [e.equipment_id]: { status: e.status, score: e.score, coverage: e.coverage, data_quality: e.data_quality } } }
  }
  if (e.type === 'recommendation') return { ...s, analyticsVersion: s.analyticsVersion + 1 }
  const alerts = { ...s.alerts }
  if (e.alert.state === 'active') alerts[e.alert.id] = e.alert
  else delete alerts[e.alert.id]
  return { ...s, alerts }
}

function reducer(s: State, a: Action): State {
  if (a.type === 'connection') return { ...s, connection: a.connection, error: a.error ?? null }
  const msg = a.msg
  if (msg.type === 'error') return { ...s, connection: 'unavailable', error: msg.message }
  if (msg.type === 'snapshot') {
    return {
      ...s,
      connection: 'open',
      error: null,
      sensors: Object.fromEntries(msg.sensors.map((x) => [x.sensor_id, x])),
      latest: Object.fromEntries(msg.latest.map((r) => [key(r), r])),
      alerts: Object.fromEntries(msg.alerts.map((x) => [x.id, x])),
      lastMessageAt: Date.now(),
    }
  }
  let next = s
  for (const e of msg.events) next = applyEvent(next, e)
  return { ...next, dropped: msg.dropped, lastMessageAt: Date.now() }
}

interface Ctx extends State {
  summary: TelemetrySummary | null
  summaryError: string | null
  metrics: Record<string, MetricDef>
  refreshSummary: () => void
  /** reload sensors from REST after registry edits (the socket only streams readings/status/alerts) */
  reloadSensors: () => Promise<void>
}

const TelemetryContext = createContext<Ctx | null>(null)

export function TelemetryProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initial)
  const summary = usePolling(telemetry.summary, 5000)
  const metrics = usePolling(telemetry.metrics, 0)

  useEffect(() => {
    let ws: WebSocket | null = null
    let retry = 0
    let timer: number | undefined
    let closed = false

    const connect = () => {
      dispatch({ type: 'connection', connection: retry ? 'reconnecting' : 'connecting' })
      ws = new WebSocket(wsUrl('/ws/telemetry'))
      ws.onmessage = (ev) => {
        retry = 0
        try {
          dispatch({ type: 'message', msg: JSON.parse(ev.data) as TelemetryMessage })
        } catch {
          /* ignore malformed frames */
        }
      }
      ws.onclose = () => {
        if (closed) return
        retry += 1
        dispatch({ type: 'connection', connection: 'reconnecting' })
        timer = window.setTimeout(connect, Math.min(10000, 1000 * 2 ** Math.min(retry, 4)))
      }
    }
    connect()
    return () => {
      closed = true
      window.clearTimeout(timer)
      ws?.close()
    }
  }, [])

  const reloadSensors = async () => {
    const page = await telemetry.sensors()
    dispatch({
      type: 'message',
      msg: {
        type: 'snapshot',
        summary: summary.data as TelemetrySummary,
        sensors: page.items,
        latest: Object.values(state.latest),
        alerts: Object.values(state.alerts),
      },
    })
    summary.refresh()
  }

  const value: Ctx = {
    ...state,
    summary: summary.data,
    summaryError: summary.error,
    metrics: Object.fromEntries((metrics.data ?? []).map((m) => [m.name, m])),
    refreshSummary: summary.refresh,
    reloadSensors,
  }
  return <TelemetryContext.Provider value={value}>{children}</TelemetryContext.Provider>
}

export function useTelemetry(): Ctx {
  const ctx = useContext(TelemetryContext)
  if (!ctx) throw new Error('useTelemetry must be used inside <TelemetryProvider>')
  return ctx
}

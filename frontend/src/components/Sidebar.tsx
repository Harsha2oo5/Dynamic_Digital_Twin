import type { LucideIcon } from 'lucide-react'
import { Activity, BellRing, Boxes, ChartLine, Cpu, Database, HeartPulse, Layers, LayoutDashboard, Radar, Radio, Settings, Video, Wrench } from 'lucide-react'
import { usePolling } from '../hooks/usePolling'
import { useTelemetry } from '../hooks/useTelemetry'
import { api } from '../services/api'
import { cx, StatusDot } from './ui'

export type PageId =
  | 'overview'
  | 'live'
  | 'inventory'
  | 'equipment'
  | 'sensors'
  | 'analytics'
  | 'alerts'
  | 'health'
  | 'anomalies'
  | 'maintenance'
  | 'details'
  | 'models'
  | 'dataset'
  | 'settings'

export const NAV: { id: PageId; label: string; icon: LucideIcon }[] = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'live', label: 'Live Monitoring', icon: Video },
  { id: 'inventory', label: 'Object Inventory', icon: Boxes },
  { id: 'equipment', label: 'Equipment Monitoring', icon: Activity },
  { id: 'sensors', label: 'Sensor Management', icon: Radio },
  { id: 'analytics', label: 'Telemetry Analytics', icon: ChartLine },
  { id: 'alerts', label: 'Alerts', icon: BellRing },
  { id: 'health', label: 'Equipment Health', icon: HeartPulse },
  { id: 'anomalies', label: 'Anomaly Monitoring', icon: Radar },
  { id: 'maintenance', label: 'Predictive Maintenance', icon: Wrench },
  { id: 'models', label: 'Model Management', icon: Layers },
  { id: 'dataset', label: 'Dataset Management', icon: Database },
  { id: 'settings', label: 'Settings', icon: Settings },
]

export function Sidebar({ page, onNavigate }: { page: PageId; onNavigate: (p: PageId) => void }) {
  const health = usePolling(api.health, 5000)
  const { alerts } = useTelemetry()
  const activeAlerts = Object.values(alerts)
  const { anomalyFeed } = useTelemetry()
  const ongoing = anomalyFeed.filter((a) => a.condition_active && a.status !== 'resolved').length
  const critical = activeAlerts.some((a) => a.severity === 'critical')
  const tone = health.error ? 'bad' : health.data?.status === 'ok' ? 'good' : health.data ? 'warn' : 'neutral'
  const statusText = health.error ? 'Backend offline' : health.data?.status === 'ok' ? 'Backend online' : health.data ? 'Model not loaded' : 'Connecting…'

  return (
    <aside className="flex w-full shrink-0 flex-col border-b border-line bg-surface-1 md:h-full md:w-60 md:border-r md:border-b-0">
      <div className="flex items-center gap-2.5 px-4 py-4">
        <div className="grid size-8 place-items-center rounded-md bg-accent/15 text-accent-strong">
          <Boxes size={18} />
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold text-ink">Dynamic Digital Twin</div>
          <div className="text-xs text-ink-3">Vision · Telemetry · Analytics · v0.3</div>
        </div>
      </div>

      <nav className="flex gap-1 overflow-x-auto px-2 pb-2 md:flex-1 md:flex-col md:overflow-visible md:pb-0">
        {NAV.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => onNavigate(id)}
            className={cx(
              'flex shrink-0 items-center gap-3 rounded-md px-3 py-2 text-left text-sm transition-colors',
              page === id ? 'bg-surface-3 text-ink' : 'text-ink-2 hover:bg-surface-2 hover:text-ink',
            )}
          >
            <Icon size={16} className={page === id ? 'text-accent-strong' : ''} />
            {label}
            {id === 'anomalies' && ongoing > 0 && (
              <span className="ml-auto rounded bg-warn/20 px-1.5 text-xs font-semibold text-warn">{ongoing}</span>
            )}
            {id === 'alerts' && activeAlerts.length > 0 && (
              <span className={cx('ml-auto rounded px-1.5 text-xs font-semibold', critical ? 'bg-bad/20 text-bad' : 'bg-warn/20 text-warn')}>
                {activeAlerts.length}
              </span>
            )}
          </button>
        ))}
      </nav>

      <div className="hidden space-y-2 border-t border-line px-4 py-3 text-xs md:block">
        <div className="flex items-center gap-2 text-ink-2">
          <StatusDot tone={tone} />
          {statusText}
        </div>
        <div className="flex items-center gap-2 text-ink-3">
          <Cpu size={13} />
          Execution device: <span className="font-mono text-ink-2">{health.data?.device?.toUpperCase() ?? '—'}</span>
        </div>
      </div>
    </aside>
  )
}

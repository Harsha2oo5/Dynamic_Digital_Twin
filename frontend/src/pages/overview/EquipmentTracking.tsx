import { Boxes } from 'lucide-react'
import { Badge, Empty, Panel } from '../../components/ui'
import { usePolling } from '../../hooks/usePolling'
import { api } from '../../services/api'

/** Overview block: unique equipment items by class (session-level identity) and tracker lifecycle counters. */
export function EquipmentTracking() {
  const summary = usePolling(api.inventorySummary, 3000)
  const sessions = usePolling(api.sessions, 3000)
  const s = summary.data
  const live = (sessions.data ?? []).filter((x) => x.live)
  const totals = live.reduce<Record<string, number>>((acc, x) => {
    for (const [k, v] of Object.entries(x.tracking)) acc[k] = (acc[k] ?? 0) + v
    return acc
  }, {})
  const max = Math.max(1, ...(s?.by_class.map((c) => c.items) ?? [1]))

  return (
    <Panel
      title="Equipment tracking"
      icon={Boxes}
      className="mt-4"
      actions={s ? <Badge tone={s.active_items ? 'good' : 'neutral'}>{s.active_items} active now</Badge> : undefined}
    >
      {s && s.total_items ? (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
          <div className="space-y-2">
            {s.by_class.map((c) => (
              <div key={c.class_name} className="grid grid-cols-[9rem_minmax(0,1fr)_4rem] items-center gap-3 text-sm">
                <span className="truncate text-ink-2">{c.class_name}</span>
                <div className="h-2 rounded bg-surface-3">
                  <div className="h-2 rounded bg-accent" style={{ width: `${(c.items / max) * 100}%` }} />
                </div>
                <span className="tabular text-right">
                  {c.items}
                  <span className="ml-1 text-xs text-ink-3">({c.active})</span>
                </span>
              </div>
            ))}
            <p className="pt-1 text-xs text-ink-3">Unique items per class across {s.sessions} session(s); (n) = visible now. IDs are session-level, not permanent.</p>
          </div>
          <div className="space-y-1.5 text-sm">
            <div className="text-xs uppercase tracking-wider text-ink-3">Live tracking ({live.length} session{live.length === 1 ? '' : 's'})</div>
            {[
              ['Active tracks', totals.active],
              ['Tentative (new)', totals.new],
              ['Lost (in buffer)', totals.lost],
              ['Removed', totals.removed],
              ['Re-linked items', totals.relinks],
            ].map(([k, v]) => (
              <div key={k as string} className="flex justify-between">
                <span className="text-ink-2">{k}</span>
                <span className="tabular">{v ?? 0}</span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <Empty icon={Boxes} title="No equipment identified yet">
          Run detection on a webcam stream or video; items appear once their track is confirmed.
        </Empty>
      )}
    </Panel>
  )
}

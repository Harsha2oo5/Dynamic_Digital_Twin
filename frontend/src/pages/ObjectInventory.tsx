import { Boxes, History, Layers, Search, Trash2, X } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Alert, Badge, Button, cx, Empty, KV, PageHeader, Panel } from '../components/ui'
import { usePolling } from '../hooks/usePolling'
import { colorFor, dateTime, pct, time } from '../lib/format'
import { api } from '../services/api'
import type { TrackedObject, TrackState } from '../types/api'
import { selectCls } from './live/Controls'

const statusTone = { active: 'good', lost: 'neutral', snapshot: 'accent' } as const
const lifecycleTone: Record<TrackState, 'good' | 'warn' | 'neutral' | 'accent'> = {
  new: 'accent',
  active: 'good',
  lost: 'warn',
  removed: 'neutral',
}

function duration(s: number | null | undefined): string {
  if (s == null) return '—'
  if (s < 60) return `${s.toFixed(1)} s`
  return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`
}

function itemColor(o: TrackedObject) {
  return colorFor(o.equipment_id ? [...o.equipment_id].reduce((a, c) => a + c.charCodeAt(0), 0) : (o.track_id ?? o.class_id))
}

function BoxPreview({ obj }: { obj: TrackedObject }) {
  const { last_bbox: b, frame_width: w, frame_height: h } = obj
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full rounded border border-line bg-surface-0">
      <rect x={b.x1} y={b.y1} width={b.x2 - b.x1} height={b.y2 - b.y1} fill="none" stroke={itemColor(obj)} strokeWidth={Math.max(w, h) / 200} />
    </svg>
  )
}

function Detail({ obj, onClose }: { obj: TrackedObject; onClose: () => void }) {
  return (
    <Panel
      title={obj.equipment_id ?? `Object ${obj.object_id}`}
      actions={
        <button onClick={onClose} className="text-ink-3 hover:text-ink" aria-label="Close">
          <X size={16} />
        </button>
      }
      bodyClassName="space-y-4 p-4"
    >
      <KV
        rows={[
          ['Class', obj.class_name],
          ['Class consistency', obj.class_consistency != null ? pct(obj.class_consistency, 0) : '—'],
          ['Component type', obj.component_type ?? <span className="text-ink-3">n/a (pretrained model)</span>],
          ['Status', <Badge tone={statusTone[obj.status]}>{obj.status}</Badge>],
          ['Lifecycle', obj.track_state ? <Badge tone={lifecycleTone[obj.track_state]}>{obj.track_state}</Badge> : '—'],
          ['Equipment ID', obj.equipment_id ? <span className="font-mono text-xs">{obj.equipment_id} (session-level)</span> : '—'],
          ['Tracker IDs', obj.track_ids?.length ? obj.track_ids.map((t) => `#${t}`).join(', ') : obj.track_id != null ? `#${obj.track_id}` : '—'],
          ['Re-linked', obj.relinked ? `${obj.relinked}× after the tracker ID changed` : 'no'],
          ['Source', `${obj.source_type} · ${obj.source_id}`],
          ['Last confidence', pct(obj.confidence, 1)],
          ['Avg / max conf.', `${pct(obj.avg_confidence, 1)} / ${pct(obj.max_confidence, 1)}`],
          ['Observations', obj.frames_seen],
          ['Duration', duration(obj.duration_s)],
          ['First detected', dateTime(obj.first_seen)],
          ['Last detected', dateTime(obj.last_seen)],
          ['Frame size', `${obj.frame_width}×${obj.frame_height}`],
        ]}
      />
      <div>
        <div className="mb-1.5 text-xs text-ink-3">Last position in frame</div>
        <BoxPreview obj={obj} />
      </div>
      <p className="text-xs text-ink-3">
        Equipment IDs are session-level: stable while the item is re-identified within one stream or video, but not permanent. The same physical item seen in
        another session gets another ID.
      </p>
    </Panel>
  )
}

export function ObjectInventory() {
  const objects = usePolling(api.objects, 1500)
  const summary = usePolling(api.inventorySummary, 3000)
  const sessions = usePolling(api.sessions, 3000)
  const [selected, setSelected] = useState<string | null>(null)
  const [status, setStatus] = useState<'all' | TrackedObject['status']>('all')
  const [cls, setCls] = useState('all')
  const [session, setSession] = useState('all')
  const [query, setQuery] = useState('')

  const classes = useMemo(() => [...new Set((objects.data ?? []).map((o) => o.class_name))].sort(), [objects.data])
  const rows = useMemo(
    () =>
      (objects.data ?? []).filter(
        (o) =>
          (status === 'all' || o.status === status) &&
          (cls === 'all' || o.class_name === cls) &&
          (session === 'all' || o.source_id === session) &&
          `${o.class_name} ${o.equipment_id ?? ''}`.toLowerCase().includes(query.toLowerCase()),
      ),
    [objects.data, status, cls, session, query],
  )
  const selectedObj = objects.data?.find((o) => o.object_id === selected) ?? null

  const clear = async () => {
    if (!window.confirm('Clear the object inventory and detection statistics?')) return
    await api.clearObjects()
    setSelected(null)
    objects.refresh()
    summary.refresh()
    sessions.refresh()
  }

  const s = summary.data
  return (
    <>
      <PageHeader
        title="Object Inventory"
        subtitle="One row per equipment item identified in a session (webcam, video or image)"
        actions={
          <Button icon={Trash2} variant="danger" onClick={clear} disabled={!objects.data?.length}>
            Clear
          </Button>
        }
      />
      {objects.error && <Alert tone="bad" title="Could not load inventory">{objects.error}</Alert>}

      <div className="mb-4 grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Panel title="Equipment by class" icon={Layers}>
          {s && s.by_class.length ? (
            <>
              <div className="flex flex-wrap gap-2">
                {s.by_class.map((c) => (
                  <button
                    key={c.class_name}
                    onClick={() => setCls(cls === c.class_name ? 'all' : c.class_name)}
                    className={cx(
                      'rounded border px-2.5 py-1.5 text-left text-sm',
                      cls === c.class_name ? 'border-accent bg-surface-3' : 'border-line bg-surface-2 hover:bg-surface-3',
                    )}
                  >
                    <div className="text-xs text-ink-3">{c.class_name}</div>
                    <div className="tabular">
                      <span className="font-semibold">{c.items}</span>
                      <span className="ml-1.5 text-xs text-ink-3">{c.active} active</span>
                    </div>
                  </button>
                ))}
              </div>
              <p className="mt-3 text-xs text-ink-3">
                {s.total_items} items in {s.sessions} session{s.sessions === 1 ? '' : 's'} · {s.note}
              </p>
            </>
          ) : (
            <Empty icon={Layers} title="No equipment identified yet" />
          )}
        </Panel>
        <Panel title="Sessions" icon={History} bodyClassName="p-0">
          {sessions.data?.length ? (
            <div className="max-h-56 overflow-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-surface-1 text-left text-xs uppercase tracking-wider text-ink-3">
                  <tr className="border-b border-line">
                    {['Session', 'Items', 'Re-links', 'Started', ''].map((h) => (
                      <th key={h} className="px-4 py-2 font-medium">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="tabular divide-y divide-line">
                  {sessions.data.map((x) => (
                    <tr
                      key={x.source_id}
                      onClick={() => setSession(session === x.source_id ? 'all' : x.source_id)}
                      className={cx('cursor-pointer hover:bg-surface-2', session === x.source_id && 'bg-surface-3')}
                    >
                      <td className="px-4 py-2">
                        <div>{x.label ?? x.source_type}</div>
                        <div className="font-mono text-xs text-ink-3">{x.source_id}</div>
                      </td>
                      <td className="px-4 py-2">
                        {x.equipment_items}
                        <span className="ml-1 text-xs text-ink-3">({x.active_items} active)</span>
                      </td>
                      <td className="px-4 py-2">{x.tracking.relinks ?? '—'}</td>
                      <td className="px-4 py-2 text-ink-2">{x.started_at ? time(x.started_at) : '—'}</td>
                      <td className="px-4 py-2">{x.live ? <Badge tone="good">live</Badge> : <Badge tone="neutral">ended</Badge>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty icon={History} title="No sessions yet" />
          )}
        </Panel>
      </div>

      <div className={cx('grid gap-4', selectedObj && 'xl:grid-cols-[minmax(0,1fr)_22rem]')}>
        <Panel
          title={`${rows.length} item${rows.length === 1 ? '' : 's'}`}
          icon={Boxes}
          bodyClassName="p-0"
          actions={
            <>
              <div className="relative">
                <Search size={13} className="absolute top-1/2 left-2 -translate-y-1/2 text-ink-3" />
                <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search" className={cx(selectCls, 'w-32 pl-7')} />
              </div>
              <select value={cls} onChange={(e) => setCls(e.target.value)} className={selectCls}>
                <option value="all">All classes</option>
                {classes.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
              <select value={status} onChange={(e) => setStatus(e.target.value as typeof status)} className={selectCls}>
                <option value="all">All statuses</option>
                <option value="active">Active</option>
                <option value="lost">Lost</option>
                <option value="snapshot">Snapshot</option>
              </select>
            </>
          }
        >
          {rows.length ? (
            <div className="max-h-[60vh] overflow-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-surface-1 text-left text-xs uppercase tracking-wider text-ink-3">
                  <tr className="border-b border-line">
                    {['Equipment', 'Class', 'Confidence', 'Duration', 'Last detected', 'Obs.', 'Lifecycle', 'Status'].map((h) => (
                      <th key={h} className="px-4 py-2 font-medium">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="tabular divide-y divide-line">
                  {rows.map((o) => (
                    <tr
                      key={o.object_id}
                      onClick={() => setSelected(o.object_id)}
                      className={cx('cursor-pointer hover:bg-surface-2', selected === o.object_id && 'bg-surface-3')}
                    >
                      <td className="px-4 py-2 font-mono text-xs">
                        <span className="inline-flex items-center gap-2">
                          <span className="size-2.5 rounded-sm" style={{ background: itemColor(o) }} />
                          {o.equipment_id ? o.equipment_id.split('@')[0] : o.object_id}
                          {!!o.relinked && <span className="text-ink-3" title="Re-linked after a tracker ID change">↺{o.relinked}</span>}
                        </span>
                      </td>
                      <td className="px-4 py-2">
                        {o.class_name}
                        {o.class_consistency != null && o.class_consistency < 0.8 && (
                          <span className="ml-1 text-xs text-warn" title="Share of the track's class votes">
                            {pct(o.class_consistency, 0)}
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-2">{pct(o.avg_confidence, 1)}</td>
                      <td className="px-4 py-2">{duration(o.duration_s)}</td>
                      <td className="px-4 py-2 text-ink-2">{time(o.last_seen)}</td>
                      <td className="px-4 py-2">{o.frames_seen}</td>
                      <td className="px-4 py-2">{o.track_state ? <Badge tone={lifecycleTone[o.track_state]}>{o.track_state}</Badge> : '—'}</td>
                      <td className="px-4 py-2">
                        <Badge tone={statusTone[o.status]}>{o.status}</Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty icon={Boxes} title={objects.loading ? 'Loading…' : 'No equipment observed yet'}>
              Items appear here once a track is confirmed (seen in several consecutive frames) in a webcam stream or video, or for each detection in an
              uploaded image.
            </Empty>
          )}
        </Panel>
        {selectedObj && <Detail obj={selectedObj} onClose={() => setSelected(null)} />}
      </div>
    </>
  )
}

import { Boxes, Search, Trash2, X } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Alert, Badge, Button, cx, Empty, KV, PageHeader, Panel } from '../components/ui'
import { usePolling } from '../hooks/usePolling'
import { colorFor, dateTime, pct, time } from '../lib/format'
import { api } from '../services/api'
import type { TrackedObject } from '../types/api'
import { selectCls } from './live/Controls'

const statusTone = { active: 'good', lost: 'neutral', snapshot: 'accent' } as const

function BoxPreview({ obj }: { obj: TrackedObject }) {
  const { last_bbox: b, frame_width: w, frame_height: h } = obj
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full rounded border border-line bg-surface-0">
      <rect x={b.x1} y={b.y1} width={b.x2 - b.x1} height={b.y2 - b.y1} fill="none" stroke={colorFor(obj.track_id ?? obj.class_id)} strokeWidth={Math.max(w, h) / 200} />
    </svg>
  )
}

function Detail({ obj, onClose }: { obj: TrackedObject; onClose: () => void }) {
  return (
    <Panel
      title={`Object ${obj.object_id}`}
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
          ['Component type', obj.component_type ?? <span className="text-ink-3">n/a (pretrained model)</span>],
          ['Status', <Badge tone={statusTone[obj.status]}>{obj.status}</Badge>],
          ['Tracking ID', obj.track_id != null ? `#${obj.track_id} (temporary)` : '—'],
          ['Source', `${obj.source_type} · ${obj.source_id}`],
          ['Last confidence', pct(obj.confidence, 1)],
          ['Avg / max conf.', `${pct(obj.avg_confidence, 1)} / ${pct(obj.max_confidence, 1)}`],
          ['Frames seen', obj.frames_seen],
          ['First detected', dateTime(obj.first_seen)],
          ['Last detected', dateTime(obj.last_seen)],
          ['Last box', `${Math.round(obj.last_bbox.x1)}, ${Math.round(obj.last_bbox.y1)} → ${Math.round(obj.last_bbox.x2)}, ${Math.round(obj.last_bbox.y2)}`],
          ['Frame size', `${obj.frame_width}×${obj.frame_height}`],
        ]}
      />
      <div>
        <div className="mb-1.5 text-xs text-ink-3">Last position in frame</div>
        <BoxPreview obj={obj} />
      </div>
      <p className="text-xs text-ink-3">
        Tracking IDs are temporary visual identifiers scoped to one stream or video. They are not permanent physical component IDs.
      </p>
    </Panel>
  )
}

export function ObjectInventory() {
  const objects = usePolling(api.objects, 1500)
  const [selected, setSelected] = useState<string | null>(null)
  const [status, setStatus] = useState<'all' | TrackedObject['status']>('all')
  const [query, setQuery] = useState('')

  const rows = useMemo(
    () => (objects.data ?? []).filter((o) => (status === 'all' || o.status === status) && o.class_name.toLowerCase().includes(query.toLowerCase())),
    [objects.data, status, query],
  )
  const selectedObj = objects.data?.find((o) => o.object_id === selected) ?? null

  const clear = async () => {
    if (!window.confirm('Clear the object inventory and detection statistics?')) return
    await api.clearObjects()
    setSelected(null)
    objects.refresh()
  }

  return (
    <>
      <PageHeader
        title="Object Inventory"
        subtitle="Objects observed by the tracker across webcam, video and image sources"
        actions={
          <Button icon={Trash2} variant="danger" onClick={clear} disabled={!objects.data?.length}>
            Clear
          </Button>
        }
      />
      {objects.error && <Alert tone="bad" title="Could not load inventory">{objects.error}</Alert>}
      <div className={cx('grid gap-4', selectedObj && 'xl:grid-cols-[minmax(0,1fr)_22rem]')}>
        <Panel
          title={`${rows.length} object${rows.length === 1 ? '' : 's'}`}
          icon={Boxes}
          bodyClassName="p-0"
          actions={
            <>
              <div className="relative">
                <Search size={13} className="absolute top-1/2 left-2 -translate-y-1/2 text-ink-3" />
                <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Filter class" className={cx(selectCls, 'w-36 pl-7')} />
              </div>
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
            <div className="max-h-[70vh] overflow-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-surface-1 text-left text-xs uppercase tracking-wider text-ink-3">
                  <tr className="border-b border-line">
                    {['Object ID', 'Class', 'Confidence', 'First detected', 'Last detected', 'Frames', 'Status'].map((h) => (
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
                          <span className="size-2.5 rounded-sm" style={{ background: colorFor(o.track_id ?? o.class_id) }} />
                          {o.object_id}
                        </span>
                      </td>
                      <td className="px-4 py-2">{o.class_name}</td>
                      <td className="px-4 py-2">{pct(o.confidence, 1)}</td>
                      <td className="px-4 py-2 text-ink-2">{time(o.first_seen)}</td>
                      <td className="px-4 py-2 text-ink-2">{time(o.last_seen)}</td>
                      <td className="px-4 py-2">{o.frames_seen}</td>
                      <td className="px-4 py-2">
                        <Badge tone={statusTone[o.status]}>{o.status}</Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty icon={Boxes} title={objects.loading ? 'Loading…' : 'No objects observed yet'}>
              Objects appear here once detection runs on a webcam stream, video or image.
            </Empty>
          )}
        </Panel>
        {selectedObj && <Detail obj={selectedObj} onClose={() => setSelected(null)} />}
      </div>
    </>
  )
}

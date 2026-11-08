import { ScanLine } from 'lucide-react'
import { colorFor, pct } from '../lib/format'
import type { Detection } from '../types/api'
import { Empty } from './ui'

export function DetectionTable({ detections }: { detections: Detection[] }) {
  if (!detections.length) {
    return (
      <Empty icon={ScanLine} title="No detections in this frame">
        Try lowering the confidence threshold.
      </Empty>
    )
  }
  return (
    <div className="max-h-80 overflow-auto">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-surface-1 text-left text-xs uppercase tracking-wider text-ink-3">
          <tr>
            <th className="py-2 pr-3 font-medium">Track / equipment</th>
            <th className="py-2 pr-3 font-medium">Class</th>
            <th className="py-2 pr-3 font-medium">State</th>
            <th className="py-2 pr-3 text-right font-medium">Conf.</th>
            <th className="py-2 font-medium">Box (x1, y1, x2, y2)</th>
          </tr>
        </thead>
        <tbody className="tabular divide-y divide-line">
          {[...detections]
            .sort((a, b) => b.confidence - a.confidence)
            .map((d, i) => (
              <tr key={`${d.track_id ?? 'n'}-${i}`}>
                <td className="py-1.5 pr-3">
                  <span className="inline-flex items-center gap-2">
                    <span className="size-2.5 rounded-sm" style={{ background: colorFor(d.track_id ?? d.class_id) }} />
                    {d.track_id != null ? `#${d.track_id}` : <span className="text-ink-3">—</span>}
                    {d.equipment_id && <span className="font-mono text-xs text-ink-2">{d.equipment_id.split('@')[0]}</span>}
                  </span>
                </td>
                <td className="py-1.5 pr-3">
                  {d.class_name}
                  {d.stable_class && d.stable_class !== d.class_name && (
                    <span className="ml-1 text-xs text-ink-3" title="Class voted over the whole track">
                      (track: {d.stable_class})
                    </span>
                  )}
                </td>
                <td className="py-1.5 pr-3 text-xs text-ink-2">{d.track_state ?? '—'}</td>
                <td className="py-1.5 pr-3 text-right">{pct(d.confidence, 1)}</td>
                <td className="py-1.5 font-mono text-xs text-ink-2">
                  {[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2].map((v) => Math.round(v)).join(', ')}
                </td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  )
}

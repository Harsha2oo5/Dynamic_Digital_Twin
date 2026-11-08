import { Activity } from 'lucide-react'
import { KV, Panel } from '../../components/ui'
import { int, num } from '../../lib/format'
import type { FrameResult } from '../../types/api'

export function FrameMetrics({ result, extra = [] }: { result: FrameResult | null; extra?: [string, React.ReactNode][] }) {
  const tracked = result?.detections.filter((d) => d.track_id != null).length
  const dets = result?.detections ?? []
  const confirmed = new Set(dets.filter((d) => d.equipment_id).map((d) => d.equipment_id)).size
  const tentative = dets.filter((d) => d.track_state === 'new').length
  const byClass = Object.entries(
    dets.filter((d) => d.equipment_id).reduce<Record<string, number>>((acc, d) => {
      const c = d.stable_class ?? d.class_name
      acc[c] = (acc[c] ?? 0) + 1
      return acc
    }, {}),
  )
  return (
    <Panel title="Frame metrics" icon={Activity}>
      <KV
        rows={[
          ['Device', <span className="font-mono">{result?.device.toUpperCase() ?? '—'}</span>],
          ['Inference time', result ? `${num(result.inference_ms)} ms` : '—'],
          ['Inference FPS', num(result?.fps)],
          ...extra,
          ['Frame', result ? `#${result.frame_id} · ${result.frame_width}×${result.frame_height}` : '—'],
          ['Detections', int(result?.detections.length)],
          ['Tracked', int(tracked)],
          ['Equipment in view', result ? `${confirmed} confirmed · ${tentative} tentative` : '—'],
          ...(byClass.length
            ? ([['By class', <span className="text-xs">{byClass.map(([c, n]) => `${c} ${n}`).join(' · ')}</span>]] as [string, React.ReactNode][])
            : []),
        ]}
      />
    </Panel>
  )
}

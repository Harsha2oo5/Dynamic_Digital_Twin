import { Activity } from 'lucide-react'
import { KV, Panel } from '../../components/ui'
import { int, num } from '../../lib/format'
import type { FrameResult } from '../../types/api'

export function FrameMetrics({ result, extra = [] }: { result: FrameResult | null; extra?: [string, React.ReactNode][] }) {
  const tracked = result?.detections.filter((d) => d.track_id != null).length
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
        ]}
      />
    </Panel>
  )
}

import { SlidersHorizontal } from 'lucide-react'
import type { ReactNode } from 'react'
import { Panel } from '../../components/ui'
import { useSettings } from '../../hooks/useSettings'

export const IMAGE_SIZES = [320, 416, 512, 640]

function Slider({ label, value, onChange, min = 0.05, max = 0.95 }: { label: string; value: number; onChange: (v: number) => void; min?: number; max?: number }) {
  return (
    <label className="block">
      <div className="mb-1 flex justify-between text-xs">
        <span className="text-ink-2">{label}</span>
        <span className="tabular font-mono text-ink">{value.toFixed(2)}</span>
      </div>
      <input type="range" min={min} max={max} step={0.01} value={value} onChange={(e) => onChange(Number(e.target.value))} className="w-full" />
    </label>
  )
}

export function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex items-center justify-between gap-3 text-xs">
      <span className="text-ink-2">{label}</span>
      {children}
    </label>
  )
}

export const selectCls = 'rounded-md border border-line bg-surface-2 px-2 py-1 text-xs text-ink focus:border-accent focus:outline-none'

export function DetectionControls({ showFrameSkip, showTracking, note }: { showFrameSkip?: boolean; showTracking?: boolean; note?: ReactNode }) {
  const { settings: s, update } = useSettings()
  return (
    <Panel title="Detection settings" icon={SlidersHorizontal} bodyClassName="space-y-4 p-4">
      <Slider label="Confidence threshold" value={s.confidence} onChange={(confidence) => update({ confidence })} />
      <Slider label="IoU threshold (NMS)" value={s.iou} onChange={(iou) => update({ iou })} />
      <Row label="Inference resolution">
        <select className={selectCls} value={s.imageSize} onChange={(e) => update({ imageSize: Number(e.target.value) })}>
          {IMAGE_SIZES.map((v) => (
            <option key={v} value={v}>
              {v} px
            </option>
          ))}
        </select>
      </Row>
      {showFrameSkip && (
        <Row label="Process every Nth frame">
          <select className={selectCls} value={s.frameSkip} onChange={(e) => update({ frameSkip: Number(e.target.value) })}>
            {[1, 2, 3, 5, 10].map((v) => (
              <option key={v} value={v}>
                {v === 1 ? 'every frame' : `every ${v}`}
              </option>
            ))}
          </select>
        </Row>
      )}
      {showTracking && (
        <Row label="ByteTrack tracking">
          <input type="checkbox" checked={s.tracking} onChange={(e) => update({ tracking: e.target.checked })} className="size-4 accent-[var(--color-accent)]" />
        </Row>
      )}
      {note && <p className="text-xs text-ink-3">{note}</p>}
    </Panel>
  )
}

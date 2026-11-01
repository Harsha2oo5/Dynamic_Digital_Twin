import { useEffect, useRef } from 'react'
import { colorFor } from '../lib/format'
import type { Detection } from '../types/api'

interface Props {
  detections: Detection[]
  /** Dimensions of the frame the backend ran inference on (box coordinate space). */
  frameWidth: number
  frameHeight: number
  /** Intrinsic size of the displayed media (video.videoWidth / img.naturalWidth). */
  mediaWidth: number
  mediaHeight: number
  highlightTrack?: number | null
}

/**
 * Canvas overlay that sits exactly on top of an `object-contain` image/video element.
 * Boxes are mapped: frame px -> media px -> letterboxed display px.
 */
export function DetectionOverlay({ detections, frameWidth, frameHeight, mediaWidth, mediaHeight, highlightTrack }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const draw = () => {
      const { clientWidth: cw, clientHeight: ch } = canvas
      const dpr = window.devicePixelRatio || 1
      canvas.width = Math.round(cw * dpr)
      canvas.height = Math.round(ch * dpr)
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, cw, ch)
      if (!frameWidth || !frameHeight || !mediaWidth || !mediaHeight) return

      // Rectangle actually occupied by the media inside the element (object-fit: contain).
      const fit = Math.min(cw / mediaWidth, ch / mediaHeight)
      const dw = mediaWidth * fit
      const dh = mediaHeight * fit
      const ox = (cw - dw) / 2
      const oy = (ch - dh) / 2
      const sx = dw / frameWidth
      const sy = dh / frameHeight

      ctx.font = '600 12px Inter, "Segoe UI", system-ui, sans-serif'
      ctx.textBaseline = 'top'
      for (const d of detections) {
        const color = colorFor(d.track_id ?? d.class_id)
        const x = ox + d.bbox.x1 * sx
        const y = oy + d.bbox.y1 * sy
        const w = (d.bbox.x2 - d.bbox.x1) * sx
        const h = (d.bbox.y2 - d.bbox.y1) * sy
        const highlighted = highlightTrack != null && d.track_id === highlightTrack

        ctx.lineWidth = highlighted ? 3 : 2
        ctx.strokeStyle = color
        ctx.strokeRect(x, y, w, h)

        const label = `${d.track_id != null ? `#${d.track_id} ` : ''}${d.class_name} ${Math.round(d.confidence * 100)}%`
        const tw = ctx.measureText(label).width + 8
        const th = 18
        const ly = y - th >= oy ? y - th : y // keep the label inside the media area
        ctx.fillStyle = color
        ctx.fillRect(x, ly, tw, th)
        ctx.fillStyle = '#0b0f14'
        ctx.fillText(label, x + 4, ly + 3)
      }
    }

    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(canvas)
    return () => ro.disconnect()
  }, [detections, frameWidth, frameHeight, mediaWidth, mediaHeight, highlightTrack])

  return <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
}

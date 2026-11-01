export const pct = (v: number | null | undefined, digits = 0) => (v == null ? '—' : `${(v * 100).toFixed(digits)}%`)

export const num = (v: number | null | undefined, digits = 1) => (v == null ? '—' : v.toFixed(digits))

export const int = (v: number | null | undefined) => (v == null ? '—' : v.toLocaleString())

export function time(iso: string | null | undefined): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export function dateTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'medium' })
}

export function duration(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  return h ? `${h}h ${m}m` : m ? `${m}m ${s}s` : `${s}s`
}

/** Same golden-ratio hue stepping as backend/app/vision/visualization.py, so colours match. */
export function colorFor(key: number): string {
  const hue = ((key * 0.618033988749895) % 1) * 360
  return `hsl(${hue.toFixed(0)} 85% 60%)`
}

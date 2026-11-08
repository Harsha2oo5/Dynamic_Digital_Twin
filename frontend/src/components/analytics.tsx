import { Check, CheckCheck } from 'lucide-react'
import { useState } from 'react'
import { Line, LineChart, ResponsiveContainer, YAxis } from 'recharts'
import type { AnomalyStatus, EvidenceLevel, HealthStatus } from '../types/analytics'
import { Badge, Button, cx } from './ui'

export const HEALTH_TONE = { healthy: 'good', warning: 'warn', degraded: 'warn', critical: 'bad', unknown: 'neutral' } as const
export const HEALTH_COLOR: Record<HealthStatus, string> = {
  healthy: 'var(--color-good)',
  warning: 'var(--color-warn)',
  degraded: '#e3732b',
  critical: 'var(--color-bad)',
  unknown: 'var(--color-ink-3)',
}

export function HealthBadge({ status }: { status: HealthStatus }) {
  return <Badge tone={HEALTH_TONE[status]}>{status}</Badge>
}

export function EvidenceBadge({ level }: { level: EvidenceLevel }) {
  return <Badge tone={level === 'high' ? 'bad' : level === 'medium' ? 'warn' : 'neutral'}>{level} evidence</Badge>
}

export function AnomalyStatusBadge({ status, active }: { status: AnomalyStatus; active?: boolean }) {
  return (
    <span className="inline-flex gap-1">
      <Badge tone={status === 'open' ? 'warn' : status === 'acknowledged' ? 'accent' : 'neutral'}>{status}</Badge>
      {active && <Badge tone="bad">ongoing</Badge>}
    </span>
  )
}

/** Health score with its band colour. `null` = unknown: no number is shown. */
export function HealthScore({ score, status, size = 'md' }: { score: number | null; status: HealthStatus; size?: 'sm' | 'md' | 'lg' }) {
  const text = { sm: 'text-sm', md: 'text-2xl', lg: 'text-4xl' }[size]
  return (
    <div className="min-w-0">
      <div className={cx('tabular font-semibold', text)} style={{ color: HEALTH_COLOR[status] }}>
        {score == null ? '—' : Math.round(score)}
        {score != null && <span className="text-xs font-normal text-ink-3"> / 100</span>}
      </div>
      {size !== 'sm' && (
        <div className="mt-1 h-1.5 w-full overflow-hidden rounded bg-surface-3">
          <div className="h-full rounded" style={{ width: `${score ?? 0}%`, background: HEALTH_COLOR[status] }} />
        </div>
      )}
    </div>
  )
}

export function Sparkline({ points, color = 'var(--color-accent)', domain }: { points: { v: number | null }[]; color?: string; domain?: [number, number] }) {
  if (points.filter((p) => p.v != null).length < 2) return <div className="h-10 text-xs text-ink-3">not enough history</div>
  return (
    <div className="h-10">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
          <YAxis hide domain={domain ?? ['auto', 'auto']} />
          <Line type="monotone" dataKey="v" stroke={color} strokeWidth={1.5} dot={false} isAnimationActive={false} connectNulls={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

const NAME_KEY = 'dt.operator'
const readName = () => {
  try {
    return localStorage.getItem(NAME_KEY) ?? ''
  } catch {
    return ''
  }
}

/** Acknowledge / resolve form (operator name remembered in this browser only). */
export function OperatorAction({
  label,
  onSubmit,
  withNote = true,
  icon = 'check',
}: {
  label: string
  onSubmit: (by: string, note: string) => Promise<unknown>
  withNote?: boolean
  icon?: 'check' | 'double'
}) {
  const [open, setOpen] = useState(false)
  const [by, setBy] = useState(readName)
  const [note, setNote] = useState('')
  const [error, setError] = useState<string | null>(null)
  const Icon = icon === 'double' ? CheckCheck : Check
  if (!open)
    return (
      <Button icon={Icon} onClick={() => setOpen(true)}>
        {label}
      </Button>
    )
  const submit = async () => {
    try {
      await onSubmit(by.trim(), note.trim())
      try {
        localStorage.setItem(NAME_KEY, by.trim())
      } catch {
        /* storage unavailable */
      }
      setOpen(false)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }
  const input = 'rounded-md border border-line bg-surface-2 px-2 py-1 text-sm text-ink focus:border-accent focus:outline-none'
  return (
    <div className="flex flex-wrap items-center gap-2">
      <input className={input + ' w-36'} placeholder="Your name" value={by} onChange={(e) => setBy(e.target.value)} />
      {withNote && <input className={input + ' w-56'} placeholder="Note (optional)" value={note} onChange={(e) => setNote(e.target.value)} />}
      <Button variant="primary" icon={Icon} disabled={!by.trim()} onClick={submit}>
        {label}
      </Button>
      <Button onClick={() => setOpen(false)}>Cancel</Button>
      {error && <span className="text-xs text-bad">{error}</span>}
    </div>
  )
}

export const detailsHref = (equipmentId: string) => `#details/${encodeURIComponent(equipmentId)}`

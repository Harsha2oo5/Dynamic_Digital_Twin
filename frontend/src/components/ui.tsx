import type { LucideIcon } from 'lucide-react'
import { CircleAlert, CircleCheck, Info, TriangleAlert } from 'lucide-react'
import type { ButtonHTMLAttributes, ReactNode } from 'react'

export function cx(...classes: (string | false | null | undefined)[]) {
  return classes.filter(Boolean).join(' ')
}

export function Panel({
  title,
  icon: Icon,
  actions,
  children,
  className,
  bodyClassName,
}: {
  title?: ReactNode
  icon?: LucideIcon
  actions?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
}) {
  return (
    <section className={cx('rounded-lg border border-line bg-surface-1', className)}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 border-b border-line px-4 py-3">
          <h2 className="flex items-center gap-2 text-sm font-semibold tracking-wide text-ink">
            {Icon && <Icon size={16} className="text-ink-3" />}
            {title}
          </h2>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={cx('p-4', bodyClassName)}>{children}</div>
    </section>
  )
}

export function StatCard({
  label,
  value,
  sub,
  icon: Icon,
}: {
  label: string
  value: ReactNode
  sub?: ReactNode
  icon?: LucideIcon
}) {
  return (
    <div className="rounded-lg border border-line bg-surface-1 p-4">
      <div className="flex items-center justify-between text-xs font-medium uppercase tracking-wider text-ink-3">
        {label}
        {Icon && <Icon size={15} />}
      </div>
      <div className="tabular mt-2 truncate text-2xl font-semibold text-ink">{value}</div>
      {sub && <div className="mt-1 truncate text-xs text-ink-2">{sub}</div>}
    </div>
  )
}

type Tone = 'good' | 'warn' | 'bad' | 'neutral' | 'accent'

const toneClass: Record<Tone, string> = {
  good: 'bg-good/15 text-good border-good/30',
  warn: 'bg-warn/15 text-warn border-warn/30',
  bad: 'bg-bad/15 text-bad border-bad/30',
  neutral: 'bg-surface-3 text-ink-2 border-line',
  accent: 'bg-accent/15 text-accent-strong border-accent/30',
}

export function Badge({ tone = 'neutral', children }: { tone?: Tone; children: ReactNode }) {
  return (
    <span className={cx('inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-xs font-medium', toneClass[tone])}>
      {children}
    </span>
  )
}

export function StatusDot({ tone }: { tone: Tone }) {
  const color = { good: 'bg-good', warn: 'bg-warn', bad: 'bg-bad', neutral: 'bg-ink-3', accent: 'bg-accent' }[tone]
  return <span className={cx('inline-block size-2 rounded-full', color)} />
}

export function Button({
  variant = 'secondary',
  icon: Icon,
  children,
  className,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'secondary' | 'danger'; icon?: LucideIcon }) {
  const styles = {
    primary: 'bg-accent text-surface-0 hover:bg-accent-strong',
    secondary: 'border border-line bg-surface-2 text-ink hover:bg-surface-3',
    danger: 'border border-bad/40 bg-bad/10 text-bad hover:bg-bad/20',
  }[variant]
  return (
    <button
      className={cx(
        'inline-flex items-center justify-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40',
        styles,
        className,
      )}
      {...rest}
    >
      {Icon && <Icon size={15} />}
      {children}
    </button>
  )
}

export function Alert({ tone, title, children }: { tone: 'info' | 'warn' | 'bad' | 'good'; title?: string; children?: ReactNode }) {
  const map = {
    info: { Icon: Info, cls: 'border-accent/30 bg-accent/10 text-accent-strong' },
    warn: { Icon: TriangleAlert, cls: 'border-warn/30 bg-warn/10 text-warn' },
    bad: { Icon: CircleAlert, cls: 'border-bad/30 bg-bad/10 text-bad' },
    good: { Icon: CircleCheck, cls: 'border-good/30 bg-good/10 text-good' },
  }[tone]
  return (
    <div className={cx('flex gap-3 rounded-md border px-3 py-2.5 text-sm', map.cls)}>
      <map.Icon size={16} className="mt-0.5 shrink-0" />
      <div className="min-w-0">
        {title && <div className="font-medium">{title}</div>}
        {children && <div className="text-ink-2">{children}</div>}
      </div>
    </div>
  )
}

export function Empty({ icon: Icon, title, children }: { icon: LucideIcon; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-10 text-center">
      <Icon size={28} className="text-ink-3" />
      <div className="text-sm font-medium text-ink-2">{title}</div>
      {children && <div className="max-w-md text-xs text-ink-3">{children}</div>}
    </div>
  )
}

export function KV({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[minmax(0,10rem)_1fr] gap-x-4 gap-y-2 text-sm">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-ink-3">{k}</dt>
          <dd className="tabular min-w-0 break-words text-ink">{v}</dd>
        </div>
      ))}
    </dl>
  )
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: string; actions?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-xl font-semibold text-ink">{title}</h1>
        {subtitle && <p className="mt-0.5 text-sm text-ink-3">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  )
}

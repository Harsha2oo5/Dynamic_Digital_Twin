import { Activity, ChevronDown, ChevronRight, Radar } from 'lucide-react'
import { Fragment, useEffect, useState } from 'react'
import { AnomalyStatusBadge, detailsHref, OperatorAction } from '../components/analytics'
import { SeverityBadge, SourceBadge, TelemetryConnectionNotice } from '../components/telemetry'
import { Alert, Badge, Button, Empty, KV, PageHeader, Panel } from '../components/ui'
import { usePolling } from '../hooks/usePolling'
import { useTelemetry } from '../hooks/useTelemetry'
import { dateTime, time, titleCase } from '../lib/format'
import { analytics } from '../services/analytics'
import type { Anomaly } from '../types/analytics'
import { selectCls } from './live/Controls'

const CATEGORIES = ['battery', 'power_supply_unit', 'equipment_cabinet', 'router', 'radio_unit', 'antenna', 'cable', 'site']
const fmt = (v: number | null, unit: string | null) => (v == null ? '—' : `${Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)}${unit && unit !== 'state' && unit !== 'ratio' ? ` ${unit}` : ''}`)

export function AnomalyDetail({ a, onChanged }: { a: Anomaly; onChanged: () => void }) {
  return (
    <div className="space-y-3 bg-surface-0/40 px-4 py-3">
      <p className="text-sm text-ink">{a.explanation}</p>
      {a.evidence.context && <Alert tone="info">Context: {a.evidence.context}</Alert>}
      <KV
        rows={[
          ['Observed / expected', `${fmt(a.observed, a.unit)} / ${fmt(a.expected, a.unit)}`],
          ['Score (now / peak)', `${a.score.toFixed(2)} / ${a.peak_score.toFixed(2)} threshold units`],
          ['Detectors', a.detectors.join(', ') || a.detector],
          ['Detected', `${dateTime(a.first_detected_at)} → ${a.condition_active ? 'ongoing' : dateTime(a.ended_at)}`],
          ['Occurrences', a.occurrences],
          ['Sensor', a.sensor_id ?? 'several (multivariate)'],
          ['Status', a.resolved_by ? `${a.status} by ${a.resolved_by}${a.resolution_note ? ` · ${a.resolution_note}` : ''}` : a.acknowledged_by ? `acknowledged by ${a.acknowledged_by}` : a.status],
        ]}
      />
      {a.evidence.findings && (
        <ul className="space-y-1 text-xs text-ink-3">
          {a.evidence.findings.map((f, i) => (
            <li key={i}>
              <span className="font-mono text-ink-2">{f.detector}</span> ({f.score.toFixed(2)}): {f.detail}
            </li>
          ))}
        </ul>
      )}
      {a.status !== 'resolved' && (
        <div className="flex flex-wrap gap-2">
          {a.status === 'open' && <OperatorAction label="Acknowledge" onSubmit={async (by, note) => { await analytics.ack(a.id, by, note); onChanged() }} />}
          <OperatorAction label="Resolve" icon="double" onSubmit={async (by, note) => { await analytics.resolve(a.id, by, note); onChanged() }} />
        </div>
      )}
      <p className="text-xs text-ink-3">Scores are distances from learned normal behaviour in threshold units, not probabilities of failure.</p>
    </div>
  )
}

export function AnomalyMonitoring() {
  const { anomalyFeed, connection, error, analyticsVersion } = useTelemetry()
  const [f, setF] = useState({ category: '', severity: '', status: '', active: '' })
  const [offset, setOffset] = useState(0)
  const [open, setOpen] = useState<number | null>(null)
  const limit = 25
  const page = usePolling(() => analytics.anomalies({ ...f, active: f.active === '' ? undefined : f.active === 'true', limit, offset }), 15000)
  useEffect(() => {
    page.refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analyticsVersion, f, offset])
  const set = (k: keyof typeof f, v: string) => {
    setF({ ...f, [k]: v })
    setOffset(0)
  }
  const total = page.data?.total ?? 0
  return (
    <>
      <PageHeader title="Anomaly Monitoring" subtitle="Abnormal equipment behaviour found by the detectors, with the evidence for each event." />
      <div className="space-y-4">
        <TelemetryConnectionNotice connection={connection} error={error} />
        <Panel title="Live feed" icon={Activity} bodyClassName="p-0" actions={<span className="text-xs text-ink-3">since this page was opened</span>}>
          {anomalyFeed.length === 0 ? (
            <Empty icon={Radar} title="No anomaly events yet" />
          ) : (
            <ul className="max-h-60 divide-y divide-line/60 overflow-y-auto">
              {anomalyFeed.slice(0, 15).map((a) => (
                <li key={a.id} className="flex flex-wrap items-center gap-2 px-4 py-2 text-xs">
                  <span className="tabular w-16 text-ink-3">{time(a.last_detected_at)}</span>
                  <SeverityBadge severity={a.severity} />
                  <a className="font-mono text-accent-strong hover:underline" href={detailsHref(a.equipment_id ?? '')}>
                    {a.equipment_id ?? 'site'}
                  </a>
                  <span className="text-ink-2">{a.metric ?? 'multivariate'}</span>
                  <span className="min-w-0 flex-1 truncate text-ink-3">{a.explanation}</span>
                  <AnomalyStatusBadge status={a.status} active={a.condition_active} />
                  {a.data_source === 'simulated' && <Badge tone="accent">sim</Badge>}
                </li>
              ))}
            </ul>
          )}
        </Panel>
        <Panel
          title={`Anomaly records (${total})`}
          bodyClassName="p-0"
          actions={
            <>
              <select className={selectCls} value={f.category} onChange={(e) => set('category', e.target.value)} aria-label="Category">
                <option value="">All categories</option>
                {CATEGORIES.map((c) => (
                  <option key={c} value={c}>
                    {titleCase(c)}
                  </option>
                ))}
              </select>
              <select className={selectCls} value={f.severity} onChange={(e) => set('severity', e.target.value)} aria-label="Severity">
                <option value="">Any severity</option>
                <option>warning</option>
                <option>critical</option>
              </select>
              <select className={selectCls} value={f.status} onChange={(e) => set('status', e.target.value)} aria-label="Status">
                <option value="">Any status</option>
                <option>open</option>
                <option>acknowledged</option>
                <option>resolved</option>
              </select>
              <select className={selectCls} value={f.active} onChange={(e) => set('active', e.target.value)} aria-label="Condition">
                <option value="">Ongoing or ended</option>
                <option value="true">Ongoing</option>
                <option value="false">Ended</option>
              </select>
            </>
          }
        >
          {page.error && (
            <div className="p-3">
              <Alert tone="bad">{page.error}</Alert>
            </div>
          )}
          {page.data && page.data.items.length === 0 ? (
            <Empty icon={Radar} title="No anomalies match" />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-left text-xs text-ink-3">
                  <tr className="border-b border-line">
                    {['', 'Detected', 'Severity', 'Equipment', 'Metric', 'Observed / expected', 'Score', 'Detector', 'Status', 'Source'].map((h) => (
                      <th key={h} className="px-3 py-2 font-medium">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(page.data?.items ?? []).map((a) => (
                    <Fragment key={a.id}>
                      <tr onClick={() => setOpen(open === a.id ? null : a.id)} className="cursor-pointer border-b border-line/60 hover:bg-surface-2">
                        <td className="px-2 py-2 text-ink-3">{open === a.id ? <ChevronDown size={14} /> : <ChevronRight size={14} />}</td>
                        <td className="px-3 py-2 text-xs whitespace-nowrap text-ink-2">{dateTime(a.first_detected_at)}</td>
                        <td className="px-3 py-2">
                          <SeverityBadge severity={a.severity} />
                        </td>
                        <td className="px-3 py-2 font-mono text-xs">
                          <a className="text-accent-strong hover:underline" href={detailsHref(a.equipment_id ?? '')} onClick={(e) => e.stopPropagation()}>
                            {a.equipment_id ?? 'site'}
                          </a>
                        </td>
                        <td className="px-3 py-2 text-xs text-ink-2">{a.metric ?? 'multivariate'}</td>
                        <td className="px-3 py-2 text-xs whitespace-nowrap text-ink-2">
                          {fmt(a.observed, a.unit)} / {fmt(a.expected, a.unit)}
                        </td>
                        <td className="tabular px-3 py-2 text-xs text-ink-2">{a.peak_score.toFixed(1)}</td>
                        <td className="px-3 py-2 text-xs text-ink-3">{a.detector}</td>
                        <td className="px-3 py-2">
                          <AnomalyStatusBadge status={a.status} active={a.condition_active} />
                        </td>
                        <td className="px-3 py-2">
                          <SourceBadge source={a.data_source} />
                        </td>
                      </tr>
                      {open === a.id && (
                        <tr>
                          <td colSpan={10}>
                            <AnomalyDetail a={a} onChanged={page.refresh} />
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {total > limit && (
            <div className="flex items-center justify-end gap-2 p-3 text-xs text-ink-3">
              {offset + 1}–{Math.min(offset + limit, total)} of {total}
              <Button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>
                Previous
              </Button>
              <Button disabled={offset + limit >= total} onClick={() => setOffset(offset + limit)}>
                Next
              </Button>
            </div>
          )}
        </Panel>
      </div>
    </>
  )
}

import { MonitorCog, RotateCcw, Server } from 'lucide-react'
import { Alert, Button, Empty, KV, PageHeader, Panel } from '../components/ui'
import { usePolling } from '../hooks/usePolling'
import { useSettings } from '../hooks/useSettings'
import { api, API_BASE } from '../services/api'
import { DetectionControls, Row, selectCls } from './live/Controls'

export function Settings() {
  const { settings, update, reset } = useSettings()
  const cfg = usePolling(api.config, 0)

  return (
    <>
      <PageHeader
        title="Settings"
        subtitle="Client-side detection defaults are stored in this browser"
        actions={
          <Button icon={RotateCcw} onClick={reset}>
            Reset defaults
          </Button>
        }
      />
      <div className="grid gap-4 xl:grid-cols-3">
        <div className="space-y-4">
          <DetectionControls showFrameSkip showTracking />
          <Panel title="Webcam streaming" icon={MonitorCog} bodyClassName="space-y-4 p-4">
            <Row label="Frame width sent to backend">
              <select className={selectCls} value={settings.streamWidth} onChange={(e) => update({ streamWidth: Number(e.target.value) })}>
                {[320, 480, 640, 960, 1280].map((v) => (
                  <option key={v} value={v}>
                    {v} px
                  </option>
                ))}
              </select>
            </Row>
            <Row label="JPEG quality">
              <select className={selectCls} value={settings.jpegQuality} onChange={(e) => update({ jpegQuality: Number(e.target.value) })}>
                {[0.6, 0.7, 0.8, 0.9].map((v) => (
                  <option key={v} value={v}>
                    {Math.round(v * 100)}%
                  </option>
                ))}
              </select>
            </Row>
            <p className="text-xs text-ink-3">Smaller frames reduce encoding and transfer time; inference resolution is set separately.</p>
          </Panel>
        </div>

        <Panel title="Backend runtime configuration" icon={Server} className="xl:col-span-2">
          {cfg.error && <Alert tone="bad" title="Backend unreachable">{cfg.error}</Alert>}
          {cfg.data ? (
            <>
              <KV rows={[['API base', API_BASE || `${window.location.origin} (dev proxy)`], ...Object.entries(cfg.data).map(([k, v]) => [k, <span className="font-mono text-xs">{String(v)}</span>] as [string, React.ReactNode])]} />
              <p className="mt-4 text-xs text-ink-3">
                Server defaults come from <code className="font-mono">backend/.env</code> and require a backend restart to change. Execution is CPU-only in this milestone.
              </p>
            </>
          ) : (
            !cfg.error && <Empty icon={Server} title="Loading…" />
          )}
        </Panel>
      </div>
    </>
  )
}

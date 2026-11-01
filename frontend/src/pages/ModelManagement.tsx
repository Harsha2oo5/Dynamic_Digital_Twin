import { Cpu, FlaskConical, Layers, Package, RefreshCw } from 'lucide-react'
import { Alert, Badge, Button, Empty, KV, PageHeader, Panel } from '../components/ui'
import { usePolling } from '../hooks/usePolling'
import { api } from '../services/api'

export function ModelManagement() {
  const info = usePolling(api.modelInfo, 0)
  const models = usePolling(api.models, 0)
  const m = info.data

  return (
    <>
      <PageHeader
        title="Model Management"
        subtitle="Loaded detection model and available checkpoints"
        actions={
          <Button
            icon={RefreshCw}
            onClick={() => {
              info.refresh()
              models.refresh()
            }}
          >
            Refresh
          </Button>
        }
      />
      <div className="mb-4 space-y-2 empty:hidden">
        {info.error && <Alert tone="bad" title="Backend unreachable">{info.error}</Alert>}
        {m && !m.loaded && <Alert tone="warn" title="Model failed to load">{m.error}</Alert>}
        {m?.fallback_active && (
          <Alert tone="warn" title="Custom model selected, but the pretrained model is running">
            {m.fallback_reason} (MODEL_FALLBACK_TO_PRETRAINED=true)
          </Alert>
        )}
        {m?.taxonomy_match === false && (
          <Alert tone="warn" title="Custom model classes differ from the 7-class Digital Twin taxonomy">
            Check that the checkpoint was trained with the project's data.yaml.
          </Alert>
        )}
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Current model" icon={Layers}>
          {m ? (
            <KV
              rows={[
                ['Model', m.model_name ?? '—'],
                ['Selected (DETECTION_MODEL)', <span className="font-mono">{m.selected_model}</span>],
                ['Running type', <Badge tone={m.model_type === 'custom' ? 'accent' : 'neutral'}>{m.model_type}</Badge>],
                [
                  'Custom checkpoint',
                  m.custom_model_available ? <Badge tone="good">available</Badge> : <Badge tone="neutral">not trained yet</Badge>,
                ],
                ...(m.taxonomy_match != null
                  ? ([['Taxonomy match', m.taxonomy_match ? <Badge tone="good">7 classes ✓</Badge> : <Badge tone="warn">mismatch</Badge>]] as [string, React.ReactNode][])
                  : []),
                ['Checkpoint', <span className="font-mono text-xs">{m.model_path}</span>],
                ['Task', m.task ?? '—'],
                ['Execution device', <span className="inline-flex items-center gap-1.5 font-mono"><Cpu size={13} /> {m.device.toUpperCase()}</span>],
                ['Status', m.loaded ? <Badge tone="good">Loaded{m.load_time_s != null ? ` in ${m.load_time_s}s` : ''}</Badge> : <Badge tone="bad">Not loaded</Badge>],
                ['Confidence threshold', m.confidence_threshold.toFixed(2)],
                ['IoU threshold', m.iou_threshold.toFixed(2)],
                ['Inference size', `${m.image_size} px`],
                ['Frame skip', m.frame_skip],
                ['Tracker', m.tracker],
              ]}
            />
          ) : (
            <Empty icon={Layers} title="Loading…" />
          )}
        </Panel>

        <Panel title="Checkpoints" icon={Package}>
          {models.data?.length ? (
            <table className="w-full text-sm">
              <thead className="text-left text-xs uppercase tracking-wider text-ink-3">
                <tr>
                  <th className="pb-2 font-medium">Name</th>
                  <th className="pb-2 font-medium">Type</th>
                  <th className="pb-2 text-right font-medium">Size</th>
                  <th className="pb-2" />
                </tr>
              </thead>
              <tbody className="tabular divide-y divide-line">
                {models.data.map((c) => (
                  <tr key={c.path}>
                    <td className="py-2">
                      <div>{c.name}</div>
                      <div className="font-mono text-xs text-ink-3">{c.path}</div>
                    </td>
                    <td className="py-2">{c.model_type}</td>
                    <td className="py-2 text-right">{c.size_mb.toFixed(1)} MB</td>
                    <td className="py-2 text-right">{c.active && <Badge tone="good">active</Badge>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Empty icon={Package} title="No checkpoints found in backend/models/" />
          )}
          <div className="mt-4">
            <Alert tone="info" title="Switching models">
              Set <code className="font-mono">DETECTION_MODEL=custom</code> (or <code className="font-mono">pretrained</code>) in{' '}
              <code className="font-mono">backend/.env</code> and restart the backend. The custom checkpoint is read from{' '}
              <code className="font-mono break-all">{m?.custom_model_path ?? 'models/custom/digital_twin_best.pt'}</code>.
            </Alert>
          </div>
        </Panel>

        <Panel title={`Detectable classes (${m?.num_classes ?? 0})`} icon={FlaskConical} className="xl:col-span-2">
          {m?.model_type === 'pretrained' && (
            <p className="mb-3 text-xs text-ink-3">
              These are the COCO classes of the pretrained model. It cannot recognise the seven Digital Twin components (antenna, router, cable, battery, power
              supply unit, equipment cabinet, radio unit) until the custom model is fine-tuned on the annotated dataset.
            </p>
          )}
          <div className="flex flex-wrap gap-1.5">
            {m?.class_names.map((c, i) => (
              <span key={c} className="rounded border border-line bg-surface-2 px-2 py-0.5 text-xs text-ink-2">
                <span className="mr-1 font-mono text-ink-3">{i}</span>
                {c}
              </span>
            ))}
          </div>
        </Panel>
      </div>
    </>
  )
}

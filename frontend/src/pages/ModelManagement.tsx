import { Activity, Cpu, FileText, FlaskConical, Layers, Package, RefreshCw } from 'lucide-react'
import { useState } from 'react'
import { Alert, Badge, Button, Empty, KV, PageHeader, Panel } from '../components/ui'
import { usePolling } from '../hooks/usePolling'
import { num } from '../lib/format'
import { api } from '../services/api'
import type { ModelEntry } from '../types/api'

function LabelSourceBadge({ source }: { source: string | null | undefined }) {
  if (source === 'reviewed') return <Badge tone="good">human-reviewed labels</Badge>
  if (source === 'pseudo') return <Badge tone="warn">pseudo-labels (not ground truth)</Badge>
  if (source === 'coco') return <Badge tone="neutral">COCO pretrained</Badge>
  return <Badge tone="neutral">{source ?? 'unknown'}</Badge>
}

function ModelCardPanel({ entry }: { entry: ModelEntry | undefined }) {
  const card = entry?.model_card
  if (!entry) return <Empty icon={FileText} title="No model selected" />
  if (!card) return <Empty icon={FileText} title={`${entry.name} has no model card`} />
  const metrics = card.validation_metrics ?? {}
  return (
    <div className="space-y-4">
      <KV
        rows={[
          ['Model', `${entry.name}${card.version != null ? ` · version ${card.version}` : ''}`],
          ['Label source', <LabelSourceBadge source={card.label_source} />],
          ...(card.pseudo_run ? ([['Pseudo-label run', <span className="font-mono">{card.pseudo_run}</span>]] as [string, React.ReactNode][]) : []),
          ...(card.created_at ? ([['Created', card.created_at.replace('T', ' ')]] as [string, React.ReactNode][]) : []),
          ...(card.training
            ? ([
                [
                  'Training',
                  `${card.training.mode ?? ''} run ${card.training.run ?? ''} · ${card.training.epochs_completed ?? '?'} epochs · ${card.training.imgsz ?? '?'} px · ${
                    card.training.duration_s != null ? `${(card.training.duration_s / 60).toFixed(1)} min` : ''
                  }`,
                ],
              ] as [string, React.ReactNode][])
            : []),
          ...(card.dataset?.images
            ? ([['Dataset images', Object.entries(card.dataset.images).map(([k, v]) => `${k} ${v}`).join(' · ')]] as [string, React.ReactNode][])
            : []),
        ]}
      />
      {Object.keys(metrics).length > 0 && (
        <div>
          <div className="mb-1.5 text-xs text-ink-3">Validation metrics — {card.metrics_meaning}</div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(metrics).map(([k, v]) => (
              <span key={k} className="rounded border border-line bg-surface-2 px-2 py-1 text-xs">
                <span className="text-ink-3">{k}</span> <span className="tabular font-semibold">{v.toFixed(3)}</span>
              </span>
            ))}
          </div>
        </div>
      )}
      {!!card.limitations?.length && (
        <Alert tone={card.label_source === 'pseudo' ? 'warn' : 'info'} title="Limitations">
          <ul className="list-disc space-y-1 pl-4">
            {card.limitations.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </Alert>
      )}
    </div>
  )
}

export function ModelManagement() {
  const info = usePolling(api.modelInfo, 0)
  const versions = usePolling(api.modelVersions, 0)
  const perf = usePolling(api.modelPerformance, 5000)
  const [busy, setBusy] = useState<string | null>(null)
  const [selectError, setSelectError] = useState<string | null>(null)
  const [inspect, setInspect] = useState<string | null>(null)
  const m = info.data
  const active = versions.data?.find((v) => v.active)
  const shown = versions.data?.find((v) => v.model_id === inspect) ?? active

  const refresh = () => {
    info.refresh()
    versions.refresh()
    perf.refresh()
  }

  const select = async (id: string) => {
    setBusy(id)
    setSelectError(null)
    try {
      await api.selectModel(id)
      setInspect(null)
    } catch (e) {
      setSelectError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
      refresh()
    }
  }

  const p = perf.data
  return (
    <>
      <PageHeader title="Model Management" subtitle="Versioned detection models, their provenance, and runtime switching" actions={<Button icon={RefreshCw} onClick={refresh}>Refresh</Button>} />
      <div className="mb-4 space-y-2 empty:hidden">
        {info.error && <Alert tone="bad" title="Backend unreachable">{info.error}</Alert>}
        {selectError && <Alert tone="bad" title="Model switch failed">{selectError}</Alert>}
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
        {active?.label_source === 'pseudo' && (
          <Alert tone="warn" title="The running model was trained on automatic pseudo-labels">
            No human verified its training labels. Its validation numbers are agreement with pseudo-labels, not accuracy; real accuracy is unknown until a
            human-reviewed test set exists.
          </Alert>
        )}
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Model versions" icon={Package} className="xl:col-span-2" bodyClassName="p-0">
          {versions.data?.length ? (
            <table className="w-full text-sm">
              <thead className="text-left text-xs uppercase tracking-wider text-ink-3">
                <tr className="border-b border-line">
                  {['Model', 'Type', 'Version', 'Labels', 'Size', ''].map((h) => (
                    <th key={h} className="px-4 py-2 font-medium">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="tabular divide-y divide-line">
                {versions.data.map((v) => (
                  <tr key={v.model_id} className="hover:bg-surface-2">
                    <td className="px-4 py-2">
                      <button className="text-left hover:underline" onClick={() => setInspect(v.model_id)}>
                        {v.name}
                      </button>
                      <div className="font-mono text-xs text-ink-3">{v.model_id}</div>
                    </td>
                    <td className="px-4 py-2">{v.model_type}</td>
                    <td className="px-4 py-2">{v.version ?? '—'}</td>
                    <td className="px-4 py-2">
                      <LabelSourceBadge source={v.label_source} />
                    </td>
                    <td className="px-4 py-2">{v.size_mb.toFixed(1)} MB</td>
                    <td className="px-4 py-2 text-right">
                      {v.active ? (
                        <Badge tone="good">active</Badge>
                      ) : (
                        <Button onClick={() => select(v.model_id)} disabled={busy != null}>
                          {busy === v.model_id ? 'Loading…' : 'Use this model'}
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Empty icon={Package} title="No checkpoints found in backend/models/" />
          )}
          <p className="px-4 py-3 text-xs text-ink-3">
            Switching takes effect immediately for new frames; running video jobs finish with the model they started with. The choice lasts until the backend
            restarts (then DETECTION_MODEL / MODEL_PATH in backend/.env apply). If a checkpoint fails to load, the current model stays active.
          </p>
        </Panel>

        <Panel title={shown ? `Model card · ${shown.name}` : 'Model card'} icon={FileText}>
          <ModelCardPanel entry={shown} />
        </Panel>

        <Panel title="Runtime performance (this backend session)" icon={Activity}>
          {p ? (
            <KV
              rows={[
                ['Model', p.model],
                ['Frames processed', p.runtime.frames_processed],
                ['Avg inference', p.runtime.avg_inference_ms != null ? `${num(p.runtime.avg_inference_ms)} ms (${num(p.runtime.avg_inference_fps)} FPS)` : '—'],
                ['Min / max inference', p.runtime.min_inference_ms != null ? `${num(p.runtime.min_inference_ms)} / ${num(p.runtime.max_inference_ms)} ms` : '—'],
                ['Avg confidence', p.runtime.avg_confidence != null ? p.runtime.avg_confidence.toFixed(3) : '—'],
              ]}
            />
          ) : (
            <Empty icon={Activity} title="Loading…" />
          )}
        </Panel>

        <Panel title="Current model" icon={Layers}>
          {m ? (
            <KV
              rows={[
                ['Model', m.model_name ?? '—'],
                ['Running type', <Badge tone={m.model_type === 'custom' ? 'accent' : 'neutral'}>{m.model_type}</Badge>],
                ...(m.taxonomy_match != null
                  ? ([['Taxonomy match', m.taxonomy_match ? <Badge tone="good">7 classes ✓</Badge> : <Badge tone="warn">mismatch</Badge>]] as [string, React.ReactNode][])
                  : []),
                ['Checkpoint', <span className="font-mono text-xs break-all">{m.model_path}</span>],
                ['Execution device', <span className="inline-flex items-center gap-1.5 font-mono"><Cpu size={13} /> {m.device.toUpperCase()}</span>],
                ['Status', m.loaded ? <Badge tone="good">Loaded{m.load_time_s != null ? ` in ${m.load_time_s}s` : ''}</Badge> : <Badge tone="bad">Not loaded</Badge>],
                ['Confidence / IoU', `${m.confidence_threshold.toFixed(2)} / ${m.iou_threshold.toFixed(2)}`],
                ['Inference size', `${m.image_size} px`],
                ['Frame skip', m.frame_skip],
                ['Tracker', m.tracker],
              ]}
            />
          ) : (
            <Empty icon={Layers} title="Loading…" />
          )}
        </Panel>

        <Panel title={`Detectable classes (${m?.num_classes ?? 0})`} icon={FlaskConical}>
          {m?.model_type === 'pretrained' && (
            <p className="mb-3 text-xs text-ink-3">
              These are the COCO classes of the pretrained model. It cannot recognise the seven Digital Twin components; select a custom model above.
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

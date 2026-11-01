import { Database, FolderTree, ListChecks, LoaderCircle, PenTool, RefreshCw, Terminal } from 'lucide-react'
import { useState } from 'react'
import { Alert, Badge, Button, Empty, KV, PageHeader, Panel } from '../components/ui'
import { usePolling } from '../hooks/usePolling'
import { dateTime, int } from '../lib/format'
import { api } from '../services/api'

const res = (r: [number, number] | null) => (r ? `${r[0]}×${r[1]}` : '—')

export function DatasetManagement() {
  const [refreshing, setRefreshing] = useState(false)
  const ds = usePolling(() => api.dataset(false), 0)
  const d = ds.data

  const rescan = async () => {
    setRefreshing(true)
    try {
      await api.dataset(true)
      ds.refresh()
    } finally {
      setRefreshing(false)
    }
  }

  return (
    <>
      <PageHeader
        title="Dataset Management"
        subtitle="Read-only inspection of the collected Digital Twin dataset"
        actions={
          <Button icon={refreshing ? LoaderCircle : RefreshCw} onClick={rescan} disabled={refreshing}>
            {refreshing ? 'Scanning…' : 'Re-scan'}
          </Button>
        }
      />
      {ds.error && <Alert tone="bad" title="Could not scan dataset">{ds.error}</Alert>}
      {ds.loading && !d && <Empty icon={LoaderCircle} title="Scanning dataset (first scan hashes every image)…" />}

      {d && (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
          <div className="space-y-4">
            <Panel title="Folders" icon={FolderTree} bodyClassName="p-0">
              {d.folders?.length ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="text-left text-xs uppercase tracking-wider text-ink-3">
                      <tr className="border-b border-line">
                        {['Folder', 'Class', 'Images', 'Labelled', 'Formats', 'Resolution range', 'Size'].map((h) => (
                          <th key={h} className="px-4 py-2 font-medium">
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="tabular divide-y divide-line">
                      {d.folders.map((f) => (
                        <tr key={f.path}>
                          <td className="px-4 py-2 font-mono text-xs">{f.path}</td>
                          <td className="px-4 py-2 whitespace-nowrap">
                            {f.class_name ? (
                              <>
                                <span className="mr-1.5 font-mono text-xs text-ink-3">{f.class_id}</span>
                                {f.class_name}
                              </>
                            ) : (
                              <span className="text-warn">unmapped</span>
                            )}
                          </td>
                          <td className="px-4 py-2">{f.images}</td>
                          <td className="px-4 py-2">{f.labelled ? f.labelled : <span className="text-warn">0</span>}</td>
                          <td className="px-4 py-2 text-xs text-ink-2">
                            {Object.entries(f.extensions)
                              .map(([e, n]) => `${e} ×${n}`)
                              .join(', ')}
                          </td>
                          <td className="px-4 py-2 text-xs text-ink-2">
                            {res(f.min_resolution)} – {res(f.max_resolution)}
                          </td>
                          <td className="px-4 py-2 whitespace-nowrap">{f.size_mb} MB</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <Empty icon={FolderTree} title="No image folders found" />
              )}
            </Panel>

            <Panel title="Annotation workflow" icon={PenTool}>
              <ol className="list-decimal space-y-1.5 pl-5 text-sm text-ink-2">
                <li>
                  Load <span className="font-mono">annotation/cvat_labels.json</span> or <span className="font-mono">label_studio_config.xml</span> and upload each
                  folder as its own task.
                </li>
                <li>Box every component instance (all 7 classes, not only the folder's) and export in YOLO format.</li>
                <li>
                  Validate: <code className="font-mono text-xs">python -m app.training.validate_annotations ../annotation/exports/&lt;export&gt;</code>
                </li>
                <li>
                  Split 70/20/10 (duplicate-aware): <code className="font-mono text-xs">python -m app.training.prepare_dataset --labels …</code>
                </li>
                <li>
                  Pilot, then full training on CPU: <code className="font-mono text-xs">python -m app.training.train --data dataset/digital_twin_yolo/data.yaml --pilot</code>
                </li>
              </ol>
              <p className="mt-2 text-xs text-ink-3">See docs/annotation-guide.md and docs/custom-model-training.md.</p>
              <p className="mt-3 text-xs text-ink-3">In-app annotation and training controls are planned for a later phase.</p>
            </Panel>
          </div>

          <div className="space-y-4">
            <Panel title="Summary" icon={Database}>
              <KV
                rows={[
                  ['Root', <span className="font-mono text-xs">{d.root}</span>],
                  ['Images', int(d.total_images)],
                  ['Labelled', int(d.labelled_images)],
                  ['Annotation format', d.annotation_format ?? '—'],
                  ['data.yaml', d.data_yaml ?? 'not found'],
                  ['macOS metadata', `${int(d.macos_metadata_files)} files (ignored)`],
                  ['Unreadable', int(d.unreadable_images?.length)],
                  ['Duplicate groups', int(d.duplicate_groups?.length)],
                  ['Scanned', dateTime(d.scanned_at)],
                  [
                    'Training-ready',
                    d.ready_for_detection_training ? <Badge tone="good">yes</Badge> : <Badge tone="warn">no</Badge>,
                  ],
                ]}
              />
            </Panel>
            <Panel title="Validation issues" icon={ListChecks} bodyClassName="space-y-2 p-4">
              {d.issues.length ? d.issues.map((i) => <Alert key={i} tone="warn">{i}</Alert>) : <Alert tone="good" title="No issues found" />}
            </Panel>
            <Panel title="CLI" icon={Terminal}>
              <code className="block rounded bg-surface-0 p-2 font-mono text-xs text-ink-2">python -m app.training.audit_dataset</code>
              <p className="mt-2 text-xs text-ink-3">Full audit with near-duplicate detection; report in experiments/dataset_audit/.</p>
            </Panel>
          </div>
        </div>
      )}
    </>
  )
}

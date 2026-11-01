import { Image as ImageIcon, LoaderCircle, RefreshCw, Upload } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { DetectionOverlay } from '../../components/DetectionOverlay'
import { DetectionTable } from '../../components/DetectionTable'
import { Alert, Button, cx, Empty, Panel } from '../../components/ui'
import { useSettings } from '../../hooks/useSettings'
import { api } from '../../services/api'
import type { ImageDetectionResponse } from '../../types/api'
import { DetectionControls } from './Controls'
import { FrameMetrics } from './Metrics'

export function ImageView() {
  const { settings } = useSettings()
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [url, setUrl] = useState<string | null>(null)
  const [dims, setDims] = useState({ w: 0, h: 0 })
  const [result, setResult] = useState<ImageDetectionResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [view, setView] = useState<'overlay' | 'server'>('overlay')

  useEffect(() => () => void (url && URL.revokeObjectURL(url)), [url])

  const run = async (f: File) => {
    setBusy(true)
    setError(null)
    try {
      setResult(await api.detectImage(f, settings))
    } catch (e) {
      setResult(null)
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const choose = (f: File | undefined) => {
    if (!f) return
    setFile(f)
    setResult(null)
    setUrl(URL.createObjectURL(f))
    run(f)
  }

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_20rem]">
      <div className="space-y-4">
        <Panel
          title={file ? file.name : 'Image detection'}
          icon={ImageIcon}
          bodyClassName="p-0"
          actions={
            <>
              {result?.annotated_image && (
                <div className="flex overflow-hidden rounded-md border border-line text-xs">
                  {(['overlay', 'server'] as const).map((v) => (
                    <button key={v} onClick={() => setView(v)} className={cx('px-2.5 py-1', view === v ? 'bg-surface-3 text-ink' : 'text-ink-3 hover:text-ink')}>
                      {v === 'overlay' ? 'Original + overlay' : 'Server-annotated'}
                    </button>
                  ))}
                </div>
              )}
              {file && (
                <Button icon={RefreshCw} onClick={() => run(file)} disabled={busy} title="Re-run with current settings">
                  Re-run
                </Button>
              )}
              <Button icon={Upload} variant="primary" onClick={() => inputRef.current?.click()}>
                Upload image
              </Button>
              <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.bmp,.webp" hidden onChange={(e) => choose(e.target.files?.[0])} />
            </>
          }
        >
          <div className="relative aspect-video w-full overflow-hidden rounded-b-lg bg-black">
            {url ? (
              <>
                {view === 'server' && result?.annotated_image ? (
                  <img src={`data:image/jpeg;base64,${result.annotated_image}`} className="absolute inset-0 h-full w-full object-contain" alt="Annotated" />
                ) : (
                  <>
                    <img
                      src={url}
                      alt="Uploaded"
                      onLoad={(e) => setDims({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
                      className="absolute inset-0 h-full w-full object-contain"
                    />
                    {result && (
                      <DetectionOverlay detections={result.detections} frameWidth={result.frame_width} frameHeight={result.frame_height} mediaWidth={dims.w} mediaHeight={dims.h} />
                    )}
                  </>
                )}
                {busy && (
                  <div className="absolute inset-0 grid place-items-center bg-black/40">
                    <LoaderCircle className="animate-spin text-accent-strong" />
                  </div>
                )}
              </>
            ) : (
              <button className="absolute inset-0 grid place-items-center" onClick={() => inputRef.current?.click()}>
                <Empty icon={Upload} title="Upload a JPG or PNG image">
                  Runs YOLO inference on the CPU backend and draws boxes over the original image.
                </Empty>
              </button>
            )}
          </div>
        </Panel>
        {error && <Alert tone="bad" title="Detection failed">{error}</Alert>}
        {result && (
          <Panel title={`Detections (${result.detections.length})`}>
            <DetectionTable detections={result.detections} />
          </Panel>
        )}
      </div>
      <div className="space-y-4">
        <DetectionControls note="Press Re-run after changing settings." />
        <FrameMetrics result={result} />
      </div>
    </div>
  )
}

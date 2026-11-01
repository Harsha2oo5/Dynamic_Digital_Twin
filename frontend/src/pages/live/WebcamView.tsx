import { Camera, Play, RefreshCw, Square, Video } from 'lucide-react'
import { useRef, useState } from 'react'
import { DetectionOverlay } from '../../components/DetectionOverlay'
import { DetectionTable } from '../../components/DetectionTable'
import { Alert, Badge, Button, Empty, Panel } from '../../components/ui'
import { useSettings } from '../../hooks/useSettings'
import { useWebcamStream } from '../../hooks/useWebcamStream'
import { num } from '../../lib/format'
import { DetectionControls } from './Controls'
import { FrameMetrics } from './Metrics'

export function WebcamView() {
  const { settings } = useSettings()
  const videoRef = useRef<HTMLVideoElement>(null)
  const [dims, setDims] = useState({ w: 0, h: 0 })
  const stream = useWebcamStream(videoRef, settings)
  const r = stream.result
  const running = stream.state === 'running'

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_20rem]">
      <div className="space-y-4">
        <Panel
          title="Laptop webcam"
          icon={Camera}
          bodyClassName="p-0"
          actions={
            <>
              {running && <Badge tone="good">● LIVE</Badge>}
              {running ? (
                <>
                  <Button icon={RefreshCw} onClick={stream.resetTracking} title="Restart tracking IDs">
                    Reset IDs
                  </Button>
                  <Button icon={Square} variant="danger" onClick={stream.stop}>
                    Stop
                  </Button>
                </>
              ) : (
                <Button icon={Play} variant="primary" onClick={stream.start} disabled={stream.state === 'starting'}>
                  {stream.state === 'starting' ? 'Starting…' : 'Start'}
                </Button>
              )}
            </>
          }
        >
          <div className="relative aspect-video w-full overflow-hidden rounded-b-lg bg-black">
            <video
              ref={videoRef}
              muted
              playsInline
              onLoadedMetadata={(e) => setDims({ w: e.currentTarget.videoWidth, h: e.currentTarget.videoHeight })}
              className="absolute inset-0 h-full w-full object-contain"
            />
            {running && r && (
              <DetectionOverlay detections={r.detections} frameWidth={r.frame_width} frameHeight={r.frame_height} mediaWidth={dims.w} mediaHeight={dims.h} />
            )}
            {stream.state === 'idle' && (
              <div className="absolute inset-0 grid place-items-center">
                <Empty icon={Video} title="Webcam stopped">
                  Press Start to stream frames to the CPU backend. Frames are sent one at a time; the next is sent as soon as the previous result arrives.
                </Empty>
              </div>
            )}
          </div>
        </Panel>
        {stream.error && <Alert tone="bad" title="Stream error">{stream.error}</Alert>}
        <Panel title="Current detections">
          <DetectionTable detections={running && r ? r.detections : []} />
        </Panel>
      </div>
      <div className="space-y-4">
        <DetectionControls showTracking note="Changes apply to the running stream immediately." />
        <FrameMetrics result={running ? r : null} extra={[['End-to-end FPS', num(running ? stream.loopFps : null)]]} />
      </div>
    </div>
  )
}

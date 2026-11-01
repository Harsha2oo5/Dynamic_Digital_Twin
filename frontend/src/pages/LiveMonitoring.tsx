import type { LucideIcon } from 'lucide-react'
import { Camera, Film, Image } from 'lucide-react'
import { useState } from 'react'
import { cx, PageHeader } from '../components/ui'
import { ImageView } from './live/ImageView'
import { VideoView } from './live/VideoView'
import { WebcamView } from './live/WebcamView'

type Source = 'webcam' | 'video' | 'image'

const SOURCES: { id: Source; label: string; icon: LucideIcon }[] = [
  { id: 'webcam', label: 'Webcam', icon: Camera },
  { id: 'video', label: 'Video file', icon: Film },
  { id: 'image', label: 'Image', icon: Image },
]

export function LiveMonitoring() {
  const [source, setSource] = useState<Source>('video')
  return (
    <>
      <PageHeader
        title="Live Monitoring"
        subtitle="YOLO detection + ByteTrack tracking, CPU inference"
        actions={
          <div className="flex overflow-hidden rounded-md border border-line bg-surface-1">
            {SOURCES.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setSource(id)}
                className={cx('flex items-center gap-2 px-3 py-1.5 text-sm', source === id ? 'bg-surface-3 text-ink' : 'text-ink-3 hover:text-ink')}
              >
                <Icon size={15} />
                {label}
              </button>
            ))}
          </div>
        }
      />
      {/* Webcam unmounts (releasing the camera); video/image stay mounted to keep their results. */}
      {source === 'webcam' && <WebcamView />}
      <div hidden={source !== 'video'}>
        <VideoView />
      </div>
      <div hidden={source !== 'image'}>
        <ImageView />
      </div>
    </>
  )
}

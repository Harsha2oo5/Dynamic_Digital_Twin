import { createContext, useContext, useState, type ReactNode } from 'react'

export interface ClientSettings {
  confidence: number
  iou: number
  imageSize: number
  frameSkip: number
  tracking: boolean
  /** Width frames are downscaled to before sending webcam frames to the backend. */
  streamWidth: number
  jpegQuality: number
}

export const DEFAULT_SETTINGS: ClientSettings = {
  confidence: 0.35,
  iou: 0.45,
  imageSize: 416,
  frameSkip: 1,
  tracking: true,
  streamWidth: 640,
  jpegQuality: 0.8,
}

const STORAGE_KEY = 'ddt.settings.v1'

function load(): ClientSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? { ...DEFAULT_SETTINGS, ...JSON.parse(raw) } : DEFAULT_SETTINGS
  } catch {
    return DEFAULT_SETTINGS
  }
}

interface Ctx {
  settings: ClientSettings
  update: (patch: Partial<ClientSettings>) => void
  reset: () => void
}

const SettingsContext = createContext<Ctx | null>(null)

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<ClientSettings>(load)

  const persist = (next: ClientSettings) => {
    setSettings(next)
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
    } catch {
      /* storage unavailable: settings stay in memory */
    }
  }

  return (
    <SettingsContext.Provider
      value={{
        settings,
        update: (patch) => persist({ ...settings, ...patch }),
        reset: () => persist(DEFAULT_SETTINGS),
      }}
    >
      {children}
    </SettingsContext.Provider>
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export function useSettings(): Ctx {
  const ctx = useContext(SettingsContext)
  if (!ctx) throw new Error('useSettings must be used inside SettingsProvider')
  return ctx
}

import { useEffect, useState } from 'react'
import { NAV, Sidebar, type PageId } from './components/Sidebar'
import { SettingsProvider } from './hooks/useSettings'
import { TelemetryProvider } from './hooks/useTelemetry'
import { Alerts } from './pages/Alerts'
import { AnomalyMonitoring } from './pages/AnomalyMonitoring'
import { EquipmentDetails } from './pages/EquipmentDetails'
import { EquipmentHealth } from './pages/EquipmentHealth'
import { PredictiveMaintenance } from './pages/PredictiveMaintenance'
import { DatasetManagement } from './pages/DatasetManagement'
import { EquipmentMonitoring } from './pages/EquipmentMonitoring'
import { LiveMonitoring } from './pages/LiveMonitoring'
import { ModelManagement } from './pages/ModelManagement'
import { ObjectInventory } from './pages/ObjectInventory'
import { Overview } from './pages/Overview'
import { SensorManagement } from './pages/SensorManagement'
import { Settings } from './pages/Settings'
import { TelemetryAnalytics } from './pages/TelemetryAnalytics'

const PAGES: Record<PageId, () => React.JSX.Element> = {
  overview: Overview,
  live: LiveMonitoring,
  inventory: ObjectInventory,
  equipment: EquipmentMonitoring,
  sensors: SensorManagement,
  analytics: TelemetryAnalytics,
  alerts: Alerts,
  health: EquipmentHealth,
  anomalies: AnomalyMonitoring,
  maintenance: PredictiveMaintenance,
  details: EquipmentDetails,
  models: ModelManagement,
  dataset: DatasetManagement,
  settings: Settings,
}

/** "#anomalies" -> anomalies; "#details/BAT-001" -> details (the page reads its own parameter). */
function pageFromHash(hash: string): PageId {
  const id = hash.slice(1).split('/')[0] as PageId
  return id === 'details' || NAV.some((n) => n.id === id) ? id : 'overview'
}

export default function App() {
  const [hash, setHash] = useState(window.location.hash)

  useEffect(() => {
    const onHash = () => setHash(window.location.hash)
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const page = pageFromHash(hash)
  const Page = PAGES[page]
  return (
    <SettingsProvider>
      <TelemetryProvider>
      <div className="flex h-full flex-col md:flex-row">
        <Sidebar page={page === 'details' ? 'health' : page} onNavigate={(p) => (window.location.hash = p)} />
        <main className="min-w-0 flex-1 overflow-y-auto px-4 py-5 md:px-6">
          <div className="mx-auto max-w-[1600px]">
            <Page key={hash} />
          </div>
        </main>
      </div>
      </TelemetryProvider>
    </SettingsProvider>
  )
}

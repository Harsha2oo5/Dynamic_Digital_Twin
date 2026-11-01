import { useEffect, useState } from 'react'
import { NAV, Sidebar, type PageId } from './components/Sidebar'
import { SettingsProvider } from './hooks/useSettings'
import { DatasetManagement } from './pages/DatasetManagement'
import { LiveMonitoring } from './pages/LiveMonitoring'
import { ModelManagement } from './pages/ModelManagement'
import { ObjectInventory } from './pages/ObjectInventory'
import { Overview } from './pages/Overview'
import { Settings } from './pages/Settings'

const PAGES: Record<PageId, () => React.JSX.Element> = {
  overview: Overview,
  live: LiveMonitoring,
  inventory: ObjectInventory,
  models: ModelManagement,
  dataset: DatasetManagement,
  settings: Settings,
}

function pageFromHash(): PageId {
  const id = window.location.hash.slice(1) as PageId
  return NAV.some((n) => n.id === id) ? id : 'overview'
}

export default function App() {
  const [page, setPage] = useState<PageId>(pageFromHash)

  useEffect(() => {
    const onHash = () => setPage(pageFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const Page = PAGES[page]
  return (
    <SettingsProvider>
      <div className="flex h-full flex-col md:flex-row">
        <Sidebar page={page} onNavigate={(p) => (window.location.hash = p)} />
        <main className="min-w-0 flex-1 overflow-y-auto px-4 py-5 md:px-6">
          <div className="mx-auto max-w-[1600px]">
            <Page />
          </div>
        </main>
      </div>
    </SettingsProvider>
  )
}

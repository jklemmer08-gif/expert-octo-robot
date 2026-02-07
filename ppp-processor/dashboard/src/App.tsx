import { useEffect, useState } from 'react'
import { JobQueue } from './components/JobQueue'
import { QAReview } from './components/QAReview'
import { WorkerStatus } from './components/WorkerStatus'
import { CostTracker } from './components/CostTracker'
import { api } from './api/client'

type Tab = 'jobs' | 'qa' | 'workers' | 'cost'

export default function App() {
  const [tab, setTab] = useState<Tab>('jobs')
  const [health, setHealth] = useState<string>('checking...')

  useEffect(() => {
    api.health()
      .then(() => setHealth('connected'))
      .catch(() => setHealth('offline'))
  }, [])

  const tabs: { id: Tab; label: string }[] = [
    { id: 'jobs', label: 'Job Queue' },
    { id: 'qa', label: 'QA Review' },
    { id: 'workers', label: 'Workers' },
    { id: 'cost', label: 'Budget' },
  ]

  return (
    <div className="min-h-screen">
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold">PPP Processor Dashboard</h1>
          <span className={`text-sm px-2 py-1 rounded ${
            health === 'connected' ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
          }`}>
            API: {health}
          </span>
        </div>
        <nav className="flex gap-4 mt-4">
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                tab === t.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="p-6">
        {tab === 'jobs' && <JobQueue />}
        {tab === 'qa' && <QAReview />}
        {tab === 'workers' && <WorkerStatus />}
        {tab === 'cost' && <CostTracker />}
      </main>
    </div>
  )
}

import { useEffect, useState } from 'react'
import { api, SystemStatus } from '../api/client'

const BUDGET_TOTAL = 75.0

export function CostTracker() {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.status()
      .then(setStatus)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <p className="text-gray-400">Loading budget data...</p>
  if (!status) return <p className="text-red-400">Failed to load data</p>

  const totalCost = status.cost.total_cost
  const remaining = BUDGET_TOTAL - totalCost
  const pct = Math.min(100, (totalCost / BUDGET_TOTAL) * 100)

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">RunPod Budget Tracker</h2>

      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <div className="flex justify-between text-sm text-gray-400 mb-2">
          <span>Spent: ${totalCost.toFixed(2)}</span>
          <span>Budget: ${BUDGET_TOTAL.toFixed(2)}</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-4">
          <div
            className={`h-4 rounded-full transition-all ${
              pct > 90 ? 'bg-red-500' : pct > 70 ? 'bg-yellow-500' : 'bg-green-500'
            }`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="text-center mt-2">
          <span className="text-2xl font-bold">${remaining.toFixed(2)}</span>
          <span className="text-gray-400 ml-1">remaining</span>
        </div>
      </div>

      {Object.keys(status.cost.by_gpu).length > 0 && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Cost by GPU</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-500 border-b border-gray-700">
                <th className="pb-2">GPU</th>
                <th className="pb-2">Cost</th>
                <th className="pb-2">Hours</th>
                <th className="pb-2">Jobs</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(status.cost.by_gpu).map(([gpu, info]) => (
                <tr key={gpu} className="border-b border-gray-800">
                  <td className="py-2">{gpu}</td>
                  <td className="py-2">${info.cost.toFixed(2)}</td>
                  <td className="py-2">{info.hours.toFixed(1)}h</td>
                  <td className="py-2">{info.jobs}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="bg-gray-800 rounded-lg p-4 mt-6">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Queue Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(status.queue)
            .filter(([k]) => k !== 'total' && k !== 'avg_time_sec')
            .map(([key, val]) => (
              <div key={key} className="text-center">
                <span className="text-2xl font-bold">{val}</span>
                <span className="block text-xs text-gray-500 mt-1">{key}</span>
              </div>
            ))}
        </div>
      </div>
    </div>
  )
}

import { useEffect, useState } from 'react'
import { api, Worker } from '../api/client'

export function WorkerStatus() {
  const [workers, setWorkers] = useState<Worker[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.getWorkers()
      .then(setWorkers)
      .catch(console.error)
      .finally(() => setLoading(false))

    const interval = setInterval(() => {
      api.getWorkers().then(setWorkers).catch(console.error)
    }, 10000)

    return () => clearInterval(interval)
  }, [])

  if (loading) return <p className="text-gray-400">Loading workers...</p>

  const local = workers.filter(w => w.worker_type === 'local_gpu')
  const cloud = workers.filter(w => w.worker_type === 'cloud')

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">Worker Status</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 className="text-sm font-medium text-gray-400 mb-3">Local GPU Workers</h3>
          {local.length === 0 ? (
            <div className="bg-gray-800 rounded-lg p-4 text-gray-500">No local workers registered</div>
          ) : (
            <div className="space-y-3">
              {local.map(w => (
                <WorkerCard key={w.id} worker={w} />
              ))}
            </div>
          )}
        </div>

        <div>
          <h3 className="text-sm font-medium text-gray-400 mb-3">Cloud Workers</h3>
          {cloud.length === 0 ? (
            <div className="bg-gray-800 rounded-lg p-4 text-gray-500">No cloud pods active</div>
          ) : (
            <div className="space-y-3">
              {cloud.map(w => (
                <WorkerCard key={w.id} worker={w} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function WorkerCard({ worker }: { worker: Worker }) {
  const statusColor = {
    idle: 'bg-gray-600',
    busy: 'bg-blue-600',
    offline: 'bg-red-600',
    error: 'bg-red-800',
  }[worker.status] || 'bg-gray-600'

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${statusColor}`} />
          <span className="font-medium text-sm">{worker.id}</span>
        </div>
        <span className="text-xs text-gray-400">{worker.status}</span>
      </div>

      <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-gray-400">
        <div>
          <span className="block text-gray-500">GPU</span>
          {worker.gpu_name || 'N/A'}
        </div>
        <div>
          <span className="block text-gray-500">Jobs</span>
          {worker.jobs_completed}
        </div>
        <div>
          <span className="block text-gray-500">Cost</span>
          ${worker.total_cost.toFixed(2)}
        </div>
      </div>

      {worker.gpu_utilization != null && (
        <div className="mt-2">
          <div className="flex justify-between text-xs text-gray-500">
            <span>GPU</span>
            <span>{worker.gpu_utilization}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-1.5 mt-1">
            <div
              className="bg-green-500 h-1.5 rounded-full"
              style={{ width: `${worker.gpu_utilization}%` }}
            />
          </div>
        </div>
      )}
    </div>
  )
}

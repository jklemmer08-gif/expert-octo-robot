import { useEffect, useState } from 'react'
import { api, Job } from '../api/client'

const STATUS_COLORS: Record<string, string> = {
  pending: 'bg-yellow-900 text-yellow-300',
  processing: 'bg-blue-900 text-blue-300',
  sampling: 'bg-purple-900 text-purple-300',
  sample_ready: 'bg-purple-800 text-purple-200',
  approved: 'bg-green-900 text-green-300',
  completed: 'bg-green-800 text-green-200',
  failed: 'bg-red-900 text-red-300',
  rejected: 'bg-red-800 text-red-200',
  skipped: 'bg-gray-700 text-gray-400',
}

export function JobQueue() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [filter, setFilter] = useState<string>('')
  const [loading, setLoading] = useState(true)

  const fetchJobs = () => {
    setLoading(true)
    api.getJobs(filter || undefined)
      .then(setJobs)
      .catch(console.error)
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchJobs() }, [filter])

  const statuses = ['', 'pending', 'processing', 'sampling', 'sample_ready',
    'approved', 'completed', 'failed', 'rejected', 'skipped']

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Job Queue</h2>
        <div className="flex gap-2">
          <select
            value={filter}
            onChange={e => setFilter(e.target.value)}
            className="bg-gray-700 text-sm rounded px-3 py-1"
          >
            {statuses.map(s => (
              <option key={s} value={s}>{s || 'All'}</option>
            ))}
          </select>
          <button
            onClick={fetchJobs}
            className="bg-blue-600 text-sm px-3 py-1 rounded hover:bg-blue-500"
          >
            Refresh
          </button>
          <button
            onClick={() => api.scanLibrary().then(fetchJobs)}
            className="bg-green-700 text-sm px-3 py-1 rounded hover:bg-green-600"
          >
            Scan Library
          </button>
        </div>
      </div>

      {loading ? (
        <p className="text-gray-400">Loading...</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-400 border-b border-gray-700">
                <th className="pb-2">Status</th>
                <th className="pb-2">Title</th>
                <th className="pb-2">Model</th>
                <th className="pb-2">Tier</th>
                <th className="pb-2">Progress</th>
                <th className="pb-2">Time</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map(job => (
                <tr key={job.id} className="border-b border-gray-800 hover:bg-gray-800/50">
                  <td className="py-2">
                    <span className={`px-2 py-0.5 rounded text-xs ${STATUS_COLORS[job.status] || 'bg-gray-700'}`}>
                      {job.status}
                    </span>
                  </td>
                  <td className="py-2 max-w-xs truncate" title={job.title || ''}>
                    {job.title || job.source_path.split('/').pop()}
                  </td>
                  <td className="py-2 text-gray-400">{job.model}</td>
                  <td className="py-2 text-gray-400">{job.tier}</td>
                  <td className="py-2">
                    {job.progress != null ? (
                      <div className="w-24 bg-gray-700 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full transition-all"
                          style={{ width: `${job.progress}%` }}
                        />
                      </div>
                    ) : (
                      <span className="text-gray-500">-</span>
                    )}
                  </td>
                  <td className="py-2 text-gray-400">
                    {job.processing_time_sec
                      ? `${(job.processing_time_sec / 60).toFixed(1)}m`
                      : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {jobs.length === 0 && (
            <p className="text-gray-500 text-center py-8">No jobs found</p>
          )}
        </div>
      )}
    </div>
  )
}

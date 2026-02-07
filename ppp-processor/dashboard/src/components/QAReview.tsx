import { useEffect, useState } from 'react'
import { api, QASample } from '../api/client'

export function QAReview() {
  const [samples, setSamples] = useState<QASample[]>([])
  const [loading, setLoading] = useState(true)
  const [notes, setNotes] = useState<Record<string, string>>({})

  const fetchSamples = () => {
    setLoading(true)
    api.getPendingQA()
      .then(setSamples)
      .catch(console.error)
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchSamples() }, [])

  const handleApprove = async (sampleId: string) => {
    await api.approveQA(sampleId, notes[sampleId])
    fetchSamples()
  }

  const handleReject = async (sampleId: string) => {
    await api.rejectQA(sampleId, notes[sampleId])
    fetchSamples()
  }

  if (loading) return <p className="text-gray-400">Loading QA samples...</p>

  if (samples.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-400 text-lg">No pending QA reviews</p>
        <p className="text-gray-500 text-sm mt-2">Samples will appear here when they need human review</p>
      </div>
    )
  }

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">QA Review ({samples.length} pending)</h2>

      <div className="space-y-6">
        {samples.map(sample => (
          <div key={sample.id} className="bg-gray-800 rounded-lg p-4">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="font-medium">{sample.title || sample.job_id}</h3>
                <div className="flex gap-4 mt-2 text-sm text-gray-400">
                  <span>SSIM: <strong className={
                    (sample.ssim ?? 0) >= 0.85 ? 'text-green-400' : 'text-yellow-400'
                  }>{sample.ssim?.toFixed(3) ?? 'N/A'}</strong></span>
                  <span>PSNR: <strong className={
                    (sample.psnr ?? 0) >= 28 ? 'text-green-400' : 'text-yellow-400'
                  }>{sample.psnr?.toFixed(1) ?? 'N/A'}</strong></span>
                  <span>Sharpness: <strong>{sample.sharpness?.toFixed(1) ?? 'N/A'}</strong></span>
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => handleApprove(sample.id)}
                  className="bg-green-700 px-4 py-2 rounded text-sm hover:bg-green-600"
                >
                  Approve
                </button>
                <button
                  onClick={() => handleReject(sample.id)}
                  className="bg-red-700 px-4 py-2 rounded text-sm hover:bg-red-600"
                >
                  Reject
                </button>
              </div>
            </div>

            <div className="mt-3 grid grid-cols-2 gap-4">
              <div className="bg-gray-900 rounded p-3">
                <p className="text-xs text-gray-500 mb-1">Original Sample</p>
                <p className="text-sm text-gray-400 truncate">{sample.original_sample_path}</p>
              </div>
              <div className="bg-gray-900 rounded p-3">
                <p className="text-xs text-gray-500 mb-1">Upscaled Sample</p>
                <p className="text-sm text-gray-400 truncate">{sample.sample_path}</p>
              </div>
            </div>

            <input
              type="text"
              placeholder="Review notes (optional)"
              value={notes[sample.id] || ''}
              onChange={e => setNotes({ ...notes, [sample.id]: e.target.value })}
              className="mt-3 w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm"
            />
          </div>
        ))}
      </div>
    </div>
  )
}

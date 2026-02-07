/**
 * Typed API client for PPP Processor backend.
 */

const BASE = '/api'

interface Job {
  id: string
  title: string | null
  source_path: string
  output_path: string | null
  tier: string
  model: string | null
  scale: number | null
  is_vr: boolean
  status: string
  priority: number
  progress: number | null
  current_stage: string | null
  processing_time_sec: number | null
  created_at: string | null
}

interface QASample {
  id: string
  job_id: string
  ssim: number | null
  psnr: number | null
  sharpness: number | null
  auto_approved: boolean | null
  human_approved: boolean | null
  sample_path: string | null
  original_sample_path: string | null
  title?: string
}

interface Worker {
  id: string
  worker_type: string
  status: string
  current_job_id: string | null
  gpu_name: string | null
  gpu_utilization: number | null
  jobs_completed: number
  total_cost: number
}

interface SystemStatus {
  queue: Record<string, number>
  cost: {
    total_cost: number
    by_gpu: Record<string, { cost: number; hours: number; jobs: number }>
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!resp.ok) throw new Error(`API ${resp.status}: ${await resp.text()}`)
  return resp.json()
}

export const api = {
  health: () => request<{ status: string }>('/health'),

  status: () => request<SystemStatus>('/status'),

  getJobs: (status?: string, limit = 100) => {
    const params = new URLSearchParams({ limit: String(limit) })
    if (status) params.set('status', status)
    return request<Job[]>(`/jobs?${params}`)
  },

  getJob: (id: string) => request<Job>(`/jobs/${id}`),

  createJob: (data: { source_path: string; model?: string; scale?: number }) =>
    request<Job>('/jobs', { method: 'POST', body: JSON.stringify(data) }),

  getPendingQA: () => request<QASample[]>('/qa/pending'),

  approveQA: (sampleId: string, notes?: string) =>
    request<{ status: string }>(`/qa/${sampleId}/approve`, {
      method: 'POST',
      body: JSON.stringify({ notes }),
    }),

  rejectQA: (sampleId: string, notes?: string) =>
    request<{ status: string }>(`/qa/${sampleId}/reject`, {
      method: 'POST',
      body: JSON.stringify({ notes }),
    }),

  getWorkers: () => request<Worker[]>('/workers'),

  scanLibrary: () =>
    request<{ status: string; new_jobs: number }>('/library/scan', { method: 'POST' }),
}

export type { Job, QASample, Worker, SystemStatus }
